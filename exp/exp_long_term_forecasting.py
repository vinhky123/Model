from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import csv

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Statistical models (ARIMA, SARIMAX) don't have trainable parameters
        if self.args.model in ['ARIMA', 'SARIMAX']:
            return None
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Statistical models (ARIMA, SARIMAX) don't require training
        # They fit during inference
        if self.args.model in ['ARIMA', 'SARIMAX']:
            print(f"\n⚠️  {self.args.model} is a statistical model - no training required.")
            print("Model will fit during inference on test data.\n")
            
            # Store dummy training stats for consistency
            self.train_stats = {
                'total_time': 0.0,
                'avg_epoch_time': 0.0,
                'num_epochs': 0,
                'throughput': 0.0
            }
            return self.model

        time_now = time.time()
        
        # Track training time
        train_start_time = time.time()
        epoch_times = []

        train_steps = len(train_loader)
        
        # Training speed benchmark mode: quick training for speed measurement
        if self.args.train_speed_benchmark:
            benchmark_epochs = 3  # Epoch 0: warm-up, Epoch 1-2: measurement
            print(f"\n🚀 TRAINING SPEED BENCHMARK MODE")
            print(f"   Training {benchmark_epochs} epochs (1 warm-up + 2 measurement)")
            print(f"   Skipping validation, early stopping, and testing")
            print(f"   Model: {self.args.model} | Dataset: {self.args.data} | Batch size: {self.args.batch_size}\n")
        else:
            benchmark_epochs = self.args.train_epochs
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(benchmark_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_duration = time.time() - epoch_time
            epoch_times.append(epoch_duration)
            
            # Calculate iterations per second
            iters_per_sec = train_steps / epoch_duration
            samples_per_sec = len(train_data) / epoch_duration
            
            if self.args.train_speed_benchmark:
                # In benchmark mode, show detailed speed metrics
                epoch_status = "WARM-UP" if epoch == 0 else "MEASUREMENT"
                print(f"Epoch {epoch + 1} [{epoch_status}]: {epoch_duration:.2f}s | "
                      f"{iters_per_sec:.2f} iters/s | {samples_per_sec:.2f} samples/s")
            else:
                print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, epoch_duration))
            
            train_loss = np.average(train_loss)
            
            # Skip validation and early stopping in benchmark mode
            if not self.args.train_speed_benchmark:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Calculate training statistics
        total_train_time = time.time() - train_start_time
        
        # In benchmark mode, calculate stable metrics (skip warm-up epoch)
        if self.args.train_speed_benchmark and len(epoch_times) >= 2:
            measurement_epochs = epoch_times[1:]  # Skip first epoch (warm-up)
            avg_epoch_time = np.mean(measurement_epochs)
            std_epoch_time = np.std(measurement_epochs)
            total_samples = len(train_data) * len(measurement_epochs)
            train_throughput = total_samples / sum(measurement_epochs)
            iters_per_sec = train_steps / avg_epoch_time
            
            print("\n" + "="*100)
            print("🏁 TRAINING SPEED BENCHMARK RESULTS")
            print("="*100)
            print(f"Model:              {self.args.model}")
            print(f"Dataset:            {self.args.data} ({self.args.data_path})")
            print(f"Features:           {self.args.features} mode")
            print(f"Batch size:         {self.args.batch_size}")
            print(f"Input channels:     {self.args.enc_in}")
            print(f"Sequence length:    {self.args.seq_len}")
            print(f"Prediction length:  {self.args.pred_len}")
            print("-"*100)
            print(f"Training samples:   {len(train_data)}")
            print(f"Steps per epoch:    {train_steps}")
            print(f"Total epochs:       {len(epoch_times)} (1 warm-up + {len(measurement_epochs)} measurement)")
            print("-"*100)
            print(f"Avg epoch time:     {avg_epoch_time:.3f}s ± {std_epoch_time:.3f}s")
            print(f"Throughput:         {train_throughput:.2f} samples/sec")
            print(f"                    {iters_per_sec:.2f} iterations/sec")
            print(f"Time per sample:    {(avg_epoch_time / len(train_data)) * 1000:.2f}ms")
            print(f"Time per batch:     {(avg_epoch_time / train_steps) * 1000:.2f}ms")
            print("="*100 + "\n")
            
            # Store stats
            self.train_stats = {
                'total_time': sum(measurement_epochs),
                'avg_epoch_time': avg_epoch_time,
                'std_epoch_time': std_epoch_time,
                'num_epochs': len(measurement_epochs),
                'throughput': train_throughput,
                'iters_per_sec': iters_per_sec
            }
            
            # Return early - no testing needed in benchmark mode
            return self.model
        else:
            # Normal training mode
            avg_epoch_time = np.mean(epoch_times)
            total_samples = len(train_data) * len(epoch_times)
            train_throughput = total_samples / total_train_time
            
            print("\n" + "="*80)
            print("Training Statistics:")
            print(f"  Total training time:   {total_train_time:.2f}s ({total_train_time/60:.2f}min)")
            print(f"  Number of epochs:      {len(epoch_times)}")
            print(f"  Avg time per epoch:    {avg_epoch_time:.2f}s")
            print(f"  Training throughput:   {train_throughput:.2f} samples/sec")
            print("="*80 + "\n")
            
            # Store training stats for later use in test()
            self.train_stats = {
                'total_time': total_train_time,
                'avg_epoch_time': avg_epoch_time,
                'num_epochs': len(epoch_times),
                'throughput': train_throughput
            }
            
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            return self.model

    def benchmark_inference(self, test_loader, n_warmup=10, n_test=100):
        """
        Benchmark pure inference speed (forward pass only)
        """
        import time
        
        self.model.eval()
        
        # Warm-up phase
        print("Warming up GPU...")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= n_warmup:
                    break
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking on {n_test} batches...")
        inference_times = []
        total_samples = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= n_test:
                    break
                    
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                total_samples += batch_x.size(0)
        
        inference_times = np.array(inference_times)
        
        return {
            'mean_time': inference_times.mean(),
            'std_time': inference_times.std(),
            'throughput': total_samples / inference_times.sum(),
            'latency': inference_times.sum() / total_samples * 1000,
            'total_samples': total_samples
        }
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # Statistical models don't have checkpoints to load
        if test and self.args.model not in ['ARIMA', 'SARIMAX']:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        
        # Save to text file
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # Save to CSV file for easy comparison

        csv_file = 'result_summary.csv'
        file_exists = os.path.isfile(csv_file)
        
        # Get model params
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Get dataset name (use actual name instead of "custom")
        dataset_name = self.args.data
        if dataset_name == 'custom':
            # Extract name from data_path
            data_file = os.path.basename(self.args.data_path)
            if 'electricity' in data_file.lower():
                dataset_name = 'ECL'
            elif 'weather' in data_file.lower():
                dataset_name = 'Weather'
            elif 'traffic' in data_file.lower():
                dataset_name = 'Traffic'
            elif 'exchange' in data_file.lower():
                dataset_name = 'Exchange'
            else:
                # Use filename without extension as fallback
                dataset_name = os.path.splitext(data_file)[0]
        
        # Get training stats (if available from train())
        train_time_s = 'N/A'
        avg_epoch_time_s = 'N/A'
        train_throughput = 'N/A'
        
        if hasattr(self, 'train_stats'):
            train_time_s = f'{self.train_stats["total_time"]:.2f}'
            avg_epoch_time_s = f'{self.train_stats["avg_epoch_time"]:.2f}'
            train_throughput = f'{self.train_stats["throughput"]:.2f}'
        
        # Benchmark inference speed if enabled
        inference_time_ms = 'N/A'
        inference_throughput = 'N/A'
        latency_ms = 'N/A'
        
        if hasattr(self.args, 'benchmark') and self.args.benchmark:
            print("\n" + "="*80)
            print("Benchmarking Inference Speed...")
            print("="*80)
            
            benchmark_results = self.benchmark_inference(test_loader, n_warmup=10, n_test=100)
            
            inference_time_ms = f'{benchmark_results["mean_time"]*1000:.2f}'
            inference_throughput = f'{benchmark_results["throughput"]:.2f}'
            latency_ms = f'{benchmark_results["latency"]:.2f}'
            
            print(f"\n📊 Inference Speed:")
            print(f"  Time per batch:  {benchmark_results['mean_time']*1000:.2f} ms")
            print(f"  Throughput:      {benchmark_results['throughput']:.2f} samples/sec")
            print(f"  Latency:         {benchmark_results['latency']:.2f} ms/sample")
            print("="*80 + "\n")
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'model', 'dataset', 'seq_len', 'pred_len', 
                    'mae', 'mse', 'rmse', 'mape', 'mspe',
                    'params_M', 'train_time_s', 'avg_epoch_time_s', 'train_throughput_samples_sec',
                    'inference_ms', 'inference_throughput', 'latency_ms'
                ])
            writer.writerow([
                self.args.model,
                dataset_name,
                self.args.seq_len,
                self.args.pred_len,
                f'{mae:.4f}',
                f'{mse:.4f}',
                f'{rmse:.4f}',
                f'{mape:.4f}',
                f'{mspe:.4f}',
                f'{total_params/1e6:.2f}',
                train_time_s,
                avg_epoch_time_s,
                train_throughput,
                inference_time_ms,
                inference_throughput,
                latency_ms
            ])

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
