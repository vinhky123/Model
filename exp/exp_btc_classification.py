import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from data_provider.btc_dataset import BTCDataset
from models.TimesNet import Model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")


class Exp_BTC_Classification(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device("cuda:0")
            print("Use GPU: cuda:0")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = TimesNet(
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            enc_in=self.args.enc_in,
            dec_in=self.args.dec_in,
            c_out=2,  # Binary classification
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_layers=self.args.d_layers,
            d_ff=self.args.d_ff,
            moving_avg=self.args.moving_avg,
            factor=self.args.factor,
            distil=self.args.distil,
            dropout=self.args.dropout,
            embed=self.args.embed,
            activation=self.args.activation,
            output_attention=False,
            channel_independence=self.args.channel_independence,
            task_name="classification",
            num_class=self.args.num_class,
        )
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == "train":
            dataset = BTCDataset(
                data_path=args.data_path,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                features=args.features,
                target=args.target,
                scale=True,
            )
        else:
            dataset = BTCDataset(
                data_path=args.data_path,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                features=args.features,
                target=args.target,
                scale=True,
            )

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(flag == "train"),
            num_workers=args.num_workers,
            drop_last=True,
        )

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        valid_data, valid_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_time_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_time_x = batch_time_x.float().to(self.device)
                batch_y = batch_y.squeeze().to(self.device)

                outputs = self.model(batch_x, batch_time_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(valid_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_time_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_time_x = batch_time_x.float().to(self.device)
                batch_y = batch_y.squeeze().to(self.device)

                outputs = self.model(batch_x, batch_time_x)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_time_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_time_x = batch_time_x.float().to(self.device)
                batch_y = batch_y.squeeze().to(self.device)

                outputs = self.model(batch_x, batch_time_x)
                pred = outputs.argmax(dim=1)

                preds.append(pred.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        # Calculate metrics
        accuracy = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)

        print(
            "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
                accuracy, precision, recall, f1
            )
        )

        return accuracy, precision, recall, f1


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))
