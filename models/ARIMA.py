import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
import warnings
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    ARIMA: AutoRegressive Integrated Moving Average
    
    Traditional statistical model for time series forecasting.
    Best suited for univariate forecasting tasks.
    
    Parameters:
    - p: order of autoregressive part (default: 5)
    - d: degree of differencing (default: 1)  
    - q: order of moving average part (default: 0)
    
    Note: This wraps statsmodels ARIMA in PyTorch nn.Module interface
    for compatibility with the framework.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # ARIMA hyperparameters
        self.p = getattr(configs, 'arima_p', 5)  # AR order
        self.d = getattr(configs, 'arima_d', 1)  # Differencing order
        self.q = getattr(configs, 'arima_q', 0)  # MA order
        
        # Store configs
        self.configs = configs
        self.models = {}  # Store fitted models for each channel
        
        # Check task compatibility
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            raise ValueError(f"ARIMA only supports forecasting tasks, got {self.task_name}")
        
        if configs.features == 'M':
            print(f"⚠️  ARIMA is univariate model. For multivariate ({configs.enc_in} channels), "
                  f"we will fit separate ARIMA for each channel.")
    
    def _fit_and_predict(self, history, pred_len):
        """
        Fit ARIMA model and make predictions
        
        Args:
            history: numpy array of shape (seq_len,)
            pred_len: number of steps to forecast
            
        Returns:
            predictions: numpy array of shape (pred_len,)
        """
        try:
            # Fit ARIMA model
            model = ARIMA_Model(history, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            
            # Make predictions
            forecast = model_fit.forecast(steps=pred_len)
            
            return forecast
        except Exception as e:
            # If ARIMA fails, return simple naive forecast (repeat last value)
            print(f"ARIMA fitting failed: {e}. Using naive forecast.")
            return np.full(pred_len, history[-1])
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting with ARIMA
        
        For each sample and each channel, fit ARIMA and predict
        """
        batch_size, seq_len, n_channels = x_enc.shape
        
        # Convert to numpy
        x_enc_np = x_enc.detach().cpu().numpy()
        
        # Initialize output
        predictions = np.zeros((batch_size, self.pred_len, n_channels))
        
        # Fit ARIMA for each sample and each channel
        for b in range(batch_size):
            for c in range(n_channels):
                history = x_enc_np[b, :, c]
                pred = self._fit_and_predict(history, self.pred_len)
                predictions[b, :, c] = pred
        
        # Convert back to torch tensor
        output = torch.from_numpy(predictions).float().to(x_enc.device)
        
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass - route to appropriate task
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # For forecasting, we need full sequence
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, pred_len, n_channels]
        
        return None


class ModelLite(Model):
    """
    Lightweight ARIMA with simplified parameters for faster fitting
    Good for quick experiments and large datasets
    """
    def __init__(self, configs):
        # Override with lighter defaults
        configs.arima_p = getattr(configs, 'arima_p', 2)
        configs.arima_d = getattr(configs, 'arima_d', 1)
        configs.arima_q = getattr(configs, 'arima_q', 0)
        super().__init__(configs)

