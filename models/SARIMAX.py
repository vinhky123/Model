import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_Model
import warnings
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    SARIMAX: Seasonal AutoRegressive Integrated Moving Average with eXogenous variables
    
    Extension of ARIMA with:
    - Seasonal components (P, D, Q, s)
    - Support for exogenous variables (time features)
    
    Parameters:
    - p, d, q: Non-seasonal ARIMA orders (default: 1, 1, 1)
    - P, D, Q: Seasonal ARIMA orders (default: 1, 1, 1)
    - s: Seasonal period (default: 24 for hourly data)
    
    Note: This wraps statsmodels SARIMAX in PyTorch nn.Module interface
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # Non-seasonal orders
        self.p = getattr(configs, 'arima_p', 1)
        self.d = getattr(configs, 'arima_d', 1)
        self.q = getattr(configs, 'arima_q', 1)
        
        # Seasonal orders
        self.P = getattr(configs, 'seasonal_P', 1)
        self.D = getattr(configs, 'seasonal_D', 1)
        self.Q = getattr(configs, 'seasonal_Q', 1)
        self.s = getattr(configs, 'seasonal_period', 24)  # Default: 24 hours
        
        # Determine seasonal period from freq if not specified
        if not hasattr(configs, 'seasonal_period'):
            freq_to_period = {
                'h': 24,      # Hourly -> daily seasonality
                't': 96,      # 15-min -> daily seasonality (24*4)
                'd': 7,       # Daily -> weekly seasonality
                'w': 52,      # Weekly -> yearly seasonality
                'm': 12,      # Monthly -> yearly seasonality
            }
            self.s = freq_to_period.get(configs.freq, 24)
        
        self.use_exog = getattr(configs, 'use_exog', True)  # Use time features as exogenous
        
        self.configs = configs
        
        # Check task compatibility
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            raise ValueError(f"SARIMAX only supports forecasting tasks, got {self.task_name}")
        
        print(f"📊 SARIMAX initialized:")
        print(f"   - Order: ({self.p}, {self.d}, {self.q})")
        print(f"   - Seasonal: ({self.P}, {self.D}, {self.Q}, {self.s})")
        print(f"   - Use exogenous: {self.use_exog}")
        
        if configs.features == 'M':
            print(f"   ⚠️  SARIMAX is univariate. Fitting separate model for each of {configs.enc_in} channels.")
    
    def _fit_and_predict(self, history, exog_train=None, exog_pred=None):
        """
        Fit SARIMAX and predict
        
        Args:
            history: (seq_len,) historical values
            exog_train: (seq_len, n_features) exogenous variables for training
            exog_pred: (pred_len, n_features) exogenous variables for prediction
            
        Returns:
            forecast: (pred_len,) predictions
        """
        try:
            # Build SARIMAX model
            if self.use_exog and exog_train is not None:
                model = SARIMAX_Model(
                    history,
                    exog=exog_train,
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = SARIMAX_Model(
                    history,
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            # Fit model
            model_fit = model.fit(disp=False, maxiter=50)
            
            # Forecast
            if self.use_exog and exog_pred is not None:
                forecast = model_fit.forecast(steps=self.pred_len, exog=exog_pred)
            else:
                forecast = model_fit.forecast(steps=self.pred_len)
            
            return forecast
            
        except Exception as e:
            # Fallback to naive forecast
            print(f"SARIMAX fitting failed: {e}. Using naive seasonal forecast.")
            # Repeat last seasonal cycle
            if len(history) >= self.s:
                last_season = history[-self.s:]
                n_repeats = (self.pred_len // self.s) + 1
                forecast = np.tile(last_season, n_repeats)[:self.pred_len]
            else:
                forecast = np.full(self.pred_len, history[-1])
            return forecast
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting with SARIMAX
        """
        batch_size, seq_len, n_channels = x_enc.shape
        
        # Convert to numpy
        x_enc_np = x_enc.detach().cpu().numpy()
        
        # Exogenous variables (time features)
        exog_train = None
        exog_pred = None
        if self.use_exog and x_mark_enc is not None and x_mark_dec is not None:
            exog_train_full = x_mark_enc.detach().cpu().numpy()
            exog_pred_full = x_mark_dec.detach().cpu().numpy()[:, -self.pred_len:, :]
        
        # Initialize output
        predictions = np.zeros((batch_size, self.pred_len, n_channels))
        
        # Fit SARIMAX for each sample and channel
        for b in range(batch_size):
            for c in range(n_channels):
                history = x_enc_np[b, :, c]
                
                # Prepare exogenous variables
                if self.use_exog and x_mark_enc is not None:
                    exog_train = exog_train_full[b]
                    exog_pred = exog_pred_full[b]
                
                pred = self._fit_and_predict(history, exog_train, exog_pred)
                predictions[b, :, c] = pred
        
        # Convert back to torch
        output = torch.from_numpy(predictions).float().to(x_enc.device)
        
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, pred_len, n_channels]
        
        return None


class ModelLite(Model):
    """
    Lightweight SARIMAX with simplified parameters
    """
    def __init__(self, configs):
        # Override with lighter defaults
        configs.arima_p = getattr(configs, 'arima_p', 1)
        configs.arima_d = getattr(configs, 'arima_d', 1)
        configs.arima_q = getattr(configs, 'arima_q', 0)
        configs.seasonal_P = getattr(configs, 'seasonal_P', 1)
        configs.seasonal_D = getattr(configs, 'seasonal_D', 0)
        configs.seasonal_Q = getattr(configs, 'seasonal_Q', 0)
        super().__init__(configs)

