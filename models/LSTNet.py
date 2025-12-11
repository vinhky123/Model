import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    LSTNet: Long- and Short-term Time-series Network
    Paper: Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (SIGIR 2018)
    
    Architecture:
    - Convolutional layer: captures short-term local patterns
    - Recurrent layer (GRU): captures long-term dependencies
    - Recurrent-skip layer: captures very long-term periodic patterns
    - Autoregressive component: linear model for scale insensitive
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # Only support forecasting tasks
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            raise ValueError(f"LSTNet only supports forecasting tasks, got {self.task_name}")
        
        # Model hyperparameters
        self.hidCNN = getattr(configs, 'hidCNN', 100)  # CNN hidden units
        self.hidRNN = getattr(configs, 'hidRNN', 100)  # RNN hidden units
        self.hidSkip = getattr(configs, 'hidSkip', 5)  # Skip-RNN hidden units
        self.CNN_kernel = getattr(configs, 'CNN_kernel', 6)  # CNN kernel size
        self.skip = getattr(configs, 'skip', 24)  # Skip length for periodic patterns
        self.highway_window = getattr(configs, 'highway_window', 24)  # AR component window
        self.dropout = configs.dropout
        
        self.channels = configs.enc_in
        self.output_dim = configs.c_out
        
        # 1. Convolutional layer
        self.conv1 = nn.Conv2d(1, self.hidCNN, kernel_size=(self.CNN_kernel, self.channels))
        self.dropout_conv = nn.Dropout(p=self.dropout)
        
        # Calculate CNN output size
        self.conv_out_size = self.seq_len - self.CNN_kernel + 1
        
        # 2. Recurrent layer (GRU)
        self.GRU1 = nn.GRU(self.hidCNN, self.hidRNN, batch_first=True)
        self.dropout_GRU = nn.Dropout(p=self.dropout)
        
        # 3. Skip-Recurrent layer for long-term patterns
        if self.skip > 0:
            self.pt = (self.conv_out_size - self.skip) // self.skip
            if self.pt > 0:
                self.GRUskip = nn.GRU(self.hidCNN, self.hidSkip, batch_first=True)
                self.linear1 = nn.Linear(self.hidRNN + self.skip * self.hidSkip, self.output_dim)
            else:
                self.linear1 = nn.Linear(self.hidRNN, self.output_dim)
        else:
            self.linear1 = nn.Linear(self.hidRNN, self.output_dim)
        
        # 4. Autoregressive component (highway network)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, 1)
        
        self.output = None
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Args:
            x_enc: [batch_size, seq_len, channels]
        Returns:
            output: [batch_size, pred_len, output_dim]
        """
        batch_size = x_enc.size(0)
        
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # --- CNN Component ---
        # Input: [batch, seq_len, channels] -> [batch, 1, seq_len, channels]
        c = x_enc.unsqueeze(1)
        
        # Conv2d: [batch, 1, seq_len, channels] -> [batch, hidCNN, conv_out_size, 1]
        c = F.relu(self.conv1(c))
        c = self.dropout_conv(c)
        
        # Remove last dimension and transpose: [batch, hidCNN, conv_out_size] -> [batch, conv_out_size, hidCNN]
        c = c.squeeze(3).transpose(1, 2)
        
        # --- RNN Component ---
        # GRU: [batch, conv_out_size, hidCNN] -> [batch, conv_out_size, hidRNN]
        r, _ = self.GRU1(c)
        r = self.dropout_GRU(r)
        
        # Take last output: [batch, hidRNN]
        r = r[:, -1, :]
        
        # --- Skip-RNN Component (for periodic patterns) ---
        if self.skip > 0 and self.pt > 0:
            # Skip connection to capture long-term periodic patterns
            # Reshape: [batch, pt, skip, hidCNN]
            s = c[:, -self.pt * self.skip:, :].contiguous()
            s = s.view(batch_size, self.pt, self.skip, self.hidCNN)
            s = s.permute(0, 2, 1, 3).contiguous()
            s = s.view(batch_size * self.skip, self.pt, self.hidCNN)
            
            # Skip-GRU
            s, _ = self.GRUskip(s)
            s = s[:, -1, :]  # Take last output
            s = s.view(batch_size, self.skip * self.hidSkip)
            
            # Concatenate RNN and Skip-RNN outputs
            r = torch.cat([r, s], dim=1)
        
        # --- Fully Connected Layer ---
        res = self.linear1(r)  # [batch, output_dim]
        
        # --- Highway Component (Autoregressive) ---
        if self.highway_window > 0:
            # Take last highway_window steps for each channel
            # x_enc: [batch, seq_len, channels]
            z = x_enc[:, -self.highway_window:, :]  # [batch, highway_window, channels]
            z = z.permute(0, 2, 1).contiguous()  # [batch, channels, highway_window]
            z = self.highway(z)  # [batch, channels, 1]
            z = z.squeeze(2)  # [batch, channels]
            
            # Add highway component
            if self.output_dim == self.channels:
                res = res + z
            else:
                # If output_dim != channels, only add to corresponding dimensions
                res[:, :self.channels] = res[:, :self.channels] + z
        
        # De-normalization
        res = res * (stdev[:, 0, :].squeeze(1))
        res = res + (means[:, 0, :].squeeze(1))
        
        # Repeat for pred_len time steps
        # [batch, output_dim] -> [batch, pred_len, output_dim]
        res = res.unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return res


class ModelRecursive(nn.Module):
    """
    LSTNet with recursive prediction for multi-step forecasting
    Better for longer prediction horizons
    """
    
    def __init__(self, configs):
        super(ModelRecursive, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # Use LSTNet base model for single-step prediction
        self.base_model = Model(configs)
        self.base_model.pred_len = 1
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Recursive multi-step prediction
        """
        batch_size = x_enc.size(0)
        predictions = []
        
        # Start with input sequence
        current_seq = x_enc.clone()
        
        for _ in range(self.pred_len):
            # Predict next step
            pred = self.base_model(current_seq, None, None, None)  # [batch, 1, output_dim]
            predictions.append(pred)
            
            # Update sequence: remove first, append prediction
            current_seq = torch.cat([current_seq[:, 1:, :], pred], dim=1)
        
        # Concatenate all predictions
        output = torch.cat(predictions, dim=1)  # [batch, pred_len, output_dim]
        
        return output

