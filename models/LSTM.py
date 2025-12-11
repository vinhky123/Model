import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    LSTM Model with support for all 5 tasks:
    - Long-term/Short-term Forecasting: Encoder-Decoder LSTM with Attention
    - Imputation: Bidirectional LSTM for context-aware reconstruction
    - Anomaly Detection: LSTM Autoencoder for reconstruction
    - Classification: LSTM Encoder + Global Pooling
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        # Embedding layer for encoder
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Task-specific architecture
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            # Decoder embedding for forecasting tasks
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            # Encoder LSTM
            self.encoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
                dropout=configs.dropout if configs.e_layers > 1 else 0,
            )
            # Decoder LSTM
            self.decoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.d_layers,
                batch_first=True,
                dropout=configs.dropout if configs.d_layers > 1 else 0,
            )
            # Projection layer (with attention context)
            self.projection = nn.Linear(2 * configs.d_model, configs.c_out)
            
        elif self.task_name == 'imputation':
            # Bidirectional LSTM for better context understanding
            self.encoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
                bidirectional=True,
                dropout=configs.dropout if configs.e_layers > 1 else 0,
            )
            # Project bidirectional output back to original dimension
            self.projection = nn.Linear(2 * configs.d_model, configs.c_out)
            
        elif self.task_name == 'anomaly_detection':
            # LSTM Encoder for anomaly detection
            self.encoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
                dropout=configs.dropout if configs.e_layers > 1 else 0,
            )
            self.projection = nn.Linear(configs.d_model, configs.c_out)
            
        elif self.task_name == 'classification':
            # LSTM Encoder for classification
            self.encoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
                dropout=configs.dropout if configs.e_layers > 1 else 0,
            )
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(configs.dropout)
            # Classification head
            self.projection = nn.Linear(configs.d_model, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Encoder-Decoder LSTM with Attention for forecasting
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(enc_out)
        
        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Transfer encoder's final state to decoder
        # Repeat last layer's hidden/cell state for all decoder layers
        decoder_hidden = encoder_hidden[-1:].repeat(self.decoder.num_layers, 1, 1)
        decoder_cell = encoder_cell[-1:].repeat(self.decoder.num_layers, 1, 1)
        
        decoder_outputs, _ = self.decoder(dec_out, (decoder_hidden, decoder_cell))
        
        # Attention mechanism
        # scores: [B, dec_len, enc_len]
        scores = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)
        # context_vectors: [B, dec_len, d_model]
        context_vectors = torch.bmm(attn_weights, encoder_outputs)
        
        # Combine context and decoder output
        combined = torch.cat((context_vectors, decoder_outputs), dim=-1)
        output = self.projection(combined)
        
        # De-Normalization
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, output.shape[1], 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, output.shape[1], 1))
        
        return output

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Bidirectional LSTM for imputation task
        """
        # Normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Bidirectional LSTM
        lstm_out, _ = self.encoder(enc_out)
        
        # Project to output
        output = self.projection(lstm_out)
        
        # De-Normalization
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return output

    def anomaly_detection(self, x_enc):
        """
        LSTM Autoencoder for anomaly detection
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # LSTM Encoder
        lstm_out, _ = self.encoder(enc_out)
        
        # Project back to original space
        output = self.projection(lstm_out)
        
        # De-Normalization
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return output

    def classification(self, x_enc, x_mark_enc):
        """
        LSTM Encoder with Global Pooling for classification
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # LSTM Encoder
        lstm_out, (hidden, cell) = self.encoder(enc_out)
        
        # Use the last hidden state from the last layer
        # hidden: [num_layers, batch, d_model]
        output = hidden[-1]  # [batch, d_model]
        
        # Apply activation and dropout
        output = self.act(output)
        output = self.dropout_layer(output)
        
        # Classification
        output = self.projection(output)  # [batch, num_class]
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Unified forward pass for all tasks
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
        
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, seq_len, c_out]
        
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, seq_len, c_out]
        
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, num_class]
        
        return None
