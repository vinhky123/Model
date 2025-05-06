import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, configs):
        super(LSTMModel, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.encoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
            )
            self.decoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=configs.d_layers,
                batch_first=True,
            )
            self.projection = nn.Linear(2 * configs.d_model, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(enc_out)

        decoder_hidden = encoder_hidden[-1:].repeat(self.decoder.num_layers, 1, 1)
        decoder_cell = encoder_cell[-1:].repeat(self.decoder.num_layers, 1, 1)

        decoder_outputs, _ = self.decoder(dec_out, (decoder_hidden, decoder_cell))

        # Attention
        scores = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)
        context_vectors = torch.bmm(attn_weights, encoder_outputs)

        combined = torch.cat((context_vectors, decoder_outputs), dim=-1)
        output = self.projection(combined)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        return None
