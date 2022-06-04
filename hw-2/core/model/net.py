import torch
import torch.nn as nn

from core.model.sequence_encoder import BiGRUEncoder, SelfAttentionEncoder
from util.misc import batch_concat_features


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm=True):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_1d = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear_block(x)
        if self.batch_norm:
            x = self.batch_norm_1d(x)
        return x


class LibriPhoneNet(nn.Module):
    def __init__(self, phone_dim, output_dim=41, **phone_net_kwargs):
        super().__init__()

        self.phone_dim = phone_dim
        self.output_dim = output_dim
        self.feature_encode_mode = phone_net_kwargs.get('feature_encode_mode')

        if self.feature_encode_mode == 'bi_gru':
            self.bi_gru_encoder = BiGRUEncoder(phone_dim, phone_net_kwargs.get('bi_gru_output_dim'), batch_first=True)
            self.phone_dim = phone_net_kwargs.get('bi_gru_output_dim')
        elif self.feature_encode_mode == 'self_attention':
            self.self_attention_encoder = SelfAttentionEncoder()
        elif self.feature_encode_mode == 'frame_concat':
            self.frame_window_size = phone_net_kwargs.get('frame_window_size')
            self.phone_dim = phone_dim * self.frame_window_size
        else:
            raise ValueError(f'feature encode mode: {self.feature_encode_mode}, does\'t support')

        self.linear_layers = nn.Sequential(
            LinearBlock(self.phone_dim, 256),
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            LinearBlock(64, output_dim)
        )

    def forward(self, batch_padded_sequences, sequences_len):
        # sequences_len.shape (B)
        batch_outputs = batch_padded_sequences
        if self.feature_encode_mode == 'bi_gru':
            batch_outputs = self.bi_gru_encoder(batch_padded_sequences, sequences_len)
        elif self.feature_encode_mode == 'frame_concat':
            batch_outputs = batch_concat_features(batch_padded_sequences, sequences_len, self.frame_window_size)

        # batch_outputs.shape (B, T, D)
        max_seq_len = batch_outputs.size(1)

        batch_seq_predictions = list()
        for i, output in enumerate(batch_outputs):
            cur_seq_len = sequences_len[i]
            predictions = self.linear_layers(output[:cur_seq_len])
            batch_seq_predictions.append(
                torch.cat([predictions, torch.zeros((max_seq_len - cur_seq_len, self.output_dim))]))

        # batch_seq_predictions.shape (B, T, output_dim)
        return torch.stack(batch_seq_predictions)
