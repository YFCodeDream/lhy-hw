import torch
import torch.nn as nn

from util.misc import reverse_padded_sequence


class BiGRUEncoder(nn.Module):
    def __init__(self, phone_dim, bi_gru_output_dim, batch_first=False):
        super().__init__()
        self.forward_gru = nn.GRU(phone_dim, bi_gru_output_dim // 2, batch_first=batch_first)
        self.backward_gru = nn.GRU(phone_dim, bi_gru_output_dim // 2, batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, batch_padded_sequences, sequences_len):
        # batch_padded_sequences -> (B, T, D)
        if not self.batch_first:
            # need (T, B, D) -> GRU(batch_first=False)
            # so, (B, T, D) -> (T, B, D)
            batch_padded_sequences = batch_padded_sequences.transpose(0, 1)

        forward_outputs, forward_hidden = self.forward_gru(batch_padded_sequences)
        backward_batch_padded_sequences = reverse_padded_sequence(batch_padded_sequences,
                                                                  sequences_len,
                                                                  batch_first=self.batch_first)
        backward_outputs, output_hidden = self.backward_gru(backward_batch_padded_sequences)
        backward_outputs = reverse_padded_sequence(backward_outputs, sequences_len, batch_first=self.batch_first)

        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)

        # output shape: (B, T, output_dim)

        return outputs


class SelfAttentionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
