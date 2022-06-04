import torch


def reverse_padded_sequence(sequences, sequences_len, batch_first=False):
    # when batch_first = False -> sequences (T, B, D)
    if not batch_first:
        # (T, B, D) -> (B, T, D)
        sequences = sequences.transpose(0, 1)

    batch_size = sequences.shape[0]
    max_seq_len = sequences.shape[1]
    seq_embedding_depth = sequences.shape[2]

    assert batch_size == len(sequences_len)

    reverse_indices = [list(range(max_seq_len)) for _ in range(batch_size)]

    for i, seq_len in enumerate(sequences_len):
        if seq_len > 0:
            reverse_indices[i][:seq_len] = reverse_indices[i][seq_len - 1::-1]

    reverse_indices = torch.LongTensor(reverse_indices).unsqueeze(2).expand_as(sequences).to(sequences.device)

    reverse_sequences = torch.gather(sequences, 1, reverse_indices)

    if not batch_first:
        reverse_sequences = reverse_sequences.transpose(0, 1)

    return reverse_sequences


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_features(x, frame_window_size):
    assert frame_window_size % 2 == 1  # n must be odd
    if frame_window_size < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, frame_window_size)
    x = x.view(seq_len, frame_window_size, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (frame_window_size // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, frame_window_size * feature_dim)


def batch_concat_features(batch_padded_sequences, sequences_len, frame_window_size):
    concat_record = list()
    batch_size = batch_padded_sequences.size(0)
    max_seq_len = batch_padded_sequences.size(1)
    output_dim = batch_padded_sequences.size(2)

    for i in range(batch_size):
        cur_padded_sequences = batch_padded_sequences[i]

        cur_seq_len = sequences_len[i]
        cur_sequences = cur_padded_sequences[:cur_seq_len]

        cur_concat_features = concat_features(cur_sequences, frame_window_size)

        concat_record.append(torch.cat([cur_concat_features, torch.zeros((max_seq_len - cur_seq_len, output_dim))]))

    return torch.stack(concat_record)
