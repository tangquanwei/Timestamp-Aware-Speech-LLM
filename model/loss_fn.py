import torch

def get_loss_fn(
    lambda_param=2,
    device="npu",
    num_token=501,
):
    # 151646 first timestamp token
    timestamp_token_ids = [151646 + i for i in range(num_token)]
    timestamp_token_ids_tensor = torch.tensor(timestamp_token_ids, device=device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-9527)

    def joint_loss(outputs, labels, lambda_param):
        # (batch_size, seq_len, vocab_size)
        logits = outputs.logits
        # (batch_size * seq_len, vocab_size)
        flat_logits = logits.view(-1, logits.size(-1))
        # (batch_size * seq_len)
        flat_labels = labels.view(-1)

        is_timestamp_token_mask = torch.isin(flat_labels, timestamp_token_ids_tensor)

        is_text_token_mask = ~is_timestamp_token_mask & (flat_labels != -9527)

        text_logits = flat_logits[is_text_token_mask]
        text_labels = flat_labels[is_text_token_mask]

        timestamp_logits = flat_logits[is_timestamp_token_mask]
        timestamp_labels = flat_labels[is_timestamp_token_mask]

        if text_labels.numel() > 0:
            loss_text = criterion(text_logits, text_labels)
        else:
            loss_text = torch.tensor(0.0, device=device)
        if timestamp_labels.numel() > 0:
            loss_timestamp = criterion(timestamp_logits, timestamp_labels)
        else:
            loss_timestamp = torch.tensor(0.0, device=device)
        loss_total = loss_text + lambda_param * loss_timestamp
        return loss_total

    return joint_loss