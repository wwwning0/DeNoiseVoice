# train.py
import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset import ValentiniDataset, collate_fn

def si_snr(estimated, reference):
    reference = reference - torch.mean(reference, dim=1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=1, keepdim=True)
    alpha = torch.sum(reference * estimated, dim=1, keepdim=True) / (ref_energy + 1e-8)
    proj = alpha * reference
    noise = estimated - proj
    si_snr = 10 * torch.log10(
        (torch.sum(proj ** 2, dim=1) + 1e-8) / (torch.sum(noise ** 2, dim=1) + 1e-8)
    )
    return torch.mean(si_snr)


try:
    from dc_crn import DCCRN as CustomDCCRN
except ImportError as e:
    print(f"Error importing DCCRN from dccrn.py: {e}")
    print("Please ensure dccrn.py, conv_stft.py, and complexnn.py are in the same directory or in your Python path.")
    print("Alternatively, copy the necessary class and function definitions directly into this file.")
    sys.exit(1)

class DCCRN_CTC(nn.Module):
    def __init__(self, vocab_size=29):
        super().__init__()

        self.enhancer = CustomDCCRN(
            rnn_units=256,      # 示例参数，可调整
            masking_mode='E',   # 示例参数，可调整
        )

        self.asr_encoder = Wav2Vec2Model.from_pretrained(
            os.path.join(current_dir, "..", "models", "wav2vec2-base-960h"),
            local_files_only=True
        )
        for p in self.asr_encoder.parameters():
            p.requires_grad = False
        self.ctc_head = nn.Linear(768, vocab_size)

    def forward(self, x):
        # x: (B, T)
        _, enhanced = self.enhancer(x) # DCCRN forward 返回 [spec, wav]，取 wav 部分

        with torch.no_grad():
            feats = self.asr_encoder(enhanced).last_hidden_state  # (B, T, 768)
        logits = self.ctc_head(feats)  # (B, T, V)
        return enhanced, logits



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    csv_path = os.path.join(current_dir, ".", "train.csv")
    assert os.path.exists(csv_path), f"train.csv not found at {csv_path}"

    model = DCCRN_CTC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        ValentiniDataset(csv_path, sample_rate=16000),
        batch_size=15,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_idx, (noisy, clean, targets) in enumerate(train_loader):
            noisy, clean, targets = noisy.to(device), clean.to(device), targets.to(device)
            enhanced, logits = model(noisy)

            # 获取两个张量的长度
            min_len = min(enhanced.size(1), clean.size(1))
            # 裁剪到相同长度
            enhanced_truncated = enhanced[:, :min_len]
            clean_truncated = clean[:, :min_len]

            # SI-SNR loss (使用裁剪后的张量)
            si_snr_loss = -si_snr(enhanced_truncated, clean_truncated)

            # CTC loss
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
    
            with torch.no_grad():
                feats_for_length = model.asr_encoder(enhanced_truncated).last_hidden_state
            wav2vec2_output_len = feats_for_length.size(1)
            input_lengths = torch.full((logits.size(0),), wav2vec2_output_len, dtype=torch.long, device=device)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T_enh_feats_padded, B, V)
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device) # [T_enh_feats_padded, ] * B
            target_lengths = (targets != 0).sum(dim=1) # [len_target_1, len_target_2, ...]
            ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0)


            loss = si_snr_loss + 1 * ctc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), f"dccrn_ctc_epoch_{epoch}.pth")
