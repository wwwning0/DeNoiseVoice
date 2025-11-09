# train.py
import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio  # 用于 GPU 梅尔谱计算
import numpy as np  # 用于 STOI 计算

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset import ValentiniDataset, collate_fn

# --- STOI Loss Function (GPU 优化版，使用梅尔倒谱近似) ---
def stoi_loss_gpu(enhanced, clean, sr=16000, n_mels=40, n_fft=1024, hop_length=256):
    """
    使用梅尔倒谱近似计算 STOI 损失（GPU 版）
    STOI 越高越好，所以损失是 -STOI
    由于直接计算 STOI 梯度困难，这里用梅尔倒谱 L1 距离作为替代
    """
    # 创建梅尔频谱变换器
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(enhanced.device)

    # 计算梅尔谱 (B, n_mels, T_mel)
    enh_mel = mel_transform(enhanced)
    clean_mel = mel_transform(clean)

    # 转 dB
    enh_db = 10 * torch.log10(enh_mel + 1e-9)
    clean_db = 10 * torch.log10(clean_mel + 1e-9)

    # L1 损失（近似 STOI 优化目标）
    # STOI 与梅尔倒谱距离负相关，所以损失越小，STOI 越高
    return F.l1_loss(enh_db, clean_db)

# --- 模型定义 ---
from asteroid.models import ConvTasNet

class ConvTasNet_CTC(nn.Module):
    def __init__(self, vocab_size=29):
        super().__init__()
        # 使用 ConvTasNet 替代 DCCRN
        self.enhancer = ConvTasNet(n_src=1, n_filters=512, n_blocks=8, n_repeats=3)

        # Wav2Vec2 ASR 编码器
        model_path = os.path.join(current_dir, "..", "models", "wav2vec2-base-960h")
        self.asr_encoder = Wav2Vec2Model.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # 冻结 ASR 编码器参数
        for p in self.asr_encoder.parameters():
            p.requires_grad = False
            
        self.ctc_head = nn.Linear(768, vocab_size)

    def forward(self, x):
        # ConvTasNet 输出 (B, 1, T) -> (B, T)
        separated = self.enhancer(x.unsqueeze(1))  # 输入 (B, T) -> (B, 1, T)
        enhanced = separated.squeeze(1)  # (B, 1, T) -> (B, T)

        # ASR 特征提取
        with torch.no_grad():
            feats = self.asr_encoder(enhanced).last_hidden_state  # (B, T, 768)
        logits = self.ctc_head(feats)  # (B, T, V)
        
        return enhanced, logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 使用 9 秒过滤后的 CSV
    csv_path = os.path.join(current_dir, "train_filtered.csv")
    assert os.path.exists(csv_path), f"Please generate {csv_path} first."

    model = ConvTasNet_CTC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 注意：这里不使用 ReduceLROnPlateau，因为 loss 可能不稳定，建议手动调整 lr

    train_loader = DataLoader(
        ValentiniDataset(csv_path, sample_rate=16000),
        batch_size=2,  # 可根据显存调整
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 避免多进程 DataLoader 在 Linux 上报错
        pin_memory=True
    )

    # --- 新的损失权重 ---
    stoi_weight = 0.5
    ctc_weight = 1.0

    for epoch in range(60):
        model.train()
        total_epoch_loss = 0
        valid_batches = 0

        for batch_idx, (noisy, clean, targets, lengths) in enumerate(train_loader):
            noisy, clean, targets = noisy.to(device), clean.to(device), targets.to(device)

            try:
                enhanced, logits = model(noisy)

                # 长度对齐
                min_len = min(enhanced.size(1), clean.size(1))
                enhanced_truncated = enhanced[:, :min_len]
                clean_truncated = clean[:, :min_len]

                # --- 新的损失函数 ---
                # 1. STOI Loss (近似)
                stoi_loss_val = stoi_loss_gpu(enhanced_truncated, clean_truncated)
                
                # 2. CTC Loss
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
                input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
                target_lengths = (targets != 0).sum(dim=1)
                ctc_loss_val = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)

                # 总损失
                loss = stoi_weight * stoi_loss_val + ctc_weight * ctc_loss_val

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_epoch_loss += loss.item()
                valid_batches += 1
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.4f} "
                          f"(STOI: {stoi_loss_val.item():.4f}, CTC: {ctc_loss_val.item():.4f})")

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"WARNING: CUDA OOM at batch {batch_idx}. Skipping.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        if valid_batches > 0:
            avg_loss = total_epoch_loss / valid_batches
            print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch} finished. No valid batches.")

        if (epoch + 1) % 20 == 0:
            model_path = f"conv_tasnet_ctc_stoi_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")