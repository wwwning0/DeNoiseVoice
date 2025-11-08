import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset import ValentiniDataset, collate_fn
from dc_crn import DCCRN as CustomDCCRN

def si_snr(estimated, reference):
    eps = 1e-8
    min_len = min(estimated.size(1), reference.size(1))
    estimated = estimated[:, :min_len]
    reference = reference[:, :min_len]
    
    reference_mean = torch.mean(reference, dim=1, keepdim=True)
    estimated_mean = torch.mean(estimated, dim=1, keepdim=True)
    
    reference = reference - reference_mean
    estimated = estimated - estimated_mean
    
    ref_energy = torch.sum(reference ** 2, dim=1, keepdim=True) + eps
    proj = torch.sum(reference * estimated, dim=1, keepdim=True) * reference / ref_energy
    noise = estimated - proj
    
    snr = 10 * torch.log10(torch.sum(proj ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps) + eps)
    return torch.mean(snr)

class DCCRN_CTC(nn.Module):
    def __init__(self, vocab_size=29, num_finetune_layers=4):
        super().__init__()
        self.enhancer = CustomDCCRN(rnn_units=256, masking_mode='E')

        model_path = os.path.join(current_dir, "..", "models", "wav2vec2-base-960h")
        self.asr_encoder = Wav2Vec2Model.from_pretrained(
            model_path,
            local_files_only=True,
            gradient_checkpointing=False
        )
        
        # 冻结基础参数，解冻顶层微调
        for param in self.asr_encoder.parameters():
            param.requires_grad = False
        if num_finetune_layers > 0:
            num_layers = len(self.asr_encoder.encoder.layers)
            num_finetune_layers = min(num_finetune_layers, num_layers)
            for layer in self.asr_encoder.encoder.layers[-num_finetune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.ctc_head = nn.Linear(768, vocab_size)

    def forward(self, x, input_lengths):
        _, enhanced = self.enhancer(x)

        # 1. 特征提取：输出 (B, C, T) = (B, 512, T_feat)
        with torch.no_grad():
            extract_features = self.asr_encoder.feature_extractor(enhanced)  # (B, 512, T_feat)
            # 转置为 (B, T, C) = (B, T_feat, 512)，适配后续所有层的维度要求
            extract_features = extract_features.transpose(1, 2)

        # 2. 特征投影（内部含 layer_norm，已适配 (B, T, C) 格式）
        # 注意：此时 feature_projection 接收 (B, T, 512)，返回 (B, T, 768)
        hidden_states = self.asr_encoder.feature_projection(extract_features)[0]  # (B, T_feat, 768)

        # 3. 精准创建掩码：基于特征序列长度（与 hidden_states 时序维度同步）
        max_feat_len = hidden_states.size(1)
        def get_feat_len(orig_len):
            def _conv_out_len(l, k, s):
                return torch.floor((l - k) / s) + 1
            l = orig_len
            for _ in range(2): l = _conv_out_len(l, 10, 5)
            for _ in range(5): l = _conv_out_len(l, 3, 2)
            return l.to(torch.long)
        feat_lengths = get_feat_len(input_lengths)  # (B,) 每个样本的特征长度
        attention_mask = torch.arange(max_feat_len, device=x.device)[None, :] < feat_lengths[:, None]
        attention_mask = attention_mask.long()
        # 转为 Transformer 编码器所需的掩码格式
        attention_mask = self.asr_encoder._get_feature_vector_attention_mask(max_feat_len, attention_mask)

        # 4. 兜底截断：确保特征序列长度 ≤ 499（规避位置卷积限制）
        max_allowed_feat_len = 499
        if hidden_states.size(1) > max_allowed_feat_len:
            hidden_states = hidden_states[:, :max_allowed_feat_len, :]
            attention_mask = attention_mask[:, :max_allowed_feat_len]
            feat_lengths = torch.clamp(feat_lengths, max=max_allowed_feat_len)

        # 5. Transformer 编码（输入格式 (B, T, 768)，掩码同步）
        encoder_outputs = self.asr_encoder.encoder(hidden_states, attention_mask=attention_mask)
        final_hidden_states = encoder_outputs[0]

        # 6. CTC 计算
        logits = self.ctc_head(final_hidden_states)
        asr_input_lengths = feat_lengths  # 修正后的特征长度，用于 CTC 损失

        return enhanced, logits, asr_input_lengths

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 必须使用 9 秒过滤后的 CSV
    csv_path = os.path.join(current_dir, "train_filtered.csv")
    assert os.path.exists(csv_path), f"请先运行 filter_long_audio.py（MAX_DURATION_SECONDS=9.0）生成 {csv_path}"

    model = DCCRN_CTC(num_finetune_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_loader = DataLoader(
        ValentiniDataset(csv_path, sample_rate=16000),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True  # 加速 GPU 数据传输
    )
    
    ctc_loss_weight = 5.0

    for epoch in range(100):
        model.train()
        total_epoch_loss = 0
        valid_batches = 0  # 统计有效批次（排除跳过的）
        
        for batch_idx, (noisy, clean, targets, noisy_lengths) in enumerate(train_loader):
            noisy, clean, targets, noisy_lengths = \
                noisy.to(device), clean.to(device), targets.to(device), noisy_lengths.to(device)

            try:
                enhanced, logits, asr_input_lengths = model(noisy, noisy_lengths)

                # 过滤 NaN/Inf 输出
                if torch.isnan(enhanced).any() or torch.isinf(enhanced).any() or torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"Warning: NaNs/Infs in model output at batch {batch_idx}. Skipping.")
                    continue
                
                # 计算 SI-SNR 损失
                si_snr_loss = -si_snr(enhanced, clean)
                if torch.isnan(si_snr_loss):
                    print(f"Warning: NaN SI-SNR loss at batch {batch_idx}. Skipping.")
                    continue

                # 计算 CTC 损失（确保输入长度 ≥ 目标长度）
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                targets_lengths = (targets != 0).sum(dim=1)
                
                valid_indices = asr_input_lengths >= targets_lengths
                if not valid_indices.all():
                    log_probs = log_probs[:, valid_indices, :]
                    targets = targets[valid_indices]
                    asr_input_lengths = asr_input_lengths[valid_indices]
                    targets_lengths = targets_lengths[valid_indices]

                    if log_probs.size(1) == 0:
                        print(f"Warning: No valid samples for CTC loss at batch {batch_idx}. Skipping.")
                        continue
                
                ctc_loss = F.ctc_loss(
                    log_probs, targets, asr_input_lengths, targets_lengths,
                    blank=0, reduction='mean', zero_infinity=True
                )
                if torch.isnan(ctc_loss):
                    print(f"Warning: NaN CTC loss at batch {batch_idx}. Skipping.")
                    continue
                    
                # 总损失与反向传播
                loss = si_snr_loss + ctc_loss_weight * ctc_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 防止梯度爆炸
                optimizer.step()

                total_epoch_loss += loss.item()
                valid_batches += 1
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f} "
                          f"(SI-SNR: {si_snr_loss.item():.4f}, CTC: {ctc_loss.item():.4f})")
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"WARNING: CUDA out of memory at batch {batch_idx}. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # 计算有效批次的平均损失（避免因跳过批次导致的计算偏差）
        if valid_batches > 0:
            avg_loss = total_epoch_loss / valid_batches
            print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch} finished. No valid batches.")
        
        scheduler.step(avg_loss if valid_batches > 0 else float('inf'))
        # 保存模型（仅保留最近 5 个 epoch，避免占用过多空间）
        model_path = f"dccrn_ctc_joint_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # 删除 5 个 epoch 前的旧模型（可选）
        old_model_path = f"dccrn_ctc_joint_epoch_{epoch-5}.pth"
        if epoch >=5 and os.path.exists(old_model_path):
            os.remove(old_model_path)