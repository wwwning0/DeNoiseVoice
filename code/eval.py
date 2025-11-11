import sys
import os
import torch
from torch.utils.data import DataLoader
import jiwer
import torchaudio
import pandas as pd
import numpy as np

from compute import compute_MCD

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.chdir(current_dir)

from dc_crn import DCCRN as CustomDCCRN  
import torch.nn as nn
from transformers import Wav2Vec2Model


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
        
        # 解冻基础层进行微调
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

        with torch.no_grad():
            extract_features = self.asr_encoder.feature_extractor(enhanced)
            extract_features = extract_features.transpose(1, 2)

        hidden_states = self.asr_encoder.feature_projection(extract_features)[0]

        max_feat_len = hidden_states.size(1)
        def get_feat_len(orig_len):
            def _conv_out_len(l, k, s):
                return torch.floor((l - k) / s) + 1
            l = orig_len
            for _ in range(2): l = _conv_out_len(l, 10, 5)
            for _ in range(5): l = _conv_out_len(l, 3, 2)
            return l.to(torch.long)
        feat_lengths = get_feat_len(input_lengths)
        attention_mask = torch.arange(max_feat_len, device=x.device)[None, :] < feat_lengths[:, None]
        attention_mask = attention_mask.long()
        attention_mask = self.asr_encoder._get_feature_vector_attention_mask(max_feat_len, attention_mask)

        max_allowed_feat_len = 499
        if hidden_states.size(1) > max_allowed_feat_len:
            hidden_states = hidden_states[:, :max_allowed_feat_len, :]
            attention_mask = attention_mask[:, :max_allowed_feat_len]
            feat_lengths = torch.clamp(feat_lengths, max=max_allowed_feat_len)

        encoder_outputs = self.asr_encoder.encoder(hidden_states, attention_mask=attention_mask)
        final_hidden_states = encoder_outputs[0]

        logits = self.ctc_head(final_hidden_states)
        asr_input_lengths = feat_lengths

        return enhanced, logits, asr_input_lengths

from dataset import ValentiniDataset, collate_fn


def si_snr(estimated, reference):
    reference = reference - torch.mean(reference, dim=1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=1, keepdim=True)
    alpha = torch.sum(reference * estimated, dim=1, keepdim=True) / (ref_energy + 1e-8)
    proj = alpha * reference
    noise = estimated - proj
    si_snr_val = 10 * torch.log10(
        (torch.sum(proj ** 2, dim=1) + 1e-8) / (torch.sum(noise ** 2, dim=1) + 1e-8)
    )
    return torch.mean(si_snr_val)


print("Initializing Whisper ASR pipeline...")
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

whisper_path = os.path.join(current_dir, "..", "models", "whisper-base")
if not os.path.exists(whisper_path):
    print(f"Whisper model not found at {whisper_path}. Please download it first.")
    sys.exit(1)

whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_path, local_files_only=True)
whisper_processor = WhisperProcessor.from_pretrained(whisper_path, local_files_only=True)

asr = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1
)
print("Whisper ASR pipeline loaded.")


def evaluate(model, test_loader, device, save_audio_count=5):  # 新增：控制保存音频数量
    model.eval()
    wer_noisy_list = []
    wer_enh_list = []
    mcd_list = [] 
    delta_sisnr_total = 0.0
    num_samples = 0
    
    tmp_noisy_path = "tmp_noisy.wav"
    tmp_enh_path = "tmp_enh.wav"
    tmp_clean_path = "tmp_clean.wav"

    audio_output_dir = "output_audio"  # 保存音频的文件夹名称
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)
        print(f"\n创建音频保存目录：{audio_output_dir}（将保存前{save_audio_count}条样本的带噪/干净/增强音频）")

    with torch.no_grad():
        num_eval_samples =50  
        for i, batch_data in enumerate(test_loader):
            if i >= num_eval_samples:
                print(f"\nReached evaluation limit of {num_eval_samples} samples.")
                break

            noisy, clean, targets, noisy_lengths = batch_data
            
            noisy = noisy.to(device)
            clean = clean.to(device)
            noisy_lengths = noisy_lengths.to(device)  

            enhanced, _, _ = model(noisy, noisy_lengths)

            min_len = min(enhanced.size(1), clean.size(1))
            enhanced = enhanced[:, :min_len]
            clean = clean[:, :min_len]
            noisy_aligned = noisy[:, :min_len]

            # 计算 ΔSI-SNR（保持不变）
            si_snr_enh = si_snr(enhanced, clean).item()
            si_snr_noisy = si_snr(noisy_aligned, clean).item()
            delta_sisnr_total += (si_snr_enh - si_snr_noisy)

            # 保存临时音频文件（保持不变）
            torchaudio.save(tmp_noisy_path, noisy_aligned.cpu(), 16000)
            torchaudio.save(tmp_enh_path, enhanced.cpu(), 16000)
            torchaudio.save(tmp_clean_path, clean.cpu(), 16000) 

            if i < save_audio_count:  # 只保存前N条样本
                noisy_save_path = os.path.join(audio_output_dir, f"{i}_noisy.wav")
                clean_save_path = os.path.join(audio_output_dir, f"{i}_clean.wav")
                enhanced_save_path = os.path.join(audio_output_dir, f"{i}_enhanced.wav")
                
                # 保存音频（复制临时文件到目标路径）
                torchaudio.save(noisy_save_path, noisy_aligned.cpu(), 16000)
                torchaudio.save(clean_save_path, clean.cpu(), 16000)
                torchaudio.save(enhanced_save_path, enhanced.cpu(), 16000)
                
                print(f"✅ 已保存第{i+1}条样本音频：")
                print(f"   - 带噪：{noisy_save_path}")
                print(f"   - 干净：{clean_save_path}")
                print(f"   - 增强：{enhanced_save_path}")
            # --------------------------------------------------------------------------

            # 计算 MCD（保持不变，保留异常捕获）
            try:
                mcd_score = compute_MCD(tmp_clean_path, tmp_enh_path)
                mcd_list.append(mcd_score)
            except Exception as e:
                print(f"Warning: Could not compute MCD for sample {i+1}. Error: {e}")
                mcd_list.append(np.nan) 

            # 使用 ASR 计算 WER（保持不变）
            ref_text = test_loader.dataset.df.iloc[i]["text"].lower()
            hyp_noisy = asr(tmp_noisy_path)["text"].lower()
            hyp_enh = asr(tmp_enh_path)["text"].lower()

            wer_noisy = jiwer.wer(ref_text, hyp_noisy)
            wer_enh = jiwer.wer(ref_text, hyp_enh)

            wer_noisy_list.append(wer_noisy)
            wer_enh_list.append(wer_enh)
            num_samples += 1

            current_mcd_str = f"{mcd_score:.2f}" if 'mcd_score' in locals() and not np.isnan(mcd_score) else "N/A"
            print(f"[{i+1}/{num_eval_samples}] "
                  f"WER noisy: {wer_noisy:.2%}, enhanced: {wer_enh:.2%}, "
                  f"MCD: {current_mcd_str}, "
                  f"ΔSI-SNR: {si_snr_enh - si_snr_noisy:.2f} dB")

    avg_delta_sisnr = delta_sisnr_total / num_samples if num_samples > 0 else 0
    avg_wer_noisy = np.mean(wer_noisy_list) if wer_noisy_list else 0
    avg_wer_enh = np.mean(wer_enh_list) if wer_enh_list else 0
    avg_mcd = np.nanmean(mcd_list) if mcd_list else 0
    wer_relative_improvement = (avg_wer_noisy - avg_wer_enh) / avg_wer_noisy if avg_wer_noisy > 0 else 0

    print("\n" + "="*60)
    print(" " * 20 + "EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Samples Evaluated: {num_samples}")
    print("\n--- Speech Enhancement Metrics ---")
    print(f"  Average ΔSI-SNR Improvement: {avg_delta_sisnr:.2f} dB")
    print(f"  Average MCD (lower is better): {avg_mcd:.2f}")
    
    print("\n--- ASR Performance (WER) ---")
    print(f"  WER on Noisy Audio:     {avg_wer_noisy:.2%}")
    print(f"  WER on Enhanced Audio:  {avg_wer_enh:.2%}")

    print("-" * 30)
    print(f"  WER Relative Improvement: {wer_relative_improvement:.2%}")
    print("="*60)

    if save_audio_count > 0:
        print(f"\n🎧 音频保存完成！共保存{min(save_audio_count, num_eval_samples)}条样本，路径：{os.path.abspath(audio_output_dir)}")
        print("   文件名说明：")
        print("   - xxx_noisy.wav：原始带噪音频")
        print("   - xxx_clean.wav：干净参考音频")
        print("   - xxx_enhanced.wav：模型增强后的音频")

    
if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading DCCRN-CTC model...")
    model = DCCRN_CTC(num_finetune_layers=4).to(device)
    model_path = "dccrn_ctc_joint_epoch_89.pth"  
    if not os.path.exists(model_path):
        print(f"FATAL: Model checkpoint '{model_path}' not found! 请替换为实际的 DCCRN 训练权重路径")
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print(f"Successfully loaded model from {model_path}")

    print("Loading test dataset...")
    test_csv = os.path.join(current_dir, "test.csv")  
    if not os.path.exists(test_csv):
        print(f"FATAL: test.csv not found at '{test_csv}'. Please generate it.")
        sys.exit(1)

    test_dataset = ValentiniDataset(test_csv, sample_rate=16000)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    print("Test dataset loaded.")

    print("\nStarting evaluation...")
    # -------------------------- 修改：调用evaluate时指定保存音频数量 --------------------------
    evaluate(model, test_loader, device, save_audio_count=5)  
    # --------------------------------------------------------------------------

    print("Cleaning up temporary audio files...")
    for f in ["tmp_noisy.wav", "tmp_enh.wav", "tmp_clean.wav"]:
        if os.path.exists(f):
            os.remove(f)
    print("Evaluation finished.")