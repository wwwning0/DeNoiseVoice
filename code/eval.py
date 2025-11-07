# eval.py
import sys
import os
import torch
from torch.utils.data import DataLoader
import jiwer
import torchaudio


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.chdir(current_dir)  


try:
    from dc_crn import DCCRN as CustomDCCRN
except ImportError as e:
    print(f"Error importing DCCRN: {e}")
    sys.exit(1)

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


import torch.nn as nn
from transformers import Wav2Vec2Model

class DCCRN_CTC(nn.Module):
    def __init__(self, vocab_size=29):
        super().__init__()
        self.enhancer = CustomDCCRN(rnn_units=256, masking_mode='E')
        model_path = os.path.join(current_dir, "..", "models", "wav2vec2-base-960h")
        self.asr_encoder = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
        for p in self.asr_encoder.parameters():
            p.requires_grad = False
        self.ctc_head = nn.Linear(768, vocab_size)

    def forward(self, x):
        _, enhanced = self.enhancer(x)  
        with torch.no_grad():
            feats = self.asr_encoder(enhanced).last_hidden_state
        logits = self.ctc_head(feats)
        return enhanced, logits


def evaluate(model, test_loader, device):
    model.eval()
    wer_noisy_list = []
    wer_enh_list = []
    delta_sisnr_total = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, (noisy, clean, _) in enumerate(test_loader):
            if i >= 100:  # 评估 100 条
                break
            noisy = noisy.to(device)
            clean = clean.to(device)

            # 推理增强语音
            enhanced, _ = model(noisy)

            # 长度对齐（与训练一致）
            min_len = min(enhanced.size(1), clean.size(1))
            enhanced = enhanced[:, :min_len]
            clean = clean[:, :min_len]
            noisy_aligned = noisy[:, :min_len]

            # ΔSI-SNR
            si_snr_enh = si_snr(enhanced, clean).item()
            si_snr_noisy = si_snr(noisy_aligned, clean).item()
            delta_sisnr_total += (si_snr_enh - si_snr_noisy)

            # 保存临时音频
            torchaudio.save("tmp_noisy.wav", noisy_aligned.cpu(), 16000)
            torchaudio.save("tmp_enh.wav", enhanced.cpu(), 16000)

            # 获取参考文本
            ref_text = test_loader.dataset.df.iloc[i]["text"].lower()

            # ASR 识别
            hyp_noisy = asr("tmp_noisy.wav")["text"].lower()
            hyp_enh = asr("tmp_enh.wav")["text"].lower()

            # 计算 WER
            wer_noisy = jiwer.wer(ref_text, hyp_noisy)
            wer_enh = jiwer.wer(ref_text, hyp_enh)

            wer_noisy_list.append(wer_noisy)
            wer_enh_list.append(wer_enh)
            num_samples += 1

            print(f"[{i+1}/100] WER noisy: {wer_noisy:.2%}, enhanced: {wer_enh:.2%}")

    # 计算平均
    avg_delta_sisnr = delta_sisnr_total / num_samples
    avg_wer_noisy = sum(wer_noisy_list) / len(wer_noisy_list)
    avg_wer_enh = sum(wer_enh_list) / len(wer_enh_list)

    print("\n" + "="*50)
    print(f"ΔSI-SNR: {avg_delta_sisnr:.2f} dB")
    print(f"WER (noisy): {avg_wer_noisy:.2%}")
    print(f"WER (enhanced): {avg_wer_enh:.2%}")
    print(f"WER Relative Improvement: {(avg_wer_noisy - avg_wer_enh) / avg_wer_noisy:.2%}")
    print("="*50)

# --- 主程序 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    model = DCCRN_CTC().to(device)
    model_path = "dccrn_ctc_epoch_36.pth"  
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found!")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # 加载测试集
    test_csv = os.path.join(current_dir, "test.csv")
    if not os.path.exists(test_csv):
        print(f"test.csv not found at {test_csv}. Please generate it.")
        sys.exit(1)

    test_dataset = ValentiniDataset(test_csv, sample_rate=16000)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 开始评估
    evaluate(model, test_loader, device)

    # 清理临时文件
    if os.path.exists("tmp_noisy.wav"):
        os.remove("tmp_noisy.wav")
    if os.path.exists("tmp_enh.wav"):
        os.remove("tmp_enh.wav")