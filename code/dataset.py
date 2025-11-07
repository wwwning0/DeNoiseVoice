# dataset.py
import os
import pandas as pd
import torch
import librosa
import torchaudio  # 仅用于 resample
from torch.nn.utils.rnn import pad_sequence

# 字符映射（VCTK 英文）
vocab = " 'abcdefghijklmnopqrstuvwxyz"
char_to_id = {ch: i+1 for i, ch in enumerate(vocab)}  # 0 = blank

def text_to_ids(text):
    return [char_to_id[ch] for ch in text.lower() if ch in char_to_id]

class ValentiniDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, sample_rate=16000):
        self.df = pd.read_csv(csv_file)
        self.sample_rate = sample_rate
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_file)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        noisy_path = os.path.join(self.root_dir, row["noisy"])
        clean_path = os.path.join(self.root_dir, row["clean"])
        text = row["text"]

        # 使用 librosa 加载（自动转为 float32, 单声道）
        noisy, sr_noisy = librosa.load(noisy_path, sr=None, mono=True)
        clean, sr_clean = librosa.load(clean_path, sr=None, mono=True)

        # 转为 PyTorch tensor (1, T)
        noisy = torch.from_numpy(noisy).unsqueeze(0).float()
        clean = torch.from_numpy(clean).unsqueeze(0).float()

        # 统一重采样到目标采样率（如 16000）
        if sr_noisy != self.sample_rate:
            noisy = torchaudio.functional.resample(noisy, sr_noisy, self.sample_rate)
        if sr_clean != self.sample_rate:
            clean = torchaudio.functional.resample(clean, sr_clean, self.sample_rate)

        # 转为目标标签
        target = torch.tensor(text_to_ids(text), dtype=torch.long)
        return noisy.squeeze(0), clean.squeeze(0), target

# 用于 DataLoader 的 collate_fn
def collate_fn(batch):
    noisy, clean, targets = zip(*batch)
    noisy = pad_sequence(noisy, batch_first=True)
    clean = pad_sequence(clean, batch_first=True)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return noisy, clean, targets