import os
import pandas as pd
import torch
import librosa
import torchaudio
from torch.nn.utils.rnn import pad_sequence

vocab = " 'abcdefghijklmnopqrstuvwxyz"
char_to_id = {ch: i + 1 for i, ch in enumerate(vocab)}

def text_to_ids(text):
    return [char_to_id[ch] for ch in text.lower() if ch in char_to_id]

class ValentiniDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, sample_rate=16000):
        self.df = pd.read_csv(csv_file)
        self.sample_rate = sample_rate
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_file)))
        # --- 最终修正: 定义一个绝对安全的样本长度上限 (9.8秒 * 16000Hz) ---
        self.max_samples = int(sample_rate * 9.8)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        noisy_path = os.path.join(self.root_dir, row["noisy"])
        clean_path = os.path.join(self.root_dir, row["clean"])
        text = row["text"]

        # 加载时直接重采样到目标采样率
        noisy, _ = librosa.load(noisy_path, sr=self.sample_rate, mono=True)
        clean, _ = librosa.load(clean_path, sr=self.sample_rate, mono=True)

        # --- 最终修正: 加入“熔断”机制，自动截断过长音频 ---
        # 这确保了即使数据未经过滤，程序也不会崩溃
        if noisy.shape[0] > self.max_samples:
            noisy = noisy[:self.max_samples]
            clean = clean[:self.max_samples]
        
        target = torch.tensor(text_to_ids(text), dtype=torch.long)
        
        # 返回 numpy 数组，collate_fn 会处理成 tensor
        return noisy, clean, target

def collate_fn(batch):
    noisy_waves, clean_waves, targets = zip(*batch)
    noisy_lengths = torch.tensor([wav.shape[0] for wav in noisy_waves], dtype=torch.long)
    
    noisy_padded = pad_sequence([torch.from_numpy(w) for w in noisy_waves], batch_first=True)
    clean_padded = pad_sequence([torch.from_numpy(w) for w in clean_waves], batch_first=True)
    targets_padded = pad_sequence([torch.tensor(t) for t in targets], batch_first=True, padding_value=0)

    return noisy_padded, clean_padded, targets_padded, noisy_lengths