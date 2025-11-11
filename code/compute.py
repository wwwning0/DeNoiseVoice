import os

import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import torch
import torchaudio.transforms as transforms as T
import whisper
import jiwer


def compute_MCD(file_original, file_reconstructed, target_sr=16000):
    try:
        # 1. 统一加载：单通道、16k采样率、float32格式
        clean, _ = librosa.load(file_original, sr=target_sr, mono=True, dtype=np.float32)
        enh, _ = librosa.load(file_reconstructed, sr=target_sr, mono=True, dtype=np.float32)
        
        # 2. 长度对齐（取较短音频的长度）
        min_len = min(len(clean), len(enh))
        clean = clean[:min_len]
        enh = enh[:min_len]
        
        # 3. 提取MFCC（语音音质评估的标准特征，鲁棒性强）
        mfcc_transform = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=13,  # 13维MFCC（核心特征）
            melkwargs={
                "n_mels": 40,
                "n_fft": 512,
                "hop_length": 160,  # 10ms帧移，适配语音时序
                "win_length": 320   # 20ms帧长
            }
        )
        
        # 4. 计算MFCC（转为张量计算，避免numpy维度问题）
        clean_tensor = torch.from_numpy(clean).unsqueeze(0)  # (1, T)
        enh_tensor = torch.from_numpy(enh).unsqueeze(0)
        clean_mfcc = mfcc_transform(clean_tensor)  # (1, 13, T_frame)
        enh_mfcc = mfcc_transform(enh_tensor)
        
        # 5. 计算帧级MCD，取平均（公式：10*sqrt(2)*||mfcc1 - mfcc2|| / log(10)）
        mfcc_diff = clean_mfcc - enh_mfcc
        frame_mcd = 10 * np.sqrt(2) * torch.norm(mfcc_diff, dim=1) / np.log(10)
        avg_mcd = torch.mean(frame_mcd).item()
        
        # 6. 兜底：排除异常值（MCD正常范围5-35 dB）
        if avg_mcd < 0 or avg_mcd > 50:
            print(f"警告：MCD值异常（{avg_mcd:.2f} dB），可能音频失真")
            return np.nan
        return avg_mcd
    
    except Exception as e:
        print(f"MCD计算失败：{str(e)}")
        return np.nan
    


def compute_WER():
    model = whisper.load_model("base")
    # 原始文本
    original_text = "At least 12 persons saw the man with the revolver in the vicinity of the Tipit crime scene, at or immediately after the shooting. By the evening of November 22, five of them had identified Lee Harvey Oswald in police lineups as the man they saw. A sixth did so the next day. Three others subsequently identified Oswald from a photograph. Two witnesses testified that Oswald resembled the man they had seen. One witness felt he was too distant from the gunman to make a positive identification. A taxi driver, William Skoggins, was eating lunch in his cab, which was parked on Patten, facing the southeast corner of Tenth Street and Patten Avenue, a few feet to the north. A police car moving east on 10th at about 10 or 12 miles an hour passed in front of his cab. About 100 feet from the corner, the police car pulled up alongside a man on the sidewalk. This man dressed in a light-colored jacket approached the car."


    file_dir = r"E:\dataset\result\test.wav"
    audio, _ = librosa.load(file_dir, sr=16000)

    result = model.transcribe(audio)
    transcribed_text = result["text"]
    # 计算 WER
    wer_score = jiwer.wer(original_text, transcribed_text)
    print(wer_score)
