# filter_long_audio.py
import pandas as pd
import librosa
import os
from tqdm import tqdm

# --- 配置 ---
# 请确保这里的路径是正确的
INPUT_CSV = "train.csv"
OUTPUT_CSV = "train_filtered.csv"
# 设置一个保守的、绝对安全的时长上限（秒）
MAX_DURATION_SECONDS = 9.0 
# ---------------------

df = pd.read_csv(INPUT_CSV)
# 假设音频文件相对于 CSV 文件的根目录是 '../'
# 如果不是，请相应修改这里的 root_dir
root_dir = os.path.dirname(os.path.abspath(INPUT_CSV))

print(f"原始数据集样本数: {len(df)}")

indices_to_keep = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="正在扫描并过滤过长音频"):
    try:
        # 我们以 noisy 音频的长度为准
        audio_path = os.path.join(root_dir, '..', row["noisy"])
        if not os.path.exists(audio_path):
            print(f"警告: 文件不存在 {audio_path}, 已跳过")
            continue
        
        duration = librosa.get_duration(path=audio_path)
        if duration <= MAX_DURATION_SECONDS:
            indices_to_keep.append(idx)
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {e}")

df_filtered = df.loc[indices_to_keep].reset_index(drop=True)

print(f"过滤后数据集样本数: {len(df_filtered)}")
print(f"共移除了 {len(df) - len(df_filtered)} 个过长样本。")

df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"已生成过滤后的数据清单: {OUTPUT_CSV}")