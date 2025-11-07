# build_csv.py
import os
import pandas as pd

root_dir = ".."  

# 处理训练集
train_noisy_dir = os.path.join(root_dir, "noisy_trainset_28spk_wav")
train_clean_dir = os.path.join(root_dir, "clean_trainset_28spk_wav")
train_txt_dir = os.path.join(root_dir, "trainset_28spk_txt")

rows = []
for noisy_file in os.listdir(train_noisy_dir):
    if not noisy_file.endswith(".wav"):
        continue
    # 提取干净语音文件名 (去掉 _snX.wav)
    clean_name = noisy_file
    clean_path = os.path.join(train_clean_dir, clean_name)
    txt_name = clean_name.replace(".wav", ".txt")
    txt_path = os.path.join(train_txt_dir, txt_name)
    if not os.path.exists(clean_path) or not os.path.exists(txt_path):
        continue
    with open(txt_path, 'r') as f:
        text = f.readline().strip().split(":", 1)[-1].strip()
    rows.append({
        "noisy": os.path.relpath(os.path.join(train_noisy_dir, noisy_file), start=root_dir),
        "clean": os.path.relpath(clean_path, start=root_dir),
        "text": text
    })

pd.DataFrame(rows).to_csv("train.csv", index=False)
print("Train CSV built successfully.")

# 处理测试集
test_noisy_dir = os.path.join(root_dir, "noisy_testset_wav")
test_clean_dir = os.path.join(root_dir, "clean_testset_wav")
test_txt_dir = os.path.join(root_dir, "testset_txt")

rows = []
for noisy_file in os.listdir(test_noisy_dir):
    if not noisy_file.endswith(".wav"):
        continue
    clean_name = noisy_file
    clean_path = os.path.join(test_clean_dir, clean_name)
    txt_name = clean_name.replace(".wav", ".txt")
    txt_path = os.path.join(test_txt_dir, txt_name)
    if not os.path.exists(clean_path) or not os.path.exists(txt_path):
        continue
    with open(txt_path, 'r') as f:
        text = f.readline().strip().split(":", 1)[-1].strip()
    rows.append({
        "noisy": os.path.relpath(os.path.join(test_noisy_dir, noisy_file), start=root_dir),
        "clean": os.path.relpath(clean_path, start=root_dir),
        "text": text
    })

pd.DataFrame(rows).to_csv("test.csv", index=False)
print("Test CSV built successfully.")