import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from PIL import Image
import argparse
import librosa
from librosa.feature import melspectrogram
from librosa import power_to_db
import logging

# 配置logger
logging.basicConfig(
    filename='dataset_statistics_1.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path to dataset", default="Project_Dataset_Release")
parser.add_argument("--image_path", help="Path to processed images", default="images_1")
args = parser.parse_args()

path = args.path
image_path = args.image_path
list_path = os.path.join(path, "LISTS")  # 添加LISTS路径

# 读取元数据
metadata = pd.read_csv(os.path.join(path, "metadata.csv"), sep=' ')

def spectral_subtraction(audio, sr=16000):
    """频谱减法降噪"""
    # 取前1秒的数据作为噪声样本
    noise_length = min(sr, len(audio))
    noise_part = audio[:noise_length]
    
    # 设置较小的n_fft值
    n_fft = min(512, noise_length)
    hop_length = n_fft // 4
    
    # 计算噪声频谱
    noise_spectrum = np.mean(np.abs(librosa.stft(noise_part, n_fft=n_fft, hop_length=hop_length)))
    
    # 对整个信号进行STFT
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_clean = S - noise_spectrum
    S_clean = np.maximum(S_clean, 0)
    
    # 重建信号
    audio_clean = librosa.istft(S_clean, hop_length=hop_length)
    return audio_clean

def audio_augmentation(audio, sr):
    """音频增强"""
    aug_type = np.random.choice(['original', 'stretch', 'pitch', 'noise'])
    if aug_type == 'stretch':
        return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    elif aug_type == 'pitch':
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 2))
    elif aug_type == 'noise':
        noise = np.random.normal(0, 0.005, len(audio))
        return audio + noise
    return audio

# 处理每一折的数据
for fold in range(5):
    # 读取当前折的训练集和验证集ID
    train_ids = pd.read_csv(os.path.join(list_path, f'train_{fold}.csv'), header=None)[0].tolist()
    val_ids = pd.read_csv(os.path.join(list_path, f'val_{fold}.csv'), header=None)[0].tolist()
    
    # 获取当前折的训练集和验证集数据
    train_rows = metadata[metadata["SUB_ID"].isin(train_ids)]
    val_rows = metadata[metadata["SUB_ID"].isin(val_ids)]
    
    # 创建当前折的目录
    fold_path = os.path.join(image_path, f'fold_{fold}')
    for type_sym in ['breathing', 'cough', 'speech']:
        type_path = os.path.join(fold_path, type_sym)
        os.makedirs(type_path, exist_ok=True)
    
    # 保存元数据
    train_rows.to_csv(os.path.join(fold_path, "train_metadata.csv"))
    val_rows.to_csv(os.path.join(fold_path, "val_metadata.csv"))
    
    # 处理音频数据
    for type_sym in ['breathing', 'cough', 'speech']:
        type_path = os.path.join(fold_path, type_sym)
        audio_path = os.path.join(path, 'AUDIO', type_sym)
        
        for f in tqdm(os.listdir(audio_path), desc=f'Processing {type_sym} fold {fold}'):
            subject_id = f[:-5]  # 移除.flac后缀
            if subject_id in train_ids or subject_id in val_ids:
                audio, sr = sf.read(os.path.join(audio_path, f))
                
                # 音频处理
                noise_reduced = spectral_subtraction(audio)
                augmented_audio = audio_augmentation(noise_reduced, sr=sr)
                
                # 生成梅尔频谱图
                melgram = melspectrogram(y=augmented_audio, sr=sr)
                melgram_db = power_to_db(melgram, ref=np.max)
                melgram_db = (melgram_db + 80)/80 * 255
                
                # 保存图像
                img = Image.fromarray(melgram_db).convert('L')
                img.save(os.path.join(type_path, f'{subject_id}.png'))

        # 打印当前折的数据统计
        print(f"\nFold {fold} Statistics:")
        print("Training Set:")
        pos = len(train_rows[train_rows['COVID_STATUS'] == 'p'])
        neg = len(train_rows[train_rows['COVID_STATUS'] == 'n'])
        # print(f"Positives: {pos}")
        # print(f"Negatives: {neg}")
        # print(f"Positive/Negative Ratio: {pos/neg * 100:.2f}%")
        train_stats = f"Print number of positives covid in train set: {pos}\n" \
        f"Print number of negatives covid in train set: {neg}\n" \
        f"Positive to negative Ratio on train set= {pos/neg * 100} %"
        print(train_stats)
        logger.info(train_stats)
        
        print("-----------------------------------------------------------------------------------------")
        logger.info("-----------------------------------------------------------------------------------------")

        print("\nValidation Set:")
        pos = len(val_rows[val_rows['COVID_STATUS'] == 'p'])
        neg = len(val_rows[val_rows['COVID_STATUS'] == 'n'])
        # 测试集统计
        test_stats = f"Print number of positives covid in test set: {pos}\n" \
                    f"Print number of negatives covid in test set: {neg}\n" \
                    f"Positive to negative Ratio on test set= {pos/neg * 100} %"
        print(test_stats)
        logger.info(test_stats)
        # print(f"Positives: {pos}")
        # print(f"Negatives: {neg}")
        # print(f"Positive/Negative Ratio: {pos/neg * 100:.2f}%")