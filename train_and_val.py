import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import pandas as pd
import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if torch.cuda.is_available():
    
    device = 'cuda'
else:
    device = 'cpu'
class CovidModel(nn.Module):
    def __init__(self,system,length=256):
        super(CovidModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        
        with torch.no_grad():
            x = torch.randn(1, 1, 128, length)
            x = self.features(x)
            fc_input_dim = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CovidDataset(Dataset):
    def __init__(self, system, csv_file, image_dir, length=256, transform=None, augment = True):
        self.data = pd.read_csv(csv_file)
        self.system = system
        self.image_dir = os.path.join(image_dir,system)
        self.length = length
        self.transform = transform
        self.positive_ids = self.data[self.data['COVID_STATUS'] == 'p']["SUB_ID"].tolist() #positive samples
        self.negative_ids = self.data[self.data['COVID_STATUS'] == 'n']["SUB_ID"].tolist() #negative samples
        self.augment = augment

    def __len__(self):

        return len(self.data)
    
    def denoise_image(self, image):
   
        # medianBlur
        # 将浮点数图像转换为uint8类型
        image_uint8 = (image * 255).astype(np.uint8)
        # 应用中值滤波
        denoised = cv2.medianBlur(image_uint8, 5)
        # 转换回浮点数
        return denoised.astype(np.float32) / 255.0

    def image_augmentation(self, image):
        """Image Augmentation"""
        aug_type = np.random.choice(['original', 'rotate', 'flip', 'noise'])
        
        if aug_type == 'rotate':
            # randomly rotate
            angle = np.random.uniform(-10, 10)
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows))
        elif aug_type == 'flip':
            # flip 
            image = cv2.flip(image, 1)
        elif aug_type == 'noise':
            # add noise
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)
            
        return image
    def __getitem__(self, idx):
        # rand choose positive samples and negative samples
        #there will be a point which we can improve our model: unbalanced-sample,since there a lot of negatives samples, so if we can set penalty to these negative samples
        if np.random.rand() > 0.5:
            id = np.random.choice(self.positive_ids)
            label = 1
        else:
            id = np.random.choice(self.negative_ids)
            label = 0

        img_name = os.path.join(self.image_dir, f"{id}.png")
        image = Image.open(img_name)
        
        # normalize
        image = np.array(image).astype(np.float32) / 255.0
        if self.augment:
            image = self.image_augmentation(image)

        image = self.denoise_image(image)
       
        if len(image.shape) == 3:
            image = image[:,:,0]
        # if image.shape != (128, 128):
        #     image = image.reshape(1, image.shape[0], self.length)  # 添加通道维度
            # 调整图像大小为固定尺寸
        target_shape = (128, self.length)
        if image.shape != target_shape:
            image = np.resize(image, target_shape)
        # randomly select parts in a long speech graphic
        # so there also will be a point to improve: solve the noisy of our sound or augument the sound data 
        if image.shape[1] > 2 * self.length:
            starting_point = np.random.choice(range(image.shape[1] - self.length - 1))
            image = image[:, starting_point:starting_point + self.length]
        else:
        # If the image is too short, widthen it into a fix size
            pad_width = max(0, self.length - image.shape[1])
            image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant')
        image = image.reshape(1, image.shape[0], self.length)
        if self.transform:
            image = self.transform(image)
        #print(f"process {idx}")
        #print(f'image.shape:{image.shape}')

        return image, float(label)

def calculate_class_weights(metadata_df):
    n_samples = len(metadata_df)
    n_classes = 2
    class_counts = metadata_df['COVID_STATUS'].value_counts()
    weights = n_samples / (n_classes * class_counts)
    class_weights = torch.FloatTensor([weights['n'], weights['p']])
    return class_weights
def calculate_metrics(predictions, targets):
    """计算多个评估指标"""
    predictions = np.array(predictions) > 0.5
    targets = np.array(targets)
    
    # 计算混淆矩阵元素
    TP = np.sum((predictions == 1) & (targets == 1))
    TN = np.sum((predictions == 0) & (targets == 0))
    FP = np.sum((predictions == 1) & (targets == 0))
    FN = np.sum((predictions == 0) & (targets == 1))
    
    # 计算各项指标
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        'accuracy': accuracy,
        'recall': recall,  # 也叫敏感度或真阳性率
        'false_alarm_rate': false_alarm_rate,  # 假阳性率
        'precision': precision,
        'f1_score': f1_score,
        'y_true':targets,
        'y_pred':predictions
    }

def plot_training_metrics(writer, fold, system):
    """绘制训练过程中的指标"""
    # 从tensorboard日志中读取数据
    log_path = f'runs/{system}/fold_{fold}'
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    
    # 获取所有指标数据
    metrics = {
        'loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1_score': []
    }
    
    # 读取损失值
    loss_events = event_acc.Scalars('Loss/train')
    val_loss_events = event_acc.Scalars('Loss/val')
    metrics['loss'] = [(e.step, e.value) for e in loss_events]
    
    # 读取其他指标
    metrics['accuracy'] = [(e.step, e.value) for e in event_acc.Scalars('Metrics/accuracy')]
    metrics['recall'] = [(e.step, e.value) for e in event_acc.Scalars('Metrics/recall')]
    metrics['precision'] = [(e.step, e.value) for e in event_acc.Scalars('Metrics/precision')]
    metrics['f1_score'] = [(e.step, e.value) for e in event_acc.Scalars('Metrics/f1_score')]
    
    # 绘图
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    steps, values = zip(*metrics['loss'])
    plt.plot(steps, values, label='Training Loss')
    if val_loss_events:
        val_steps, val_values = zip(*[(e.step, e.value) for e in val_loss_events])
        plt.plot(val_steps, val_values, label='Validation Loss')
    plt.title(f'{system} Model Loss - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制其他指标
    plt.subplot(2, 2, 2)
    for metric in ['accuracy', 'recall', 'precision']:
        steps, values = zip(*metrics[metric])
        plt.plot(steps, values, label=metric.capitalize())
    plt.title(f'{system} Model Metrics - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'metrics_{system}_fold_{fold}.png')
    plt.close()
def plot_confusion_matrix(y_true, y_pred, system, fold):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {system} Model (Fold {fold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{system}_fold_{fold}.png')
    plt.close()

def train_and_validate_model(model, optimizer, train_loader, val_loader, epochs, model_name, class_weights, fold,trained_path):
    """训练并验证模型"""
    logger = setup_logger(f'{model_name}_fold_{fold}', f'{model_name}_fold_{fold}.log')
    criterion = nn.BCELoss()  # 移除weight参数，因为我们的输出是单个概率值
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    writer = SummaryWriter(f'runs/{model_name}/fold_{fold}')
    
    best_val_loss = float('inf')
   
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.to(torch.float32).view(-1, 1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段（新用户验证）
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = target.to(torch.float32).view(-1, 1)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                val_predictions.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 计算验证集指标
        val_predictions = np.array(val_predictions) > 0.5
        val_targets = np.array(val_targets)
        
       
        metrics = calculate_metrics(val_predictions, val_targets)
        metrics['loss'] = avg_val_loss
        writer.add_scalar('Loss/val', metrics['loss'], epoch)
        writer.add_scalar('Metrics/accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/recall', metrics['recall'], epoch)
        writer.add_scalar('Metrics/precision', metrics['precision'], epoch)
        writer.add_scalar('Metrics/f1_score', metrics['f1_score'], epoch)
        # 记录日志
        logger.info(f'Fold {fold}, Epoch {epoch}:')
        logger.info(f'Accuracy: {metrics["accuracy"]:.4f}')
        logger.info(f'Recall: {metrics["recall"]:.4f}')
        logger.info(f'False Alarm Rate: {metrics["false_alarm_rate"]:.4f}')
        logger.info(f'Precision: {metrics["precision"]:.4f}')
        logger.info(f'F1 Score: {metrics["f1_score"]:.4f}')
        
        writer.add_scalars('Loss', {
            'train': avg_train_loss,
            'val': avg_val_loss
        }, epoch)
        writer.add_scalars('Metrics', {
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall'],
            'precision':metrics['precision'],
            'false_alarm_rate': metrics['false_alarm_rate']
        }, epoch)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(trained_path, f"{model_name}_fold_{fold}_best.pth"))
         # 训练结束后绘制图表
        plot_training_metrics(writer, fold, model_name)
        plot_confusion_matrix(metrics['y_true'], metrics['y_pred'], model_name, fold)
    writer.close()


def create_model(system,length=256):
    net = CovidModel(system,length).to(device)
    return net

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger



# print(net)
# breathing_model = create_model("breathing",length)
# speech_model = create_model("speech",length)
# cough_model = create_model("cough",length)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# criterion = nn.NLLLoss()



    
