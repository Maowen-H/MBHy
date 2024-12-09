import os
import argparse
import pandas as pd
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import shutil
from train_and_val_1 import (
    create_model, 
    CovidDataset, 
    train_and_validate_model,
    calculate_class_weights
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="exp_1", help="Experiment name for saving results")
    parser.add_argument("--image_path", default="images_1")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trained_path", default="train_para")
    parser.add_argument("--augment", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 创建实验目录
    experiment_dir = f"experiments/{args.experiment_name}"
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    log_dir = os.path.join(experiment_dir, "logs")
    model_dir = os.path.join(experiment_dir, "models")
    plot_dir = os.path.join(experiment_dir, "plots")
    runs_dir = os.path.join(experiment_dir, "runs")
    trained_path = os.path.join(experiment_dir, "train_para")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(trained_path, exist_ok=True)

    
    for fold in range(5):
        print(f"\nProcessing fold {fold}")
        
        # 读取当前折的数据
        fold_path = os.path.join(args.image_path, f'fold_{fold}')
        train_metadata = pd.read_csv(os.path.join(fold_path, 'train_metadata.csv'))
        val_metadata = pd.read_csv(os.path.join(fold_path, 'val_metadata.csv'))
        
        # 计算类别权重
        class_weights = calculate_class_weights(train_metadata)
        
        for system in ['breathing', 'cough', 'speech']:
            print(f"\nTraining {system} model for fold {fold}")
            
            # 创建数据加载器
            train_csv = os.path.join(fold_path, 'train_metadata.csv')
            val_csv = os.path.join(fold_path, 'val_metadata.csv')
            train_dataset = CovidDataset(system, train_csv, fold_path, args.length)
            train_sampler = train_dataset.get_sampler()
            val_dataset = CovidDataset(system, val_csv, fold_path, args.length)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
            
            # 创建模型和优化器
            model = create_model(system, args.length).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-7)
            
            # 训练和验证
            train_and_validate_model(
                model, optimizer, train_loader, val_loader,
                args.epochs, f"{system}_fold_{fold}", 
                class_weights, fold,trained_path,
                model_dir,  # 模型保存路径
                log_dir,    # 日志保存路径
                plot_dir,   # 图表保存路径
                runs_dir    # tensorboard运行记录路径
            )

if __name__ == "__main__":
    main()