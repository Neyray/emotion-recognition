"""
数据加载器
功能：
1. 创建PyTorch Dataset类
2. 创建DataLoader
3. 批量加载和处理数据
4. 支持多标签分类
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import List, Dict, Tuple
import json


# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LABEL_MAPPING_PATH = PROJECT_ROOT / "data" / "label_mapping.json"


class EmotionDataset(Dataset):
    """情绪识别数据集"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer=None, max_length=128, num_labels=28):
        """
        参数:
            texts: 文本列表
            labels: 标签列表（多标签格式）
            tokenizer: 分词器（如果使用预训练模型）
            max_length: 最大序列长度
            num_labels: 标签总数
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer           
        self.max_length = max_length
        self.num_labels = num_labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_ids = self.labels[idx]
        
        # 创建多标签one-hot编码
        label_vector = torch.zeros(self.num_labels, dtype=torch.float)
        for label_id in label_ids:
            if 0 <= label_id < self.num_labels:
                label_vector[label_id] = 1.0
        
        # 如果提供了tokenizer，进行分词
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': label_vector,
                'text': text
            }
        else:
            # 简单返回文本和标签
            return {
                'text': text,
                'labels': label_vector
            }


def load_data(data_path: str, use_augmented=False):
    """
    加载数据
    
    参数:
        data_path: 数据文件路径
        use_augmented: 是否使用增强数据
    
    返回:
        texts, labels
    """
    # 如果是训练集且使用增强数据
    if 'train' in str(data_path) and use_augmented:
        aug_path = Path(data_path).parent / "train_augmented.csv"
        if aug_path.exists():
            print(f"✓ 使用增强数据: {aug_path}")
            data_path = aug_path
    
    df = pd.read_csv(data_path)
    
    # 解析labels列
    df['labels'] = df['labels'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    texts = df['text'].tolist()
    labels = df['labels'].tolist()
    
    return texts, labels


def create_data_loaders(train_path, val_path, test_path,
                       tokenizer=None, batch_size=32, 
                       max_length=128, num_workers=0,
                       use_augmented=False):
    """
    创建数据加载器
    
    参数:
        train_path: 训练集路径
        val_path: 验证集路径
        test_path: 测试集路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载线程数
        use_augmented: 是否使用增强数据
    
    返回:
        train_loader, val_loader, test_loader
    """
    print("\n📂 加载数据集...")
    
    # 加载数据
    train_texts, train_labels = load_data(train_path, use_augmented=use_augmented)
    val_texts, val_labels = load_data(val_path)
    test_texts, test_labels = load_data(test_path)
    
    print(f"✓ 训练集: {len(train_texts)} 条")
    print(f"✓ 验证集: {len(val_texts)} 条")
    print(f"✓ 测试集: {len(test_texts)} 条")
    
    # 创建数据集
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("✓ 数据加载器创建完成")
    
    return train_loader, val_loader, test_loader


def main():
    """测试数据加载器"""
    print("=" * 70)
    print(" " * 20 + "数据加载器测试")
    print("=" * 70)
    
    # 定义路径
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    # 创建数据加载器（不使用tokenizer）
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path, val_path, test_path,
        tokenizer=None,
        batch_size=16,
        use_augmented=False
    )
    
    # 测试一个批次
    print("\n🧪 测试数据加载...")
    for batch in train_loader:
        print(f"\n批次信息:")
        print(f"  文本数量: {len(batch['text'])}")
        print(f"  标签形状: {batch['labels'].shape}")
        print(f"\n第一个样本:")
        print(f"  文本: {batch['text'][0][:100]}...")
        print(f"  标签向量: {batch['labels'][0]}")
        print(f"  激活的标签ID: {torch.where(batch['labels'][0] == 1)[0].tolist()}")
        break
    
    print("\n✅ 数据加载器测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()