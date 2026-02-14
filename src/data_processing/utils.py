"""
工具函数
包含常用的辅助函数
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_label_mapping(path=None):
    """加载标签映射"""
    if path is None:
        path = PROJECT_ROOT / "data" / "label_mapping.json"
    
    with open(path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    # 转换键为整数
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    
    return label_mapping


def get_label_statistics(labels_list: List[List[int]], label_mapping: Dict[int, str]):
    """
    获取标签统计信息
    
    参数:
        labels_list: 标签列表
        label_mapping: 标签映射字典
    
    返回:
        统计字典
    """
    from collections import Counter
    
    # 展平所有标签
    all_labels = []
    for labels in labels_list:
        all_labels.extend(labels)
    
    # 统计
    label_counts = Counter(all_labels)
    
    # 按频率排序
    stats = {}
    for label_id, count in label_counts.most_common():
        emotion = label_mapping.get(label_id, f"未知_{label_id}")
        stats[emotion] = {
            'id': label_id,
            'count': count,
            'percentage': count / len(labels_list) * 100
        }
    
    return stats


def plot_label_distribution(labels_list: List[List[int]], 
                            label_mapping: Dict[int, str],
                            top_n=15,
                            save_path=None):
    """
    绘制标签分布图
    
    参数:
        labels_list: 标签列表
        label_mapping: 标签映射
        top_n: 显示前N个标签
        save_path: 保存路径
    """
    from collections import Counter
    
    # 统计标签
    all_labels = []
    for labels in labels_list:
        all_labels.extend(labels)
    
    label_counts = Counter(all_labels)
    top_labels = label_counts.most_common(top_n)
    
    # 准备绘图数据
    emotions = [label_mapping.get(lid, f"未知_{lid}") for lid, _ in top_labels]
    counts = [count for _, count in top_labels]
    
    # 绘图
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", top_n)
    bars = plt.barh(range(len(emotions)), counts, color=colors, edgecolor='black')
    plt.yticks(range(len(emotions)), emotions, fontsize=10)
    plt.xlabel('样本数量', fontsize=11)
    plt.ylabel('情绪标签', fontsize=11)
    plt.title(f'Top {top_n} 情绪标签分布', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count + max(counts)*0.01, i, f'{count:,}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至: {save_path}")
    
    plt.show()


def save_dataset_info(train_size, val_size, test_size, save_path=None):
    """保存数据集信息"""
    if save_path is None:
        save_path = PROJECT_ROOT / "docs" / "dataset_info.json"
    
    info = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'total_size': train_size + val_size + test_size,
        'split_ratio': {
            'train': round(train_size / (train_size + val_size + test_size), 3),
            'val': round(val_size / (train_size + val_size + test_size), 3),
            'test': round(test_size / (train_size + val_size + test_size), 3)
        }
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 数据集信息已保存至: {save_path}")
    
    return info


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 加载标签映射
    label_mapping = load_label_mapping()
    print(f"✓ 标签映射加载完成，共 {len(label_mapping)} 个标签")
    
    print("\n前5个标签:")
    for i in range(5):
        print(f"  {i}: {label_mapping[i]}")