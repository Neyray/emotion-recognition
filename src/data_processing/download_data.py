"""
数据集下载脚本
下载 GoEmotions 和 EmoBank 数据集
"""

import os
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def download_goemotions():
    """
    下载 GoEmotions 数据集
    GoEmotions 是 Google 发布的细粒度情绪分类数据集，包含 28 种情绪标签
    """
    print("=" * 50)
    print("开始下载 GoEmotions 数据集...")
    print("=" * 50)
    
    try:
        # 从 HuggingFace 下载数据集
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
        
        # 创建保存目录
        goemotions_dir = RAW_DATA_DIR / "goemotions"
        goemotions_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为 CSV 文件
        for split in ['train', 'validation', 'test']:
            df = pd.DataFrame(dataset[split])
            save_path = goemotions_dir / f"{split}.csv"
            df.to_csv(save_path, index=False, encoding='utf-8')
            print(f"✓ 已保存 {split} 集: {len(df)} 条数据 -> {save_path}")
        
        print(f"\n✅ GoEmotions 数据集下载完成！")
        print(f"   保存位置: {goemotions_dir}")
        
        # 显示数据集基本信息
        print(f"\n数据集统计:")
        print(f"  训练集: {len(dataset['train'])} 条")
        print(f"  验证集: {len(dataset['validation'])} 条")
        print(f"  测试集: {len(dataset['test'])} 条")
        
        # 显示样例数据
        print(f"\n样例数据 (前3条):")
        sample_df = pd.DataFrame(dataset['train'][:3])
        print(sample_df)
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False


def download_emobank():
    """
    下载 EmoBank 数据集
    EmoBank 是基于 VAD (Valence-Arousal-Dominance) 模型的情绪数据集
    """
    print("\n" + "=" * 50)
    print("开始下载 EmoBank 数据集...")
    print("=" * 50)
    
    try:
        # EmoBank 需要手动下载，这里提供下载链接和说明
        print("⚠️  EmoBank 数据集需要手动下载")
        print("\n下载步骤:")
        print("1. 访问: https://github.com/JULIELab/EmoBank")
        print("2. 下载 corpus 文件夹中的数据文件")
        print("3. 将文件放置到: data/raw/emobank/")
        
        emobank_dir = RAW_DATA_DIR / "emobank"
        emobank_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n目标文件夹已创建: {emobank_dir}")
        print("请手动下载后放入该文件夹")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建目录失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("\n" + "🚀" * 25)
    print(" " * 15 + "数据集下载工具")
    print("🚀" * 25 + "\n")
    
    # 确保数据目录存在
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载 GoEmotions
    goemotions_success = download_goemotions()
    
    # 下载 EmoBank (手动)
    emobank_success = download_emobank()
    
    # 总结
    print("\n" + "=" * 50)
    print("下载任务完成！")
    print("=" * 50)
    if goemotions_success:
        print("✅ GoEmotions: 已下载")
    else:
        print("❌ GoEmotions: 下载失败")
    
    if emobank_success:
        print("⚠️  EmoBank: 需要手动下载")
    
    print("\n下一步: 运行 preprocess.py 进行数据预处理")


if __name__ == "__main__":
    main()