"""
数据预处理脚本
功能：
1. 文本清洗（去除特殊字符、URL、HTML标签等）
2. 文本标准化（小写化、去除多余空格）
3. 处理缺失值和异常值
4. 划分数据集（train/val/test）
5. 保存预处理后的数据
"""

import pandas as pd
import numpy as np
import re
import json
import ast
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "goemotions"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        # 加载标签映射
        label_mapping_path = PROJECT_ROOT / "data" / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)
                # 转换键为整数
                self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        else:
            # 如果文件不存在，使用默认映射
            self.label_mapping = {
                0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance',
                4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity',
                8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
                12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
                16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
                20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
                24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
            }
    
    def clean_text(self, text):
        """
        清洗文本
        - 去除URL
        - 去除HTML标签
        - 去除特殊字符（保留基本标点）
        - 去除多余空格
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 去除URL
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 去除邮箱地址
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除多个感叹号或问号（保留1-2个）
        text = re.sub(r'[!]{3,}', '!!', text)
        text = re.sub(r'[?]{3,}', '??', text)
        
        # 去除多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def normalize_text(self, text):
        """
        标准化文本
        - 转小写
        - 去除多余空格
        """
        if pd.isna(text) or text == "":
            return ""
        
        # 转小写
        text = text.lower()
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def parse_labels(self, label_str):
        """将字符串形式的列表转换为真正的列表"""
        if pd.isna(label_str):
            return []
        
        try:
            if isinstance(label_str, list):
                return label_str
            return ast.literal_eval(label_str)
        except:
            return []
    
    def filter_valid_samples(self, df):
        """
        过滤有效样本
        - 文本不为空
        - 至少有一个标签
        - 文本长度在合理范围内（2-100词）
        """
        valid_mask = (
            (df['text_clean'].str.len() > 0) &  # 文本不为空
            (df['labels'].apply(len) > 0) &      # 至少有一个标签
            (df['text_length'] >= 2) &           # 最少2个词
            (df['text_length'] <= 100)           # 最多100个词
        )
        
        return df[valid_mask].copy()
    
    def process_dataframe(self, df, split_name):
        """处理单个数据集"""
        print(f"\n处理 {split_name} 数据集...")
        print(f"原始数据: {len(df)} 条")
        
        # 解析labels列
        df['labels'] = df['labels'].apply(self.parse_labels)
        
        # 清洗文本
        tqdm.pandas(desc="清洗文本")
        df['text_clean'] = df['text'].progress_apply(self.clean_text)
        
        # 标准化文本
        tqdm.pandas(desc="标准化文本")
        df['text_normalized'] = df['text_clean'].progress_apply(self.normalize_text)
        
        # 计算文本长度
        df['text_length'] = df['text_normalized'].apply(lambda x: len(x.split()))
        
        # 过滤有效样本
        df_valid = self.filter_valid_samples(df)
        print(f"过滤后数据: {len(df_valid)} 条 (移除 {len(df) - len(df_valid)} 条)")
        
        # 选择需要的列
        df_final = df_valid[['id', 'text_normalized', 'labels', 'text_length']].copy()
        df_final.rename(columns={'text_normalized': 'text'}, inplace=True)
        
        return df_final


def main():
    """主函数"""
    print("=" * 70)
    print(" " * 20 + "数据预处理工具")
    print("=" * 70)
    
    # 初始化预处理器
    preprocessor = TextPreprocessor()
    
    # 加载原始数据
    print("\n📂 加载原始数据...")
    train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    val_df = pd.read_csv(RAW_DATA_DIR / "validation.csv")
    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
    
    print(f"✓ 训练集: {len(train_df)} 条")
    print(f"✓ 验证集: {len(val_df)} 条")
    print(f"✓ 测试集: {len(test_df)} 条")
    
    # 处理各个数据集
    train_processed = preprocessor.process_dataframe(train_df, "训练集")
    val_processed = preprocessor.process_dataframe(val_df, "验证集")
    test_processed = preprocessor.process_dataframe(test_df, "测试集")
    
    # 保存预处理后的数据
    print("\n💾 保存预处理后的数据...")
    train_processed.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False, encoding='utf-8')
    val_processed.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False, encoding='utf-8')
    test_processed.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False, encoding='utf-8')
    
    print(f"✓ 训练集已保存: {len(train_processed)} 条")
    print(f"✓ 验证集已保存: {len(val_processed)} 条")
    print(f"✓ 测试集已保存: {len(test_processed)} 条")
    
    # 生成统计报告
    print("\n" + "=" * 70)
    print("预处理统计报告:")
    print("=" * 70)
    
    total_original = len(train_df) + len(val_df) + len(test_df)
    total_processed = len(train_processed) + len(val_processed) + len(test_processed)
    
    print(f"原始总数据: {total_original:,} 条")
    print(f"预处理后总数据: {total_processed:,} 条")
    print(f"数据保留率: {total_processed/total_original*100:.2f}%")
    
    # 文本长度统计
    print(f"\n文本长度统计 (训练集):")
    print(f"  平均长度: {train_processed['text_length'].mean():.2f} 词")
    print(f"  最短: {train_processed['text_length'].min()} 词")
    print(f"  最长: {train_processed['text_length'].max()} 词")
    print(f"  中位数: {train_processed['text_length'].median():.2f} 词")
    
    # 标签统计
    all_labels = []
    for labels in train_processed['labels']:
        all_labels.extend(labels)
    
    from collections import Counter
    label_counts = Counter(all_labels)
    
    print(f"\n标签统计 (训练集):")
    print(f"  总标签数: {len(all_labels):,}")
    print(f"  唯一标签数: {len(label_counts)}")
    print(f"  平均标签/样本: {len(all_labels)/len(train_processed):.2f}")
    
    print("\n✅ 数据预处理完成！")
    print(f"📁 保存位置: {PROCESSED_DATA_DIR}")
    print("\n下一步: 运行 data_augmentation.py 进行数据增强")
    print("=" * 70)


if __name__ == "__main__":
    main()