"""
数据增强脚本
功能：
1. 同义词替换 (Synonym Replacement)
2. 随机插入 (Random Insertion)
3. 随机交换 (Random Swap)
4. 随机删除 (Random Deletion)
"""

import pandas as pd
import numpy as np
import random
import nlpaug.augmenter.word as naw
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class TextAugmenter:
    """文本增强器"""
    
    def __init__(self):
        # 同义词替换增强器
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        
        # 随机词增强器
        self.insert_aug = naw.RandomWordAug(action='insert')
        self.swap_aug = naw.RandomWordAug(action='swap')
        self.delete_aug = naw.RandomWordAug(action='delete')
    
    def _ensure_string(self, result, original_text):
        """确保返回字符串而不是列表"""
        if result is None:
            return original_text
        if isinstance(result, list):
            return result[0] if len(result) > 0 else original_text
        return result
    
    def augment_synonym(self, text, n=1):
        """同义词替换"""
        try:
            augmented = self.synonym_aug.augment(text, n=n)
            return self._ensure_string(augmented, text)
        except:
            return text
    
    def augment_insert(self, text):
        """随机插入词"""
        try:
            augmented = self.insert_aug.augment(text)
            return self._ensure_string(augmented, text)
        except:
            return text
    
    def augment_swap(self, text):
        """随机交换词序"""
        try:
            augmented = self.swap_aug.augment(text)
            return self._ensure_string(augmented, text)
        except:
            return text
    
    def augment_delete(self, text):
        """随机删除词"""
        try:
            augmented = self.delete_aug.augment(text)
            return self._ensure_string(augmented, text)
        except:
            return text
    
    def augment_text(self, text, method='synonym'):
        """
        增强单条文本
        method: 'synonym', 'insert', 'swap', 'delete', 'random'
        """
        if method == 'synonym':
            return self.augment_synonym(text)
        elif method == 'insert':
            return self.augment_insert(text)
        elif method == 'swap':
            return self.augment_swap(text)
        elif method == 'delete':
            return self.augment_delete(text)
        elif method == 'random':
            # 随机选择一种方法
            methods = ['synonym', 'insert', 'swap', 'delete']
            chosen_method = random.choice(methods)
            return self.augment_text(text, method=chosen_method)
        else:
            return text


def augment_dataset(df, augmenter, augment_ratio=0.3, methods=['synonym', 'swap']):
    """
    增强数据集
    
    参数:
        df: 原始数据框
        augmenter: 增强器实例
        augment_ratio: 增强比例（例如0.3表示增强30%的数据）
        methods: 使用的增强方法列表
    
    返回:
        增强后的数据框
    """
    print(f"\n开始数据增强...")
    print(f"原始数据: {len(df)} 条")
    print(f"增强比例: {augment_ratio*100}%")
    print(f"增强方法: {', '.join(methods)}")
    
    # 计算需要增强的样本数
    n_augment = int(len(df) * augment_ratio)
    
    # 随机选择需要增强的样本
    augment_indices = np.random.choice(df.index, size=n_augment, replace=False)
    
    augmented_samples = []
    
    for idx in tqdm(augment_indices, desc="增强数据"):
        row = df.loc[idx]
        text = str(row['text'])  # 确保是字符串
        
        # 对每种方法生成一个增强样本
        for method in methods:
            aug_text = augmenter.augment_text(text, method=method)
            
            # 再次确保是字符串
            aug_text = str(aug_text)
            
            # 创建新样本
            new_sample = row.copy()
            new_sample['text'] = aug_text
            new_sample['id'] = f"{row['id']}_aug_{method}"
            new_sample['text_length'] = len(aug_text.split())
            
            augmented_samples.append(new_sample)
    
    # 合并原始数据和增强数据
    augmented_df = pd.DataFrame(augmented_samples)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # 打乱顺序
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"✓ 增强后数据: {len(combined_df)} 条 (新增 {len(augmented_samples)} 条)")
    
    return combined_df


def main():
    """主函数"""
    print("=" * 70)
    print(" " * 20 + "数据增强工具")
    print("=" * 70)
    
    # 加载预处理后的数据
    print("\n📂 加载预处理数据...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    
    # 解析labels列
    import ast
    train_df['labels'] = train_df['labels'].apply(ast.literal_eval)
    
    print(f"✓ 训练集: {len(train_df)} 条")
    
    # 初始化增强器
    print("\n🔧 初始化增强器...")
    augmenter = TextAugmenter()
    print("✓ 增强器初始化完成")
    
    # 数据增强
    augmented_train = augment_dataset(
        train_df, 
        augmenter,
        augment_ratio=0.2,  # 增强20%的数据
        methods=['synonym', 'swap']  # 使用同义词替换和随机交换
    )
    
    # 保存增强后的数据
    print("\n💾 保存增强数据...")
    output_path = PROCESSED_DATA_DIR / "train_augmented.csv"
    augmented_train.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ 已保存至: {output_path}")
    
    # 统计报告
    print("\n" + "=" * 70)
    print("增强统计报告:")
    print("=" * 70)
    print(f"原始训练集: {len(train_df):,} 条")
    print(f"增强后训练集: {len(augmented_train):,} 条")
    print(f"增强率: {(len(augmented_train) - len(train_df)) / len(train_df) * 100:.2f}%")
    
    print("\n✅ 数据增强完成！")
    print(f"📁 保存位置: {output_path}")
    print("\n下一步: 运行 data_loader.py 创建数据加载器")
    print("=" * 70)


if __name__ == "__main__":
    main()