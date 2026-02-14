# 细粒度对话情绪识别

## 数据处理模块

### 使用方法

1. 下载数据集
```bash
python src/data_processing/download_data.py
```

2. 数据预处理
```bash
python src/data_processing/preprocess.py
```

3. 数据增强
```bash
python src/data_processing/data_augmentation.py
```

4. 测试数据加载器
```bash
python src/data_processing/data_loader.py
```

### 文件结构
```
data/
├── raw/              # 原始数据
├── processed/        # 预处理后的数据
└── label_mapping.json  # 标签映射

src/data_processing/
├── download_data.py     # 数据下载
├── preprocess.py        # 数据预处理
├── data_augmentation.py # 数据增强
├── data_loader.py       # 数据加载
└── utils.py            # 工具函数

docs/
└── data_report.md      # 数据报告

notebooks/
└── data_exploration.ipynb  # 数据探索
```

### 数据集信息
- 数据来源：GoEmotions
- 训练集：43,275条（增强后：60,585条）
- 验证集：5,408条
- 测试集：5,410条
- 情绪类别：28种