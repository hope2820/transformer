## 依赖与环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包概览

| **包名**       | **版本** | **用途**           |
|----------------|----------|--------------------|
| PyTorch        | ≥2.0     | 深度学习框架       |
| NumPy          | ≥1.21    | 数值计算           |
| Matplotlib     | ≥3.5     | 可视化             |
| tqdm           | ≥4.64    | 进度条             |
| PyYAML         | ≥6.0     | 配置文件           |
| datasets       | ≥2.10    | IWSLT数据集处理    |
| sacremoses     | ≥0.0.53  | 文本分词           |

> **表 1.** 项目依赖包列表

---

## 代码目录结构

```bash
transformer-from-scratch/
├── configs/                    # 配置文件目录
│   └── iwslt.yaml              # IWSLT专用配置
├── data/                       # 数据目录
│   └── iwslt2017/              # IWSLT数据集
├── src/                        # 源代码
│   ├── data_loader.py          # 数据加载和预处理
│   ├── model.py                # Transformer模型实现
│   ├── train.py                # 训练流程
│   ├── utils.py                # 工具函数
│   └── ablation_study.py       # 消融实验
├── results/                    # 实验结果
│   └── training_curves.png     # 训练可视化
├── checkpoints/                # 模型检查点
│   ├── best_model.pth          # 最佳模型
│   └── checkpoint_epoch_*.pth  # 训练检查点
├── run.sh                      # Linux/macOS自动化脚本
├── run.bat                     # Windows自动化脚本
└── README.md                   # 项目文档
```

---

## 命令行使用示例

### 训练模型及消融实验

```bash
# 使用提供的自动化脚本
./run.sh
```

---

## 预期运行时间与硬件要求

### 使用的硬件规格

| **组件** | **规格** |
|-----------|-----------|
| CPU | 2核心 |
| GPU | NVIDIA T4 × 2 (16GB显存) |
| 内存 | 16GB DDR4 |
| 存储 | 20GB 临时空间 |

> **表 2.** 硬件配置要求

---

### 运行时间预估

| **实验类型** | **数据集** | **训练轮数** | **预估时间** | **GPU显存** |
|---------------|-------------|---------------|----------------|----------------|
| IWSLT训练 | IWSLT2017 | 30 | ~6小时 | 8-10 GB |
| 消融实验 | 本地文本 | 10 | ~2小时 | 4-6 GB |
| 推理 | 任意 | - | < 1秒 | 2-3 GB |

> **表 3.** 实验运行时间预估

---
