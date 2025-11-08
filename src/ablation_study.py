import torch
import torch.nn as nn
from model import Transformer
from data_loader import IWSLTLocalDataset, create_masks, TextDataset
import json
import os
import sys

# 添加路径
sys.path.append('.')


class AblationTransformer(Transformer):
    """用于消融实验的Transformer变体"""

    def __init__(self, *args, ablation_type='full', **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_type = ablation_type

        if ablation_type == 'no_pos_encoding':
            # 移除位置编码
            self.positional_encoding = nn.Identity()
        elif ablation_type == 'single_head':
            # 单头注意力
            for layer in self.encoder_layers:
                layer.self_attn.num_heads = 1
                layer.self_attn.d_k = self.d_model
        elif ablation_type == 'no_residual':
            # 移除残差连接
            for layer in self.encoder_layers:
                layer.norm1 = nn.Identity()
                layer.norm2 = nn.Identity()
        elif ablation_type == 'no_layernorm':
            # 移除LayerNorm
            for layer in self.encoder_layers:
                layer.norm1 = nn.Identity()
                layer.norm2 = nn.Identity()


def run_ablation_study(config):
    """运行消融实验"""
    ablation_types = [
        'full',  # 完整模型
        'no_pos_encoding',  # 无位置编码
        'single_head',  # 单头注意力
        'no_residual',  # 无残差连接
        'no_layernorm'  # 无LayerNorm
    ]

    results = {}

    for ablation_type in ablation_types:
        print(f"\n正在运行消融实验: {ablation_type}")

        # 加载IWSLT数据
        train_dataset = IWSLTLocalDataset(
            data_path=config['data_path'],
            split='train',
            max_length=config['max_seq_length'],
            vocab_size=config.get('vocab_size', 10000),
            src_lang=config.get('src_lang', 'en'),
            tgt_lang=config.get('tgt_lang', 'de')
        )

        # 创建模型
        model = AblationTransformer(
            src_vocab_size=train_dataset.vocab_size,
            tgt_vocab_size=train_dataset.vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_length=config['max_seq_length'],
            dropout=config['dropout'],
            ablation_type=ablation_type
        )

        # 训练和评估
        trainer = AblationTrainer(config, model, train_dataset)
        final_val_loss, final_perplexity = trainer.train_and_evaluate()

        results[ablation_type] = {
            'final_val_loss': final_val_loss,
            'final_perplexity': final_perplexity,
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses
        }

        print(f"{ablation_type} - 最终验证损失: {final_val_loss:.4f}, 困惑度: {final_perplexity:.4f}")

    # 保存结果
    results_dir = config.get('results_dir', 'results_iwslt')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制消融实验结果
    plot_ablation_results(results, results_dir)

    return results


# ablation_study.py

class AblationTrainer:
    """消融实验训练器 - 修复版本"""

    def __init__(self, config, model, train_dataset=None):
        """
        初始化训练器
        Args:
            config: 配置字典
            model: 模型实例
            train_dataset: 训练数据集（可选）
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # 数据加载 - 修复：正确处理数据集
        if train_dataset is not None:
            # 使用传入的数据集
            self.train_dataset = train_dataset
            # 创建验证集，使用相同的词汇表
            self.val_dataset = IWSLTLocalDataset(
                data_path=config.get('data_path', 'data'),
                split='val',
                max_length=config.get('max_seq_length', 128),
                vocab_size=config.get('vocab_size', 10000),
                src_lang=config.get('src_lang', 'en'),
                tgt_lang=config.get('tgt_lang', 'de'),
                vocab=train_dataset.vocab,  # 共享词汇表
                idx2char=train_dataset.idx2char
            )
        else:
            # 创建新的数据集
            self.train_dataset = IWSLTLocalDataset(
                data_path=config.get('data_path', 'data'),
                split='train',
                max_length=config.get('max_seq_length', 128),
                vocab_size=config.get('vocab_size', 10000),
                src_lang=config.get('src_lang', 'en'),
                tgt_lang=config.get('tgt_lang', 'de')
            )
            self.val_dataset = IWSLTLocalDataset(
                data_path=config.get('data_path', 'data'),
                split='val',
                max_length=config.get('max_seq_length', 128),
                vocab_size=config.get('vocab_size', 10000),
                src_lang=config.get('src_lang', 'en'),
                tgt_lang=config.get('tgt_lang', 'de'),
                vocab=self.train_dataset.vocab,  # 共享词汇表
                idx2char=self.train_dataset.idx2char
            )

        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.get('batch_size', 32)
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 训练记录
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for src, tgt in self.train_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 准备输入和目标序列
            tgt_input = tgt[:, :-1]  # 解码器输入（右移）
            tgt_target = tgt[:, 1:]  # 解码器目标

            # 创建注意力掩码
            src_mask, tgt_mask = create_masks(src, tgt_input, self.device)

            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # 计算损失
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_target.contiguous().view(-1)
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]

                src_mask, tgt_mask = create_masks(src, tgt_input, self.device)
                output = self.model(src, tgt_input, src_mask, tgt_mask)

                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_target.contiguous().view(-1)
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def train_and_evaluate(self):
        """训练并返回最终验证结果"""
        num_epochs = self.config.get('num_epochs', 10)  # 消融实验使用较少的epochs

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, perplexity = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 2 == 0:  # 每2个epoch打印一次
                print(f'Epoch {epoch + 1}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 困惑度={perplexity:.4f}')

        return val_loss, perplexity


def run_ablation_study(config):
    """运行消融实验 - 修复版本"""
    ablation_types = [
        'full',  # 完整模型
        'no_pos_encoding',  # 无位置编码
        'single_head',  # 单头注意力
        'no_residual',  # 无残差连接
        'no_layernorm'  # 无LayerNorm
    ]

    results = {}

    # 首先创建训练数据集（用于构建词汇表）
    train_dataset = IWSLTLocalDataset(
        data_path=config['data_path'],
        split='train',
        max_length=config['max_seq_length'],
        vocab_size=config.get('vocab_size', 10000),
        src_lang=config.get('src_lang', 'en'),
        tgt_lang=config.get('tgt_lang', 'de')
    )

    for ablation_type in ablation_types:
        print(f"\n正在运行消融实验: {ablation_type}")
        print("=" * 50)

        # 创建消融模型变体
        model = AblationTransformer(
            src_vocab_size=train_dataset.vocab_size,
            tgt_vocab_size=train_dataset.vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_length=config['max_seq_length'],
            dropout=config['dropout'],
            ablation_type=ablation_type
        )

        # 修复：只传递3个参数
        trainer = AblationTrainer(config, model, train_dataset)

        # 训练和评估
        final_val_loss, final_perplexity = trainer.train_and_evaluate()

        results[ablation_type] = {
            'final_val_loss': final_val_loss,
            'final_perplexity': final_perplexity,
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses
        }

        print(f"{ablation_type} - 最终验证损失: {final_val_loss:.4f}, 困惑度: {final_perplexity:.4f}")

    # 保存结果
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制消融实验结果
    plot_ablation_results(results, results_dir)

    return results


def plot_ablation_results(results, results_dir='results'):
    """绘制消融实验结果"""
    import matplotlib.pyplot as plt

    ablation_types = list(results.keys())
    final_losses = [results[ab_type]['final_val_loss'] for ab_type in ablation_types]
    final_perplexities = [results[ab_type]['final_perplexity'] for ab_type in ablation_types]

    # 创建中文标签映射
    labels = {
        'full': '完整模型',
        'no_pos_encoding': '无位置编码',
        'single_head': '单头注意力',
        'no_residual': '无残差连接',
        'no_layernorm': '无LayerNorm'
    }

    x_labels = [labels.get(ab_type, ab_type) for ab_type in ablation_types]

    plt.figure(figsize=(12, 5))

    # 验证损失图
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x_labels, final_losses, color='skyblue', alpha=0.7)
    plt.ylabel('最终验证损失')
    plt.title('不同模型变体的验证损失')
    plt.xticks(rotation=45)

    # 在柱状图上添加数值
    for bar, value in zip(bars1, final_losses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # 困惑度图
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x_labels, final_perplexities, color='lightcoral', alpha=0.7)
    plt.ylabel('最终困惑度')
    plt.title('不同模型变体的困惑度')
    plt.xticks(rotation=45)

    # 在柱状图上添加数值
    for bar, value in zip(bars2, final_perplexities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 消融实验配置
    config = {
        'data_path': '../data/iwslt2017',
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 2,
        'd_ff': 512,
        'max_seq_length': 128,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'num_epochs': 10,  # 消融实验使用较少的epochs
        'results_dir': 'results',
        'src_lang': 'en',
        'tgt_lang': 'de'
    }

    os.makedirs('results', exist_ok=True)

    try:
        results = run_ablation_study(config)
        print("\n消融实验完成!")
        print("结果已保存到 results/ablation_results.json")
        print("图表已保存到 results/ablation_comparison.png")
    except Exception as e:
        print(f"消融实验失败: {e}")
        import traceback

        traceback.print_exc()
