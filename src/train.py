#!/usr/bin/env python3
"""
Transformer在IWSLT 2017数据集上的训练
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import yaml

sys.path.append('.')

from model import Transformer
from data_loader import IWSLTLocalDataset
from utils import set_seed, count_parameters


def create_masks(src, tgt_input, device):
    """创建注意力mask"""
    batch_size, src_len = src.shape
    _, tgt_len = tgt_input.shape

    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    tgt_pad_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(2)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)

    tgt_pad_mask_expanded = tgt_pad_mask.expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_pad_mask_expanded & tgt_sub_mask

    return src_mask, tgt_mask


def generate_translation(model, src_text, vocab, idx2char, max_length=100):
    """生成翻译"""
    model.eval()
    device = next(model.parameters()).device

    # 编码源文本
    src_encoded = [vocab.get(char, vocab['<unk>']) for char in src_text]
    src_tensor = torch.tensor(src_encoded).unsqueeze(0).to(device)

    # 起始标记
    start_token = vocab.get('<start>', vocab['<unk>'])
    tgt_encoded = [start_token]
    tgt_tensor = torch.tensor(tgt_encoded).unsqueeze(0).to(device)

    generated = ""

    with torch.no_grad():
        for _ in range(max_length):
            src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = create_masks(tgt_tensor, tgt_tensor, device)[1]

            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)

            # 获取最后一个预测
            last_logits = output[0, -1, :]
            next_token_id = torch.argmax(last_logits).item()
            next_char = idx2char.get(next_token_id, '<unk>')

            # 如果遇到结束标记或未知字符，停止生成
            if next_char in ['<end>', '<unk>', '.', '!', '?'] and len(generated) > 10:
                break

            generated += next_char
            tgt_encoded.append(next_token_id)
            tgt_tensor = torch.tensor(tgt_encoded).unsqueeze(0).to(device)

    return generated


class IWSLTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()

        # 从嵌套配置中获取种子值
        seed = config.get('experiment', {}).get('seed', 42)
        set_seed(seed)

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []

        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"模型参数数量: {count_parameters(self.model):,}")

    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("使用CPU")
        return device

    def _setup_data(self):
        """设置数据加载器"""
        print("初始化IWSLT数据加载器...")

        # 从嵌套配置中获取参数
        data_config = self.config.get('data', {})
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})

        data_path = data_config.get('path', 'data')
        max_length = model_config.get('max_seq_length', 128)
        vocab_size = data_config.get('vocab_size', 10000)
        src_lang = data_config.get('src_lang', 'en')
        tgt_lang = data_config.get('tgt_lang', 'de')
        batch_size = training_config.get('batch_size', 32)

        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)

        self.train_dataset = IWSLTLocalDataset(
            data_path=data_path,
            split='train',
            max_length=max_length,
            vocab_size=vocab_size,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )

        # 验证集使用训练集的词汇表，确保一致性
        self.val_dataset = IWSLTLocalDataset(
            data_path=data_path,
            split='val',
            max_length=max_length,
            vocab_size=vocab_size,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            vocab=self.train_dataset.vocab,
            idx2char=self.train_dataset.idx2char
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size
        )

        # 保存词汇表信息
        self.vocab = self.train_dataset.vocab
        self.idx2char = self.train_dataset.idx2char

    def _setup_model(self):
        print("初始化Transformer模型...")

        # 从嵌套配置中获取模型参数
        model_config = self.config.get('model', {})
        data_config = self.config.get('data', {})

        src_lang = data_config.get('src_lang', 'en')
        tgt_lang = data_config.get('tgt_lang', 'de')

        # 注意：这里假设源语言和目标语言使用相同的词汇表
        # 对于真正的翻译任务，可能需要为每种语言使用不同的词汇表
        vocab_size = len(self.vocab)

        self.model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=model_config.get('d_model', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 3),
            d_ff=model_config.get('d_ff', 1024),
            max_seq_length=model_config.get('max_seq_length', 128),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)

    def _setup_optimizer(self):
        # 从嵌套配置中获取训练参数
        training_config = self.config.get('training', {})

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.0003),
            weight_decay=training_config.get('weight_decay', 0.01)
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_config.get('scheduler_step_size', 10),
            gamma=training_config.get('scheduler_gamma', 0.5)
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        # 从嵌套配置中获取训练参数
        training_config = self.config.get('training', {})
        max_grad_norm = training_config.get('max_grad_norm', 1.0)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input, self.device)

            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_target.contiguous().view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="验证"):
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
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'perplexities': self.perplexities,
            'best_val_loss': self.best_val_loss,
            'vocab': self.vocab,
            'idx2char': self.idx2char,
            'config': self.config
        }

        # 从嵌套配置中获取检查点目录
        experiment_config = self.config.get('experiment', {})
        checkpoints_dir = experiment_config.get('checkpoints_dir', 'checkpoints_iwslt')

        os.makedirs(checkpoints_dir, exist_ok=True)
        torch.save(checkpoint, f'{checkpoints_dir}/{filename}')
        print(f"保存检查点: {checkpoints_dir}/{filename}")

    def train(self, num_epochs=None):
        # 从嵌套配置中获取训练参数
        training_config = self.config.get('training', {})
        experiment_config = self.config.get('experiment', {})

        if num_epochs is None:
            num_epochs = training_config.get('num_epochs', 50)

        save_interval = experiment_config.get('save_interval', 10)
        log_interval = experiment_config.get('log_interval', 100)

        print(f"开始训练 {num_epochs} 个epoch...")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            train_loss = self.train_epoch()
            val_loss, perplexity = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.perplexities.append(perplexity)

            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"困惑度: {perplexity:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_iwslt_model.pth')
                print(f"🔥 新的最佳模型! 验证损失: {val_loss:.4f}")

            # 按配置间隔保存检查点
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
                self.generate_examples()

        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time / 60:.2f}分钟")

        self.save_checkpoint('final_iwslt_model.pth')
        self.plot_results()

    def generate_examples(self):
        """生成翻译示例"""
        self.model.eval()

        print("\n翻译示例:")
        print("-" * 50)

        # 示例源文本
        test_sentences = [
            "Hello, how are you?",
            "What is your name?",
            "The weather is nice today.",
            "I love machine learning.",
            "This is a test sentence."
        ]

        with torch.no_grad():
            for src_text in test_sentences:
                translation = generate_translation(
                    self.model, src_text, self.vocab, self.idx2char
                )
                print(f"源: {src_text}")
                print(f"译: {translation}")
                print("-" * 30)

    def plot_results(self):
        """绘制训练结果"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练曲线')

        plt.subplot(1, 2, 2)
        plt.plot(self.perplexities)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('验证集困惑度')

        plt.tight_layout()

        # 从嵌套配置中获取结果目录
        experiment_config = self.config.get('experiment', {})
        results_dir = experiment_config.get('results_dir', 'results_iwslt')
        os.makedirs(results_dir, exist_ok=True)

        plt.savefig(f'{results_dir}/iwslt_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"训练结果已保存到 {results_dir}/iwslt_training_results.png")


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 加载YAML配置
    config_path = "configs/iwslt.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = {
            'model': {
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 3,
                'd_ff': 1024,
                'max_seq_length': 128,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 50,
                'learning_rate': 0.0003,
                'weight_decay': 0.01,
                'max_grad_norm': 1.0,
                'scheduler_step_size': 10,
                'scheduler_gamma': 0.5
            },
            'data': {
                'path': 'data',
                'vocab_size': 10000,
                'src_lang': 'en',
                'tgt_lang': 'de'
            },
            'experiment': {
                'seed': 42,
                'data_dir': 'data',
                'results_dir': 'results_iwslt',
                'checkpoints_dir': 'checkpoints_iwslt',
                'log_interval': 100,
                'save_interval': 10
            }
        }
    else:
        config = load_config(config_path)
        print(f"已加载配置文件: {config_path}")

    # 创建必要的目录
    experiment_config = config.get('experiment', {})
    checkpoints_dir = experiment_config.get('checkpoints_dir', 'checkpoints_iwslt')
    results_dir = experiment_config.get('results_dir', 'results_iwslt')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 训练模型
    trainer = IWSLTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()