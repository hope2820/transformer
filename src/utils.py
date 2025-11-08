import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_curves(train_losses: List[float], val_losses: List[float],
                         perplexities: List[float], save_path: str):
    """保存训练曲线"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和验证损失')

    plt.subplot(1, 3, 2)
    plt.plot(perplexities)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('验证集困惑度')

    plt.subplot(1, 3, 3)
    plt.semilogy(train_losses, label='训练损失(log)')
    plt.semilogy(val_losses, label='验证损失(log)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('对数尺度损失')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_text(model, start_string, vocab, idx2char, max_length=100, temperature=1.0):
    """使用训练好的模型生成文本 - 修复设备问题"""
    model.eval()

    # 获取模型所在的设备
    device = next(model.parameters()).device

    # 编码起始字符串
    input_ids = [vocab.get(char, vocab['<unk>']) for char in start_string]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    generated = start_string

    with torch.no_grad():
        for _ in range(max_length):
            # 创建mask
            src_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2)

            # 前向传播
            output = model(input_tensor, input_tensor, src_mask, src_mask)

            # 获取最后一个时间步的预测
            last_logits = output[0, -1, :] / temperature
            probabilities = torch.softmax(last_logits, dim=-1)

            # 采样下一个字符
            next_char_id = torch.multinomial(probabilities, 1).item()
            next_char = idx2char.get(next_char_id, '<unk>')

            generated += next_char

            # 更新输入，确保新张量也在正确的设备上
            input_ids.append(next_char_id)
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

            if next_char in ['.', '!', '?', '\n'] and len(generated) > 50:
                break

    return generated


def analyze_attention_weights(model, input_text, vocab, layer_idx=0, head_idx=0):
    """分析注意力权重"""
    model.eval()

    # 获取设备
    device = next(model.parameters()).device

    # 编码输入文本
    input_ids = [vocab.get(char, vocab['<unk>']) for char in input_text]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        # 获取编码器输出和注意力权重
        src_embedded = model.dropout(
            model.positional_encoding(model.src_embedding(input_tensor) *
                                      np.sqrt(model.d_model))
        )

        # 通过指定层获取注意力权重
        enc_output = src_embedded
        for i, layer in enumerate(model.encoder_layers):
            if i == layer_idx:
                # 保存注意力权重
                attn_output, attn_weights = layer.self_attn(
                    enc_output, enc_output, enc_output
                )
                break
            enc_output = layer(enc_output)

    # 提取指定头的注意力权重
    head_weights = attn_weights[0, head_idx].cpu().numpy()

    return head_weights