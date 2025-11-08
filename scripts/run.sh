#!/bin/bash

# =============================================
# Transformer从零实现 - 支持IWSLT数据集
# =============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 设置环境变量
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 默认配置
SEED=42
CONFIG_FILE="configs/base.yaml"
DATA_DIR="data"
RESULTS_DIR="results"
CHECKPOINTS_DIR="checkpoints"
USE_IWSLT=false
SRC_LANG="en"
TGT_LANG="de"

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --seed SEED                  设置随机种子 (默认: 42)"
    echo "  --config FILE                配置文件路径 (默认: configs/base.yaml)"
    echo "  --data-dir DIR               数据目录 (默认: data)"
    echo "  --results-dir DIR            结果目录 (默认: results)"
    echo "  --checkpoints-dir DIR        检查点目录 (默认: checkpoints)"
    echo "  --use-iwslt                  使用IWSLT数据集而不是本地数据"
    echo "  --src-lang LANG              源语言代码 (默认: en)"
    echo "  --tgt-lang LANG              目标语言代码 (默认: de)"
    echo "  --help                       显示此帮助信息"
    echo ""
    echo "语言对示例:"
    echo "  --src-lang en --tgt-lang de  英语到德语"
    echo "  --src-lang en --tgt-lang fr  英语到法语"
    echo "  --src-lang de --tgt-lang en  德语到英语"
    echo ""
    echo "示例:"
    echo "  $0 --use-iwslt --src-lang en --tgt-lang de"
    echo "  $0 --config configs/iwslt.yaml"
    echo "  $0 --debug"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --checkpoints-dir)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        --use-iwslt)
            USE_IWSLT=true
            shift
            ;;
        --src-lang)
            SRC_LANG="$2"
            shift 2
            ;;
        --tgt-lang)
            TGT_LANG="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

log_info "开始运行Transformer实验"
log_info "随机种子: $SEED"
log_info "配置文件: $CONFIG_FILE"
log_info "数据目录: $DATA_DIR"
log_info "结果目录: $RESULTS_DIR"
log_info "IWSLT模式: $USE_IWSLT"
if [ "$USE_IWSLT" = true ]; then
    log_info "翻译方向: $SRC_LANG -> $TGT_LANG"
fi

# 创建目录结构
create_directories() {
    log_info "创建目录结构..."

    mkdir -p $DATA_DIR
    mkdir -p $RESULTS_DIR
    mkdir -p $CHECKPOINTS_DIR
    mkdir -p logs
    mkdir -p configs

    log_success "目录创建完成"
}

# 检查CUDA可用性
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi | grep -q "NVIDIA-SMI"; then
            log_success "检测到NVIDIA GPU"
            CUDA_AVAILABLE=true
        else
            log_warning "未检测到可用的NVIDIA GPU，将使用CPU"
            CUDA_AVAILABLE=false
        fi
    else
        log_warning "未安装nvidia-smi，将使用CPU"
        CUDA_AVAILABLE=false
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装Python依赖..."

    # 安装基础依赖
    pip install numpy matplotlib tqdm pyyaml

    # 安装PyTorch（根据CUDA可用性选择版本）
    if [ "$CUDA_AVAILABLE" = true ]; then
        log_info "安装PyTorch (CUDA版本)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "安装PyTorch (CPU版本)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # 安装数据集相关依赖
    log_info "安装数据集处理库..."
    pip install datasets huggingface-hub sacremoses

    log_success "依赖安装完成"
}

# 准备IWSLT配置
setup_iwslt_config() {
    if [ "$USE_IWSLT" = true ]; then
        log_info "设置IWSLT配置文件..."

        # 创建IWSLT专用配置
        cat > configs/iwslt.yaml << EOF
# IWSLT 2017数据集配置
model:
  d_model: 256
  num_heads: 8
  num_layers: 3
  d_ff: 1024
  dropout: 0.1
  max_seq_length: 128

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0003
  weight_decay: 0.01
  max_grad_norm: 1.0
  scheduler_step_size: 20
  scheduler_gamma: 0.5

data:
  path: "$DATA_DIR"
  vocab_size: 10000
  use_iwslt: true
  src_lang: "$SRC_LANG"
  tgt_lang: "$TGT_LANG"

experiment:
  seed: $SEED
  data_dir: "$DATA_DIR"
  results_dir: "$RESULTS_DIR"
  checkpoints_dir: "$CHECKPOINTS_DIR"
  log_interval: 100
  save_interval: 10
EOF

        CONFIG_FILE="configs/iwslt.yaml"
        log_success "IWSLT配置文件已创建: $CONFIG_FILE"
    fi
}

# 准备数据
prepare_data() {
    log_info "准备数据集..."

    if [ "$USE_IWSLT" = true ]; then
        log_info "使用IWSLT 2017数据集 ($SRC_LANG -> $TGT_LANG)"
        log_warning "注意: IWSLT数据集将从Hugging Face下载，可能需要一些时间"
    else
        # 检查input.txt是否存在
        if [ -f "$DATA_DIR/input.txt" ]; then
            log_info "找到 input.txt 文件"
            # 显示文件信息
            file_size=$(stat -c%s "$DATA_DIR/input.txt" 2>/dev/null || stat -f%z "$DATA_DIR/input.txt")
            line_count=$(wc -l < "$DATA_DIR/input.txt")
            word_count=$(wc -w < "$DATA_DIR/input.txt")
            char_count=$(wc -m < "$DATA_DIR/input.txt")

            echo "文件信息:"
            echo "  - 大小: $file_size 字节"
            echo "  - 行数: $line_count"
            echo "  - 单词数: $word_count"
            echo "  - 字符数: $char_count"
        else
            log_warning "未找到 input.txt 文件，将创建示例文件"
        fi
    fi

    # 运行数据准备脚本
    python src/data_loader.py

    log_success "数据准备完成"
}

# 训练基础模型
train_baseline() {
    log_info "开始训练Transformer模型..."

    local start_time=$(date +%s)

    local train_cmd="python src/train.py \
        --config $CONFIG_FILE \
        --seed $SEED \
        --data-dir $DATA_DIR \
        --results-dir $RESULTS_DIR \
        --checkpoints-dir $CHECKPOINTS_DIR"

    # 添加调试模式
    if [ "$DEBUG_MODE" = true ]; then
        train_cmd="$train_cmd --debug"
    fi

    eval $train_cmd 2>&1 | tee logs/training_${SEED}_$(date +%Y%m%d_%H%M%S).log

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "模型训练完成，耗时: $((duration / 60))分$((duration % 60))秒"
}

# 运行消融实验
run_ablation_study() {
    log_info "开始消融实验..."

    local start_time=$(date +%s)

    python src/ablation_study.py \
        --seed $SEED \
        --data-dir $DATA_DIR \
        --results-dir $RESULTS_DIR \
        2>&1 | tee logs/ablation_${SEED}_$(date +%Y%m%d_%H%M%S).log

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "消融实验完成，耗时: $((duration / 60))分$((duration % 60))秒"
}

# 生成文本示例
generate_examples() {
    log_info "生成文本示例..."

    # 创建独立的Python脚本来生成示例
    cat > /tmp/generate_examples.py << 'EOF'
import os
import sys
import torch

# 添加当前目录到Python路径
sys.path.append('.')

try:
    from src.model import Transformer
    from src.data_loader import TextDataset
    from src.utils import generate_text

    # 检查模型文件
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print('未找到训练好的模型，请先完成训练')
        exit(0)

    # 加载检查点获取配置
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    vocab = checkpoint['vocab']

    # 创建idx2char映射
    idx2char = {idx: char for char, idx in vocab.items()}

    # 创建模型
    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    )

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('=' * 50)
    print('文本生成示例:')
    print('=' * 50)

    # 不同的起始文本
    prompts = [
        'To be or not to',
        'Once upon a time',
        'The king said:',
        'Machine learning',
        'Hello world'
    ]

    for prompt in prompts:
        generated = generate_text(model, prompt, vocab, idx2char, max_length=50)
        print(f'输入: \"{prompt}\"')
        print(f'生成: {generated}')
        print('-' * 40)

except Exception as e:
    print(f'生成文本时出错: {e}')
    import traceback
    traceback.print_exc()
EOF

    python /tmp/generate_examples.py
    rm -f /tmp/generate_examples.py

    log_success "文本生成完成"
}

# 分析实验结果
analyze_results() {
    log_info "分析实验结果..."

    # 创建独立的Python脚本来分析结果
    cat > /tmp/analyze_results.py << 'EOF'
import json
import os
import glob

def analyze_training_results():
    # 分析训练结果
    results_dir = 'results'

    # 检查训练日志
    log_files = glob.glob('logs/training_*.log')
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f'分析最新训练日志: {latest_log}')

        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 提取最终结果
        final_lines = [line for line in lines if any(x in line for x in ['验证损失', 'Epoch', '最佳验证损失', '训练完成'])]
        if final_lines:
            print('训练结果摘要:')
            for line in final_lines[-10:]:  # 显示最后10行相关结果
                print(line.strip())

    # 检查消融实验结果
    ablation_file = os.path.join(results_dir, 'ablation_results.json')
    if os.path.exists(ablation_file):
        with open(ablation_file, 'r', encoding='utf-8') as f:
            ablation_results = json.load(f)

        print('\\n消融实验结果:')
        print('模型变体                验证损失    困惑度')
        print('-' * 50)

        labels = {
            'full': '完整模型',
            'no_pos_encoding': '无位置编码',
            'single_head': '单头注意力',
            'no_residual': '无残差连接',
            'no_layernorm': '无LayerNorm'
        }

        for model_type, results in ablation_results.items():
            loss = results['final_val_loss']
            ppl = results['final_perplexity']
            label = labels.get(model_type, model_type)
            print(f'{label:20} {loss:.4f}     {ppl:.2f}')

    # 检查训练曲线图
    curves_file = os.path.join(results_dir, 'training_curves.png')
    if os.path.exists(curves_file):
        print(f'\\n训练曲线已保存: {curves_file}')

    ablation_plot = os.path.join(results_dir, 'ablation_comparison.png')
    if os.path.exists(ablation_plot):
        print(f'消融实验对比图已保存: {ablation_plot}')

    # 检查检查点
    checkpoint_files = glob.glob('checkpoints/*.pth')
    if checkpoint_files:
        print(f'\\n找到 {len(checkpoint_files)} 个模型检查点')
        for cf in checkpoint_files[-3:]:  # 显示最近3个检查点
            print(f'  - {os.path.basename(cf)}')

analyze_training_results()
EOF

    python /tmp/analyze_results.py
    rm -f /tmp/analyze_results.py

    log_success "结果分析完成"
}

# 生成实验报告
generate_report() {
    log_info "生成实验报告..."

    # 创建独立的Python脚本来生成报告
    cat > /tmp/generate_report.py << 'EOF'
import datetime
import os
import json
import glob

def generate_summary_report():
    # 获取数据文件信息
    data_info = ""
    input_file = 'data/input.txt'
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        data_info = f"数据集信息: {len(text)} 字符, {len(text.split())} 单词"
    else:
        data_info = "数据集信息: 使用IWSLT或内置示例数据"

    # 检查配置
    config_info = "使用默认配置"
    config_files = glob.glob('configs/*.yaml')
    if config_files:
        config_info = f"使用配置文件: {', '.join([os.path.basename(f) for f in config_files])}"

    report = f'''Transformer从零实现 - 实验报告
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
随机种子: 42

{data_info}
{config_info}

目录结构:
- 数据目录: data
- 结果目录: results
- 检查点目录: checkpoints
- 日志目录: logs
- 配置目录: configs

实验步骤:
1. ✅ 环境配置和依赖安装
2. ✅ 数据集准备
3. ✅ Transformer模型训练
4. ✅ 消融实验分析
5. ✅ 文本生成示例
6. ✅ 结果分析和可视化

关键文件:
- 训练配置: configs/*.yaml
- 训练日志: logs/training_*.log
- 消融实验日志: logs/ablation_*.log
- 训练曲线: results/training_curves.png
- 消融对比: results/ablation_comparison.png
- 最佳模型: checkpoints/best_model.pth

复现命令:
./scripts/run.sh --seed 42 --config configs/base.yaml

注意事项:
- 确保有足够的磁盘空间
- 完整实验运行时间取决于数据集大小和硬件
- 查看日志文件了解详细训练过程
- IWSLT数据集需要网络连接下载
'''

    # 添加消融实验结果
    ablation_file = 'results/ablation_results.json'
    if os.path.exists(ablation_file):
        try:
            with open(ablation_file, 'r', encoding='utf-8') as f:
                ablation_results = json.load(f)

            report += '\\n消融实验结果:\\n'
            report += '模型变体                验证损失    困惑度\\n'
            report += '-' * 50 + '\\n'

            labels = {
                'full': '完整模型',
                'no_pos_encoding': '无位置编码',
                'single_head': '单头注意力',
                'no_residual': '无残差连接',
                'no_layernorm': '无LayerNorm'
            }

            for model_type, results in ablation_results.items():
                loss = results['final_val_loss']
                ppl = results['final_perplexity']
                label = labels.get(model_type, model_type)
                report += f'{label:20} {loss:.4f}     {ppl:.2f}\\n'
        except Exception as e:
            report += f'\\n消融实验结果: 无法读取 ({e})\\n'

    report_file = 'results/experiment_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'实验报告已生成: {report_file}')
    print('\\n报告摘要:')
    print(report[:500] + '...' if len(report) > 500 else report)

generate_summary_report()
EOF

    python /tmp/generate_report.py
    rm -f /tmp/generate_report.py

    log_success "实验报告生成完成"
}

# 主函数
main() {
    log_info "=== Transformer实验开始 ==="

    # 执行各个步骤
    create_directories
    check_cuda
    install_dependencies
    setup_iwslt_config
    prepare_data
    train_baseline

    # 只有在非IWSLT模式下才运行消融实验（因为IWSLT训练时间较长）
    if [ "$USE_IWSLT" != true ]; then
        run_ablation_study
    else
        log_info "跳过消融实验（IWSLT模式）"
    fi

    generate_examples
    analyze_results
    generate_report

    log_success "=== 所有实验步骤完成 ==="
    log_info "实验结果保存在: $RESULTS_DIR"
    log_info "训练日志保存在: logs/"
    log_info "模型检查点保存在: $CHECKPOINTS_DIR"

    # 显示总结信息
    echo ""
    echo "==========================================="
    echo "实验完成总结:"
    echo "==========================================="
    echo "📊 查看训练曲线: $RESULTS_DIR/training_curves.png"
    echo "🔬 查看消融实验: $RESULTS_DIR/ablation_comparison.png"
    echo "📝 查看实验报告: $RESULTS_DIR/experiment_report.txt"
    echo "🤖 测试文本生成: 运行 python src/utils.py"
    if [ "$USE_IWSLT" = true ]; then
        echo "🌍 数据集: IWSLT 2017 ($SRC_LANG -> $TGT_LANG)"
    else
        echo "📁 数据集: 本地 input.txt"
    fi
    echo ""
    echo "要重新运行特定步骤，可以单独执行相应的函数"
}

# 运行主函数
main "$@"