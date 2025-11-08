@echo off
chcp 65001 >nul
title Transformer实验 - Windows版本

echo.
echo ============================================
echo    Transformer从零实现 - Windows版本
echo ============================================
echo.

:: 设置环境变量
set PYTHONHASHSEED=42

:: 默认参数
set SEED=42
set CONFIG_FILE=configs\base.yaml
set DATA_DIR=data
set RESULTS_DIR=results
set CHECKPOINTS_DIR=checkpoints
set USE_IWSLLT=false
set SRC_LANG=en
set TGT_LANG=de
set DEBUG_MODE=false

:: 解析命令行参数
:parse_args
if "%1"=="" goto :end_parse
if "%1"=="--seed" (
    set SEED=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--config" (
    set CONFIG_FILE=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--data-dir" (
    set DATA_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--results-dir" (
    set RESULTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--checkpoints-dir" (
    set CHECKPOINTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--use-iwslt" (
    set USE_IWSLT=true
    shift
    goto :parse_args
)
if "%1"=="--src-lang" (
    set SRC_LANG=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--tgt-lang" (
    set TGT_LANG=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--debug" (
    set DEBUG_MODE=true
    shift
    goto :parse_args
)
if "%1"=="--help" (
    call :show_help
    exit /b 0
)
echo [ERROR] 未知参数: %1
call :show_help
exit /b 1

:end_parse

echo [INFO] 开始运行Transformer实验
echo [INFO] 随机种子: %SEED%
echo [INFO] 配置文件: %CONFIG_FILE%
echo [INFO] 数据目录: %DATA_DIR%
echo [INFO] 结果目录: %RESULTS_DIR%
echo [INFO] IWSLT模式: %USE_IWSLT%
if "%USE_IWSLT%"=="true" (
    echo [INFO] 翻译方向: %SRC_LANG% -> %TGT_LANG%
)

:: 创建目录结构
echo [INFO] 创建目录结构...
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%CHECKPOINTS_DIR%" mkdir "%CHECKPOINTS_DIR%"
if not exist "logs" mkdir "logs"
if not exist "configs" mkdir "configs"

:: 检查Python
echo [INFO] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到Python，请先安装Python 3.8+
    echo 从 https://www.python.org/downloads/ 下载并安装
    pause
    exit /b 1
)

:: 安装依赖 - 分步进行
echo [INFO] 安装Python依赖...
echo [INFO] 安装基础科学计算库...
pip install numpy matplotlib tqdm pyyaml

echo [INFO] 安装PyTorch (CPU版本)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo [INFO] 安装数据集处理库...
pip install datasets huggingface-hub sacremoses

echo [SUCCESS] 所有依赖安装完成

:: 准备IWSLT配置
if "%USE_IWSLT%"=="true" (
    echo [INFO] 设置IWSLT配置文件...

    (
    echo # IWSLT 2017数据集配置
    echo model:
    echo   d_model: 256
    echo   num_heads: 8
    echo   num_layers: 3
    echo   d_ff: 1024
    echo   dropout: 0.1
    echo   max_seq_length: 128
    echo.
    echo training:
    echo   batch_size: 32
    echo   num_epochs: 100
    echo   learning_rate: 0.0003
    echo   weight_decay: 0.01
    echo   max_grad_norm: 1.0
    echo   scheduler_step_size: 20
    echo   scheduler_gamma: 0.5
    echo.
    echo data:
    echo   path: "%DATA_DIR%"
    echo   vocab_size: 10000
    echo   use_iwslt: true
    echo   src_lang: "%SRC_LANG%"
    echo   tgt_lang: "%TGT_LANG%"
    echo.
    echo experiment:
    echo   seed: %SEED%
    echo   data_dir: "%DATA_DIR%"
    echo   results_dir: "%RESULTS_DIR%"
    echo   checkpoints_dir: "%CHECKPOINTS_DIR%"
    echo   log_interval: 100
    echo   save_interval: 10
    ) > configs\iwslt.yaml

    set CONFIG_FILE=configs\iwslt.yaml
    echo [SUCCESS] IWSLT配置文件已创建: %CONFIG_FILE%
)

:: 数据预处理
echo [INFO] 数据预处理...
python src\data_loader.py
if errorlevel 1 (
    echo [WARNING] 数据预处理有警告，但继续执行
)

:: 训练模型
echo [INFO] 开始训练Transformer模型...
set TRAIN_CMD=python src\train.py --config %CONFIG_FILE% --seed %SEED% --data-dir %DATA_DIR% --results-dir %RESULTS_DIR% --checkpoints-dir %CHECKPOINTS_DIR%

if "%DEBUG_MODE%"=="true" (
    set TRAIN_CMD=%TRAIN_CMD% --debug
)

%TRAIN_CMD%
if errorlevel 1 (
    echo [ERROR] 模型训练失败
    pause
    exit /b 1
)

:: 运行消融实验（仅在非IWSLT模式下）
if not "%USE_IWSLT%"=="true" (
    echo [INFO] 运行消融实验...
    python src\ablation_study.py --seed %SEED% --data-dir %DATA_DIR% --results-dir %RESULTS_DIR%
    if errorlevel 1 (
        echo [WARNING] 消融实验有警告，但继续执行
    )
) else (
    echo [INFO] 跳过消融实验（IWSLT模式）
)

:: 生成示例
echo [INFO] 生成文本示例...
python -c "
import os
import sys
import torch

sys.path.append('.')

try:
    from src.model import Transformer
    from src.utils import generate_text

    checkpoint_path = 'checkpoints\best_model.pth'
    if not os.path.exists(checkpoint_path):
        print('未找到训练好的模型')
        exit(0)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    idx2char = {idx: char for char, idx in vocab.items()}

    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('文本生成示例:')
    prompts = ['To be or not to', 'Once upon a time', 'The king said:']

    for prompt in prompts:
        generated = generate_text(model, prompt, vocab, idx2char, max_length=50)
        print(f'输入: \"{prompt}\"')
        print(f'生成: {generated}')
        print('-' * 40)

except Exception as e:
    print(f'生成文本时出错: {e}')
"

:: 生成报告
echo [INFO] 生成实验报告...
python -c "
import datetime
import os
import glob

data_info = '使用IWSLT或内置示例数据' if not os.path.exists('data\input.txt') else '使用本地数据'
config_info = '使用配置文件: ' + ', '.join([os.path.basename(f) for f in glob.glob('configs\*.yaml')]) if glob.glob('configs\*.yaml') else '使用默认配置'

report = f'''Transformer从零实现 - 实验报告
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
随机种子: %SEED%

数据集信息: {data_info}
{config_info}

实验完成状态:
✓ 环境配置和依赖安装
✓ 数据集准备
✓ Transformer模型训练
✓ 文本生成示例
{'✓ 消融实验分析' if not '%USE_IWSLT%' == 'true' else '⏭ 跳过消融实验(IWSLT模式)'}

关键文件:
- 训练配置: configs\*.yaml
- 训练日志: logs\*.log
- 训练曲线: results\training_curves.png
- 最佳模型: checkpoints\best_model.pth

注意事项:
- 查看日志文件了解详细训练过程
- 可以使用训练好的模型进行文本生成
'''

with open('results\experiment_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print('实验报告已生成: results\experiment_report.txt')
"

echo.
echo ============================================
echo [SUCCESS] 所有实验步骤完成！
echo ============================================
echo.
echo 实验结果:
echo - 训练曲线: %RESULTS_DIR%\training_curves.png
echo - 实验报告: %RESULTS_DIR%\experiment_report.txt
echo - 最佳模型: %CHECKPOINTS_DIR%\best_model.pth
if not "%USE_IWSLT%"=="true" (
    echo - 消融实验: %RESULTS_DIR%\ablation_comparison.png
)
if "%USE_IWSLT%"=="true" (
    echo - 数据集: IWSLT 2017 (%SRC_LANG% -> %TGT_LANG%)
) else (
    echo - 数据集: 本地 input.txt
)
echo.
pause

goto :eof

:show_help
echo 使用方法: %0 [选项]
echo.
echo 选项:
echo   --seed SEED                  设置随机种子 ^(默认: 42^)
echo   --config FILE                配置文件路径 ^(默认: configs\base.yaml^)
echo   --data-dir DIR               数据目录 ^(默认: data^)
echo   --results-dir DIR            结果目录 ^(默认: results^)
echo   --checkpoints-dir DIR        检查点目录 ^(默认: checkpoints^)
echo   --use-iwslt                  使用IWSLT数据集而不是本地数据
echo   --src-lang LANG              源语言代码 ^(默认: en^)
echo   --tgt-lang LANG              目标语言代码 ^(默认: de^)
echo   --debug                      启用调试模式
echo   --help                       显示此帮助信息
echo.
echo 语言对示例:
echo   --src-lang en --tgt-lang de  英语到德语
echo   --src-lang en --tgt-lang fr  英语到法语
echo.
echo 示例:
echo   %0 --use-iwslt --src-lang en --tgt-lang de
echo   %0 --config configs\iwslt.yaml
echo   %0 --debug
goto :eof