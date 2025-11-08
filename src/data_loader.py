import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
import random
import xml.etree.ElementTree as ET
import glob
import re


class IWSLTXMLParser:
    """专门处理IWSLT数据格式的解析器"""

    def __init__(self):
        pass

    def parse_training_files(self, src_file, tgt_file):
        """解析训练文件 - 处理混合格式（XML标签 + 纯文本）"""
        print(f"解析训练文件: {src_file} -> {tgt_file}")

        src_sentences = self._parse_mixed_format_file(src_file)
        tgt_sentences = self._parse_mixed_format_file(tgt_file)

        print(f"从训练文件中提取了 {len(src_sentences)} 个源语句和 {len(tgt_sentences)} 个目标语句")

        # 确保对齐
        min_len = min(len(src_sentences), len(tgt_sentences))
        return src_sentences[:min_len], tgt_sentences[:min_len]

    def _parse_mixed_format_file(self, file_path):
        """解析混合格式文件（包含XML标签和纯文本）"""
        sentences = []

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        current_sentence = ""
        in_text_section = False

        for line in lines:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 跳过XML标签行（但记录是否进入正文）
            if line.startswith('<'):
                if line.startswith('<description>') or any(tag in line for tag in ['<p>', '<seg']):
                    in_text_section = True
                elif line.startswith('</description>') or any(tag in line for tag in ['</p>', '</seg']):
                    in_text_section = False
                    # 保存当前句子
                    if current_sentence and len(current_sentence) > 10:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                continue

            # 如果是正文部分，收集句子
            if in_text_section:
                # 简单的句子分割（按句号、问号、感叹号）
                parts = re.split(r'[.!?]', line)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 5:  # 最小长度
                        sentences.append(part)
            else:
                # 非XML标签的纯文本行（可能是演讲内容）
                if line and not line.startswith('<') and len(line) > 10:
                    # 同样进行句子分割
                    parts = re.split(r'[.!?]', line)
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 5:
                            sentences.append(part)

        # 处理最后一个句子
        if current_sentence and len(current_sentence) > 10:
            sentences.append(current_sentence.strip())

        # 过滤和清理
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if cleaned and len(cleaned) > 5:
                cleaned_sentences.append(cleaned)

        print(f"从 {file_path} 中提取了 {len(cleaned_sentences)} 个有效句子")
        return cleaned_sentences

    def _clean_sentence(self, sentence):
        """清理句子"""
        # 移除多余的空白字符
        sentence = re.sub(r'\s+', ' ', sentence)
        # 移除特殊字符但保留字母、数字和基本标点
        sentence = re.sub(r'[^\w\s.,!?;:\'\"-]', '', sentence)
        return sentence.strip()

    def parse_xml_files(self, src_file, tgt_file):
        """解析XML格式的验证/测试文件 - 确保按id对齐"""
        print(f"解析XML文件: {src_file} -> {tgt_file}")

        # 按seg id提取句子，确保对齐
        src_sentences_dict = self._parse_xml_file_with_id(src_file)
        tgt_sentences_dict = self._parse_xml_file_with_id(tgt_file)

        # 找到共同的id，确保对齐
        common_ids = sorted(set(src_sentences_dict.keys()) & set(tgt_sentences_dict.keys()))
        
        src_sentences = [src_sentences_dict[id] for id in common_ids]
        tgt_sentences = [tgt_sentences_dict[id] for id in common_ids]

        print(f"从XML文件中提取了 {len(src_sentences)} 个对齐的句子对")

        return src_sentences, tgt_sentences
    
    def _parse_xml_file_with_id(self, xml_file):
        """解析XML文件，返回按id索引的句子字典"""
        sentences_dict = {}

        try:
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 提取seg标签及其id属性
            seg_pattern = r'<seg[^>]*id="(\d+)"[^>]*>(.*?)</seg>'
            matches = re.findall(seg_pattern, content, re.DOTALL)

            for seg_id, sentence in matches:
                sentence = sentence.strip()
                if sentence and len(sentence) > 5:
                    cleaned = self._clean_sentence(sentence)
                    if cleaned:
                        sentences_dict[int(seg_id)] = cleaned

            # 如果没有找到带id的seg标签，回退到按顺序提取
            if not sentences_dict:
                sentences_list = self._parse_xml_file(xml_file)
                for idx, sentence in enumerate(sentences_list):
                    sentences_dict[idx] = sentence

            print(f"从 {xml_file} 中提取了 {len(sentences_dict)} 个句子")

        except Exception as e:
            print(f"解析XML文件 {xml_file} 失败: {e}")
            # 备选方案：按顺序提取
            sentences_list = self._parse_xml_file(xml_file)
            for idx, sentence in enumerate(sentences_list):
                sentences_dict[idx] = sentence

        return sentences_dict

    def _parse_xml_file(self, xml_file):
        """解析单个XML文件"""
        sentences = []

        try:
            # 读取整个文件内容
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 使用正则表达式提取所有seg标签内容
            seg_pattern = r'<seg[^>]*>(.*?)</seg>'
            matches = re.findall(seg_pattern, content, re.DOTALL)

            for match in matches:
                sentence = match.strip()
                if sentence and len(sentence) > 5:
                    cleaned = self._clean_sentence(sentence)
                    if cleaned:
                        sentences.append(cleaned)

            print(f"从 {xml_file} 中提取了 {len(sentences)} 个句子")

        except Exception as e:
            print(f"解析XML文件 {xml_file} 失败: {e}")
            # 备选方案：逐行读取
            sentences = self._parse_xml_file_fallback(xml_file)

        return sentences

    def _parse_xml_file_fallback(self, xml_file):
        """备选的XML解析方法"""
        sentences = []

        with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
            in_seg = False
            current_sentence = ""

            for line in f:
                line = line.strip()

                if line.startswith('<seg'):
                    in_seg = True
                    # 提取seg标签内的内容
                    match = re.search(r'<seg[^>]*>(.*)', line)
                    if match:
                        current_sentence = match.group(1)
                        # 检查是否在同一行结束
                        if '</seg>' in current_sentence:
                            sentence = current_sentence.split('</seg>')[0]
                            cleaned = self._clean_sentence(sentence)
                            if cleaned:
                                sentences.append(cleaned)
                            in_seg = False
                            current_sentence = ""
                elif in_seg:
                    if '</seg>' in line:
                        sentence = current_sentence + ' ' + line.split('</seg>')[0]
                        cleaned = self._clean_sentence(sentence)
                        if cleaned:
                            sentences.append(cleaned)
                        in_seg = False
                        current_sentence = ""
                    else:
                        current_sentence += ' ' + line

        return sentences


class IWSLTLocalDataset(Dataset):
    def __init__(self, data_path, split='train', max_length=128, vocab_size=10000,
                 src_lang='en', tgt_lang='de', train_ratio=0.9, vocab=None, idx2char=None):
        self.max_length = max_length
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.vocab_size = vocab_size
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.parser = IWSLTXMLParser()

        # 支持的语言列表
        self.supported_languages = ['en', 'de', 'fr', 'it', 'nl', 'ro', 'ar', 'ja', 'ko', 'zh']

        # 如果提供了词汇表，使用它（用于验证集共享训练集词汇表）
        if vocab is not None and idx2char is not None:
            self.vocab = vocab
            self.idx2char = idx2char
            self.vocab_size = len(vocab)
            print(f"使用提供的词汇表，大小: {self.vocab_size}")

        # 验证语言对
        self._validate_language_pair()

        # 加载本地IWSLT数据集
        self._load_iwslt_data_corrected()

        print(f"加载 {split} 数据集 ({src_lang}->{tgt_lang}): {len(self.samples)} 样本")
        if hasattr(self, 'vocab'):
            print(f"词汇表大小: {self.vocab_size}")

    def _validate_language_pair(self):
        """验证语言对是否有效"""
        if self.src_lang not in self.supported_languages:
            raise ValueError(f"不支持的源语言: {self.src_lang}，支持的语言: {self.supported_languages}")
        if self.tgt_lang not in self.supported_languages:
            raise ValueError(f"不支持的目标语言: {self.tgt_lang}，支持的语言: {self.supported_languages}")
        if self.src_lang == self.tgt_lang:
            raise ValueError("源语言和目标语言不能相同")

    def _load_iwslt_data_corrected(self):
        """修正的数据加载方法，正确处理IWSLT格式"""
        print(f"加载IWSLT数据集 ({self.src_lang}->{self.tgt_lang})...")

        try:
            lang_pair = f"{self.src_lang}-{self.tgt_lang}"

            if self.split == 'train':
                # 训练集：处理混合格式
                self._load_train_data(lang_pair)
            else:
                # 验证/测试集：处理XML格式
                self._load_eval_data(lang_pair)

        except Exception as e:
            print(f"加载IWSLT数据集失败: {e}")
            print("使用演示数据代替...")
            self._create_demo_dataset()

    def _load_train_data(self, lang_pair):
        """加载训练数据 - 确保逐行对齐"""
        print("加载训练数据（逐行对齐）...")

        # 查找训练文件
        src_file, tgt_file = self._find_train_files(lang_pair)

        if not src_file or not tgt_file:
            raise FileNotFoundError(f"无法找到训练数据文件")

        print(f"训练源文件: {src_file}")
        print(f"训练目标文件: {tgt_file}")

        # 逐行读取，确保对齐
        src_sentences = []
        tgt_sentences = []
        
        with open(src_file, 'r', encoding='utf-8', errors='ignore') as src_f, \
             open(tgt_file, 'r', encoding='utf-8', errors='ignore') as tgt_f:
            
            src_lines = src_f.readlines()
            tgt_lines = tgt_f.readlines()
            
            # 确保两个文件的行数一致
            min_lines = min(len(src_lines), len(tgt_lines))
            print(f"源文件行数: {len(src_lines)}, 目标文件行数: {len(tgt_lines)}, 对齐后行数: {min_lines}")
            
            for i in range(min_lines):
                src_line = src_lines[i].strip()
                tgt_line = tgt_lines[i].strip()
                
                # 跳过空行和XML标签行
                if not src_line or not tgt_line:
                    continue
                if src_line.startswith('<') or tgt_line.startswith('<'):
                    continue
                
                # 清理句子
                src_cleaned = self._clean_sentence(src_line)
                tgt_cleaned = self._clean_sentence(tgt_line)
                
                # 只保留有效长度的句子对
                if src_cleaned and tgt_cleaned and len(src_cleaned) > 5 and len(tgt_cleaned) > 5:
                    src_sentences.append(src_cleaned)
                    tgt_sentences.append(tgt_cleaned)

        print(f"从训练数据中提取了 {len(src_sentences)} 个对齐的句子对")

        if len(src_sentences) == 0:
            raise ValueError("未能从训练文件中提取到有效的句子对")

        # 构建词汇表
        self._build_vocab_from_pairs(src_sentences, tgt_sentences)

        # 划分训练集
        split_idx = int(len(src_sentences) * self.train_ratio)
        src_train = src_sentences[:split_idx]
        tgt_train = tgt_sentences[:split_idx]

        self.samples = self._create_samples_from_pairs(src_train, tgt_train)

    def _extract_train_sentences(self, content):
        """从训练数据中提取句子 - 改进的方法"""
        sentences = []
        lines = content.split('\n')

        current_sentence = ""
        in_speech = False

        for line in lines:
            line = line.strip()

            # 跳过空行和XML标签行
            if not line or line.startswith('<'):
                # 如果遇到新的演讲开始标记
                if '<speaker>' in line or '<title>' in line or '<description>' in line:
                    in_speech = True
                    continue
                # 如果遇到段落结束，保存当前句子
                elif current_sentence and len(current_sentence) > 10:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                continue

            # 如果是演讲内容
            if in_speech or len(line) > 20:  # 假设长文本是演讲内容
                # 简单的句子分割
                parts = re.split(r'[.!?]', line)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 10:  # 最小长度
                        # 清理文本
                        cleaned = self._clean_sentence(part)
                        if cleaned and len(cleaned) > 10:
                            sentences.append(cleaned)
            else:
                # 处理短行（可能是对话）
                if line and len(line) > 5:
                    cleaned = self._clean_sentence(line)
                    if cleaned:
                        current_sentence += " " + cleaned

        # 处理最后一个句子
        if current_sentence and len(current_sentence) > 10:
            sentences.append(current_sentence.strip())

        # 去重和过滤
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            if sentence not in seen and len(sentence) > 10:
                seen.add(sentence)
                unique_sentences.append(sentence)

        print(f"提取了 {len(unique_sentences)} 个唯一句子")
        return unique_sentences

    def _load_eval_data(self, lang_pair):
        """加载验证/测试数据 - 处理XML格式"""
        print(f"加载{self.split}数据（XML格式）...")

        if self.split == 'val':
            src_file, tgt_file = self._find_validation_files(lang_pair)
        else:  # test
            src_file, tgt_file = self._find_test_files(lang_pair)

        if not src_file or not tgt_file:
            print(f"未找到{self.split}集文件，从训练集划分")
            self._create_eval_from_train(lang_pair)
            return

        print(f"{self.split}源文件: {src_file}")
        print(f"{self.split}目标文件: {tgt_file}")

        # 使用XML解析器
        src_sentences, tgt_sentences = self.parser.parse_xml_files(src_file, tgt_file)
        
        print(f"从{self.split}数据中提取了 {len(src_sentences)} 个对齐的句子对")

        if len(src_sentences) == 0:
            print(f"警告：未能从{self.split}文件中提取到有效句子对，尝试从训练集划分")
            self._create_eval_from_train(lang_pair)
            return

        # 如果还没有词汇表，从验证集构建（虽然不理想，但作为备选）
        if not hasattr(self, 'vocab') or self.vocab is None:
            print("警告：验证集没有词汇表，从验证数据构建词汇表（建议使用训练集的词汇表）...")
            self._build_vocab_from_pairs(src_sentences, tgt_sentences)

        self.samples = self._create_samples_from_pairs(src_sentences, tgt_sentences)

    def _create_eval_from_train(self, lang_pair):
        """从训练集创建验证/测试集 - 确保逐行对齐"""
        src_file, tgt_file = self._find_train_files(lang_pair)

        if not src_file or not tgt_file:
            raise FileNotFoundError("无法找到训练数据文件以创建验证/测试集")

        print(f"从训练文件创建{self.split}集: {src_file}, {tgt_file}")
        
        # 逐行读取，确保对齐
        src_sentences = []
        tgt_sentences = []
        
        with open(src_file, 'r', encoding='utf-8', errors='ignore') as src_f, \
             open(tgt_file, 'r', encoding='utf-8', errors='ignore') as tgt_f:
            
            src_lines = src_f.readlines()
            tgt_lines = tgt_f.readlines()
            
            # 确保两个文件的行数一致
            min_lines = min(len(src_lines), len(tgt_lines))
            
            for i in range(min_lines):
                src_line = src_lines[i].strip()
                tgt_line = tgt_lines[i].strip()
                
                # 跳过空行和XML标签行
                if not src_line or not tgt_line:
                    continue
                if src_line.startswith('<') or tgt_line.startswith('<'):
                    continue
                
                # 清理句子
                src_cleaned = self._clean_sentence(src_line)
                tgt_cleaned = self._clean_sentence(tgt_line)
                
                # 只保留有效长度的句子对
                if src_cleaned and tgt_cleaned and len(src_cleaned) > 5 and len(tgt_cleaned) > 5:
                    src_sentences.append(src_cleaned)
                    tgt_sentences.append(tgt_cleaned)

        print(f"从训练数据中提取了 {len(src_sentences)} 个对齐的句子对")

        if len(src_sentences) == 0:
            raise ValueError("未能从训练文件中提取到有效的句子对")

        # 构建词汇表（如果还没有）
        if not hasattr(self, 'vocab') or self.vocab is None:
            print("从训练数据构建词汇表...")
            self._build_vocab_from_pairs(src_sentences, tgt_sentences)

        # 划分数据集
        total_size = len(src_sentences)
        train_size = int(total_size * self.train_ratio)
        val_size = int((total_size - train_size) * 0.5)  # 剩余的一半作为验证集

        if self.split == 'val':
            src_data = src_sentences[train_size:train_size + val_size]
            tgt_data = tgt_sentences[train_size:train_size + val_size]
        else:  # test
            src_data = src_sentences[train_size + val_size:]
            tgt_data = tgt_sentences[train_size + val_size:]

        print(f"创建{self.split}集: {len(src_data)} 个句子对")
        self.samples = self._create_samples_from_pairs(src_data, tgt_data)

    def _build_vocab_from_pairs(self, src_texts, tgt_texts):
        """从文本对构建词汇表 - 改进版本"""
        print("正在构建词汇表...")
        counter = Counter()

        # 收集所有文本
        all_text = ""
        batch_size = 10000  # 分批处理以避免内存问题

        for i in range(0, len(src_texts), batch_size):
            batch_src = src_texts[i:i + batch_size]
            batch_tgt = tgt_texts[i:i + batch_size]

            for src_text, tgt_text in zip(batch_src, batch_tgt):
                all_text += src_text + " " + tgt_text + " "

            print(f"已处理 {min(i + batch_size, len(src_texts))}/{len(src_texts)} 个句对...")

        # 按字符构建词汇表
        counter.update(all_text)
        print(f"发现 {len(counter)} 个唯一字符")

        # 显示最常见的字符
        print(f"最常见的30个字符: {counter.most_common(30)}")

        # 创建词汇表，保留位置给特殊标记
        most_common = counter.most_common(self.vocab_size - 4)

        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }

        for char, count in most_common:
            if len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)

        self.idx2char = {idx: char for char, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"词汇表构建完成，大小: {self.vocab_size}")

    def _find_train_files(self, lang_pair):
        """修复的文件查找方法"""
        print(f"查找训练文件，语言对: {lang_pair}")

        # 明确指定期望的文件名
        expected_src_file = os.path.join(self.data_path, f"train.tags.{lang_pair}.{self.src_lang}")
        expected_tgt_file = os.path.join(self.data_path, f"train.tags.{lang_pair}.{self.tgt_lang}")

        print(f"期望的源文件: {expected_src_file}")
        print(f"期望的目标文件: {expected_tgt_file}")

        # 首先尝试直接使用期望的文件名
        if os.path.exists(expected_src_file) and os.path.exists(expected_tgt_file):
            print("找到期望的文件")
            return expected_src_file, expected_tgt_file

        # 如果找不到，尝试反向语言对
        reverse_lang_pair = f"{self.tgt_lang}-{self.src_lang}"
        print(f"尝试反向语言对: {reverse_lang_pair}")

        # 对于反向语言对，源文件和目标文件需要交换
        if self.src_lang == 'en' and self.tgt_lang == 'de':
            # 如果是英德方向，但数据集是德英，需要交换文件
            expected_src_file_reverse = os.path.join(self.data_path, f"train.tags.{reverse_lang_pair}.{self.tgt_lang}")
            expected_tgt_file_reverse = os.path.join(self.data_path, f"train.tags.{reverse_lang_pair}.{self.src_lang}")

            print(f"反向源文件: {expected_src_file_reverse}")
            print(f"反目标文件: {expected_tgt_file_reverse}")

            if os.path.exists(expected_src_file_reverse) and os.path.exists(expected_tgt_file_reverse):
                print("找到反向语言对的文件，交换源和目标")
                # 注意：这里需要交换返回的文件顺序
                return expected_tgt_file_reverse, expected_src_file_reverse

        # 如果还找不到，使用原有的备选方案
        possible_patterns = [
            f"train.tags.{lang_pair}.{self.src_lang}",
            f"train.tags.{self.src_lang}-{self.tgt_lang}.{self.src_lang}",
            f"train.{self.src_lang}",
            f"*train*{self.src_lang}*",
            f"*{self.src_lang}*train*"
        ]

        src_file = None
        tgt_file = None

        # 查找源文件
        for pattern in possible_patterns:
            matches = glob.glob(os.path.join(self.data_path, pattern))
            for match in matches:
                # 检查文件内容，确保是英文
                if self._is_english_file(match):
                    src_file = match
                    print(f"找到源文件: {src_file}")
                    break
            if src_file:
                break

        # 查找目标文件
        tgt_patterns = [
            f"train.tags.{lang_pair}.{self.tgt_lang}",
            f"train.tags.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}",
            f"train.{self.tgt_lang}",
            f"*train*{self.tgt_lang}*",
            f"*{self.tgt_lang}*train*"
        ]

        for pattern in tgt_patterns:
            matches = glob.glob(os.path.join(self.data_path, pattern))
            for match in matches:
                # 检查文件内容，确保是德文
                if self._is_german_file(match):
                    tgt_file = match
                    print(f"找到目标文件: {tgt_file}")
                    break
            if tgt_file:
                break

        if not src_file or not tgt_file:
            print(f"无法找到正确的训练文件")
            print(f"可用的文件:")
            for file in os.listdir(self.data_path):
                if 'train' in file:
                    print(f"  {file}")

        return src_file, tgt_file

    def _is_english_file(self, file_path):
        """检查文件内容是否为英文"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # 读取前1000个字符
                # 简单的英文检测
                english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'for']
                english_count = sum(1 for word in english_words if word in content.lower())
                return english_count > 5
        except:
            return False

    def _is_german_file(self, file_path):
        """检查文件内容是否为德文"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # 读取前1000个字符
                # 简单的德文检测
                german_words = ['der', 'die', 'das', 'und', 'ist', 'zu', 'den', 'von', 'sich', 'mit']
                german_count = sum(1 for word in german_words if word in content.lower())
                return german_count > 5
        except:
            return False

    def _find_validation_files(self, lang_pair):
        """查找验证集XML文件 - 修复文件匹配逻辑"""
        print(f"查找验证集文件，语言对: {lang_pair}, 源语言: {self.src_lang}, 目标语言: {self.tgt_lang}")
        
        possible_patterns = [
            f"IWSLT17.TED.dev2010.{lang_pair}.{self.src_lang}.xml",
            f"dev2010.{lang_pair}.{self.src_lang}.xml",
            f"*dev*{self.src_lang}*.xml",
            f"*valid*{self.src_lang}*.xml",
            f"*val*{self.src_lang}*.xml"
        ]

        src_file = None
        tgt_file = None

        for pattern_idx, pattern in enumerate(possible_patterns):
            full_pattern = os.path.join(self.data_path, pattern)
            print(f"尝试模式 {pattern_idx+1}/{len(possible_patterns)}: {pattern}")
            src_matches = glob.glob(full_pattern)
            
            if src_matches:
                # 如果有多个匹配，选择最精确的（包含源语言的文件）
                if len(src_matches) > 1:
                    # 优先选择包含完整语言对且以源语言结尾的文件
                    preferred = [f for f in src_matches 
                                if lang_pair in os.path.basename(f) 
                                and f.endswith(f".{self.src_lang}.xml")]
                    if preferred:
                        src_file = preferred[0]
                    else:
                        # 其次选择以源语言结尾的文件
                        src_lang_files = [f for f in src_matches if f.endswith(f".{self.src_lang}.xml")]
                        if src_lang_files:
                            src_file = src_lang_files[0]
                        else:
                            # 最后选择包含语言对的文件
                            lang_pair_files = [f for f in src_matches if lang_pair in os.path.basename(f)]
                            if lang_pair_files:
                                src_file = lang_pair_files[0]
                            else:
                                src_file = src_matches[0]
                else:
                    src_file = src_matches[0]
                
                # 验证选中的文件确实是源语言文件
                if not src_file.endswith(f".{self.src_lang}.xml"):
                    print(f"  警告: 选中的文件不是源语言文件: {os.path.basename(src_file)}")
                    # 尝试从匹配中找到源语言文件
                    src_lang_files = [f for f in src_matches if f.endswith(f".{self.src_lang}.xml")]
                    if src_lang_files:
                        src_file = src_lang_files[0]
                        print(f"  已更正为: {os.path.basename(src_file)}")
                
                print(f"  找到源文件: {os.path.basename(src_file)}")
                
                # 基于源文件路径，替换最后的语言代码来找到目标文件
                # 例如: IWSLT17.TED.dev2010.en-de.en.xml -> IWSLT17.TED.dev2010.en-de.de.xml
                if src_file.endswith(f".{self.src_lang}.xml"):
                    # 方法1: 直接替换文件路径中的语言代码
                    tgt_file = src_file[:-len(f".{self.src_lang}.xml")] + f".{self.tgt_lang}.xml"
                    print(f"  尝试目标文件: {os.path.basename(tgt_file)}")
                    
                    if os.path.exists(tgt_file):
                        print(f"[OK] 找到验证集文件对:")
                        print(f"  源文件: {os.path.basename(src_file)}")
                        print(f"  目标文件: {os.path.basename(tgt_file)}")
                        return src_file, tgt_file
                    else:
                        print(f"  目标文件不存在，尝试其他方法...")
                        
                        # 方法2: 使用模式匹配查找目标文件
                        # 构建目标文件的模式
                        if lang_pair in pattern:
                            # 如果模式中包含语言对，精确替换
                            tgt_pattern = pattern.replace(f"{lang_pair}.{self.src_lang}.xml", f"{lang_pair}.{self.tgt_lang}.xml")
                        else:
                            # 否则只替换最后的语言代码
                            tgt_pattern = pattern.replace(f".{self.src_lang}.xml", f".{self.tgt_lang}.xml")
                        
                        tgt_full_pattern = os.path.join(self.data_path, tgt_pattern)
                        tgt_matches = glob.glob(tgt_full_pattern)
                        
                        if tgt_matches:
                            tgt_file = tgt_matches[0]
                            print(f"[OK] 通过模式匹配找到验证集文件对:")
                            print(f"  源文件: {os.path.basename(src_file)}")
                            print(f"  目标文件: {os.path.basename(tgt_file)}")
                            return src_file, tgt_file
                        else:
                            print(f"  模式匹配也未找到目标文件")
                            # 继续尝试下一个模式
                            continue
                else:
                    print(f"  警告: 源文件格式不符合预期 (不以 .{self.src_lang}.xml 结尾)")
                    # 尝试使用模式匹配
                    tgt_pattern = pattern.replace(f".{self.src_lang}.xml", f".{self.tgt_lang}.xml")
                    tgt_full_pattern = os.path.join(self.data_path, tgt_pattern)
                    tgt_matches = glob.glob(tgt_full_pattern)
                    if tgt_matches:
                        tgt_file = tgt_matches[0]
                        print(f"[OK] 找到验证集文件对:")
                        print(f"  源文件: {os.path.basename(src_file)}")
                        print(f"  目标文件: {os.path.basename(tgt_file)}")
                        return src_file, tgt_file
            else:
                print(f"  未找到匹配文件")

        if not src_file or not tgt_file:
            print(f"[WARNING] 未找到验证集文件对")
            print(f"  已尝试的模式: {possible_patterns}")
            print(f"  数据目录: {self.data_path}")
            if os.path.exists(self.data_path):
                print(f"  目录中的XML文件:")
                for f in os.listdir(self.data_path):
                    if f.endswith('.xml') and ('dev' in f or 'val' in f):
                        print(f"    {f}")

        return src_file, tgt_file

    def _find_test_files(self, lang_pair):
        """查找测试集XML文件 - 修复文件匹配逻辑"""
        possible_patterns = [
            f"IWSLT17.TED.tst2010.{lang_pair}.{self.src_lang}.xml",
            f"IWSLT17.TED.tst2011.{lang_pair}.{self.src_lang}.xml",
            f"IWSLT17.TED.tst2012.{lang_pair}.{self.src_lang}.xml",
            f"IWSLT17.TED.tst2013.{lang_pair}.{self.src_lang}.xml",
            f"IWSLT17.TED.tst2014.{lang_pair}.{self.src_lang}.xml",
            f"IWSLT17.TED.tst2015.{lang_pair}.{self.src_lang}.xml",
            f"*test*{self.src_lang}*.xml",
            f"*tst*{self.src_lang}*.xml"
        ]

        src_file = None
        tgt_file = None

        for pattern in possible_patterns:
            src_matches = glob.glob(os.path.join(self.data_path, pattern))
            if src_matches:
                src_file = src_matches[0]
                # 基于源文件路径，替换最后的语言代码来找到目标文件
                # 例如: IWSLT17.TED.tst2010.en-de.en.xml -> IWSLT17.TED.tst2010.en-de.de.xml
                if src_file.endswith(f".{self.src_lang}.xml"):
                    tgt_file = src_file[:-len(f".{self.src_lang}.xml")] + f".{self.tgt_lang}.xml"
                    if os.path.exists(tgt_file):
                        print(f"找到测试集文件对:")
                        print(f"  源文件: {os.path.basename(src_file)}")
                        print(f"  目标文件: {os.path.basename(tgt_file)}")
                        break
                    else:
                        # 如果直接替换失败，尝试使用模式匹配
                        tgt_pattern = pattern.replace(f".{self.src_lang}.xml", f".{self.tgt_lang}.xml")
                        # 确保不会替换掉语言对中的语言代码
                        if lang_pair in pattern:
                            tgt_pattern = pattern.replace(f"{lang_pair}.{self.src_lang}.xml", f"{lang_pair}.{self.tgt_lang}.xml")
                        tgt_matches = glob.glob(os.path.join(self.data_path, tgt_pattern))
                        if tgt_matches:
                            tgt_file = tgt_matches[0]
                            print(f"找到测试集文件对:")
                            print(f"  源文件: {os.path.basename(src_file)}")
                            print(f"  目标文件: {os.path.basename(tgt_file)}")
                            break
                else:
                    # 如果文件名格式不符合预期，尝试简单的替换
                    tgt_pattern = pattern.replace(f".{self.src_lang}.xml", f".{self.tgt_lang}.xml")
                    tgt_matches = glob.glob(os.path.join(self.data_path, tgt_pattern))
                    if tgt_matches:
                        tgt_file = tgt_matches[0]
                        print(f"找到测试集文件对:")
                        print(f"  源文件: {os.path.basename(src_file)}")
                        print(f"  目标文件: {os.path.basename(tgt_file)}")
                        break

        return src_file, tgt_file

    def _clean_sentence(self, sentence):
        """清理句子 - 改进版本"""
        # 移除多余的空白字符
        sentence = re.sub(r'\s+', ' ', sentence)
        # 移除XML残留标签
        sentence = re.sub(r'<[^>]+>', '', sentence)
        # 移除特殊字符但保留字母、数字和基本标点
        sentence = re.sub(r'[^\w\s.,!?;:\'\"-]', '', sentence)
        return sentence.strip()

    def _create_samples_from_pairs(self, src_texts, tgt_texts):
        """从文本对创建样本"""
        print(f"正在创建样本，共有 {len(src_texts)} 个句对...")
        samples = []

        skipped_count = 0
        for i, (src_text, tgt_text) in enumerate(zip(src_texts, tgt_texts)):
            src_encoded = self._encode_text(src_text)
            tgt_encoded = self._encode_text(tgt_text)

            # 检查序列长度
            if len(src_encoded) > self.max_length or len(tgt_encoded) > self.max_length:
                skipped_count += 1
                continue

            # 填充到固定长度
            src_padded = self._pad_sequence(src_encoded, self.max_length)
            tgt_padded = self._pad_sequence(tgt_encoded, self.max_length)

            samples.append({
                'input': src_padded,
                'target': tgt_padded
            })

            if i % 10000 == 0 and i > 0:
                print(f"已创建 {i} 个样本...")

        print(f"样本创建完成，共 {len(samples)} 个样本 (跳过 {skipped_count} 个过长样本)")

        # 显示前几个样本用于验证
        if samples:
            print("\n前3个样本示例:")
            for i in range(min(3, len(samples))):
                src_sample = samples[i]['input']
                tgt_sample = samples[i]['target']

                src_text = ''.join([self.idx2char.get(idx, '?') for idx in src_sample if idx != 0])
                tgt_text = ''.join([self.idx2char.get(idx, '?') for idx in tgt_sample if idx != 0])

                print(f"样本 {i}:")
                print(f"  输入: {src_text}")
                print(f"  目标: {tgt_text}")
                print()

        return samples

    def _pad_sequence(self, sequence, length):
        """填充序列到指定长度"""
        if len(sequence) >= length:
            return sequence[:length]
        else:
            return sequence + [self.vocab['<pad>']] * (length - len(sequence))

    def _encode_text(self, text):
        """编码文本为索引序列"""
        encoded = []
        # 添加开始标记
        encoded.append(self.vocab['<start>'])

        for char in text:
            encoded.append(self.vocab.get(char, self.vocab['<unk>']))

        # 添加结束标记
        encoded.append(self.vocab['<end>'])
        return encoded

    def _create_demo_dataset(self):
        """创建演示数据集（备用）"""
        print("使用演示数据集...")
        # 根据语言对选择演示数据
        demo_pairs = self._get_demo_pairs()

        # 构建演示词汇表
        all_text = " ".join([src + " " + tgt for src, tgt in demo_pairs])
        self._build_vocab_from_text(all_text)

        self.samples = []
        for src_text, tgt_text in demo_pairs:
            src_encoded = self._encode_text(src_text)
            tgt_encoded = self._encode_text(tgt_text)

            src_padded = self._pad_sequence(src_encoded, self.max_length)
            tgt_padded = self._pad_sequence(tgt_encoded, self.max_length)

            self.samples.append({
                'input': src_padded,
                'target': tgt_padded
            })

    def _get_demo_pairs(self):
        """根据语言对返回相应的演示数据"""
        demo_data = {
            'en-de': [
                ("Hello world", "Hallo Welt"),
                ("How are you", "Wie geht es dir"),
                ("Good morning", "Guten Morgen"),
                ("Thank you", "Danke schön"),
                ("I love programming", "Ich liebe Programmierung")
            ],
            'en-fr': [
                ("Hello world", "Bonjour le monde"),
                ("How are you", "Comment allez-vous"),
                ("Good morning", "Bonjour"),
                ("Thank you", "Merci"),
                ("I love programming", "J'aime la programmation")
            ],
            'en-zh': [
                ("Hello world", "你好世界"),
                ("How are you", "你好吗"),
                ("Good morning", "早上好"),
                ("Thank you", "谢谢"),
                ("I love programming", "我喜欢编程")
            ],
            # 可以添加更多语言对的演示数据...
        }

        key = f"{self.src_lang}-{self.tgt_lang}"
        if key in demo_data:
            return demo_data[key]
        else:
            # 默认返回英语-德语的演示数据
            return demo_data['en-de']

    def _build_vocab_from_text(self, text):
        """从文本构建词汇表（演示用）"""
        counter = Counter(text)
        most_common = counter.most_common(self.vocab_size - 4)

        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }

        for char, count in most_common:
            if len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)

        self.idx2char = {idx: char for char, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample['input']), torch.tensor(sample['target'])


# 为了兼容性，保留原来的TextDataset类
class TextDataset(IWSLTLocalDataset):
    def __init__(self, data_path, split='train', max_length=128, vocab_size=10000,
                 src_lang='en', tgt_lang='de', use_iwslt=True):
        # 使用本地IWSLT数据
        super().__init__(data_path=data_path, split=split, max_length=max_length,
                         vocab_size=vocab_size, src_lang=src_lang, tgt_lang=tgt_lang)


def create_masks(src, tgt, device):
    """创建注意力mask"""
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape

    # 源语言mask（忽略pad位置）
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    # 目标语言mask（忽略pad位置，且防止看到未来信息）
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)

    return src_mask, tgt_mask


def prepare_dataset_from_input():
    """准备数据集"""
    data_dir = 'data'
    input_file = os.path.join(data_dir, 'input.txt')

    if not os.path.exists(input_file):
        print("input.txt 不存在，创建示例文件...")
        os.makedirs(data_dir, exist_ok=True)

        sample_text = """This is a sample text file for training the Transformer model.
You should replace this with your own input.txt file.
The model will learn to predict the next character in the sequence."""

        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)

        print(f"创建示例文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"从 {input_file} 加载数据集")
    print(f"数据集大小: {len(text)} 字符")

    return True


if __name__ == "__main__":
    # 测试多种语言对的数据加载
    test_language_pairs = [
        ('en', 'de'),
        # 可以添加更多语言对测试
    ]

    for src_lang, tgt_lang in test_language_pairs:
        try:
            print(f"\n测试 {src_lang}-{tgt_lang} 语言对...")

            print("测试训练集加载...")
            train_dataset = IWSLTLocalDataset(
                data_path="data/iwslt2017",
                split='train',
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            print(f"成功加载 {src_lang}-{tgt_lang} 训练集，样本数: {len(train_dataset)}")

            print("测试验证集加载...")
            val_dataset = IWSLTLocalDataset(
                data_path="data/iwslt2017",
                split='val',
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            print(f"成功加载 {src_lang}-{tgt_lang} 验证集，样本数: {len(val_dataset)}")

        except Exception as e:
            print(f"加载 {src_lang}-{tgt_lang} 失败: {e}")
            import traceback

            traceback.print_exc()