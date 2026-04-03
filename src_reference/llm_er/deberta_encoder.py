"""
DeBERTa 编码器
使用 DeBERTa 模型计算语义期望向量 U
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import sys
# Add src directory to path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils import (
    get_path, load_config, logger,
    get_device, progress_bar,
    save_dataframe
)


class ConceptOccurrenceDataset(Dataset):
    """概念出现数据集"""

    def __init__(
        self,
        texts: List[str],
        concept_terms: List[str],
        start_positions: List[int],
        tokenizer,
        max_length: int = 512,
        pos_types: Optional[List[str]] = None
    ):
        self.texts = texts
        self.concept_terms = concept_terms
        self.start_positions = start_positions
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 词性类型：adjective, noun, adjective_phrase, noun_phrase
        self.pos_types = pos_types if pos_types else ['noun'] * len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        term = self.concept_terms[idx]
        start_pos = self.start_positions[idx]
        pos_type = self.pos_types[idx]

        # 创建带 [MASK] 的文本
        masked_text = self._create_masked_text(text, term, start_pos, pos_type)

        # Tokenize
        encoding = self.tokenizer(
            masked_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 找到 [MASK] token 的位置
        mask_token_id = self.tokenizer.mask_token_id
        input_ids = encoding['input_ids'].squeeze()
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        # 如果没有找到 [MASK]，使用第一个非特殊 token
        if len(mask_positions) == 0:
            mask_pos = 1  # 跳过 [CLS]
        else:
            mask_pos = mask_positions[0].item()

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'mask_position': mask_pos,
            'idx': idx
        }

    def _create_masked_text(self, text: str, term: str, start_pos: int, pos_type: str = 'noun') -> str:
        """
        将目标词替换为 [MASK]

        对于形容词类型(pos_type='adjective')：
        检测其后是否紧跟名词，如果是则把形容词和名词一起 mask
        例如："autonomous systems" -> "[MASK]"
        """
        term_lower = term.lower()
        text_lower = text.lower()

        # 如果是形容词，需要检查后面的名词
        if pos_type == 'adjective':
            return self._mask_adjective_with_noun(text, term, start_pos)

        # 非形容词情况：直接 mask 术语本身
        # 尝试精确位置
        if start_pos >= 0 and start_pos < len(text):
            end_pos = start_pos + len(term)
            if text_lower[start_pos:end_pos] == term_lower:
                return text[:start_pos] + '[MASK]' + text[end_pos:]

        # 否则搜索术语
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        match = pattern.search(text)

        if match:
            return text[:match.start()] + '[MASK]' + text[match.end():]

        # 如果找不到，返回原文（会使用默认位置）
        return text

    def _mask_adjective_with_noun(self, text: str, adjective: str, start_pos: int) -> str:
        """
        形容词+名词一起 mask

        查找形容词后面紧跟的词（假设是名词），一起替换为 [MASK]
        例如: "autonomous systems" -> "[MASK]"
              "the autonomous vehicle was" -> "the [MASK] was"
        """
        adj_lower = adjective.lower()
        text_lower = text.lower()

        # 首先找到形容词的位置
        adj_start = -1
        adj_end = -1

        # 尝试精确位置
        if start_pos >= 0 and start_pos < len(text):
            end_pos = start_pos + len(adjective)
            if text_lower[start_pos:end_pos] == adj_lower:
                adj_start = start_pos
                adj_end = end_pos

        # 如果精确位置没找到，搜索
        if adj_start < 0:
            pattern = re.compile(r'\b' + re.escape(adjective) + r'\b', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                adj_start = match.start()
                adj_end = match.end()

        if adj_start < 0:
            # 找不到形容词，返回原文
            return text

        # 查找形容词后面的词（名词）
        # 跳过空白字符，找到下一个词
        remaining = text[adj_end:]
        noun_match = re.match(r'^(\s+)(\w+)', remaining)

        if noun_match:
            # 找到了后续的词，把形容词和这个词一起 mask
            whitespace = noun_match.group(1)
            noun = noun_match.group(2)
            mask_end = adj_end + len(whitespace) + len(noun)
            return text[:adj_start] + '[MASK]' + text[mask_end:]
        else:
            # 形容词后面没有词，只 mask 形容词
            return text[:adj_start] + '[MASK]' + text[adj_end:]


class DeBERTaEncoder:
    """
    DeBERTa 编码器

    使用 DeBERTa 的 Masked Language Model 特性计算语义期望：
    将目标词替换为 [MASK]，取 [MASK] 位置的隐藏状态作为 U
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        初始化 DeBERTa 编码器

        Args:
            model_name: 模型名称
            device: 计算设备
            batch_size: 批处理大小
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # 设置设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """延迟加载 tokenizer"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """加载 DeBERTa 模型"""
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading DeBERTa model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Hidden size: {self._model.config.hidden_size}")

    @property
    def hidden_size(self) -> int:
        """获取隐藏层维度"""
        return self.model.config.hidden_size

    def encode_single(
        self,
        text: str,
        target_term: str,
        start_position: int = -1
    ) -> np.ndarray:
        """
        编码单个文本

        Args:
            text: 输入文本
            target_term: 目标术语
            start_position: 术语在文本中的起始位置

        Returns:
            语义期望向量 U (hidden_size,)
        """
        dataset = ConceptOccurrenceDataset(
            texts=[text],
            concept_terms=[target_term],
            start_positions=[start_position],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        item = dataset[0]

        with torch.no_grad():
            input_ids = item['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)
            mask_position = item['mask_position']

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 取 [MASK] 位置的隐藏状态
            hidden_states = outputs.last_hidden_state
            mask_embedding = hidden_states[0, mask_position, :]

        return mask_embedding.cpu().numpy()

    def encode_batch(
        self,
        occurrences_df: pd.DataFrame,
        text_column: str = 'text_block',
        term_column: str = 'matched_term',
        position_column: str = 'start_char',
        pos_type_column: str = 'pos_type',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        批量编码

        Args:
            occurrences_df: concept_occurrences DataFrame
            text_column: 文本列名
            term_column: 术语列名
            position_column: 位置列名
            pos_type_column: 词性类型列名 (adjective/noun/adjective_phrase/noun_phrase)
            show_progress: 是否显示进度

        Returns:
            语义期望矩阵 U (n_samples, hidden_size)
        """
        texts = occurrences_df[text_column].tolist()
        terms = occurrences_df[term_column].tolist()
        positions = occurrences_df[position_column].tolist() if position_column in occurrences_df.columns else [-1] * len(texts)

        # 获取词性类型（如果存在）
        if pos_type_column in occurrences_df.columns:
            pos_types = occurrences_df[pos_type_column].tolist()
        else:
            pos_types = ['noun'] * len(texts)

        dataset = ConceptOccurrenceDataset(
            texts=texts,
            concept_terms=terms,
            start_positions=positions,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pos_types=pos_types
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # 避免多进程问题
        )

        all_embeddings = []

        iterator = dataloader
        if show_progress:
            iterator = progress_bar(
                iterator,
                desc="Encoding with DeBERTa",
                total=len(dataloader)
            )

        with torch.no_grad():
            for batch in iterator:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mask_positions = batch['mask_position']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                hidden_states = outputs.last_hidden_state

                # 提取每个样本的 [MASK] 位置嵌入
                batch_embeddings = []
                for i, pos in enumerate(mask_positions):
                    emb = hidden_states[i, pos, :]
                    batch_embeddings.append(emb.cpu().numpy())

                all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings)
        logger.info(f"Encoded {len(embeddings)} samples, shape: {embeddings.shape}")

        return embeddings


def compute_semantic_expectations(
    occurrences_df: pd.DataFrame,
    model_name: str = "microsoft/deberta-v3-base",
    batch_size: int = 32,
    device: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    便捷函数：计算语义期望向量

    Args:
        occurrences_df: concept_occurrences DataFrame
        model_name: DeBERTa 模型名
        batch_size: 批处理大小
        device: 计算设备

    Returns:
        (U_matrix, updated_df) 元组
    """
    encoder = DeBERTaEncoder(
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )

    U_matrix = encoder.encode_batch(occurrences_df)

    # 将向量添加到 DataFrame（作为列表存储）
    occurrences_df = occurrences_df.copy()
    occurrences_df['U_vector'] = list(U_matrix)

    return U_matrix, occurrences_df


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse
    from utils import load_dataframe

    parser = argparse.ArgumentParser(description="Compute semantic expectations with DeBERTa")
    parser.add_argument(
        "--input",
        type=str,
        default="concept_occurrences.parquet",
        help="Input file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="semantic_expectations.npy",
        help="Output file for U matrix"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="DeBERTa model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )

    args = parser.parse_args()

    # 加载数据
    df = load_dataframe(args.input, subdir="processed")
    logger.info(f"Loaded {len(df)} occurrences")

    # 计算语义期望
    encoder = DeBERTaEncoder(
        model_name=args.model,
        batch_size=args.batch_size
    )
    U_matrix = encoder.encode_batch(df)

    # 保存
    output_path = get_path(f"data/processed/{args.output}")
    np.save(output_path, U_matrix)
    logger.info(f"Saved U matrix to {output_path}")

    print(f"\nU matrix shape: {U_matrix.shape}")
    print(f"Mean norm: {np.linalg.norm(U_matrix, axis=1).mean():.4f}")
