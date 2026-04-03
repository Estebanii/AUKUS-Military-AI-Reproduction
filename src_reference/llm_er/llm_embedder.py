"""
LLM 嵌入器
使用大语言模型的输入嵌入层作为语义词典 V
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import sys
# Add src directory to path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils import (
    get_path, load_config, logger,
    get_device
)


class LLMEmbedder:
    """
    LLM 嵌入器

    使用大语言模型的输入嵌入层获取词汇的语义表示
    这些嵌入作为"语义词典"V，为跨国比较提供统一的参照系
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        use_auth_token: Optional[str] = None
    ):
        """
        初始化 LLM 嵌入器

        Args:
            model_name: 模型名称（支持 Llama, GPT-2 等）
            device: 计算设备
            use_auth_token: HuggingFace 认证 token（用于访问受限模型）
        """
        self.model_name = model_name
        self.use_auth_token = use_auth_token

        # 设置设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        self._model = None
        self._tokenizer = None
        self._embedding_layer = None

    @property
    def tokenizer(self):
        """延迟加载 tokenizer"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def embedding_layer(self):
        """获取嵌入层"""
        if self._embedding_layer is None:
            self._load_model()
        return self._embedding_layer

    def _load_model(self):
        """加载模型"""
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

        logger.info(f"Loading LLM model: {self.model_name}")

        # 根据模型类型选择加载方式
        try:
            # 尝试作为 Causal LM 加载
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.use_auth_token,
                trust_remote_code=True
            )

            # 只加载嵌入层，不加载完整模型以节省内存
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.use_auth_token,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # 获取嵌入层
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                # Llama 风格
                self._embedding_layer = model.model.embed_tokens
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                # GPT-2 风格
                self._embedding_layer = model.transformer.wte
            elif hasattr(model, 'get_input_embeddings'):
                self._embedding_layer = model.get_input_embeddings()
            else:
                raise ValueError("Cannot find embedding layer in model")

            self._embedding_layer.to(self.device)
            self._embedding_layer.eval()

            # 释放其他部分的内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.info("Falling back to GPT-2")

            self.model_name = "gpt2"
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModel.from_pretrained("gpt2")
            self._embedding_layer = model.wte
            self._embedding_layer.to(self.device)
            self._embedding_layer.eval()

        logger.info(f"Embedding layer loaded, dim: {self._embedding_layer.weight.shape[1]}")

    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.embedding_layer.weight.shape[1]

    @property
    def vocab_size(self) -> int:
        """获取词表大小"""
        return self.embedding_layer.weight.shape[0]

    def get_token_embedding(self, token: str) -> np.ndarray:
        """
        获取单个 token 的嵌入

        Args:
            token: 输入 token

        Returns:
            嵌入向量
        """
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)

        if len(token_ids) == 0:
            logger.warning(f"Token '{token}' not in vocabulary")
            return np.zeros(self.embedding_dim)

        with torch.no_grad():
            token_id = token_ids[0]  # 取第一个 token
            embedding = self.embedding_layer.weight[token_id]

        return embedding.cpu().numpy().astype(np.float32)

    def get_phrase_embedding(
        self,
        phrase: str,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        获取短语的嵌入

        Args:
            phrase: 输入短语
            aggregation: 聚合方式 ('mean', 'first', 'last')

        Returns:
            嵌入向量
        """
        token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)

        if len(token_ids) == 0:
            logger.warning(f"Phrase '{phrase}' produced no tokens")
            return np.zeros(self.embedding_dim)

        with torch.no_grad():
            token_tensor = torch.tensor(token_ids, device=self.device)
            embeddings = self.embedding_layer(token_tensor)

            if aggregation == 'mean':
                result = embeddings.mean(dim=0)
            elif aggregation == 'first':
                result = embeddings[0]
            elif aggregation == 'last':
                result = embeddings[-1]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        return result.cpu().numpy().astype(np.float32)

    def get_concept_embeddings(
        self,
        concepts: Dict[str, List[str]],
        aggregation: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        获取多个概念的嵌入

        Args:
            concepts: {concept_id: [term1, term2, ...]} 映射
            aggregation: 聚合方式

        Returns:
            {concept_id: embedding} 映射
        """
        embeddings = {}

        for concept_id, terms in concepts.items():
            if not terms:
                embeddings[concept_id] = np.zeros(self.embedding_dim)
                continue

            # 对每个术语获取嵌入，然后平均
            term_embeddings = []
            for term in terms:
                emb = self.get_phrase_embedding(term, aggregation='mean')
                term_embeddings.append(emb)

            # 平均所有术语的嵌入
            embeddings[concept_id] = np.mean(term_embeddings, axis=0)

        return embeddings

    def get_all_embeddings(self) -> np.ndarray:
        """
        获取完整的嵌入矩阵

        Returns:
            嵌入矩阵 (vocab_size, embedding_dim)
        """
        with torch.no_grad():
            embeddings = self.embedding_layer.weight.cpu().numpy()
        return embeddings.astype(np.float32)


def get_concept_embeddings(
    concept_terms: Dict[str, List[str]],
    model_name: str = "meta-llama/Llama-3.2-1B",
    device: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    便捷函数：获取概念嵌入

    Args:
        concept_terms: {concept_id: [terms]} 映射
        model_name: LLM 模型名
        device: 计算设备

    Returns:
        (embeddings_dict, embedding_dim) 元组
    """
    embedder = LLMEmbedder(model_name=model_name, device=device)
    embeddings = embedder.get_concept_embeddings(concept_terms)
    return embeddings, embedder.embedding_dim


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get LLM embeddings for concepts")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",  # 默认使用 GPT-2（不需要认证）
        help="LLM model name"
    )
    parser.add_argument(
        "--test-phrase",
        type=str,
        default="artificial intelligence",
        help="Test phrase to embed"
    )

    args = parser.parse_args()

    embedder = LLMEmbedder(model_name=args.model)

    print(f"\nModel: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    print(f"Vocabulary size: {embedder.vocab_size}")

    # 测试短语嵌入
    test_embedding = embedder.get_phrase_embedding(args.test_phrase)
    print(f"\nEmbedding for '{args.test_phrase}':")
    print(f"  Shape: {test_embedding.shape}")
    print(f"  Norm: {np.linalg.norm(test_embedding):.4f}")
    print(f"  First 5 dims: {test_embedding[:5]}")
