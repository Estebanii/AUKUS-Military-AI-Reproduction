"""
LLM-ER 向量化主模块
整合 DeBERTa 编码、LLM 嵌入、矩阵训练的完整流程
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
# Add src directory to path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils import (
    get_path, load_config, logger,
    save_dataframe, load_dataframe,
    progress_bar
)
from .deberta_encoder import DeBERTaEncoder
from .llm_embedder import LLMEmbedder
from .matrix_trainer import MatrixTrainer


class LLMERVectorizer:
    """
    LLM-ER 向量化器

    完整的向量化流程：
    1. 使用 DeBERTa 计算每次概念出现的语义期望 U
    2. 使用 LLM 获取概念的语义词典 V
    3. 训练转换矩阵 A
    4. 生成最终语义向量 Y = A @ U
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化向量化器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()

        # 获取配置
        llm_er_config = self.config.get('llm_er', {})

        deberta_config = llm_er_config.get('deberta', {})
        self.deberta_model = deberta_config.get('model_name', 'microsoft/deberta-v3-base')
        self.deberta_batch_size = deberta_config.get('batch_size', 32)
        self.deberta_max_length = deberta_config.get('max_length', 512)

        llm_config = llm_er_config.get('llm_embedding', {})
        self.llm_model = llm_config.get('model_name', 'gpt2')  # 默认使用 GPT-2

        matrix_config = llm_er_config.get('matrix_training', {})
        self.reg_lambda = matrix_config.get('regularization_lambda', 0.1)

        # 子模块（延迟初始化）
        self._deberta_encoder = None
        self._llm_embedder = None
        self._matrix_trainer = None

        # 数据
        self.U_matrix = None
        self.V_dict = None
        self.A_matrix = None

    @property
    def deberta_encoder(self):
        if self._deberta_encoder is None:
            self._deberta_encoder = DeBERTaEncoder(
                model_name=self.deberta_model,
                batch_size=self.deberta_batch_size,
                max_length=self.deberta_max_length
            )
        return self._deberta_encoder

    @property
    def llm_embedder(self):
        if self._llm_embedder is None:
            self._llm_embedder = LLMEmbedder(model_name=self.llm_model)
        return self._llm_embedder

    @property
    def matrix_trainer(self):
        if self._matrix_trainer is None:
            self._matrix_trainer = MatrixTrainer(
                regularization_lambda=self.reg_lambda
            )
        return self._matrix_trainer

    def run(
        self,
        occurrences_df: Optional[pd.DataFrame] = None,
        concept_terms: Optional[Dict[str, List[str]]] = None,
        force_reload: bool = False
    ) -> Dict:
        """
        运行完整的向量化流程

        Args:
            occurrences_df: concept_occurrences DataFrame
            concept_terms: {concept_id: [terms]} 映射
            force_reload: 是否强制重新计算

        Returns:
            包含各阶段结果的字典
        """
        results = {}

        # ============================================================
        # 加载数据
        # ============================================================
        if occurrences_df is None:
            logger.info("Loading concept occurrences...")
            occurrences_df = load_dataframe("concept_occurrences.parquet", subdir="processed")

        results['occurrences'] = occurrences_df
        logger.info(f"Loaded {len(occurrences_df)} concept occurrences")

        # 获取概念-术语映射
        if concept_terms is None:
            concept_terms = self._get_concept_terms(occurrences_df)

        # ============================================================
        # Step 1: 计算语义期望 U
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 1: Computing semantic expectations (U)")
        logger.info("="*60)

        u_path = get_path("data/processed/U_matrix.npy")

        if u_path.exists() and not force_reload:
            logger.info(f"Loading existing U matrix from {u_path}")
            self.U_matrix = np.load(u_path)
        else:
            self.U_matrix = self.deberta_encoder.encode_batch(occurrences_df)
            np.save(u_path, self.U_matrix)

        results['U_matrix'] = self.U_matrix
        logger.info(f"U matrix shape: {self.U_matrix.shape}")

        # ============================================================
        # Step 2: 获取语义词典 V
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 2: Getting semantic dictionary (V)")
        logger.info("="*60)

        v_path = get_path("data/processed/V_dict.npz")

        if v_path.exists() and not force_reload:
            logger.info(f"Loading existing V dictionary from {v_path}")
            v_data = np.load(v_path)
            self.V_dict = {k: v_data[k] for k in v_data.files}
        else:
            self.V_dict = self.llm_embedder.get_concept_embeddings(concept_terms)
            np.savez(v_path, **self.V_dict)

        results['V_dict'] = self.V_dict
        logger.info(f"V dictionary: {len(self.V_dict)} concepts")

        # ============================================================
        # Step 3: 计算每个概念的平均 U
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 3: Computing concept-level U averages")
        logger.info("="*60)

        U_by_concept = self._compute_concept_averages(
            occurrences_df,
            self.U_matrix
        )
        results['U_by_concept'] = U_by_concept

        # ============================================================
        # Step 4: 训练转换矩阵 A
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 4: Training transformation matrix (A)")
        logger.info("="*60)

        a_path = get_path("data/processed/A_matrix.npz")

        if a_path.exists() and not force_reload:
            logger.info(f"Loading existing A matrix from {a_path}")
            self.matrix_trainer.load(a_path)
        else:
            # 使用概念出现次数作为权重
            weights = occurrences_df['concept_id'].value_counts().to_dict()
            self.matrix_trainer.train(U_by_concept, self.V_dict, weights)
            self.matrix_trainer.save(a_path)

        self.A_matrix = self.matrix_trainer.A
        results['A_matrix'] = self.A_matrix
        logger.info(f"A matrix shape: {self.A_matrix.shape}")

        # ============================================================
        # Step 5: 生成最终语义向量 Y
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 5: Generating semantic vectors (Y)")
        logger.info("="*60)

        Y_matrix = self.matrix_trainer.transform(self.U_matrix)
        results['Y_matrix'] = Y_matrix
        logger.info(f"Y matrix shape: {Y_matrix.shape}")

        # 保存 Y 矩阵
        y_path = get_path("data/processed/Y_matrix.npy")
        np.save(y_path, Y_matrix)

        # ============================================================
        # Step 6: 创建最终数据集
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 6: Creating semantic vectors dataset")
        logger.info("="*60)

        semantic_df = self._create_semantic_dataframe(occurrences_df, Y_matrix)
        save_dataframe(semantic_df, "semantic_vectors.parquet", subdir="processed")
        results['semantic_vectors'] = semantic_df

        # 打印摘要
        self._print_summary(results)

        logger.info("\nVectorization completed!")
        return results

    def _get_concept_terms(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """从出现数据中提取概念-术语映射"""
        concept_terms = {}
        for concept_id, group in df.groupby('concept_id'):
            terms = group['matched_term'].str.lower().unique().tolist()
            concept_terms[concept_id] = terms
        return concept_terms

    def _compute_concept_averages(
        self,
        df: pd.DataFrame,
        U_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """计算每个概念的平均 U 向量"""
        U_by_concept = {}

        for concept_id, indices in df.groupby('concept_id').indices.items():
            concept_U = U_matrix[indices]
            U_by_concept[concept_id] = concept_U.mean(axis=0)

        logger.info(f"Computed averages for {len(U_by_concept)} concepts")
        return U_by_concept

    def _create_semantic_dataframe(
        self,
        occurrences_df: pd.DataFrame,
        Y_matrix: np.ndarray
    ) -> pd.DataFrame:
        """创建包含语义向量的 DataFrame"""
        # 复制元数据
        semantic_df = occurrences_df[[
            'occurrence_id', 'concept_id', 'concept_label',
            'matched_term', 'doc_id', 'country',
            'date', 'year', 'month', 'post_aukus', 'source_type'
        ]].copy()

        # 添加 Y 向量（作为列表存储）
        semantic_df['Y_vector'] = list(Y_matrix)

        # 添加向量范数
        semantic_df['Y_norm'] = np.linalg.norm(Y_matrix, axis=1)

        logger.info(f"Created semantic vectors dataset: {len(semantic_df)} records")
        return semantic_df

    def _print_summary(self, results: Dict):
        """打印摘要"""
        print("\n" + "="*60)
        print("LLM-ER Vectorization Summary")
        print("="*60)

        print(f"\nInput:")
        print(f"  Concept occurrences: {len(results['occurrences']):,}")
        print(f"  Unique concepts: {results['occurrences']['concept_id'].nunique()}")

        print(f"\nDimensions:")
        print(f"  U (DeBERTa): {results['U_matrix'].shape[1]}")
        print(f"  V (LLM): {list(results['V_dict'].values())[0].shape[0] if results['V_dict'] else 'N/A'}")
        print(f"  Y (output): {results['Y_matrix'].shape[1]}")

        print(f"\nMatrix A shape: {results['A_matrix'].shape}")

        print(f"\nOutput files:")
        print(f"  data/processed/U_matrix.npy")
        print(f"  data/processed/V_dict.npz")
        print(f"  data/processed/A_matrix.npz")
        print(f"  data/processed/Y_matrix.npy")
        print(f"  data/processed/semantic_vectors.parquet")


def run_vectorization(force_reload: bool = False) -> Dict:
    """
    便捷函数：运行向量化流程

    Args:
        force_reload: 是否强制重新计算

    Returns:
        结果字典
    """
    vectorizer = LLMERVectorizer()
    return vectorizer.run(force_reload=force_reload)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-ER vectorization")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation"
    )
    parser.add_argument(
        "--deberta-model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="DeBERTa model name"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt2",
        help="LLM model for embeddings"
    )

    args = parser.parse_args()

    results = run_vectorization(force_reload=args.force)
