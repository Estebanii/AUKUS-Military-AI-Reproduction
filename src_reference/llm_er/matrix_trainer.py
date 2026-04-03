"""
转换矩阵训练器
训练从 DeBERTa 空间到 LLM 空间的转换矩阵 A
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg

import sys
# Add src directory to path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils import (
    get_path, load_config, logger,
    center_vectors, normalize_vectors
)


class MatrixTrainer:
    """
    转换矩阵训练器

    训练线性变换矩阵 A，使得 V ≈ A @ U

    优化目标:
    A = argmin_A Σ_w ω_w ||V_w - A @ U_w||² + λ||A - A_0||²_F

    其中:
    - V_w: 概念 w 的 LLM 嵌入（语义词典）
    - U_w: 概念 w 所有出现的平均语义期望
    - A_0: 基于 Procrustes 的先验对齐矩阵
    - λ: 正则化系数
    """

    def __init__(
        self,
        regularization_lambda: float = 0.1,
        use_procrustes_prior: bool = True,
        whiten: bool = True
    ):
        """
        初始化训练器

        Args:
            regularization_lambda: 正则化系数
            use_procrustes_prior: 是否使用 Procrustes 先验
            whiten: 是否对向量进行白化处理
        """
        self.lambda_ = regularization_lambda
        self.use_procrustes_prior = use_procrustes_prior
        self.whiten = whiten

        self.A = None  # 训练后的转换矩阵
        self.A0 = None  # 先验矩阵
        self.U_mean = None  # U 的均值（用于中心化）
        self.V_mean = None  # V 的均值（用于中心化）

    def compute_procrustes_prior(
        self,
        U: np.ndarray,
        V: np.ndarray
    ) -> np.ndarray:
        """
        使用 Procrustes 分析计算最优正交对齐矩阵

        Args:
            U: 源空间矩阵 (n_concepts, d_U)
            V: 目标空间矩阵 (n_concepts, d_V)

        Returns:
            正交对齐矩阵 A_0
        """
        # SVD 分解: V.T @ U = W @ Σ @ Z.T
        # 最优正交矩阵: A_0 = W @ Z.T

        # 中心化
        U_centered = U - U.mean(axis=0)
        V_centered = V - V.mean(axis=0)

        # SVD
        M = V_centered.T @ U_centered
        W, _, Zt = linalg.svd(M, full_matrices=False)

        # 正交对齐矩阵
        A0 = W @ Zt

        return A0

    def compute_whitening_transform(
        self,
        X: np.ndarray,
        eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算白化变换矩阵

        Args:
            X: 输入矩阵 (n_samples, d)
            eps: 数值稳定性参数

        Returns:
            (whitening_matrix, mean) 元组
        """
        mean = X.mean(axis=0)
        X_centered = X - mean

        # 协方差矩阵
        cov = X_centered.T @ X_centered / (X.shape[0] - 1)

        # 特征分解
        eigenvalues, eigenvectors = linalg.eigh(cov)

        # 确保数值稳定性：将负数或过小的特征值裁剪到 eps
        eigenvalues = np.clip(eigenvalues, eps, None)

        # 白化变换
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        whitening_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T

        return whitening_matrix, mean

    def train(
        self,
        U_by_concept: Dict[str, np.ndarray],
        V_by_concept: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        训练转换矩阵

        Args:
            U_by_concept: {concept_id: 平均U向量} 映射
            V_by_concept: {concept_id: V向量} 映射
            weights: {concept_id: 权重} 映射（可选）

        Returns:
            训练后的转换矩阵 A
        """
        # 确保概念对齐
        common_concepts = set(U_by_concept.keys()) & set(V_by_concept.keys())
        if len(common_concepts) == 0:
            raise ValueError("No common concepts between U and V")

        logger.info(f"Training on {len(common_concepts)} concepts")

        # 构建矩阵
        concepts = sorted(common_concepts)
        U_matrix = np.array([U_by_concept[c] for c in concepts])  # (K, d_U)
        V_matrix = np.array([V_by_concept[c] for c in concepts])  # (K, d_V)

        # 权重
        if weights is None:
            W = np.ones(len(concepts))
        else:
            W = np.array([weights.get(c, 1.0) for c in concepts])
        W = W / W.sum()  # 归一化

        # 中心化
        self.U_mean = U_matrix.mean(axis=0)
        self.V_mean = V_matrix.mean(axis=0)
        U_centered = U_matrix - self.U_mean
        V_centered = V_matrix - self.V_mean

        # 可选白化
        if self.whiten:
            logger.info("Applying whitening transform")
            W_u, _ = self.compute_whitening_transform(U_centered)
            U_centered = U_centered @ W_u

        # 计算先验
        if self.use_procrustes_prior:
            logger.info("Computing Procrustes prior")
            self.A0 = self.compute_procrustes_prior(U_centered, V_centered)
        else:
            self.A0 = np.zeros((V_centered.shape[1], U_centered.shape[1]))

        # 带权重的岭回归
        # A = (V.T @ diag(W) @ U + λ * A0) @ (U.T @ diag(W) @ U + λ * I)^{-1}

        W_diag = np.diag(W)
        d_U = U_centered.shape[1]

        # 正规方程
        UTU = U_centered.T @ W_diag @ U_centered + self.lambda_ * np.eye(d_U)
        VTU = V_centered.T @ W_diag @ U_centered + self.lambda_ * self.A0

        # 求解
        self.A = linalg.solve(UTU.T, VTU.T).T

        # 计算训练误差
        V_pred = U_centered @ self.A.T
        mse = np.mean((V_centered - V_pred) ** 2)
        logger.info(f"Training MSE: {mse:.6f}")

        # 如果应用了白化，需要调整 A
        if self.whiten:
            self.A = self.A @ W_u

        return self.A

    def train_from_occurrences(
        self,
        U_matrix: np.ndarray,
        word_labels: np.ndarray,
        V_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        从出现级别数据训练转换矩阵（新方法）

        每次出现都作为独立的训练数据点，而不是按词聚合。
        这样可以捕捉更丰富的语境变化信息。

        Args:
            U_matrix: 每次出现的DeBERTa编码 (N_occurrences, d_U)
            word_labels: 每次出现对应的词 (N_occurrences,)
            V_dict: {word: V向量} LLM语义词典

        Returns:
            训练后的转换矩阵 A
        """
        # 构建对应的V矩阵：每次出现对应其词的V向量
        V_list = []
        valid_indices = []

        for i, word in enumerate(word_labels):
            word_lower = str(word).lower()
            if word_lower in V_dict:
                V_list.append(V_dict[word_lower])
                valid_indices.append(i)

        if len(V_list) == 0:
            raise ValueError("No matching words found in V_dict")

        # 过滤有效数据
        U_valid = U_matrix[valid_indices]  # (N_valid, d_U)
        V_valid = np.array(V_list)  # (N_valid, d_V)

        logger.info(f"Training on {len(valid_indices)} occurrences ({len(valid_indices)/len(word_labels)*100:.1f}% matched)")

        # 中心化
        self.U_mean = U_valid.mean(axis=0)
        self.V_mean = V_valid.mean(axis=0)
        U_centered = U_valid - self.U_mean
        V_centered = V_valid - self.V_mean

        d_U = U_centered.shape[1]
        n_samples = U_centered.shape[0]

        # 可选白化（只在样本数 > 维度数时应用）
        W_u = None
        if self.whiten and n_samples > d_U:
            logger.info("Applying whitening transform")
            try:
                W_u, _ = self.compute_whitening_transform(U_centered)
                U_centered = U_centered @ W_u
            except Exception as e:
                logger.warning(f"Whitening failed: {e}, skipping")
                W_u = None

        # 计算先验（可选）
        if self.use_procrustes_prior and n_samples > d_U:
            logger.info("Computing Procrustes prior")
            try:
                self.A0 = self.compute_procrustes_prior(U_centered, V_centered)
            except Exception as e:
                logger.warning(f"Procrustes prior failed: {e}, using zero prior")
                self.A0 = np.zeros((V_centered.shape[1], U_centered.shape[1]))
        else:
            self.A0 = np.zeros((V_centered.shape[1], U_centered.shape[1]))

        # 岭回归（不使用权重，因为每个出现都是独立数据点）
        # A = (V.T @ U + λ * A0) @ (U.T @ U + λ * I)^{-1}

        # 正规方程
        UTU = U_centered.T @ U_centered + self.lambda_ * np.eye(d_U)
        VTU = V_centered.T @ U_centered + self.lambda_ * self.A0

        # 求解
        self.A = linalg.solve(UTU.T, VTU.T).T

        # 计算训练误差
        V_pred = U_centered @ self.A.T
        mse = np.mean((V_centered - V_pred) ** 2)
        logger.info(f"Training MSE: {mse:.6f}")

        # 如果应用了白化，需要调整 A
        if W_u is not None:
            self.A = self.A @ W_u

        return self.A

    def transform(self, U: np.ndarray) -> np.ndarray:
        """
        应用转换矩阵

        Args:
            U: 输入向量或矩阵 (n, d_U) 或 (d_U,)

        Returns:
            转换后的向量 Y
        """
        if self.A is None:
            raise ValueError("Matrix not trained. Call train() first.")

        # 处理单个向量
        single = U.ndim == 1
        if single:
            U = U.reshape(1, -1)

        # 中心化
        U_centered = U - self.U_mean

        # 转换
        Y = U_centered @ self.A.T

        # 反中心化（可选）
        # Y = Y + self.V_mean

        if single:
            Y = Y.squeeze()

        return Y

    def save(self, path: str):
        """保存训练好的矩阵"""
        np.savez(
            path,
            A=self.A,
            A0=self.A0,
            U_mean=self.U_mean,
            V_mean=self.V_mean,
            lambda_=self.lambda_
        )
        logger.info(f"Saved transformation matrix to {path}")

    def load(self, path: str):
        """加载训练好的矩阵"""
        data = np.load(path)
        self.A = data['A']
        self.A0 = data['A0']
        self.U_mean = data['U_mean']
        self.V_mean = data['V_mean']
        self.lambda_ = float(data['lambda_'])
        logger.info(f"Loaded transformation matrix from {path}")


def train_transformation_matrix(
    U_by_concept: Dict[str, np.ndarray],
    V_by_concept: Dict[str, np.ndarray],
    regularization_lambda: float = 0.1,
    weights: Optional[Dict[str, float]] = None
) -> MatrixTrainer:
    """
    便捷函数：训练转换矩阵

    Args:
        U_by_concept: {concept_id: 平均U向量} 映射
        V_by_concept: {concept_id: V向量} 映射
        regularization_lambda: 正则化系数
        weights: 权重映射

    Returns:
        训练好的 MatrixTrainer 实例
    """
    trainer = MatrixTrainer(regularization_lambda=regularization_lambda)
    trainer.train(U_by_concept, V_by_concept, weights)
    return trainer


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train transformation matrix")
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.1,
        dest="lambda_",
        help="Regularization parameter"
    )

    args = parser.parse_args()

    # 示例：使用随机数据测试
    print("Testing MatrixTrainer with random data...")

    n_concepts = 50
    d_U = 768
    d_V = 2048

    # 生成随机数据
    U_by_concept = {f"C_{i}": np.random.randn(d_U) for i in range(n_concepts)}
    V_by_concept = {f"C_{i}": np.random.randn(d_V) for i in range(n_concepts)}

    # 训练
    trainer = train_transformation_matrix(
        U_by_concept,
        V_by_concept,
        regularization_lambda=args.lambda_
    )

    print(f"\nTransformation matrix shape: {trainer.A.shape}")
    print(f"Matrix norm: {np.linalg.norm(trainer.A):.4f}")

    # 测试转换
    test_U = np.random.randn(d_U)
    Y = trainer.transform(test_U)
    print(f"\nInput shape: {test_U.shape}")
    print(f"Output shape: {Y.shape}")
