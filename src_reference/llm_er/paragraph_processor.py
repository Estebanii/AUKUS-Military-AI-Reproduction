"""
LLM-ER 段落级处理器（预处理数据版本）

设计理念：
- 所有数据在运行前已预处理完成
- 不再遍历原始文章，大幅减少 I/O 和内存开销
- 使用锚点词（非目标术语的高频词）训练国家特定的矩阵A
- 每个国家有独立的认知映射矩阵

预处理流程（在运行分析前执行）：
1. python scripts/discover_and_save_anchor_words.py  # 发现锚点词
2. python scripts/preprocess_for_hpc.py              # 预处理所有数据

分析流程：
1. 加载预处理的目标词出现（term_occurrences.parquet）
2. 加载预处理的锚点词出现（anchor_occurrences_*.parquet）
3. 计算 U 向量（DeBERTa）和 V 向量（LLM）
4. 训练国家特定的矩阵 A
5. 应用转换: Y = A @ U
"""

import csv
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 导入 LLM-ER 组件
from .llm_embedder import LLMEmbedder
from .matrix_trainer import MatrixTrainer
from .anchor_extractor import (
    load_target_terms,
    load_fixed_anchor_words,
    DEFAULT_ANCHOR_WORDS
)


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    return current.parent.parent.parent


def get_path(relative_path: str) -> Path:
    """将相对路径转换为绝对路径"""
    return _get_project_root() / relative_path


def load_config() -> Dict:
    """加载配置文件"""
    config_path = get_path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def progress_bar(iterable, desc=None, total=None):
    """简单的进度条封装"""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        return iterable


class ParagraphLevelProcessor:
    """
    段落级 LLM-ER 处理器

    将概念标注模块的输出转换为 LLM-ER 向量化所需的格式，
    并使用段落作为上下文计算语义向量。

    流程：
    1. 加载术语出现数据（来自 ConceptAnnotator）
    2. 使用段落作为上下文，计算语义期望 U
    3. 计算转换矩阵 A
    4. 生成语义偏差向量 Y = X - U 或 Y = A @ U
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化处理器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()

        # 获取配置
        llm_er_config = self.config.get('llm_er', {})
        deberta_config = llm_er_config.get('deberta', {})

        self.model_name = deberta_config.get('model_name', 'microsoft/deberta-v3-base')
        self.batch_size = deberta_config.get('batch_size', 32)
        self.max_length = deberta_config.get('max_length', 512)

        # LLM 嵌入配置
        llm_config = llm_er_config.get('llm_embedding', {})
        self.llm_model = llm_config.get('model_name', 'gpt2')

        # 矩阵训练配置
        matrix_config = llm_er_config.get('matrix_training', {})
        self.reg_lambda = matrix_config.get('regularization_lambda', 0.1)

        # 子模块（延迟初始化）
        self._encoder = None
        self._llm_embedder = None
        self._matrix_trainer = None

    @property
    def encoder(self):
        """延迟加载 DeBERTa 编码器"""
        if self._encoder is None:
            from .deberta_encoder import DeBERTaEncoder
            self._encoder = DeBERTaEncoder(
                model_name=self.model_name,
                batch_size=self.batch_size,
                max_length=self.max_length
            )
        return self._encoder

    @property
    def llm_embedder(self):
        """延迟加载 LLM 嵌入器"""
        if self._llm_embedder is None:
            self._llm_embedder = LLMEmbedder(model_name=self.llm_model)
        return self._llm_embedder

    @property
    def matrix_trainer(self):
        """延迟加载矩阵训练器"""
        if self._matrix_trainer is None:
            self._matrix_trainer = MatrixTrainer(
                regularization_lambda=self.reg_lambda
            )
        return self._matrix_trainer

    def prepare_data(self, occurrences_df: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据格式

        将 ConceptAnnotator 输出转换为 DeBERTa 编码器所需格式

        Args:
            occurrences_df: 来自 ConceptAnnotator 的 DataFrame

        Returns:
            格式化后的 DataFrame
        """
        # 复制数据
        df = occurrences_df.copy()

        # 重命名/添加必要的列
        # paragraph -> text_block (上下文)
        # term -> matched_term
        # char_start -> start_char

        if 'text_block' not in df.columns and 'paragraph' in df.columns:
            df['text_block'] = df['paragraph']

        if 'matched_term' not in df.columns and 'term' in df.columns:
            df['matched_term'] = df['term']

        if 'start_char' not in df.columns and 'char_start' in df.columns:
            df['start_char'] = df['char_start']

        # 为每个出现生成唯一ID
        if 'occurrence_id' not in df.columns:
            df['occurrence_id'] = range(len(df))

        # 统一概念ID（所有术语属于同一个概念：军事AI）
        if 'concept_id' not in df.columns:
            df['concept_id'] = 'military_ai'

        if 'concept_label' not in df.columns:
            df['concept_label'] = 'Military Artificial Intelligence'

        # 文档ID
        if 'doc_id' not in df.columns:
            df['doc_id'] = df['article_id']

        # 日期处理
        if 'date' not in df.columns:
            df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'

        # 来源类型
        if 'source_type' not in df.columns:
            df['source_type'] = 'news'

        logger.info(f"Prepared {len(df)} records for LLM-ER processing")

        return df

    def compute_semantic_vectors(
        self,
        occurrences_df: pd.DataFrame,
        anchor_occurrences: Dict[str, pd.DataFrame],
        show_progress: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        计算语义向量（预处理数据版本）

        流程：
        1. 计算目标术语的 U 向量（DeBERTa编码）
        2. 获取锚点词的 V 嵌入（LLM语义空间）
        3. 使用预处理的锚点词数据训练国家特定的矩阵 A
        4. 应用转换: Y = A @ U

        Args:
            occurrences_df: 术语出现数据（预处理的 term_occurrences.parquet）
            anchor_occurrences: {country: DataFrame} 预处理的锚点词出现
            show_progress: 是否显示进度

        Returns:
            (Y_matrix, semantic_df)
        """
        # 准备数据
        df = self.prepare_data(occurrences_df)
        countries = df['country'].unique().tolist()
        logger.info(f"Processing {len(df)} occurrences from {len(countries)} countries: {countries}")

        # ============================================================
        # Step 1: 计算所有目标术语的 U 矩阵
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 1: Computing U vectors for target terms...")
        logger.info("="*60)

        U_matrix = self.encoder.encode_batch(
            df,
            text_column='text_block',
            term_column='matched_term',
            position_column='start_char',
            show_progress=show_progress
        )
        logger.info(f"U matrix shape: {U_matrix.shape}")

        # ============================================================
        # Step 2: 获取锚点词的 LLM 嵌入 V
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 2: Loading anchor words and getting LLM embeddings...")
        logger.info("="*60)

        # 加载目标术语（需要排除）
        target_terms = load_target_terms()

        # 优先使用预计算的固定锚点词列表（HPC部署模式）
        anchor_words = load_fixed_anchor_words(top_n=100)

        if len(anchor_words) == 0:
            raise ValueError(
                "Fixed anchor word list not found at data/intermediate/anchor_words.json. "
                "This file is required for reproduction."
            )
                anchor_words = [w for w in DEFAULT_ANCHOR_WORDS
                               if w.lower() not in target_terms and
                               not any(t in w.lower() or w.lower() in t for t in target_terms)]

        logger.info(f"Using {len(anchor_words)} anchor words")

        # 获取锚点词的 V 嵌入
        V_anchor_dict = {}
        for word in progress_bar(anchor_words, desc="Getting anchor V embeddings"):
            try:
                V_anchor_dict[word.lower()] = self.llm_embedder.get_phrase_embedding(word)
            except Exception as e:
                logger.warning(f"Failed to get embedding for anchor '{word}': {e}")

        logger.info(f"Got V embeddings for {len(V_anchor_dict)} anchor words")

        # ============================================================
        # Step 3: 训练国家特定的矩阵 A（使用预处理的锚点词数据）
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 3: Training country-specific transformation matrices...")
        logger.info("="*60)

        # 必须提供预处理的锚点词数据
        if anchor_occurrences is None or len(anchor_occurrences) == 0:
            raise ValueError(
                "anchor_occurrences is required!\n"
                "Please run: python scripts/preprocess_for_hpc.py first"
            )

        logger.info("Using preprocessed anchor occurrences...")
        country_trainers = self._train_matrices_from_preprocessed(
            anchor_occurrences, anchor_words, V_anchor_dict, show_progress
        )

        # ============================================================
        # Step 4: 应用国家特定的矩阵 A 生成 Y
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("Step 4: Applying country-specific transformations (Y = A @ U)...")
        logger.info("="*60)

        Y_matrix = np.zeros_like(U_matrix)

        for country in countries:
            mask = df['country'] == country
            indices = df[mask].index.tolist()
            row_indices = [df.index.get_loc(idx) for idx in indices]

            if country in country_trainers and country_trainers[country].A is not None:
                trainer = country_trainers[country]
                U_country = U_matrix[row_indices]
                Y_country = trainer.transform(U_country)
                Y_matrix[row_indices] = Y_country
                logger.info(f"  {country}: Transformed {len(row_indices)} vectors using A_{country}")
            else:
                # 如果没有该国的矩阵，使用全局平均或直接用 U
                logger.warning(f"  {country}: No matrix available, using U directly")
                Y_matrix[row_indices] = U_matrix[row_indices]

        logger.info(f"Y matrix shape: {Y_matrix.shape}")

        # 保存国家特定的矩阵
        for country, trainer in country_trainers.items():
            if trainer.A is not None:
                a_path = get_path(f"data/processed/A_matrix_{country}.npz")
                trainer.save(str(a_path))
                logger.info(f"Saved A_{country} to {a_path}")

        # ============================================================
        # 创建语义向量 DataFrame
        # ============================================================
        columns_to_copy = [
            'occurrence_id', 'term', 'matched_term', 'article_id', 'doc_id',
            'country', 'year', 'month', 'post_aukus', 'ai_similarity'
        ]
        # 添加可选的文本溯源列（如果存在）
        optional_columns = ['pos_type', 'text_block', 'context', 'source_type']
        for col in optional_columns:
            if col in df.columns:
                columns_to_copy.append(col)

        semantic_df = df[columns_to_copy].copy()
        semantic_df['U_vector'] = list(U_matrix)
        semantic_df['Y_vector'] = list(Y_matrix)
        semantic_df['Y_norm'] = np.linalg.norm(Y_matrix, axis=1)
        semantic_df['concept_id'] = 'military_ai'
        # 添加 concept_label（基于 matched_term）
        semantic_df['concept_label'] = semantic_df['matched_term']

        logger.info(f"Created semantic vectors for {len(semantic_df)} occurrences")

        return Y_matrix, semantic_df

    def _train_matrices_from_preprocessed(
        self,
        anchor_occurrences: Dict[str, pd.DataFrame],
        anchor_words: List[str],
        V_anchor_dict: Dict[str, np.ndarray],
        show_progress: bool = True
    ) -> Dict[str, MatrixTrainer]:
        """
        从预处理的锚点词数据训练矩阵（HPC模式）

        Args:
            anchor_occurrences: {country: DataFrame} 预处理的锚点词出现
            anchor_words: 锚点词列表
            V_anchor_dict: {word: V向量}
            show_progress: 是否显示进度

        Returns:
            {country: MatrixTrainer}
        """
        country_trainers = {}

        for country, anchor_df in anchor_occurrences.items():
            logger.info(f"\n  Processing {country} ({len(anchor_df)} preprocessed occurrences)...")

            if len(anchor_df) < 100:
                logger.warning(f"  {country}: Only {len(anchor_df)} anchor occurrences, skipping")
                country_trainers[country] = MatrixTrainer(regularization_lambda=self.reg_lambda)
                continue

            # 限制样本数量
            max_samples = 30000
            if len(anchor_df) > max_samples:
                anchor_df = anchor_df.sample(n=max_samples, random_state=42)
                logger.info(f"  {country}: Sampled {max_samples} anchor occurrences")

            # 计算 U
            anchor_df = anchor_df.copy()
            anchor_df['matched_term'] = anchor_df['anchor_word']
            logger.info(f"  {country}: Computing U for {len(anchor_df)} anchor occurrences...")

            U_anchors = self.encoder.encode_batch(
                anchor_df,
                text_column='text_block',
                term_column='matched_term',
                position_column='start_char',
                show_progress=show_progress
            )

            # 训练矩阵
            trainer = MatrixTrainer(regularization_lambda=self.reg_lambda)
            word_labels = anchor_df['anchor_word'].values

            try:
                trainer.train_from_occurrences(U_anchors, word_labels, V_anchor_dict)
                logger.info(f"  {country}: Matrix A trained, shape: {trainer.A.shape}")
            except Exception as e:
                logger.error(f"  {country}: Failed to train matrix: {e}")

            country_trainers[country] = trainer

        return country_trainers

    def aggregate_monthly(
        self,
        semantic_df: pd.DataFrame,
        weight_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        月度聚合

        Args:
            semantic_df: 语义向量 DataFrame
            weight_column: 权重列（如 'ai_similarity'）

        Returns:
            月度聚合 DataFrame
        """
        results = []

        for (country, year, month), group in semantic_df.groupby(['country', 'year', 'month']):
            Y_vectors = np.vstack(group['Y_vector'].values)

            # 计算加权或简单平均
            if weight_column and weight_column in group.columns:
                weights = group[weight_column].values
                weights = weights / weights.sum()  # 归一化
                Y_mean = np.average(Y_vectors, axis=0, weights=weights)
            else:
                Y_mean = Y_vectors.mean(axis=0)

            results.append({
                'country': country,
                'year': year,
                'month': month,
                'post_aukus': group['post_aukus'].iloc[0],
                'Y_vector': Y_mean,
                'Y_norm': np.linalg.norm(Y_mean),
                'n_occurrences': len(group),
                'n_unique_terms': group['term'].nunique()
            })

        monthly_df = pd.DataFrame(results)

        logger.info(f"Aggregated to {len(monthly_df)} monthly observations")

        return monthly_df

    def save_results(
        self,
        semantic_df: pd.DataFrame,
        monthly_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None
    ):
        """
        保存结果

        Args:
            semantic_df: 个体级语义向量
            monthly_df: 月度聚合结果
            output_dir: 输出目录
        """
        output_dir = output_dir or get_path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存个体级数据
        semantic_path = output_dir / "semantic_vectors_paragraph.parquet"
        semantic_df.to_parquet(semantic_path, index=False)
        logger.info(f"Saved individual-level vectors to {semantic_path}")

        # 保存月度数据
        if monthly_df is not None:
            monthly_path = output_dir / "semantic_vectors_monthly.parquet"
            monthly_df.to_parquet(monthly_path, index=False)
            logger.info(f"Saved monthly vectors to {monthly_path}")

        # 保存 Y 矩阵（numpy 格式）
        Y_matrix = np.vstack(semantic_df['Y_vector'].values)
        y_path = output_dir / "Y_matrix_paragraph.npy"
        np.save(y_path, Y_matrix)
        logger.info(f"Saved Y matrix to {y_path}")

    def run(
        self,
        occurrences_df: pd.DataFrame,
        anchor_occurrences: Dict[str, pd.DataFrame],
        compute_monthly: bool = True,
        save_results: bool = True,
        show_progress: bool = True
    ) -> Dict:
        """
        运行完整处理流程（预处理数据版本）

        Args:
            occurrences_df: 术语出现数据（预处理的 term_occurrences.parquet）
            anchor_occurrences: {country: DataFrame} 预处理的锚点词出现
            compute_monthly: 是否计算月度聚合
            save_results: 是否保存结果
            show_progress: 是否显示进度

        Returns:
            结果字典
        """
        results = {}

        # Step 1: 计算语义向量
        logger.info("\n" + "="*60)
        logger.info("Step 1: Computing paragraph-level semantic vectors")
        logger.info("="*60)

        Y_matrix, semantic_df = self.compute_semantic_vectors(
            occurrences_df,
            anchor_occurrences=anchor_occurrences,
            show_progress=show_progress
        )

        results['semantic_df'] = semantic_df
        results['Y_matrix'] = Y_matrix

        # Step 2: 月度聚合
        if compute_monthly:
            logger.info("\n" + "="*60)
            logger.info("Step 2: Monthly aggregation")
            logger.info("="*60)

            monthly_df = self.aggregate_monthly(semantic_df)
            results['monthly_df'] = monthly_df

        # Step 3: 保存结果
        if save_results:
            logger.info("\n" + "="*60)
            logger.info("Step 3: Saving results")
            logger.info("="*60)

            self.save_results(
                semantic_df,
                monthly_df if compute_monthly else None
            )

        # 打印摘要
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """打印摘要"""
        print("\n" + "="*60)
        print("Paragraph-Level LLM-ER Summary")
        print("="*60)

        semantic_df = results['semantic_df']

        print(f"\nIndividual-level data:")
        print(f"  Total occurrences: {len(semantic_df):,}")
        print(f"  Unique terms: {semantic_df['term'].nunique()}")
        print(f"  Vector dimension: {results['Y_matrix'].shape[1]}")

        print(f"\nBy country:")
        for country in sorted(semantic_df['country'].unique()):
            count = len(semantic_df[semantic_df['country'] == country])
            print(f"  {country}: {count:,}")

        if 'monthly_df' in results:
            monthly_df = results['monthly_df']
            print(f"\nMonthly aggregation:")
            print(f"  Total months: {len(monthly_df)}")
            for country in sorted(monthly_df['country'].unique()):
                count = len(monthly_df[monthly_df['country'] == country])
                print(f"  {country}: {count} months")


def run_paragraph_level_llm_er(
    occurrences_path: Optional[Path] = None,
    compute_monthly: bool = True
) -> Dict:
    """
    便捷函数：运行段落级 LLM-ER

    Args:
        occurrences_path: 术语出现数据路径
        compute_monthly: 是否计算月度聚合

    Returns:
        结果字典
    """
    # 加载术语出现数据
    occurrences_path = occurrences_path or get_path("data/intermediate/term_occurrences.parquet")
    occurrences_df = pd.read_parquet(occurrences_path)
    logger.info(f"Loaded {len(occurrences_df)} term occurrences")

    # 运行处理
    processor = ParagraphLevelProcessor()
    return processor.run(occurrences_df, compute_monthly=compute_monthly)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run paragraph-level LLM-ER processing")
    parser.add_argument(
        "--input",
        type=str,
        default="data/intermediate/term_occurrences.parquet",
        help="Input file path"
    )
    parser.add_argument(
        "--no-monthly",
        action="store_true",
        help="Skip monthly aggregation"
    )

    args = parser.parse_args()

    results = run_paragraph_level_llm_er(
        occurrences_path=get_path(args.input),
        compute_monthly=not args.no_monthly
    )
