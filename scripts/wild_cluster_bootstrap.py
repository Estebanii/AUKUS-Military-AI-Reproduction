"""
Wild Cluster Bootstrap 检验脚本
=====================================

目的: 运行Wild Cluster Bootstrap，获得聚类稳健标准误和p值
对比经典OLS标准误与Bootstrap标准误，验证论文中Bootstrap描述的正确性

方法:
1. 加载Y_vector_global数据
2. PCA降维到前3个主成分（与pc_regression_test.py完全一致）
3. 对每个PC运行OLS回归获取原始系数和残差
4. 以doc_id为聚类变量，使用Rademacher权重进行1000次Bootstrap重抽样
5. 计算Bootstrap标准误和p值
6. 输出OLS vs Bootstrap对比表

参考实现: colab_processed/src/regression/hypothesis_testing.py:116-199

"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# ============================================================
# 路径设置
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """加载带有全局A矩阵转换的语义向量数据"""
    data_path = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"加载了 {len(df)} 条记录")

    required_cols = ['country', 'year', 'Y_vector_global', 'post_aukus', 'doc_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    准备回归特征矩阵（含线性时间趋势，匹配论文方程）

    模型: Y = b0 + b_UK*D_UK + b_AU*D_AU + b_time*t + b_post*post_aukus +
               b_UK_post*(UK*post) + b_AU_post*(AU*post) + e

    time = (year - 2014), 标准化起始点
    """
    features = []
    feature_names = []

    # 国家哑变量 (US为基准)
    df_uk = (df['country'] == 'UK').astype(float).values
    df_au = (df['country'] == 'AU').astype(float).values
    features.extend([df_uk, df_au])
    feature_names.extend(['UK', 'AU'])

    # 线性时间趋势
    time_var = (df['year'] - 2014).astype(float).values
    features.append(time_var)
    feature_names.append('time')

    # AUKUS后期哑变量
    post_aukus = df['post_aukus'].astype(float).values
    features.append(post_aukus)
    feature_names.append('post_aukus')

    # 交互项 (DID核心)
    uk_x_post = df_uk * post_aukus
    au_x_post = df_au * post_aukus
    features.extend([uk_x_post, au_x_post])
    feature_names.extend(['UK_x_post', 'AU_x_post'])

    X = np.column_stack(features)
    return X, feature_names


def wild_cluster_bootstrap_pc(
    X_with_intercept: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    对单个PC维度运行Wild Cluster Bootstrap

    Args:
        X_with_intercept: 含截距的设计矩阵 (n_samples, n_params)
        y: 因变量向量 (n_samples,)
        clusters: 聚类标签 (n_samples,)
        n_bootstrap: Bootstrap重抽样次数
        seed: 随机种子

    Returns:
        包含原始OLS结果和Bootstrap结果的字典
    """
    np.random.seed(seed)

    n_samples, n_params = X_with_intercept.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # ---- 原始OLS估计 ----
    XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
    beta_original = XtX_inv @ X_with_intercept.T @ y
    residuals = y - X_with_intercept @ beta_original

    # 经典OLS标准误
    sigma2 = np.sum(residuals ** 2) / (n_samples - n_params)
    V_ols = sigma2 * XtX_inv
    se_ols = np.sqrt(np.diag(V_ols))
    t_stat_ols = beta_original / se_ols
    p_val_ols = 2 * (1 - stats.t.cdf(np.abs(t_stat_ols), n_samples - n_params))

    # ---- Wild Cluster Bootstrap ----
    # 预计算：将样本按聚类分组的索引
    cluster_to_idx = {}
    for i, c in enumerate(clusters):
        if c not in cluster_to_idx:
            cluster_to_idx[c] = []
        cluster_to_idx[c].append(i)

    # 将cluster映射为有序数组以加速
    cluster_ids = np.array([c for c in unique_clusters])

    bootstrap_betas = np.zeros((n_bootstrap, n_params))

    for b in range(n_bootstrap):
        # Rademacher权重：对每个cluster抽取{-1, +1}
        weights = np.random.choice([-1, 1], size=n_clusters)

        # 构造每个样本的权重
        sample_weights = np.empty(n_samples)
        for j, c in enumerate(cluster_ids):
            idx = cluster_to_idx[c]
            sample_weights[idx] = weights[j]

        # 扰动残差
        perturbed_residuals = residuals * sample_weights

        # 构造Bootstrap Y*
        Y_star = X_with_intercept @ beta_original + perturbed_residuals

        # 重新估计（使用预计算的XtX_inv）
        beta_star = XtX_inv @ X_with_intercept.T @ Y_star
        bootstrap_betas[b] = beta_star

    # Bootstrap标准误
    se_bootstrap = np.std(bootstrap_betas, axis=0, ddof=1)

    # Bootstrap p值（双侧，中心化尾部概率）
    # p = Pr(|β*| >= |β̂|) 其中β*以0为中心
    # 标准做法：将bootstrap β* 中心化到0（因为H0下β=0）
    # 使用 |β* - β̂| >= |β̂| 等价于 |β*| >= |2β̂| 或
    # 更标准的：p = Pr(|β* - β̂| >= |β̂ - 0|) = Pr(|β*_centered| >= |β̂|)
    # 其中 β*_centered = β* - β̂
    centered_betas = bootstrap_betas - beta_original[np.newaxis, :]
    p_val_bootstrap = np.mean(
        np.abs(centered_betas) >= np.abs(beta_original[np.newaxis, :]),
        axis=0
    )

    return {
        'beta': beta_original,
        'se_ols': se_ols,
        't_stat_ols': t_stat_ols,
        'p_val_ols': p_val_ols,
        'se_bootstrap': se_bootstrap,
        'p_val_bootstrap': p_val_bootstrap,
        'n_bootstrap': n_bootstrap,
        'n_clusters': n_clusters,
        'n_samples': n_samples,
        'n_params': n_params
    }


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("Wild Cluster Bootstrap 检验")
    logger.info("=" * 70)

    start_time = time.time()

    # 1. 加载数据
    logger.info("\n--- Step 1: 加载数据 ---")
    df = load_data()

    print(f"\n数据概览:")
    print(f"  总样本量: {len(df):,}")
    print(f"  国家分布: {df['country'].value_counts().to_dict()}")
    print(f"  唯一文档数 (聚类数): {df['doc_id'].nunique():,}")

    # 2. 准备特征
    logger.info("\n--- Step 2: 准备特征 ---")
    X, feature_names = prepare_features(df)
    Y = np.vstack(df['Y_vector_global'].values)
    clusters = df['doc_id'].values

    # 3. PCA降维
    logger.info("\n--- Step 3: PCA降维 ---")
    pca = PCA(n_components=3)
    Y_pca = pca.fit_transform(Y)

    explained_variance = pca.explained_variance_ratio_
    logger.info(f"PCA解释方差: {explained_variance}")

    # 4. 添加截距
    n_samples = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    feature_names_with_intercept = ['intercept'] + feature_names

    # 5. 对每个PC运行Bootstrap
    N_BOOTSTRAP = 1000
    SEED_BASE = 42

    all_results = {}

    for pc_idx in range(3):
        pc_name = f'PC{pc_idx + 1}'
        logger.info(f"\n--- Step 5.{pc_idx+1}: {pc_name} Bootstrap ({N_BOOTSTRAP} iterations) ---")

        y = Y_pca[:, pc_idx]

        result = wild_cluster_bootstrap_pc(
            X_with_intercept, y, clusters,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED_BASE + pc_idx
        )

        all_results[pc_name] = result

        logger.info(f"  {pc_name}: 完成 ({result['n_clusters']} clusters, {N_BOOTSTRAP} iterations)")

    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed:.1f}秒")

    # 6. 打印对比表
    print("\n" + "=" * 100)
    print("OLS vs Wild Cluster Bootstrap 对比")
    print("=" * 100)

    for pc_name, result in all_results.items():
        print(f"\n{pc_name} (n_clusters={result['n_clusters']}, n_bootstrap={result['n_bootstrap']}):")
        print(f"{'变量':<12} {'系数':>10} {'OLS_SE':>10} {'Boot_SE':>10} {'SE比值':>10} {'OLS_p':>12} {'Boot_p':>12} {'结论一致':>10}")
        print("-" * 90)

        for i, name in enumerate(feature_names_with_intercept):
            coef = result['beta'][i]
            se_ols = result['se_ols'][i]
            se_boot = result['se_bootstrap'][i]
            se_ratio = se_boot / se_ols if se_ols > 0 else float('nan')
            p_ols = result['p_val_ols'][i]
            p_boot = result['p_val_bootstrap'][i]

            # 判断结论是否一致（同一显著性水平区间）
            def sig_level(p):
                if p < 0.001: return '***'
                elif p < 0.01: return '**'
                elif p < 0.05: return '*'
                elif p < 0.1: return '.'
                else: return 'ns'

            consistent = sig_level(p_ols) == sig_level(p_boot)

            print(f"{name:<12} {coef:>10.6f} {se_ols:>10.6f} {se_boot:>10.6f} {se_ratio:>10.4f} {p_ols:>12.2e} {p_boot:>12.3f} {'✓' if consistent else '✗':>10}")

    # 7. 加载原始JSON进行交叉验证
    logger.info("\n--- Step 7: 与原始JSON交叉验证 ---")
    original_json_path = RESULTS_DIR / 'global_A_regression_results.json'

    with open(original_json_path, 'r') as f:
        original = json.load(f)

    print("\n" + "=" * 100)
    print("与原始JSON系数交叉验证")
    print("=" * 100)

    cross_validation = {'matches': [], 'mismatches': []}

    for var in feature_names:
        orig = original['regression_coefficients'].get(var, {})
        for pc_idx in range(3):
            pc_name = f'PC{pc_idx + 1}'
            orig_coef = orig.get(f'{pc_name}_coef')
            computed_coef = all_results[pc_name]['beta'][feature_names_with_intercept.index(var)]

            if orig_coef is not None:
                diff = abs(orig_coef - computed_coef)
                rel_diff = diff / abs(orig_coef) if orig_coef != 0 else diff
                match = rel_diff < 1e-10

                entry = {'variable': var, 'pc': pc_name, 'diff': diff, 'match': match}
                if match:
                    cross_validation['matches'].append(entry)
                else:
                    cross_validation['mismatches'].append(entry)

    print(f"系数匹配: {len(cross_validation['matches'])}/{len(cross_validation['matches']) + len(cross_validation['mismatches'])}")

    # 8. 保存结果
    logger.info("\n--- Step 8: 保存结果 ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'method': 'Wild Cluster Bootstrap',
        'description': 'OLS vs Wild Cluster Bootstrap标准误和p值对比',
        'n_bootstrap': N_BOOTSTRAP,
        'seed_base': SEED_BASE,
        'n_samples': int(n_samples),
        'n_clusters': int(all_results['PC1']['n_clusters']),
        'avg_cluster_size': float(n_samples / all_results['PC1']['n_clusters']),
        'pca_explained_variance': {f'PC{i+1}': float(v) for i, v in enumerate(explained_variance)},
        'comparison': {}
    }

    for pc_name, result in all_results.items():
        pc_comparison = {}
        for i, name in enumerate(feature_names_with_intercept):
            pc_comparison[name] = {
                'coef': float(result['beta'][i]),
                'se_ols': float(result['se_ols'][i]),
                'se_bootstrap': float(result['se_bootstrap'][i]),
                'se_ratio': float(result['se_bootstrap'][i] / result['se_ols'][i]) if result['se_ols'][i] > 0 else None,
                'p_ols': float(result['p_val_ols'][i]),
                'p_bootstrap': float(result['p_val_bootstrap'][i]),
                't_stat_ols': float(result['t_stat_ols'][i])
            }
        output['comparison'][pc_name] = pc_comparison

    # 汇总统计
    all_se_ratios = []
    conclusion_changes = []
    for pc_name, result in all_results.items():
        for i, name in enumerate(feature_names_with_intercept):
            if name == 'intercept':
                continue
            ratio = result['se_bootstrap'][i] / result['se_ols'][i]
            all_se_ratios.append(ratio)

            def sig(p):
                return p < 0.05

            ols_sig = sig(result['p_val_ols'][i])
            boot_sig = sig(result['p_val_bootstrap'][i])
            if ols_sig != boot_sig:
                conclusion_changes.append({
                    'variable': name,
                    'pc': pc_name,
                    'ols_p': float(result['p_val_ols'][i]),
                    'bootstrap_p': float(result['p_val_bootstrap'][i]),
                    'ols_significant': bool(ols_sig),
                    'bootstrap_significant': bool(boot_sig)
                })

    output['summary'] = {
        'se_ratio_mean': float(np.mean(all_se_ratios)),
        'se_ratio_median': float(np.median(all_se_ratios)),
        'se_ratio_min': float(np.min(all_se_ratios)),
        'se_ratio_max': float(np.max(all_se_ratios)),
        'n_conclusion_changes': len(conclusion_changes),
        'conclusion_changes': conclusion_changes,
        'overall_assessment': (
            'OLS和Bootstrap结果高度一致，所有假设检验结论不变'
            if len(conclusion_changes) == 0
            else f'有{len(conclusion_changes)}个系数的显著性结论发生变化'
        )
    }

    output_path = OUTPUT_DIR / 'wild_cluster_bootstrap_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"结果已保存: {output_path}")

    # 9. 打印汇总
    print("\n" + "=" * 100)
    print("Wild Cluster Bootstrap 汇总")
    print("=" * 100)
    print(f"\n聚类数: {all_results['PC1']['n_clusters']:,}")
    print(f"平均聚类大小: {n_samples / all_results['PC1']['n_clusters']:.2f}")
    print(f"Bootstrap次数: {N_BOOTSTRAP}")
    print(f"\nSE比值（Bootstrap/OLS）统计:")
    print(f"  均值: {np.mean(all_se_ratios):.4f}")
    print(f"  中位数: {np.median(all_se_ratios):.4f}")
    print(f"  范围: [{np.min(all_se_ratios):.4f}, {np.max(all_se_ratios):.4f}]")
    print(f"\n结论变化数: {len(conclusion_changes)}")

    if conclusion_changes:
        print("变化详情:")
        for change in conclusion_changes:
            print(f"  {change['variable']} ({change['pc']}): OLS p={change['ols_p']:.4f} → Boot p={change['bootstrap_p']:.3f}")
    else:
        print("所有假设检验结论在Bootstrap下保持不变。")

    print(f"\n总耗时: {elapsed:.1f}秒")
    print("=" * 100)

    return output


if __name__ == '__main__':
    output = main()
