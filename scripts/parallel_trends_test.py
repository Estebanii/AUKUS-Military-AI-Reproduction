"""
平行趋势检验 (Parallel Trends Test)
=====================================

DID方法的核心假设检验：
在AUKUS之前，UK和AU相对于US的概念化趋势是否平行？

方法：事件研究法 (Event Study Design)
- 以2021年（AUKUS签署年）为基准年
- 估计每个年份的 UK×Year 和 AU×Year 交互项系数
- AUKUS前的系数应接近0且不显著（平行趋势）
- AUKUS后的系数可能显著（政策效应）

"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# ============================================================
# 路径设置
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """加载带有全局A矩阵转换的语义向量数据"""
    data_path = PROJECT_ROOT / 'data' / 'semantic_vectors_paragraph_global_A.parquet'

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"加载了 {len(df)} 条记录")

    # 确保有必要的列
    required_cols = ['country', 'year', 'Y_vector_global', 'doc_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    return df


def wild_cluster_bootstrap_se(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> dict:
    """
    计算每个系数的Wild Cluster Bootstrap标准误和p值。

    与wild_cluster_bootstrap.py和did_robustness_full.py使用相同的方法：
    Rademacher权重，doc_id聚类，1000次重抽样。

    Args:
        X: 含截距的设计矩阵 (n_samples, n_params)
        y: 因变量 (n_samples,)
        clusters: 聚类标签 (n_samples,)
        n_bootstrap: Bootstrap次数
        seed: 随机种子

    Returns:
        dict with keys: beta, se_ols, se_bootstrap, p_ols, p_bootstrap
    """
    rng = np.random.RandomState(seed)
    n_samples, n_params = X.shape

    # OLS估计
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta

    # OLS标准误
    sigma2 = np.sum(residuals ** 2) / (n_samples - n_params)
    se_ols = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat_ols = beta / se_ols
    p_ols = 2 * (1 - stats.t.cdf(np.abs(t_stat_ols), n_samples - n_params))

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 聚类Bootstrap
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {}
    for i, c in enumerate(clusters):
        if c not in cluster_to_idx:
            cluster_to_idx[c] = []
        cluster_to_idx[c].append(i)
    cluster_ids = list(unique_clusters)

    bootstrap_betas = np.zeros((n_bootstrap, n_params))
    for b in range(n_bootstrap):
        weights = rng.choice([-1, 1], size=n_clusters)
        sample_weights = np.empty(n_samples)
        for j, c in enumerate(cluster_ids):
            sample_weights[cluster_to_idx[c]] = weights[j]

        Y_star = X @ beta + residuals * sample_weights
        bootstrap_betas[b] = XtX_inv @ X.T @ Y_star

    se_bootstrap = np.std(bootstrap_betas, axis=0, ddof=1)

    # Bootstrap p值（双侧）
    centered = bootstrap_betas - beta[np.newaxis, :]
    p_bootstrap = np.mean(np.abs(centered) >= np.abs(beta[np.newaxis, :]), axis=0)

    return {
        'beta': beta,
        'se_ols': se_ols,
        'se_bootstrap': se_bootstrap,
        'p_ols': p_ols,
        'p_bootstrap': p_bootstrap,
        'r2': r2,
        'bootstrap_betas': bootstrap_betas,
        'n_clusters': n_clusters
    }


def prepare_event_study_features(
    df: pd.DataFrame,
    base_year: int = 2021,
    min_year: int = 2014,
    max_year: int = 2024
) -> Tuple[np.ndarray, List[str], np.ndarray, pd.DataFrame]:
    """
    准备事件研究的特征矩阵（仅US-UK比较）

    Args:
        df: 数据DataFrame
        base_year: 基准年份（AUKUS签署年，系数=0）
        min_year: 最小年份
        max_year: 最大年份

    Returns:
        X: 特征矩阵
        feature_names: 特征名称
        Y: 因变量矩阵
        df_filtered: 过滤后的数据（用于诊断/描述）
    """
    # 过滤至US+UK、指定年份范围
    df_filtered = df[
        (df['year'] >= min_year) & (df['year'] <= max_year) &
        (df['country'].isin(['US', 'UK']))
    ].copy()
    logger.info(f"过滤后记录数（US+UK, {min_year}-{max_year}）: {len(df_filtered)}")

    # 获取年份列表（排除基准年）
    years = sorted([y for y in df_filtered['year'].unique() if y != base_year])
    logger.info(f"年份列表（不含基准年{base_year}）: {years}")

    features = []
    feature_names = []

    # 国家哑变量（US为基准）
    df_uk = (df_filtered['country'] == 'UK').astype(float).values
    features.append(df_uk)
    feature_names.append('UK')

    # 年份哑变量（base_year为基准）
    for year in years:
        year_dummy = (df_filtered['year'] == year).astype(float).values
        features.append(year_dummy)
        feature_names.append(f'year_{year}')

    # UK×年份交互项（核心：事件研究系数）
    for year in years:
        year_dummy = (df_filtered['year'] == year).astype(float).values
        uk_x_year = df_uk * year_dummy
        features.append(uk_x_year)
        feature_names.append(f'UK_x_{year}')

    X = np.column_stack(features)

    # 提取Y矩阵
    Y = np.vstack(df_filtered['Y_vector_global'].values)

    logger.info(f"特征矩阵形状: {X.shape}")
    logger.info(f"因变量矩阵形状: {Y.shape}")

    return X, feature_names, Y, df_filtered


def run_pca_regression(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    n_components: int = 3,
    pca_obj=None,
    clusters: np.ndarray = None,
    n_bootstrap: int = 1000,
    seed_base: int = 42
) -> Dict:
    """
    运行PCA降维后的回归分析

    Args:
        X: 特征矩阵
        Y: 因变量矩阵
        feature_names: 特征名称
        n_components: PCA主成分数
        pca_obj: 预拟合的PCA对象（如提供，则使用transform而非fit_transform，
                 确保PCA基与DID脚本全样本拟合一致）
        clusters: 聚类标签（doc_id），如提供则使用Wild Cluster Bootstrap SE
        n_bootstrap: Bootstrap重抽样次数
        seed_base: Bootstrap随机种子基数

    Returns:
        回归结果字典
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = X.shape
    use_bootstrap = clusters is not None

    # PCA降维：使用预拟合PCA或现场拟合
    if pca_obj is not None:
        pca = pca_obj
        Y_pca = pca.transform(Y)
        logger.info("使用预拟合PCA（全样本基，与DID脚本一致）")
    else:
        pca = PCA(n_components=n_components)
        Y_pca = pca.fit_transform(Y)

    explained_variance = pca.explained_variance_ratio_
    logger.info(f"PCA解释方差: {explained_variance}")
    if use_bootstrap:
        n_clusters = len(np.unique(clusters))
        logger.info(f"使用Wild Cluster Bootstrap SE（{n_bootstrap}次，{n_clusters}个doc_id聚类）")

    # 添加截距
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    feature_names_with_intercept = ['intercept'] + feature_names
    n_params = n_features + 1

    results = {
        'pca_explained_variance': explained_variance.tolist(),
        'n_samples': n_samples,
        'n_features': n_features,
        'se_method': 'wild_cluster_bootstrap' if use_bootstrap else 'ols',
        'coefficients': {},
        'base_year': 2021,
        'feature_names': ['intercept'] + feature_names,
        '_bootstrap_betas': {},  # per-PC bootstrap coefficient matrices for Wald test
    }

    # 对每个主成分运行回归
    for pc_idx in range(n_components):
        y = Y_pca[:, pc_idx]
        pc_name = f'PC{pc_idx + 1}'

        if use_bootstrap:
            # Wild Cluster Bootstrap：聚类稳健SE和p值
            boot = wild_cluster_bootstrap_se(
                X_with_intercept, y, clusters,
                n_bootstrap=n_bootstrap,
                seed=seed_base + pc_idx
            )
            beta = boot['beta']
            se = boot['se_bootstrap']
            p_val = boot['p_bootstrap']
            r2 = boot['r2']
            se_ols = boot['se_ols']
            bootstrap_betas_pc = boot['bootstrap_betas']  # (n_bootstrap, n_params)
            logger.info(f"  {pc_name}: Bootstrap完成, 平均SE膨胀={np.mean(se/se_ols):.2f}x")
        else:
            # 经典OLS标准误
            XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
            beta = XtX_inv @ X_with_intercept.T @ y
            residuals = y - X_with_intercept @ beta
            sigma2 = np.sum(residuals ** 2) / (n_samples - n_params)
            se = np.sqrt(np.diag(sigma2 * XtX_inv))
            t_stat = beta / se
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_params))
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

        results['coefficients'][pc_name] = {
            'r2': r2
        }
        if use_bootstrap:
            results['_bootstrap_betas'][pc_name] = bootstrap_betas_pc

        # 存储每个特征的系数
        for i, name in enumerate(feature_names_with_intercept):
            results['coefficients'][pc_name][name] = {
                'coef': float(beta[i]),
                'se': float(se[i]),
                't': float(beta[i] / se[i]) if se[i] > 0 else 0.0,
                'p': float(p_val[i])
            }

    return results


def extract_event_study_coefficients(results: Dict) -> pd.DataFrame:
    """
    提取事件研究系数（UK×Year和AU×Year交互项）

    Args:
        results: 回归结果

    Returns:
        事件研究系数表
    """
    records = []

    for pc_name, pc_results in results['coefficients'].items():
        for feature_name, coef_info in pc_results.items():
            if isinstance(coef_info, dict) and 'coef' in coef_info:
                # 解析UK_x_YEAR或AU_x_YEAR格式
                if '_x_' in feature_name:
                    parts = feature_name.split('_x_')
                    if len(parts) == 2:
                        country = parts[0]
                        year = int(parts[1])

                        records.append({
                            'pc': pc_name,
                            'country': country,
                            'year': year,
                            'coef': coef_info['coef'],
                            'se': coef_info['se'],
                            't': coef_info['t'],
                            'p': coef_info['p'],
                            'significant': coef_info['p'] < 0.05,
                            'ci_lower': coef_info['coef'] - 1.96 * coef_info['se'],
                            'ci_upper': coef_info['coef'] + 1.96 * coef_info['se']
                        })

    df = pd.DataFrame(records)

    # 添加AUKUS前后标记
    base_year = results.get('base_year', 2021)
    df['period'] = df['year'].apply(lambda x: 'pre_aukus' if x < base_year else 'post_aukus')

    return df


def plot_event_study(
    coef_df: pd.DataFrame,
    output_path: Path,
    base_year: int = 2021
):
    """
    绘制事件研究图

    Args:
        coef_df: 事件研究系数表
        output_path: 输出路径
        base_year: 基准年份
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pcs = ['PC1', 'PC2', 'PC3']

    for col, pc in enumerate(pcs):
            ax = axes[col]
            country = 'UK'

            # 筛选数据
            data = coef_df[(coef_df['country'] == country) & (coef_df['pc'] == pc)]
            data = data.sort_values('year')

            if len(data) == 0:
                ax.set_title(f'{country} - {pc} (无数据)')
                continue

            years = data['year'].values
            coefs = data['coef'].values
            ci_lower = data['ci_lower'].values
            ci_upper = data['ci_upper'].values
            significant = data['significant'].values

            # 绘制置信区间
            ax.fill_between(years, ci_lower, ci_upper, alpha=0.3, color='steelblue')

            # 绘制系数点
            colors = ['red' if sig else 'steelblue' for sig in significant]
            ax.scatter(years, coefs, c=colors, s=80, zorder=5, edgecolors='black')
            ax.plot(years, coefs, color='steelblue', linewidth=1.5, alpha=0.7)

            # 添加零线和AUKUS线
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.axvline(x=base_year, color='red', linestyle='--', linewidth=2,
                      label=f'AUKUS ({base_year})')

            # 标注显著性
            for i, (yr, coef, sig) in enumerate(zip(years, coefs, significant)):
                if sig:
                    ax.annotate('*', (yr, coef), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=14, color='red')

            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel('Coefficient', fontsize=11)
            ax.set_title(f'{country} × Year Interaction ({pc})', fontsize=12)
            ax.grid(True, alpha=0.3)

            # 设置x轴刻度
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=45)

    # 添加图例和说明
    fig.suptitle('Event Study: UK × Year Interaction (vs US baseline)\n'
                 '(Red dots = significant at p<0.05; Shaded area = 95% CI; 2014-2024)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"事件研究图已保存: {output_path}")


def test_parallel_trends(coef_df: pd.DataFrame, regression_results: Dict = None) -> Dict:
    """
    检验平行趋势假设（完整Bootstrap协方差矩阵Wald）

    对AUKUS前的系数进行联合检验：H0: 所有pre_aukus系数 = 0
    使用Bootstrap系数矩阵构造完整协方差矩阵，避免对角Wald的偏误。

    Args:
        coef_df: 事件研究系数表
        regression_results: 回归结果（含_bootstrap_betas和feature_names）

    Returns:
        检验结果
    """
    results = {}
    feature_names = regression_results.get('feature_names', []) if regression_results else []

    for pc in ['PC1', 'PC2', 'PC3']:
        results[pc] = {}

        for country in ['UK']:
            pre_data = coef_df[
                (coef_df['country'] == country) &
                (coef_df['pc'] == pc) &
                (coef_df['period'] == 'pre_aukus')
            ]

            if len(pre_data) == 0:
                results[pc][country] = {
                    'n_years': 0,
                    'conclusion': 'insufficient_data'
                }
                continue

            coefs = pre_data['coef'].values
            ses = pre_data['se'].values
            pre_years = sorted(pre_data['year'].values)

            # 使用完整Bootstrap协方差矩阵构造Wald统计量
            boot_betas = regression_results.get('_bootstrap_betas', {}).get(pc) if regression_results else None
            if boot_betas is not None and len(feature_names) > 0:
                # 找到pre-AUKUS UK×Year交互项在feature_names中的索引
                pre_indices = []
                for yr in pre_years:
                    fname = f'UK_x_{yr}'
                    if fname in feature_names:
                        pre_indices.append(feature_names.index(fname))

                if len(pre_indices) == len(pre_years):
                    # 提取这些系数的Bootstrap样本子矩阵
                    boot_sub = boot_betas[:, pre_indices]  # (n_bootstrap, n_pre_years)
                    # 完整协方差矩阵
                    cov_matrix = np.cov(boot_sub, rowvar=False, ddof=1)
                    # Wald = β' Σ⁻¹ β
                    try:
                        cov_inv = np.linalg.inv(cov_matrix)
                        wald_stat = float(coefs @ cov_inv @ coefs)
                        df = len(coefs)
                        p_value = 1 - stats.chi2.cdf(wald_stat, df)
                        logger.info(f"  {pc} {country}: 完整协方差Wald χ²={wald_stat:.2f}, df={df}, p={p_value:.4f}")
                    except np.linalg.LinAlgError:
                        logger.warning(f"  {pc} {country}: 协方差矩阵奇异，回退到对角Wald")
                        wald_stat = float(np.sum((coefs / ses) ** 2))
                        df = len(coefs)
                        p_value = 1 - stats.chi2.cdf(wald_stat, df)
                else:
                    logger.warning(f"  {pc} {country}: 特征名匹配失败，回退到对角Wald")
                    wald_stat = float(np.sum((coefs / ses) ** 2))
                    df = len(coefs)
                    p_value = 1 - stats.chi2.cdf(wald_stat, df)
            else:
                # 回退：对角Wald
                wald_stat = float(np.sum((coefs / ses) ** 2))
                df = len(coefs)
                p_value = 1 - stats.chi2.cdf(wald_stat, df)

            # 检验每个系数是否显著
            individual_sig = pre_data['significant'].values
            n_significant = individual_sig.sum()

            # 判断是否满足平行趋势
            # 标准1：联合检验不显著 (p > 0.05)
            # 标准2：大多数个别系数不显著
            joint_test_pass = p_value > 0.05
            individual_test_pass = n_significant <= 1  # 最多1个显著

            parallel_trends_satisfied = joint_test_pass and individual_test_pass
            # 三级分类：satisfied (p>0.1), marginal (0.05<p<0.1), violated (p<0.05)
            if p_value > 0.1:
                conclusion = 'satisfied'
            elif p_value > 0.05:
                conclusion = 'marginal'
            else:
                conclusion = 'violated'

            results[pc][country] = {
                'n_years': len(coefs),
                'wald_stat': float(wald_stat),
                'df': df,
                'p_value': float(p_value),
                'n_significant': int(n_significant),
                'joint_test_pass': bool(joint_test_pass),
                'individual_test_pass': bool(individual_test_pass),
                'parallel_trends_satisfied': bool(parallel_trends_satisfied),
                'conclusion': conclusion
            }

    return results


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("平行趋势检验 (Parallel Trends Test)")
    logger.info("="*60)

    # 输出目录
    output_dir = PROJECT_ROOT / 'figures' / 'parallel_trends'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    logger.info("\n--- Step 1: 加载数据 ---")
    df = load_data()

    # 2. PCA降维（在全样本上拟合，与DID脚本保持一致）
    logger.info("\n--- Step 2: PCA降维（全样本拟合） ---")
    from sklearn.decomposition import PCA
    Y_full = np.vstack(df['Y_vector_global'].values)
    pca_full = PCA(n_components=3)
    pca_full.fit(Y_full)
    logger.info(f"PCA解释方差（全样本，N={len(df)}）: {pca_full.explained_variance_ratio_}")

    # 3. 准备事件研究特征（US+UK, 2014-2024）
    logger.info("\n--- Step 3: 准备事件研究特征（US+UK, 2014-2024） ---")
    X, feature_names, Y, df_filtered = prepare_event_study_features(
        df,
        base_year=2021,  # AUKUS签署年为基准
        min_year=2014,   # 从2014年开始（US+UK重叠时段）
        max_year=2024    # 到2024年（2025不完整）
    )

    # 4. 运行PCA回归（全样本PCA基 + 聚类Bootstrap SE）
    logger.info("\n--- Step 4: 运行PCA回归（全样本PCA基 + Wild Cluster Bootstrap SE） ---")
    clusters = df_filtered['doc_id'].values
    results = run_pca_regression(
        X, Y, feature_names, n_components=3,
        pca_obj=pca_full,
        clusters=clusters,
        n_bootstrap=1000,
        seed_base=42
    )

    # 5. 提取事件研究系数
    logger.info("\n--- Step 5: 提取事件研究系数 ---")
    coef_df = extract_event_study_coefficients(results)

    # 打印系数表
    print("\n事件研究系数表:")
    print(coef_df.to_string(index=False))

    # 5. 检验平行趋势
    logger.info("\n--- Step 5: 检验平行趋势 ---")
    parallel_trends_results = test_parallel_trends(coef_df, regression_results=results)

    print("\n平行趋势检验结果:")
    for pc, country_results in parallel_trends_results.items():
        print(f"\n{pc}:")
        for country, result in country_results.items():
            if result.get('n_years', 0) > 0:
                print(f"  {country}: Wald χ² = {result['wald_stat']:.2f} (df={result['df']}), "
                      f"p = {result['p_value']:.4f}, "
                      f"显著系数数 = {result['n_significant']}, "
                      f"结论: {result['conclusion']}")

    # 6. 绘制事件研究图
    logger.info("\n--- Step 6: 绘制事件研究图 ---")
    plot_event_study(
        coef_df,
        output_dir / 'event_study_parallel_trends.png',
        base_year=2021
    )

    # 7. 保存结果
    logger.info("\n--- Step 7: 保存结果 ---")

    # 保存系数表
    coef_df.to_csv(output_dir / 'event_study_coefficients.csv', index=False)

    # 保存完整结果
    full_results = {
        'method': 'Event Study for Parallel Trends Test',
        'base_year': 2021,
        'n_samples': int(results['n_samples']),
        'pca_explained_variance': results['pca_explained_variance'],
        'parallel_trends_test': parallel_trends_results,
        'interpretation': generate_interpretation(parallel_trends_results)
    }

    with open(output_dir / 'parallel_trends_results.json', 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n结果已保存到: {output_dir}")
    logger.info("="*60)
    logger.info("平行趋势检验完成!")
    logger.info("="*60)

    return full_results


def generate_interpretation(results: Dict) -> str:
    """生成平行趋势检验的解释"""
    interpretations = []

    # 检查UK的结果
    uk_satisfied = all(
        results[pc]['UK'].get('parallel_trends_satisfied', False)
        for pc in ['PC1', 'PC2', 'PC3']
        if results[pc]['UK'].get('n_years', 0) > 0
    )

    if uk_satisfied:
        interpretations.append(
            "英国样本：平行趋势假设得到满足。AUKUS前的年份系数均不显著异于零，"
            "表明英国与美国在AUKUS之前的概念化趋势是平行的。DID估计对英国样本有效。"
        )
    else:
        interpretations.append(
            "英国样本：平行趋势假设存在争议。部分AUKUS前年份的系数显著，"
            "可能表明两国在AUKUS之前已存在不同的演变趋势。需要谨慎解释DID结果。"
        )

    return "\n\n".join(interpretations)


if __name__ == '__main__':
    main()
