"""
PC回归系数验证脚本
=====================================

目的: 验证 global_A_regression_results.json 中的PC回归系数
复现论文中H1(差异模式解读)的关键数据

论文数据 (Table 3b):
- PC1: AU beta=-0.057, p<0.001; UK beta=-0.010, p=0.051
- PC2: UK beta=0.013, p<0.01
- PC3: UK beta=-0.023, p<0.001

方法:
1. 加载Y_vector_global数据
2. PCA降维到前3个主成分
3. 对每个PC运行OLS回归: PC_i = b0 + b_UK*D_UK + b_AU*D_AU + b_post*post_aukus + ...
4. 比较系数与论文数据

"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# ============================================================
# 路径设置 (fuxian文件夹内的相对路径)
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

    # 确保有必要的列
    required_cols = ['country', 'year', 'Y_vector_global', 'post_aukus']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    准备回归特征矩阵

    模型: Y = b0 + b_UK*D_UK + b_AU*D_AU + b_post*post_aukus +
               b_UK_post*(UK*post) + b_AU_post*(AU*post) + e

    Returns:
        X: 特征矩阵 (n_samples, n_features)
        feature_names: 特征名称列表
    """
    features = []
    feature_names = []

    # 国家哑变量 (US为基准)
    df_uk = (df['country'] == 'UK').astype(float).values
    df_au = (df['country'] == 'AU').astype(float).values
    features.extend([df_uk, df_au])
    feature_names.extend(['UK', 'AU'])

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

    logger.info(f"特征矩阵形状: {X.shape}")
    logger.info(f"特征名称: {feature_names}")

    return X, feature_names


def run_pc_regression(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    n_components: int = 3
) -> Dict:
    """
    运行PCA降维后的回归分析

    Args:
        X: 特征矩阵
        Y: 因变量矩阵 (n_samples, 768)
        feature_names: 特征名称
        n_components: PCA主成分数

    Returns:
        回归结果字典
    """
    n_samples, n_features = X.shape

    # PCA降维
    pca = PCA(n_components=n_components)
    Y_pca = pca.fit_transform(Y)

    explained_variance = pca.explained_variance_ratio_
    logger.info(f"PCA解释方差: {explained_variance}")
    logger.info(f"累计解释方差: {np.cumsum(explained_variance)}")

    # 添加截距
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    feature_names_with_intercept = ['intercept'] + feature_names
    n_params = n_features + 1

    results = {
        'pca_explained_variance': {f'PC{i+1}': float(v) for i, v in enumerate(explained_variance)},
        'n_samples': n_samples,
        'n_features': n_features,
        'regression_coefficients': {},
        'model_fit': {}
    }

    # 对每个主成分运行回归
    for pc_idx in range(n_components):
        y = Y_pca[:, pc_idx]

        # OLS估计: beta = (X'X)^{-1} X'y
        XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
        beta = XtX_inv @ X_with_intercept.T @ y
        residuals = y - X_with_intercept @ beta

        # 计算标准误（经典OLS）
        sigma2 = np.sum(residuals ** 2) / (n_samples - n_params)
        V = sigma2 * XtX_inv
        se = np.sqrt(np.diag(V))

        # t统计量和p值
        t_stat = beta / se
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_params))

        # 计算R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        pc_name = f'PC{pc_idx + 1}'
        results['model_fit'][f'{pc_name}_r2'] = float(r2)

        # 存储每个特征的系数
        for i, name in enumerate(feature_names_with_intercept):
            if name not in results['regression_coefficients']:
                results['regression_coefficients'][name] = {}

            results['regression_coefficients'][name][f'{pc_name}_coef'] = float(beta[i])
            results['regression_coefficients'][name][f'{pc_name}_se'] = float(se[i])
            results['regression_coefficients'][name][f'{pc_name}_t'] = float(t_stat[i])
            results['regression_coefficients'][name][f'{pc_name}_p'] = float(p_val[i])

    return results


def verify_against_json(results: Dict) -> Dict:
    """
    验证回归结果与原JSON数据是否一致
    """
    original_json_path = RESULTS_DIR / 'global_A_regression_results.json'

    if not original_json_path.exists():
        return {'error': f'原JSON文件不存在: {original_json_path}'}

    with open(original_json_path, 'r') as f:
        original_results = json.load(f)

    verification = {'matches': [], 'mismatches': []}

    for var in ['UK', 'AU', 'post_aukus', 'UK_x_post', 'AU_x_post']:
        orig = original_results['regression_coefficients'].get(var, {})
        computed = results['regression_coefficients'].get(var, {})

        for pc in ['PC1', 'PC2', 'PC3']:
            orig_coef = orig.get(f'{pc}_coef', None)
            comp_coef = computed.get(f'{pc}_coef', None)

            if orig_coef is not None and comp_coef is not None:
                diff = abs(orig_coef - comp_coef)
                rel_diff = diff / abs(orig_coef) if orig_coef != 0 else diff
                match = rel_diff < 1e-10

                entry = {
                    'variable': var,
                    'pc': pc,
                    'original': orig_coef,
                    'computed': comp_coef,
                    'diff': diff,
                    'rel_diff': rel_diff,
                    'match': match
                }

                if match:
                    verification['matches'].append(entry)
                else:
                    verification['mismatches'].append(entry)

    return verification


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("PC回归系数验证")
    logger.info("="*60)

    # 1. 加载数据
    logger.info("\n--- Step 1: 加载数据 ---")
    df = load_data()

    # 打印数据概览
    print(f"\n数据概览:")
    print(f"  总样本量: {len(df):,}")
    print(f"  国家分布: {df['country'].value_counts().to_dict()}")
    print(f"  AUKUS前: {(df['post_aukus'] == 0).sum():,}")
    print(f"  AUKUS后: {(df['post_aukus'] == 1).sum():,}")

    # 2. 准备特征
    logger.info("\n--- Step 2: 准备特征 ---")
    X, feature_names = prepare_features(df)

    # 提取Y矩阵
    Y = np.vstack(df['Y_vector_global'].values)
    logger.info(f"Y矩阵形状: {Y.shape}")

    # 3. 运行PC回归
    logger.info("\n--- Step 3: 运行PC回归 ---")
    results = run_pc_regression(X, Y, feature_names, n_components=3)

    # 4. 打印回归系数表
    print("\n" + "="*70)
    print("PC回归系数表 (与论文Table 3b对比)")
    print("="*70)

    print("\n{:<12} {:<15} {:<15} {:<15}".format("变量", "PC1", "PC2", "PC3"))
    print("-"*60)

    for var in ['intercept', 'UK', 'AU', 'post_aukus', 'UK_x_post', 'AU_x_post']:
        coefs = results['regression_coefficients'].get(var, {})
        pc1 = f"{coefs.get('PC1_coef', 0):.4f}" if coefs else "-"
        pc2 = f"{coefs.get('PC2_coef', 0):.4f}" if coefs else "-"
        pc3 = f"{coefs.get('PC3_coef', 0):.4f}" if coefs else "-"

        # 添加显著性标记
        p1 = coefs.get('PC1_p', 1)
        p2 = coefs.get('PC2_p', 1)
        p3 = coefs.get('PC3_p', 1)

        pc1 += "***" if p1 < 0.001 else ("**" if p1 < 0.01 else ("*" if p1 < 0.05 else ("." if p1 < 0.1 else "")))
        pc2 += "***" if p2 < 0.001 else ("**" if p2 < 0.01 else ("*" if p2 < 0.05 else ("." if p2 < 0.1 else "")))
        pc3 += "***" if p3 < 0.001 else ("**" if p3 < 0.01 else ("*" if p3 < 0.05 else ("." if p3 < 0.1 else "")))

        print(f"{var:<12} {pc1:<15} {pc2:<15} {pc3:<15}")

    print("-"*60)
    print("显著性: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

    # 5. 与原JSON文件对比
    logger.info("\n--- Step 4: 与原JSON对比 ---")
    verification = verify_against_json(results)

    print("\n" + "="*70)
    print("与原global_A_regression_results.json对比")
    print("="*70)

    if 'error' in verification:
        print(f"\n错误: {verification['error']}")
    else:
        print(f"\n匹配数: {len(verification['matches'])}")
        print(f"不匹配数: {len(verification['mismatches'])}")

        if verification['matches']:
            print("\n匹配项:")
            for item in verification['matches']:
                print(f"  {item['variable']} {item['pc']}: 原始={item['original']:.6f}, 计算={item['computed']:.6f}, 差异={item['diff']:.2e} ✓")

        if verification['mismatches']:
            print("\n不匹配项:")
            for item in verification['mismatches']:
                print(f"  {item['variable']} {item['pc']}: 原始={item['original']:.6f}, 计算={item['computed']:.6f}, 差异={item['diff']:.2e} ✗")

    # 6. 保存验证结果
    logger.info("\n--- Step 5: 保存结果 ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / 'pc_regression_verification.json'
    verification_output = {
        'method': 'PC Regression Verification',
        'description': '验证global_A_regression_results.json中的PC回归系数',
        'n_samples': results['n_samples'],
        'pca_explained_variance': results['pca_explained_variance'],
        'regression_coefficients': results['regression_coefficients'],
        'model_fit': results['model_fit'],
        'verification': verification
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(verification_output, f, indent=2, ensure_ascii=False)

    logger.info(f"验证结果已保存: {output_path}")

    print("\n" + "="*70)
    print("PC回归验证完成!")
    print("="*70)

    return results, verification


if __name__ == '__main__':
    results, verification = main()
