"""
DID H3 距离变化验证脚本
=====================================

目的: 验证 global_A_regression_results.json 中的H3距离变化结果
复现论文中H3(AUKUS后分化)的关键数据

论文数据 (H3_convergence):
- pre_aukus_avg_distance: 0.1477
- post_aukus_avg_distance: 0.1759
- percent_change: +19.1%
- 国家对距离变化:
  - US-UK: +13.2%
  - US-AU: +17.1%
  - UK-AU: +28.5%

方法:
1. 加载Y_vector_global数据
2. 按国家和AUKUS前后分组计算Y均值
3. 计算欧氏距离
4. 比较与原JSON数据

"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

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

    return df


def compute_country_period_means(df: pd.DataFrame) -> Dict:
    """
    计算每个国家在AUKUS前后的Y向量均值

    Returns:
        {country: {'pre': mean_Y, 'post': mean_Y, 'pre_count': n, 'post_count': n}}
    """
    results = {}

    for country in ['US', 'UK', 'AU']:
        country_df = df[df['country'] == country]

        pre_mask = country_df['post_aukus'] == 0
        post_mask = country_df['post_aukus'] == 1

        Y_pre = np.vstack(country_df.loc[pre_mask, 'Y_vector_global'].values)
        Y_post = np.vstack(country_df.loc[post_mask, 'Y_vector_global'].values)

        results[country] = {
            'pre': Y_pre.mean(axis=0),
            'post': Y_post.mean(axis=0),
            'pre_count': len(Y_pre),
            'post_count': len(Y_post)
        }

        logger.info(f"{country}: pre={len(Y_pre)}, post={len(Y_post)}")

    return results


def compute_pairwise_distances(country_means: Dict) -> Dict:
    """
    计算AUKUS前后国家对之间的欧氏距离

    Returns:
        {pair: {'pre': dist, 'post': dist, 'change_percent': pct}}
    """
    pairs = [('US', 'UK'), ('US', 'AU'), ('UK', 'AU')]
    results = {}

    for c1, c2 in pairs:
        pair_name = f"{c1}_{c2}"

        # AUKUS前距离
        pre_dist = np.linalg.norm(country_means[c1]['pre'] - country_means[c2]['pre'])

        # AUKUS后距离
        post_dist = np.linalg.norm(country_means[c1]['post'] - country_means[c2]['post'])

        # 变化百分比
        change_pct = (post_dist - pre_dist) / pre_dist * 100

        results[pair_name] = {
            'pre': pre_dist,
            'post': post_dist,
            'change_percent': change_pct
        }

    return results


def compute_average_distance(pairwise_distances: Dict) -> Dict:
    """
    计算平均距离
    """
    pre_distances = [v['pre'] for v in pairwise_distances.values()]
    post_distances = [v['post'] for v in pairwise_distances.values()]

    pre_avg = np.mean(pre_distances)
    post_avg = np.mean(post_distances)
    change_pct = (post_avg - pre_avg) / pre_avg * 100

    return {
        'pre_aukus_avg_distance': pre_avg,
        'post_aukus_avg_distance': post_avg,
        'percent_change': change_pct
    }


def verify_against_json(computed: Dict, pairwise: Dict) -> Dict:
    """
    验证计算结果与原JSON数据是否一致
    """
    original_json_path = RESULTS_DIR / 'global_A_regression_results.json'

    if not original_json_path.exists():
        return {'error': f'原JSON文件不存在: {original_json_path}'}

    with open(original_json_path, 'r') as f:
        original = json.load(f)

    h3 = original['hypothesis_tests']['H3_convergence']

    verification = {
        'matches': [],
        'mismatches': []
    }

    # 验证平均距离
    checks = [
        ('pre_aukus_avg_distance', computed['pre_aukus_avg_distance'], h3['pre_aukus_avg_distance']),
        ('post_aukus_avg_distance', computed['post_aukus_avg_distance'], h3['post_aukus_avg_distance']),
        ('percent_change', computed['percent_change'], h3['percent_change'])
    ]

    for name, calc, orig in checks:
        diff = abs(calc - orig)
        rel_diff = diff / abs(orig) if orig != 0 else diff
        match = rel_diff < 1e-10  # 相对误差小于1e-10

        entry = {
            'name': name,
            'computed': calc,
            'original': orig,
            'diff': diff,
            'rel_diff': rel_diff,
            'match': match
        }

        if match:
            verification['matches'].append(entry)
        else:
            verification['mismatches'].append(entry)

    # 验证国家对距离
    for pair_name, pair_data in pairwise.items():
        orig_pair = h3['country_pair_distances'].get(pair_name, {})

        for metric in ['pre', 'post', 'change_percent']:
            calc = pair_data.get(metric)
            orig = orig_pair.get(metric) if metric != 'change_percent' else orig_pair.get('change_percent')

            if calc is not None and orig is not None:
                diff = abs(calc - orig)
                rel_diff = diff / abs(orig) if orig != 0 else diff
                match = rel_diff < 1e-10

                entry = {
                    'name': f'{pair_name}_{metric}',
                    'computed': calc,
                    'original': orig,
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
    logger.info("DID H3 距离变化验证")
    logger.info("="*60)

    # 1. 加载数据
    logger.info("\n--- Step 1: 加载数据 ---")
    df = load_data()

    # 打印数据概览
    print(f"\n数据概览:")
    print(f"  总样本量: {len(df):,}")
    print(f"  AUKUS前: {(df['post_aukus'] == 0).sum():,}")
    print(f"  AUKUS后: {(df['post_aukus'] == 1).sum():,}")

    # 2. 计算国家-时期均值
    logger.info("\n--- Step 2: 计算国家-时期Y均值 ---")
    country_means = compute_country_period_means(df)

    # 3. 计算国家对距离
    logger.info("\n--- Step 3: 计算国家对欧氏距离 ---")
    pairwise_distances = compute_pairwise_distances(country_means)

    # 4. 计算平均距离
    logger.info("\n--- Step 4: 计算平均距离 ---")
    avg_distances = compute_average_distance(pairwise_distances)

    # 5. 打印结果
    print("\n" + "="*70)
    print("H3 距离变化结果")
    print("="*70)

    print(f"\n平均距离:")
    print(f"  AUKUS前: {avg_distances['pre_aukus_avg_distance']:.6f}")
    print(f"  AUKUS后: {avg_distances['post_aukus_avg_distance']:.6f}")
    print(f"  变化: {avg_distances['percent_change']:+.2f}%")

    print(f"\n国家对距离:")
    print(f"{'国家对':<10} {'AUKUS前':<12} {'AUKUS后':<12} {'变化':<10}")
    print("-"*50)
    for pair, data in pairwise_distances.items():
        print(f"{pair:<10} {data['pre']:.6f}     {data['post']:.6f}     {data['change_percent']:+.2f}%")

    # 6. 验证与原JSON对比
    logger.info("\n--- Step 5: 与原JSON对比 ---")
    verification = verify_against_json(avg_distances, pairwise_distances)

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
                print(f"  {item['name']}: 原始={item['original']:.6f}, 计算={item['computed']:.6f}, 差异={item['diff']:.2e} ✓")

        if verification['mismatches']:
            print("\n不匹配项:")
            for item in verification['mismatches']:
                print(f"  {item['name']}: 原始={item['original']:.6f}, 计算={item['computed']:.6f}, 差异={item['diff']:.2e} ✗")

    # 7. 结论
    print("\n" + "="*70)
    conclusion = "divergence" if avg_distances['percent_change'] > 0 else "convergence"
    print(f"结论: {conclusion} (AUKUS后距离{'增加' if conclusion == 'divergence' else '减少'})")
    print("="*70)

    # 8. 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'did_h3_verification.json'

    # Convert numpy bool to Python bool for JSON serialization
    verification_serializable = {
        'matches': [{k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in item.items()} for item in verification.get('matches', [])],
        'mismatches': [{k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in item.items()} for item in verification.get('mismatches', [])]
    }

    output_data = {
        'method': 'DID H3 Distance Verification',
        'description': '验证global_A_regression_results.json中的H3距离变化结果',
        'computed_results': {
            'average_distances': {k: float(v) for k, v in avg_distances.items()},
            'pairwise_distances': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pairwise_distances.items()}
        },
        'verification': verification_serializable,
        'conclusion': conclusion
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n验证结果已保存: {output_path}")

    return avg_distances, pairwise_distances, verification


if __name__ == '__main__':
    avg, pairwise, verification = main()
