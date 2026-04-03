#!/usr/bin/env python3
"""
数据完整性统计检验：检测英美国防数据中的异常缺漏
Statistical Data Completeness Check for US/UK Defense Data

检验方法:
1. 月度观测数分布拟合 (Poisson / Negative Binomial)
2. 异常值检测 (Z-score, IQR)
3. 结构性断裂检测 (Chow-type test)
4. 零值月份分析
5. 年度均匀性检验 (Chi-square)
6. 年际增长率分析
7. 月度季节性模式检测
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载数据并按国家/年/月聚合"""
    df = pd.read_parquet(DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet')
    return df


def monthly_counts(df, country):
    """生成某国的月度观测数时间序列（含零值月份）"""
    sub = df[df['country'] == country].copy()
    year_min, year_max = sub['year'].min(), sub['year'].max()

    # 生成完整的年月网格
    all_ym = []
    for y in range(year_min, year_max + 1):
        for m in range(1, 13):
            all_ym.append((y, m))
    grid = pd.DataFrame(all_ym, columns=['year', 'month'])

    # 实际计数
    actual = sub.groupby(['year', 'month']).size().reset_index(name='count')
    merged = grid.merge(actual, on=['year', 'month'], how='left').fillna(0)
    merged['count'] = merged['count'].astype(int)

    # 唯一文章数
    art = sub.groupby(['year', 'month'])['article_id'].nunique().reset_index(name='n_articles')
    merged = merged.merge(art, on=['year', 'month'], how='left').fillna(0)
    merged['n_articles'] = merged['n_articles'].astype(int)

    return merged


def test_1_distribution_fit(counts, country):
    """测试1: 月度计数的分布拟合 (Poisson vs Negative Binomial)"""
    logger.info(f"  Test 1: Distribution fit for {country}")

    c = counts['count'].values
    # 排除首尾不完整年份的部分月份
    # 用所有非零或有意义的月份

    mean_c = np.mean(c)
    var_c = np.var(c, ddof=1)
    dispersion = var_c / mean_c if mean_c > 0 else float('inf')

    # Poisson GoF test
    # H0: data follows Poisson(lambda=mean)
    expected_poisson = stats.poisson.pmf(range(int(max(c)) + 1), mean_c) * len(c)

    # 简化：用卡方检验比较观测分布与Poisson期望
    # 将计数分箱（确保单调递增）
    max_val = int(max(c)) + 1
    raw_bins = [0, 1, 10, 50, 100, 200, 500, max_val]
    bins = sorted(set(b for b in raw_bins if b <= max_val))
    if bins[-1] < max_val:
        bins.append(max_val)
    observed_hist, _ = np.histogram(c, bins=bins)
    expected_hist = np.array([
        np.sum(stats.poisson.pmf(range(int(bins[i]), int(bins[i+1])), mean_c))
        for i in range(len(bins)-1)
    ]) * len(c)

    # 移除期望为0的箱
    mask = expected_hist > 0
    if np.sum(mask) > 1:
        chi2, p_poisson = stats.chisquare(observed_hist[mask], expected_hist[mask])
    else:
        chi2, p_poisson = float('nan'), float('nan')

    # Negative Binomial参数估计 (method of moments)
    if var_c > mean_c and mean_c > 0:
        p_nb = mean_c / var_c
        r_nb = mean_c * p_nb / (1 - p_nb)
        nb_fit = True
    else:
        p_nb, r_nb = 0.5, 1
        nb_fit = False

    return {
        'mean': float(mean_c),
        'variance': float(var_c),
        'dispersion_ratio': float(dispersion),
        'overdispersed': bool(dispersion > 1.5),
        'poisson_chi2': float(chi2),
        'poisson_p': float(p_poisson),
        'poisson_rejected': bool(p_poisson < 0.05) if not np.isnan(p_poisson) else None,
        'nb_fit_possible': nb_fit,
        'nb_r': float(r_nb) if nb_fit else None,
        'nb_p': float(p_nb) if nb_fit else None,
        'interpretation': 'Negative Binomial更适合（过度离散）' if dispersion > 1.5 else 'Poisson可接受'
    }


def test_2_outlier_detection(counts, country):
    """测试2: 异常值检测"""
    logger.info(f"  Test 2: Outlier detection for {country}")

    c = counts['count'].values

    # Z-score method
    z_scores = stats.zscore(c)
    z_outliers = counts[np.abs(z_scores) > 2.5].copy()
    z_outliers['z_score'] = z_scores[np.abs(z_scores) > 2.5]

    # IQR method
    q1, q3 = np.percentile(c, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers = counts[(c < lower) | (c > upper)].copy()

    # 零值月份
    zero_months = counts[counts['count'] == 0]

    # 极低值月份（<该国月均值的10%）
    threshold_low = np.mean(c) * 0.1
    low_months = counts[(counts['count'] > 0) & (counts['count'] < threshold_low)]

    z_outlier_list = []
    for _, row in z_outliers.iterrows():
        z_outlier_list.append({
            'year': int(row['year']), 'month': int(row['month']),
            'count': int(row['count']), 'z_score': float(row['z_score'])
        })

    zero_list = [{'year': int(r['year']), 'month': int(r['month'])} for _, r in zero_months.iterrows()]
    low_list = [{'year': int(r['year']), 'month': int(r['month']), 'count': int(r['count'])}
                for _, r in low_months.iterrows()]

    return {
        'iqr_bounds': {'lower': float(lower), 'upper': float(upper)},
        'n_z_outliers': len(z_outlier_list),
        'z_outliers': z_outlier_list,
        'n_zero_months': len(zero_list),
        'zero_months': zero_list,
        'n_low_months': len(low_list),
        'low_months': low_list,
        'zero_rate': float(len(zero_list) / len(counts)),
    }


def test_3_structural_breaks(counts, country):
    """测试3: 结构性断裂检测 (滑动Chow-type检验)"""
    logger.info(f"  Test 3: Structural break detection for {country}")

    c = counts['count'].values
    n = len(c)

    if n < 12:
        return {'error': 'Insufficient data for structural break test'}

    # 滑动F统计量 (Chow test at each point)
    min_window = max(6, n // 6)
    f_stats = []

    for split in range(min_window, n - min_window):
        y1 = c[:split]
        y2 = c[split:]

        # 比较两段的均值
        n1, n2 = len(y1), len(y2)
        mean1, mean2 = np.mean(y1), np.mean(y2)
        var1, var2 = np.var(y1, ddof=1), np.var(y2, ddof=1)

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(y1, y2, equal_var=False)

        ym = counts.iloc[split]
        f_stats.append({
            'split_idx': split,
            'year': int(ym['year']),
            'month': int(ym['month']),
            'mean_before': float(mean1),
            'mean_after': float(mean2),
            'ratio': float(mean2 / mean1) if mean1 > 0 else float('inf'),
            't_stat': float(t_stat),
            'p_value': float(p_val)
        })

    # 找到最显著的断裂点
    f_df = pd.DataFrame(f_stats)
    significant_breaks = f_df[f_df['p_value'] < 0.01].sort_values('p_value')

    # 找到最大的均值比变化点
    top_breaks = []
    if len(significant_breaks) > 0:
        # 选择不重叠的断裂点（至少间隔6个月）
        used = set()
        for _, row in significant_breaks.iterrows():
            idx = row['split_idx']
            if not any(abs(idx - u) < 6 for u in used):
                used.add(idx)
                top_breaks.append({
                    'year': int(row['year']),
                    'month': int(row['month']),
                    'mean_before': round(row['mean_before'], 1),
                    'mean_after': round(row['mean_after'], 1),
                    'ratio': round(row['ratio'], 2),
                    'p_value': float(row['p_value'])
                })
            if len(top_breaks) >= 5:
                break

    return {
        'n_candidate_splits': len(f_stats),
        'n_significant_breaks': len(significant_breaks),
        'top_breaks': top_breaks,
        'interpretation': f'发现{len(top_breaks)}个显著结构性断裂点' if top_breaks else '无显著结构性断裂'
    }


def test_4_zero_inflation(counts, country):
    """测试4: 零膨胀分析"""
    logger.info(f"  Test 4: Zero-inflation analysis for {country}")

    c = counts['count'].values
    n_zeros = np.sum(c == 0)
    n_total = len(c)

    mean_c = np.mean(c)
    # 期望零值比例 (under Poisson)
    expected_zero_rate = stats.poisson.pmf(0, mean_c) if mean_c > 0 else 1.0
    observed_zero_rate = n_zeros / n_total

    # 零值聚集检测：连续零值
    consecutive_zeros = []
    current_run = 0
    start_idx = None
    for i, val in enumerate(c):
        if val == 0:
            if current_run == 0:
                start_idx = i
            current_run += 1
        else:
            if current_run > 0:
                row_start = counts.iloc[start_idx]
                row_end = counts.iloc[start_idx + current_run - 1]
                consecutive_zeros.append({
                    'start': f"{int(row_start['year'])}-{int(row_start['month']):02d}",
                    'end': f"{int(row_end['year'])}-{int(row_end['month']):02d}",
                    'length': current_run
                })
            current_run = 0
    if current_run > 0:
        row_start = counts.iloc[start_idx]
        row_end = counts.iloc[start_idx + current_run - 1]
        consecutive_zeros.append({
            'start': f"{int(row_start['year'])}-{int(row_start['month']):02d}",
            'end': f"{int(row_end['year'])}-{int(row_end['month']):02d}",
            'length': current_run
        })

    # 只保留长度>=2的连续零值段
    long_gaps = [z for z in consecutive_zeros if z['length'] >= 2]

    # Vuong test approximation: compare zero proportion
    if mean_c > 0:
        z_vuong = (observed_zero_rate - expected_zero_rate) / np.sqrt(
            expected_zero_rate * (1 - expected_zero_rate) / n_total
        )
    else:
        z_vuong = 0

    return {
        'n_zeros': int(n_zeros),
        'n_total': int(n_total),
        'observed_zero_rate': float(observed_zero_rate),
        'expected_zero_rate_poisson': float(expected_zero_rate),
        'zero_excess_ratio': float(observed_zero_rate / expected_zero_rate) if expected_zero_rate > 0 else float('inf'),
        'vuong_z': float(z_vuong),
        'zero_inflated': bool(z_vuong > 1.96),
        'consecutive_zero_gaps': long_gaps,
        'max_gap_months': max([z['length'] for z in consecutive_zeros], default=0),
    }


def test_5_yearly_uniformity(counts, country):
    """测试5: 各年内月度均匀性检验"""
    logger.info(f"  Test 5: Yearly uniformity test for {country}")

    results = []
    for year in sorted(counts['year'].unique()):
        year_data = counts[counts['year'] == year]
        monthly = year_data['count'].values

        # 跳过全为零的年份
        if np.sum(monthly) == 0:
            continue

        # 排除部分年份（首年和末年可能不完整）
        n_nonzero = np.sum(monthly > 0)
        total = np.sum(monthly)

        if n_nonzero < 6:  # 少于6个月有数据
            results.append({
                'year': int(year),
                'n_months_with_data': int(n_nonzero),
                'total_obs': int(total),
                'chi2': None,
                'p_value': None,
                'uniform': None,
                'note': 'Insufficient months (< 6) for uniformity test',
                'cv': float(np.std(monthly[monthly > 0]) / np.mean(monthly[monthly > 0])) if n_nonzero > 0 else None
            })
            continue

        # 只对有数据的月份做均匀性检验
        nonzero_months = monthly[monthly > 0]
        expected = np.full(len(nonzero_months), np.mean(nonzero_months))
        chi2, p = stats.chisquare(nonzero_months, expected)
        cv = float(np.std(nonzero_months) / np.mean(nonzero_months))

        results.append({
            'year': int(year),
            'n_months_with_data': int(n_nonzero),
            'total_obs': int(total),
            'chi2': float(chi2),
            'p_value': float(p),
            'uniform': bool(p > 0.05),
            'cv': cv,
            'note': None
        })

    n_tested = sum(1 for r in results if r['chi2'] is not None)
    n_uniform = sum(1 for r in results if r.get('uniform') == True)
    n_nonuniform = sum(1 for r in results if r.get('uniform') == False)

    return {
        'yearly_results': results,
        'n_years_tested': n_tested,
        'n_uniform': n_uniform,
        'n_nonuniform': n_nonuniform,
        'uniformity_rate': float(n_uniform / n_tested) if n_tested > 0 else None
    }


def test_6_yoy_growth(counts, country):
    """测试6: 年际增长率分析"""
    logger.info(f"  Test 6: Year-over-year growth analysis for {country}")

    yearly = counts.groupby('year')['count'].sum().reset_index()
    yearly.columns = ['year', 'annual_count']

    # 文章数
    # 重新从原始数据计算
    yearly['growth_rate'] = yearly['annual_count'].pct_change()
    yearly['log_count'] = np.log1p(yearly['annual_count'])

    # 检测异常增长/下降
    growth_rates = yearly['growth_rate'].dropna().values
    if len(growth_rates) > 2:
        mean_growth = np.mean(growth_rates)
        std_growth = np.std(growth_rates, ddof=1)

        anomalies = []
        for _, row in yearly.iterrows():
            if pd.notna(row['growth_rate']):
                z = (row['growth_rate'] - mean_growth) / std_growth if std_growth > 0 else 0
                if abs(z) > 2:
                    anomalies.append({
                        'year': int(row['year']),
                        'count': int(row['annual_count']),
                        'growth_rate': float(row['growth_rate']),
                        'z_score': float(z)
                    })
    else:
        mean_growth, std_growth = 0, 0
        anomalies = []

    # 趋势线拟合
    years = yearly['year'].values.astype(float)
    log_counts = yearly['log_count'].values
    if len(years) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_counts)
        trend = {
            'slope': float(slope),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'interpretation': f'年均增长率约{(np.exp(slope)-1)*100:.1f}%'
        }
    else:
        trend = None

    yearly_list = []
    for _, row in yearly.iterrows():
        yearly_list.append({
            'year': int(row['year']),
            'count': int(row['annual_count']),
            'growth_rate': float(row['growth_rate']) if pd.notna(row['growth_rate']) else None
        })

    return {
        'yearly_counts': yearly_list,
        'mean_growth_rate': float(mean_growth),
        'std_growth_rate': float(std_growth),
        'growth_anomalies': anomalies,
        'trend': trend
    }


def test_7_seasonal_pattern(counts, country):
    """测试7: 月度季节性模式检测"""
    logger.info(f"  Test 7: Seasonal pattern detection for {country}")

    # 只使用有充分数据的年份
    year_totals = counts.groupby('year')['count'].sum()
    valid_years = year_totals[year_totals > 50].index.tolist()

    valid_data = counts[counts['year'].isin(valid_years)]

    # 各月平均
    monthly_avg = valid_data.groupby('month')['count'].mean()
    monthly_std = valid_data.groupby('month')['count'].std()

    # Kruskal-Wallis检验（月份间是否有显著差异）
    groups = [valid_data[valid_data['month'] == m]['count'].values for m in range(1, 13)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h_stat, p_kw = stats.kruskal(*groups)
    else:
        h_stat, p_kw = float('nan'), float('nan')

    monthly_profile = []
    for m in range(1, 13):
        monthly_profile.append({
            'month': m,
            'mean': float(monthly_avg.get(m, 0)),
            'std': float(monthly_std.get(m, 0)) if m in monthly_std.index else 0
        })

    return {
        'valid_years': valid_years,
        'n_valid_years': len(valid_years),
        'monthly_profile': monthly_profile,
        'kruskal_wallis_h': float(h_stat),
        'kruskal_wallis_p': float(p_kw),
        'seasonal_effect_significant': bool(p_kw < 0.05) if not np.isnan(p_kw) else None,
        'interpretation': '存在显著的月度季节性模式' if (not np.isnan(p_kw) and p_kw < 0.05) else '无显著月度季节性差异'
    }


def test_8_runs_test(counts, country):
    """测试8: 游程检验 (Runs test) — 检测数据是否存在非随机的聚集模式"""
    logger.info(f"  Test 8: Runs test for {country}")

    c = counts['count'].values
    median_c = np.median(c)

    # 转为二值序列：高于/低于中位数
    binary = (c > median_c).astype(int)

    # 计算游程数
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1

    n1 = np.sum(binary == 1)
    n0 = np.sum(binary == 0)
    n = n1 + n0

    if n1 > 0 and n0 > 0:
        # 期望游程数和标准差
        expected_runs = 1 + (2 * n1 * n0) / n
        std_runs = np.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1)))

        if std_runs > 0:
            z_runs = (runs - expected_runs) / std_runs
            p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
        else:
            z_runs, p_runs = 0, 1
    else:
        expected_runs, z_runs, p_runs = runs, 0, 1

    return {
        'n_runs': int(runs),
        'expected_runs': float(expected_runs),
        'z_statistic': float(z_runs),
        'p_value': float(p_runs),
        'random': bool(p_runs > 0.05),
        'interpretation': '数据序列随机（无明显聚集）' if p_runs > 0.05 else '数据存在非随机聚集模式（可能提示系统性缺漏）'
    }


def comprehensive_assessment(us_results, uk_results):
    """综合评估"""
    issues = []

    # UK早期数据
    uk_zeros = uk_results['test_4_zero_inflation']
    if uk_zeros['max_gap_months'] > 3:
        issues.append({
            'severity': 'HIGH',
            'country': 'UK',
            'issue': f"存在最长{uk_zeros['max_gap_months']}个月的连续零值缺口",
            'detail': uk_zeros['consecutive_zero_gaps'],
            'impact': '早期数据稀疏，但主要分析集中在2017+，影响有限'
        })

    # UK 2018 Apr-May零值
    uk_outliers = uk_results['test_2_outlier_detection']
    for z in uk_outliers['zero_months']:
        if z['year'] >= 2014:  # 只关注分析期内的零值
            issues.append({
                'severity': 'MEDIUM',
                'country': 'UK',
                'issue': f"{z['year']}-{z['month']:02d} 观测数为零",
                'impact': '单月缺失对年度聚合影响有限'
            })

    # US异常低值
    us_outliers = us_results['test_2_outlier_detection']
    for low in us_outliers['low_months']:
        issues.append({
            'severity': 'LOW',
            'country': 'US',
            'issue': f"{low['year']}-{low['month']:02d} 观测数异常低 ({low['count']})",
            'impact': '单月低值不影响年度分析'
        })

    # 结构性断裂
    for country, results in [('US', us_results), ('UK', uk_results)]:
        breaks = results['test_3_structural_breaks']
        for b in breaks.get('top_breaks', []):
            if b['ratio'] > 3 or b['ratio'] < 0.33:
                issues.append({
                    'severity': 'MEDIUM',
                    'country': country,
                    'issue': f"在{b['year']}-{b['month']:02d}附近存在结构性断裂 (均值比={b['ratio']})",
                    'impact': '可能反映数据源变化或采集方式调整'
                })

    # 总体评估
    high_issues = [i for i in issues if i['severity'] == 'HIGH']
    medium_issues = [i for i in issues if i['severity'] == 'MEDIUM']

    if len(high_issues) > 0:
        overall = '存在需要关注的数据缺漏问题'
    elif len(medium_issues) > 2:
        overall = '存在一些数据不均匀性，但不太可能严重影响结论'
    else:
        overall = '数据总体完整，未发现影响结论的系统性缺漏'

    return {
        'issues': issues,
        'n_high': len(high_issues),
        'n_medium': len(medium_issues),
        'n_low': len([i for i in issues if i['severity'] == 'LOW']),
        'overall_assessment': overall
    }


def main():
    print("=" * 70)
    print("英美国防数据完整性统计检验")
    print("=" * 70)

    df = load_data()

    all_results = {}

    for country in ['US', 'UK']:
        print(f"\n{'=' * 50}")
        print(f"  {country} 数据分析")
        print(f"{'=' * 50}")

        counts = monthly_counts(df, country)

        # 排除明确不完整的首月和末月
        sub = df[df['country'] == country]
        print(f"  年份范围: {sub['year'].min()}-{sub['year'].max()}")
        print(f"  总观测数: {len(sub)}")
        print(f"  月度数据点: {len(counts)}")

        results = {}
        results['data_range'] = {
            'year_min': int(sub['year'].min()),
            'year_max': int(sub['year'].max()),
            'n_obs': int(len(sub)),
            'n_months': int(len(counts))
        }

        results['test_1_distribution'] = test_1_distribution_fit(counts, country)
        results['test_2_outlier_detection'] = test_2_outlier_detection(counts, country)
        results['test_3_structural_breaks'] = test_3_structural_breaks(counts, country)
        results['test_4_zero_inflation'] = test_4_zero_inflation(counts, country)
        results['test_5_yearly_uniformity'] = test_5_yearly_uniformity(counts, country)
        results['test_6_yoy_growth'] = test_6_yoy_growth(counts, country)
        results['test_7_seasonal_pattern'] = test_7_seasonal_pattern(counts, country)
        results['test_8_runs_test'] = test_8_runs_test(counts, country)

        all_results[country] = results

        # 打印关键发现
        print(f"\n  --- 关键发现 ---")

        t1 = results['test_1_distribution']
        print(f"  分布: mean={t1['mean']:.1f}, var={t1['variance']:.1f}, 离散比={t1['dispersion_ratio']:.2f}")
        print(f"    → {t1['interpretation']}")

        t2 = results['test_2_outlier_detection']
        print(f"  零值月份: {t2['n_zero_months']}个, 低值月份: {t2['n_low_months']}个")
        if t2['zero_months']:
            zero_str = ', '.join([f"{z['year']}-{z['month']:02d}" for z in t2['zero_months'][:10]])
            print(f"    零值月: {zero_str}")

        t3 = results['test_3_structural_breaks']
        if t3.get('top_breaks'):
            for b in t3['top_breaks'][:3]:
                print(f"  断裂点: {b['year']}-{b['month']:02d} (前均值={b['mean_before']}, 后均值={b['mean_after']}, p={b['p_value']:.4f})")

        t4 = results['test_4_zero_inflation']
        print(f"  零膨胀: 观测零率={t4['observed_zero_rate']:.3f}, 期望零率={t4['expected_zero_rate_poisson']:.6f}")
        if t4['consecutive_zero_gaps']:
            for gap in t4['consecutive_zero_gaps']:
                print(f"    连续零值: {gap['start']} → {gap['end']} ({gap['length']}个月)")

        t7 = results['test_7_seasonal_pattern']
        print(f"  季节性: Kruskal-Wallis p={t7['kruskal_wallis_p']:.4f} → {t7['interpretation']}")

        t8 = results['test_8_runs_test']
        print(f"  游程检验: p={t8['p_value']:.4f} → {t8['interpretation']}")

    # 综合评估
    assessment = comprehensive_assessment(all_results['US'], all_results['UK'])
    all_results['comprehensive_assessment'] = assessment

    print(f"\n{'=' * 70}")
    print("综合评估")
    print(f"{'=' * 70}")

    for issue in assessment['issues']:
        print(f"  [{issue['severity']}] {issue['country']}: {issue['issue']}")
        print(f"       影响: {issue['impact']}")

    print(f"\n  总体结论: {assessment['overall_assessment']}")
    print(f"  问题数: HIGH={assessment['n_high']}, MEDIUM={assessment['n_medium']}, LOW={assessment['n_low']}")

    # 保存结果
    output_path = OUTPUT_DIR / 'data_completeness_check.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  结果已保存: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
