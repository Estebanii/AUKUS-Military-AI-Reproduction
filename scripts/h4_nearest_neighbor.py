"""
H3 语义近邻分析脚本（改进版）
=====================================

目的: 对各国军事人工智能概念化进行语义近邻分析（H3: 功能分化）
改进: (1) 按AUKUS前后分期分析 (2) 过滤GPT-2 BPE子词片段 (3) 大小写去重

方法:
1. 加载 Y_vector_global 数据，按 post_aukus 分为 pre/post 两期
2. 为每期每国计算平均概念向量
3. 加载 GPT-2 词嵌入矩阵，过滤BPE子词片段（仅保留Ġ前缀的完整词）
4. 计算余弦相似度，提取Top-30近邻词（按lowercase去重）
5. 分析三国共同词和独特词

论文数据: H3段落 (line 423-440), 图4, 结论H3段 (line 450-452)

"""

import sys
import os
import json
import logging
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 路径设置 (review文件夹内的相对路径)
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = PROJECT_ROOT / 'figures' / 'nearest_neighbors'
MODEL_DIR = PROJECT_ROOT / 'models' / 'gpt2'

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 固定随机种子
SEED = 42
np.random.seed(SEED)


# ============================================================
# GPT-2 词表加载与过滤
# ============================================================

def load_gpt2_embeddings():
    """
    加载 GPT-2 的词嵌入矩阵和词表

    Returns:
        embeddings: (50257, 768) 嵌入矩阵
        vocab: {token_string: token_id} 词表映射
        tokenizer: GPT-2 tokenizer 实例
    """
    # 优先从 review/models/gpt2/ 加载本地模型（无需网络）
    # 若本地模型不存在，回退到 HuggingFace 缓存或在线下载
    if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists():
        logger.info(f"Loading GPT-2 from local model directory: {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModel.from_pretrained(str(MODEL_DIR))
    else:
        logger.info("Local model directory not found, trying HuggingFace cache...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
            model = AutoModel.from_pretrained("gpt2", local_files_only=True)
            logger.info("Loaded GPT-2 from HuggingFace cache")
        except OSError:
            logger.info("Cache not found, downloading GPT-2...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModel.from_pretrained("gpt2")

    # 提取词嵌入矩阵 (wte = word token embedding)
    # AutoModel returns GPT2Model directly, wte is at model.wte (not model.transformer.wte)
    with torch.no_grad():
        embeddings = model.wte.weight.cpu().numpy().astype(np.float32)

    vocab = tokenizer.get_vocab()  # {token_string: token_id}
    logger.info(f"GPT-2 loaded: vocab_size={len(vocab)}, embedding_dim={embeddings.shape[1]}")

    return embeddings, vocab, tokenizer


def filter_vocabulary(vocab):
    """
    过滤 GPT-2 词表，仅保留有意义的完整英文词

    GPT-2 使用 Byte-Level BPE 编码：
    - Ġ (U+0120) 前缀标记词首位置（原文中前有空格的token）
    - 无 Ġ 前缀的 token 多为子词续接片段（如 "ipment", "istics"）
    - 少量无 Ġ 的 token 为句首词，但对近邻分析不影响

    过滤策略：
    1. 仅保留以 Ġ 开头的 token（词首完整词）
    2. 剥离 Ġ 前缀
    3. 仅保留纯字母 token（排除数字、标点、特殊字符）
    4. 排除单字符 token
    5. 按 lowercase 记录，用于后续去重

    Args:
        vocab: {token_string: token_id} 映射

    Returns:
        filtered: [(clean_word, token_id, lowercase_form), ...] 过滤后的token列表
        stats: 过滤统计信息
    """
    stats = {
        "total_vocab": len(vocab),
        "special_tokens_removed": 0,
        "non_G_prefix_removed": 0,
        "non_alpha_removed": 0,
        "single_char_removed": 0,
        "kept": 0,
    }

    filtered = []

    for token_str, token_id in vocab.items():
        # 1. 跳过特殊 token
        if token_str == '<|endoftext|>':
            stats["special_tokens_removed"] += 1
            continue

        # 2. 仅保留 Ġ 前缀的 token
        if not token_str.startswith('\u0120'):
            stats["non_G_prefix_removed"] += 1
            continue

        # 3. 剥离 Ġ 前缀
        clean_word = token_str[1:]  # 去掉 Ġ

        # 4. 仅保留纯字母 token
        if not clean_word.isalpha():
            stats["non_alpha_removed"] += 1
            continue

        # 5. 排除单字符
        if len(clean_word) <= 1:
            stats["single_char_removed"] += 1
            continue

        lowercase = clean_word.lower()
        filtered.append((clean_word, token_id, lowercase))
        stats["kept"] += 1

    logger.info(
        f"Vocabulary filter: {stats['total_vocab']} → {stats['kept']} tokens "
        f"(removed: {stats['non_G_prefix_removed']} non-Ġ, "
        f"{stats['non_alpha_removed']} non-alpha, "
        f"{stats['single_char_removed']} single-char)"
    )

    return filtered, stats


# ============================================================
# 数据加载与分期
# ============================================================

def load_data():
    """加载 parquet 数据"""
    data_path = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def compute_country_means(df, period_name):
    """
    计算每个国家的平均 Y_vector_global

    Args:
        df: 数据子集
        period_name: 时期名称（用于日志）

    Returns:
        means: {country: mean_vector (768-dim)}
        sizes: {country: n_samples}
    """
    means = {}
    sizes = {}

    for country in ['US', 'UK', 'AU']:
        mask = df['country'] == country
        country_df = df[mask]
        n = len(country_df)

        if n == 0:
            logger.warning(f"{period_name}: {country} has 0 samples, skipping")
            continue

        # 将 Y_vector_global 列堆叠为矩阵
        Y_matrix = np.stack(country_df['Y_vector_global'].values)
        mean_vec = Y_matrix.mean(axis=0)

        means[country] = mean_vec
        sizes[country] = n
        logger.info(f"  {period_name} {country}: n={n}, mean_norm={np.linalg.norm(mean_vec):.4f}")

    return means, sizes


# ============================================================
# 近邻计算
# ============================================================

def find_nearest_neighbors(country_mean, embeddings, filtered_tokens, top_k=30):
    """
    在过滤后的 GPT-2 词表中查找最近邻词

    Args:
        country_mean: (768,) 国家平均向量
        embeddings: (50257, 768) 完整嵌入矩阵
        filtered_tokens: [(clean_word, token_id, lowercase), ...] 过滤后的token列表
        top_k: 返回前 top_k 个（按 lowercase 去重后）

    Returns:
        list of {"word": str, "similarity": float, "raw_token": str}
    """
    # 提取过滤后 token 的嵌入
    token_ids = [t[1] for t in filtered_tokens]
    filtered_embeddings = embeddings[token_ids]  # (n_filtered, 768)

    # 计算余弦相似度
    sims = cosine_similarity(country_mean.reshape(1, -1), filtered_embeddings)[0]

    # 按相似度降序排序
    sorted_indices = np.argsort(-sims)

    # 去重：按 lowercase 保留最高相似度的变体
    results = []
    seen_lowercase = set()

    for idx in sorted_indices:
        clean_word, token_id, lowercase = filtered_tokens[idx]
        if lowercase in seen_lowercase:
            continue
        seen_lowercase.add(lowercase)
        results.append({
            "word": lowercase,  # 统一使用小写
            "similarity": round(float(sims[idx]), 6),
            "original_form": clean_word,
        })
        if len(results) >= top_k:
            break

    return results


# ============================================================
# 共同/独特词分析
# ============================================================

def analyze_common_unique(results_by_country, top_n=15):
    """
    分析三国的共同词和独特词

    Args:
        results_by_country: {country: [{"word": ..., "similarity": ...}, ...]}
        top_n: 取前 top_n 个词进行比较

    Returns:
        common_words: 三国共有的词列表
        unique_words: {country: [独特词列表]}
    """
    word_sets = {}
    for country, results in results_by_country.items():
        word_sets[country] = set(r["word"] for r in results[:top_n])

    all_countries = list(word_sets.keys())
    if len(all_countries) < 3:
        return [], {c: list(word_sets.get(c, set())) for c in ['US', 'UK', 'AU']}

    # 共同词
    common = word_sets[all_countries[0]]
    for c in all_countries[1:]:
        common = common & word_sets[c]

    # 独特词
    unique = {}
    for c in all_countries:
        others = set()
        for c2 in all_countries:
            if c2 != c:
                others = others | word_sets[c2]
        unique[c] = sorted(word_sets[c] - others)

    return sorted(common), unique


def compute_pre_post_changes(pre_results, post_results, top_n=15):
    """
    计算 pre→post 的词汇变化

    Returns:
        {country: {"entered": [...], "exited": [...]}}
    """
    changes = {}
    for country in ['US', 'UK', 'AU']:
        if country not in pre_results or country not in post_results:
            changes[country] = {"entered": [], "exited": []}
            continue

        pre_words = set(r["word"] for r in pre_results[country][:top_n])
        post_words = set(r["word"] for r in post_results[country][:top_n])

        changes[country] = {
            "entered": sorted(post_words - pre_words),
            "exited": sorted(pre_words - post_words),
        }

    return changes


# ============================================================
# 可视化
# ============================================================

def generate_figures(post_results, pre_results, figures_dir):
    """
    生成近邻词对比图

    Args:
        post_results: {country: [{"word": ..., "similarity": ...}, ...]}
        pre_results: {country: [...]}
        figures_dir: 图片保存目录
    """
    try:
        # 设置 matplotlib 缓存目录（避免 fontconfig 警告）
        os.environ.setdefault('MPLCONFIGDIR', str(figures_dir / '.mpl_cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        country_labels = {'US': 'United States', 'UK': 'United Kingdom', 'AU': 'Australia'}
        colors = {'US': '#1f77b4', 'UK': '#ff7f0e', 'AU': '#2ca02c'}

        for i, country in enumerate(['US', 'UK', 'AU']):
            ax = axes[i]
            if country not in post_results:
                continue

            words = [r["word"] for r in post_results[country][:15]]
            sims = [r["similarity"] for r in post_results[country][:15]]

            # 水平条形图（从下到上，最高相似度在顶部）
            y_pos = range(len(words))
            ax.barh(y_pos, sims, color=colors[country], alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel('Cosine Similarity', fontsize=11)
            ax.set_title(f'{country_labels[country]} (Post-AUKUS)', fontsize=13, fontweight='bold')
            ax.invert_yaxis()

            # 设置x轴范围
            if sims:
                ax.set_xlim(min(sims) * 0.9, max(sims) * 1.05)

        plt.suptitle('Nearest Semantic Neighbors of Military AI Conceptualization\n(Post-AUKUS, BPE-Filtered GPT-2 Vocabulary)',
                      fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存英文版
        fig_path_en = figures_dir / 'nearest_neighbors_post_aukus.png'
        plt.savefig(fig_path_en, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure: {fig_path_en}")

        plt.close()

        # 生成 pre vs post 对比图
        fig2, axes2 = plt.subplots(1, 3, figsize=(20, 12))

        for i, country in enumerate(['US', 'UK', 'AU']):
            ax = axes2[i]
            if country not in post_results or country not in pre_results:
                continue

            post_words = [r["word"] for r in post_results[country][:15]]
            post_sims = [r["similarity"] for r in post_results[country][:15]]
            pre_word_sim = {r["word"]: r["similarity"] for r in pre_results[country][:30]}

            y_pos = np.arange(len(post_words))
            bar_height = 0.35

            # Post-AUKUS bars
            ax.barh(y_pos - bar_height/2, post_sims, bar_height,
                    color=colors[country], alpha=0.9, label='Post-AUKUS')

            # Pre-AUKUS bars (for matching words)
            pre_sims = [pre_word_sim.get(w, 0) for w in post_words]
            ax.barh(y_pos + bar_height/2, pre_sims, bar_height,
                    color=colors[country], alpha=0.3, label='Pre-AUKUS')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(post_words, fontsize=9)
            ax.set_xlabel('Cosine Similarity', fontsize=11)
            ax.set_title(f'{country_labels[country]}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.invert_yaxis()

        plt.suptitle('Pre- vs Post-AUKUS Nearest Semantic Neighbors\n(BPE-Filtered GPT-2 Vocabulary)',
                      fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        fig_path_compare = figures_dir / 'nearest_neighbors_pre_post_comparison.png'
        plt.savefig(fig_path_compare, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure: {fig_path_compare}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("H3 语义近邻分析（改进版）")
    print("=" * 70)
    print(f"改进项: (1) 按AUKUS前后分期 (2) BPE子词过滤 (3) 大小写去重")
    print()

    # ---- Step 1: 加载 GPT-2 ----
    print("Step 1: Loading GPT-2 embeddings...")
    embeddings, vocab, tokenizer = load_gpt2_embeddings()

    # ---- Step 2: 过滤词表 ----
    print("\nStep 2: Filtering vocabulary...")
    filtered_tokens, filter_stats = filter_vocabulary(vocab)
    print(f"  Total: {filter_stats['total_vocab']} → Effective: {filter_stats['kept']} tokens")

    # ---- Step 3: 加载数据并分期 ----
    print("\nStep 3: Loading data and splitting by period...")
    df = load_data()

    # 确认 post_aukus 列的类型
    if df['post_aukus'].dtype == bool:
        pre_mask = ~df['post_aukus']
        post_mask = df['post_aukus']
    else:
        pre_mask = df['post_aukus'] == 0
        post_mask = df['post_aukus'] == 1

    df_pre = df[pre_mask]
    df_post = df[post_mask]

    print(f"  Pre-AUKUS: {len(df_pre)} samples")
    print(f"  Post-AUKUS: {len(df_post)} samples")
    print(f"  Total: {len(df)} samples")

    # 计算每期国别平均向量
    print("\n  Computing pre-AUKUS country means:")
    pre_means, pre_sizes = compute_country_means(df_pre, "pre-AUKUS")
    print("\n  Computing post-AUKUS country means:")
    post_means, post_sizes = compute_country_means(df_post, "post-AUKUS")
    print("\n  Computing full-period country means:")
    all_means, all_sizes = compute_country_means(df, "full-period")

    # ---- Step 4: 计算近邻词 ----
    print("\nStep 4: Computing nearest neighbors...")

    results = {}
    for period_name, means in [("pre_aukus", pre_means), ("post_aukus", post_means), ("full_period", all_means)]:
        results[period_name] = {}
        for country, mean_vec in means.items():
            neighbors = find_nearest_neighbors(mean_vec, embeddings, filtered_tokens, top_k=30)
            results[period_name][country] = neighbors
            top3 = ", ".join(f"{n['word']}({n['similarity']:.3f})" for n in neighbors[:3])
            print(f"  {period_name} {country}: Top-3 = {top3}")

    # ---- Step 5: 共同/独特词分析 ----
    print("\nStep 5: Analyzing common and unique words...")

    analyses = {}
    for period_name in ["pre_aukus", "post_aukus", "full_period"]:
        common, unique = analyze_common_unique(results[period_name], top_n=15)
        analyses[period_name] = {"common": common, "unique": unique}
        print(f"\n  {period_name}:")
        print(f"    Common (top-15): {common}")
        for country in ['US', 'UK', 'AU']:
            if country in unique:
                print(f"    {country} unique: {unique[country]}")

    # Pre→Post 变化分析
    pre_post_changes = compute_pre_post_changes(results["pre_aukus"], results["post_aukus"])
    print("\n  Pre→Post changes (top-15):")
    for country in ['US', 'UK', 'AU']:
        ch = pre_post_changes[country]
        print(f"    {country}: entered={ch['entered']}, exited={ch['exited']}")

    # ---- Step 6: 构建 JSON 输出 ----
    print("\nStep 6: Building output JSON...")

    output = {
        "analysis_type": "nearest_neighbor_words_improved",
        "description": "H3语义近邻分析（改进版）：按AUKUS前后分期 + BPE子词过滤 + 大小写去重",
        "llm_model": "GPT-2",
        "vocab_size": 50257,
        "embedding_dim": 768,
        "seed": SEED,
        "vocabulary_filter": {
            "method": "Keep only Ġ-prefixed (word-initial) tokens, strip prefix, alpha-only, length>1, lowercase dedup by max similarity",
            "total_vocab": filter_stats["total_vocab"],
            "special_tokens_removed": filter_stats["special_tokens_removed"],
            "non_G_prefix_removed": filter_stats["non_G_prefix_removed"],
            "non_alpha_removed": filter_stats["non_alpha_removed"],
            "single_char_removed": filter_stats["single_char_removed"],
            "effective_tokens_kept": filter_stats["kept"],
        },
        "sample_sizes": {
            "total": all_sizes,
            "pre_aukus": pre_sizes,
            "post_aukus": post_sizes,
        },
        "post_aukus_analysis": {
            "description": "Primary analysis: nearest neighbors using post-AUKUS country mean vectors",
            "nearest_words": results["post_aukus"],
            "common_words_top15": analyses["post_aukus"]["common"],
            "unique_words_top15": analyses["post_aukus"]["unique"],
        },
        "pre_aukus_analysis": {
            "description": "Comparison: nearest neighbors using pre-AUKUS country mean vectors",
            "nearest_words": results["pre_aukus"],
            "common_words_top15": analyses["pre_aukus"]["common"],
            "unique_words_top15": analyses["pre_aukus"]["unique"],
        },
        "full_period_analysis": {
            "description": "Full-period analysis (all time periods) for backward compatibility",
            "nearest_words": results["full_period"],
            "common_words_top15": analyses["full_period"]["common"],
            "unique_words_top15": analyses["full_period"]["unique"],
        },
        "pre_post_comparison": {
            "description": "Words that changed in/out of top-15 between pre and post AUKUS",
            **pre_post_changes,
        },
        "methodology_notes": [
            "Country mean vectors computed separately for each time period (pre-AUKUS, post-AUKUS, full)",
            "Vocabulary filtered to Ġ-prefixed (word-initial) GPT-2 tokens only, eliminating BPE subword fragments",
            "Only purely alphabetic tokens with length > 1 retained",
            "Case-folded to lowercase, keeping highest-similarity variant per word for deduplication",
            "Cosine similarity computed between country mean Y_vector_global and GPT-2 wte embeddings",
            "Top-30 nearest words reported per country per period; common/unique analysis uses top-15",
        ],
        "timestamp": datetime.now().isoformat(),
    }

    # 保存到 outputs
    output_path = OUTPUT_DIR / 'h4_nearest_neighbor_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {output_path}")

    # 覆盖旧的 results/nearest_neighbor_analysis.json
    results_path = RESULTS_DIR / 'nearest_neighbor_analysis.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    # ---- Step 7: 生成图表 ----
    print("\nStep 7: Generating figures...")
    generate_figures(results["post_aukus"], results["pre_aukus"], FIGURES_DIR)

    # ---- 最终摘要 ----
    print("\n" + "=" * 70)
    print("SUMMARY: Post-AUKUS Nearest Neighbor Analysis (BPE-filtered)")
    print("=" * 70)

    for country in ['US', 'UK', 'AU']:
        if country in results["post_aukus"]:
            n = post_sizes.get(country, 0)
            words = results["post_aukus"][country]
            print(f"\n{country} (n={n}):")
            print(f"  Top-10: {', '.join(w['word'] for w in words[:10])}")
            unique = analyses["post_aukus"]["unique"].get(country, [])
            if unique:
                print(f"  Unique (in top-15): {', '.join(unique)}")
            else:
                print(f"  Unique (in top-15): (none)")

    print(f"\nCommon words (all three, top-15): {', '.join(analyses['post_aukus']['common'])}")
    print(f"\nVocabulary: {filter_stats['total_vocab']} → {filter_stats['kept']} effective tokens")
    print("=" * 70)


if __name__ == "__main__":
    main()
