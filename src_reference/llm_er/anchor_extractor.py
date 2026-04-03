"""
锚点词提取器
用于训练国家特定的转换矩阵A

方法论：
- 从语料中数据驱动地选择高频、非目标术语的词作为锚点词
- 从每个国家的语料中提取锚点词的所有出现
- 用(U, V)对训练该国的矩阵A
- 避免"用术语训练A，再用A处理术语"的循环依赖

HPC部署：
- 锚点词列表已预先计算并保存在 data/intermediate/anchor_words.json
- 使用 load_fixed_anchor_words() 函数直接加载，无需重新计算
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd

import sys
# Add src directory to path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils import logger, progress_bar


def load_fixed_anchor_words(
    anchor_words_path: str = None,
    top_n: int = 100
) -> List[str]:
    """
    从预计算的文件加载固定的锚点词列表

    用于HPC部署，避免每次运行都重新从语料发现锚点词

    Args:
        anchor_words_path: 锚点词文件路径（JSON或TXT格式）
        top_n: 使用前N个锚点词

    Returns:
        锚点词列表
    """
    if anchor_words_path is None:
        anchor_words_path = Path(__file__).parent.parent.parent / "data/intermediate/anchor_words.json"
    else:
        anchor_words_path = Path(anchor_words_path)

    if not anchor_words_path.exists():
        logger.warning(f"Anchor words file not found: {anchor_words_path}")
        logger.warning("Please run scripts/discover_and_save_anchor_words.py first")
        logger.warning("Falling back to predefined DEFAULT_ANCHOR_WORDS")
        return DEFAULT_ANCHOR_WORDS[:top_n]

    try:
        if anchor_words_path.suffix == '.json':
            with open(anchor_words_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                words = data.get('words', [])
        elif anchor_words_path.suffix == '.txt':
            with open(anchor_words_path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        else:
            # CSV格式
            import csv
            with open(anchor_words_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                words = [row['word'] for row in reader]

        words = words[:top_n]
        logger.info(f"Loaded {len(words)} fixed anchor words from {anchor_words_path}")
        return words

    except Exception as e:
        logger.error(f"Failed to load anchor words from {anchor_words_path}: {e}")
        logger.warning("Falling back to predefined DEFAULT_ANCHOR_WORDS")
        return DEFAULT_ANCHOR_WORDS[:top_n]


# 英语停用词（不适合作为锚点词）
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'also',
    'said', 'says', 'one', 'two', 'first', 'new', 'about', 'into', 'over',
    'after', 'before', 'between', 'through', 'during', 'without', 'again',
    'further', 'then', 'once', 'here', 'there', 'any', 'up', 'down', 'out',
    'off', 'under', 'above', 'below', 'being', 'having', 'doing', 'their',
    'them', 'his', 'her', 'him', 'my', 'your', 'our', 'us', 'me', 'if',
    'while', 'because', 'until', 'although', 'since', 'unless', 'however',
    'therefore', 'thus', 'hence', 'yet', 'still', 'even', 'already', 'always',
    'never', 'ever', 'often', 'sometimes', 'usually', 'perhaps', 'maybe',
    'probably', 'certainly', 'definitely', 'actually', 'really', 'quite',
    'rather', 'much', 'many', 'well', 'back', 'way', 'get', 'got', 'make',
    'made', 'take', 'took', 'come', 'came', 'go', 'went', 'see', 'saw',
    'know', 'knew', 'think', 'thought', 'look', 'looked', 'want', 'give',
    'gave', 'use', 'find', 'found', 'tell', 'told', 'ask', 'asked', 'work',
    'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin', 'show',
    'hear', 'play', 'run', 'move', 'like', 'live', 'believe', 'hold', 'bring',
    'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet',
    'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
    'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add',
    'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
    'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
    'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest',
    'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull',
}


def discover_anchor_words_from_corpus(
    articles_by_country: Dict[str, List[Dict]],
    target_terms: Set[str],
    min_word_length: int = 4,
    max_word_length: int = 20,
    min_total_frequency: int = 100,
    top_n: int = 100,
    content_field: str = 'content'
) -> List[str]:
    """
    从语料中数据驱动地发现锚点词

    选择标准：
    1. 高频：在语料中出现频率高
    2. 非目标术语：与 AI 研究目标无关
    3. 非停用词：排除功能词
    4. 合适长度：4-20 字符

    Args:
        articles_by_country: {country: [articles]} 原始文章
        target_terms: 需要排除的目标术语
        min_word_length: 最小词长
        max_word_length: 最大词长
        min_total_frequency: 最小总频率
        top_n: 返回前 N 个高频词

    Returns:
        锚点词列表（按频率排序）
    """
    logger.info("Discovering anchor words from corpus...")

    # 统计词频
    word_counter = Counter()
    total_articles = 0

    for country, articles in articles_by_country.items():
        for article in progress_bar(articles, desc=f"Counting words in {country}"):
            content = article.get(content_field, '')
            if not content:
                continue

            total_articles += 1

            # 提取单词（只保留字母组成的词）
            words = re.findall(r'\b[a-zA-Z]+\b', content.lower())

            for word in words:
                # 基本过滤
                if len(word) < min_word_length or len(word) > max_word_length:
                    continue
                if word in STOPWORDS:
                    continue

                word_counter[word] += 1

    logger.info(f"Processed {total_articles} articles, found {len(word_counter)} unique words")

    # 准备目标术语集合（小写）
    target_lower = set()
    for term in target_terms:
        # 添加完整术语
        target_lower.add(term.lower())
        # 也添加术语中的单词
        for word in term.lower().split():
            if len(word) >= min_word_length:
                target_lower.add(word)

    logger.info(f"Excluding {len(target_lower)} target terms/words")

    # 过滤并排序
    filtered_words = []
    for word, freq in word_counter.most_common():
        # 检查频率
        if freq < min_total_frequency:
            break  # 已按频率排序，后面的都不够频繁

        # 检查是否与目标术语重叠
        is_target = False
        for target in target_lower:
            if word == target or word in target or target in word:
                is_target = True
                break

        if is_target:
            continue

        filtered_words.append((word, freq))

        if len(filtered_words) >= top_n:
            break

    logger.info(f"Selected {len(filtered_words)} anchor words")

    # 打印前20个锚点词
    logger.info("Top 20 anchor words by frequency:")
    for i, (word, freq) in enumerate(filtered_words[:20]):
        logger.info(f"  {i+1}. {word}: {freq}")

    return [word for word, freq in filtered_words]


# 保留预定义列表作为备选
DEFAULT_ANCHOR_WORDS = [
    # 政府/军事通用词
    "government", "military", "defense", "security", "force", "forces",
    "army", "navy", "air", "marine", "command", "operations", "mission",
    "strategy", "policy", "program", "project", "budget", "funding",

    # 技术通用词（非AI特定）
    "technology", "system", "systems", "platform", "capability", "capabilities",
    "equipment", "weapon", "weapons", "vehicle", "aircraft", "ship",
    "communication", "information", "data", "network", "software", "hardware",

    # 国际关系
    "alliance", "partner", "partners", "cooperation", "agreement", "treaty",
    "threat", "enemy", "adversary", "competition", "conflict",

    # 通用动作/状态词
    "development", "research", "training", "deployment", "integration",
    "support", "service", "services", "acquisition", "procurement",

    # 时间/规模词
    "future", "modern", "advanced", "new", "current", "existing",
    "national", "international", "global", "regional", "joint",
]


class AnchorExtractor:
    """
    锚点词提取器

    从语料中提取锚点词的出现，用于训练国家特定的转换矩阵
    """

    def __init__(
        self,
        anchor_words: Optional[List[str]] = None,
        target_terms: Optional[Set[str]] = None,
        min_occurrences_per_country: int = 100,
        max_occurrences_per_word: int = 500,
        context_window: int = 200
    ):
        """
        初始化锚点词提取器

        Args:
            anchor_words: 锚点词列表（默认使用预定义列表）
            target_terms: 目标术语集合（需要排除）
            min_occurrences_per_country: 每个国家每个词的最小出现次数
            max_occurrences_per_word: 每个词的最大出现次数（避免过度采样）
            context_window: 上下文窗口大小（字符数）
        """
        self.anchor_words = anchor_words or DEFAULT_ANCHOR_WORDS
        self.target_terms = target_terms or set()
        self.min_occurrences = min_occurrences_per_country
        self.max_occurrences = max_occurrences_per_word
        self.context_window = context_window

        # 过滤掉与目标术语重叠的锚点词
        self._filter_anchor_words()

        logger.info(f"Initialized AnchorExtractor with {len(self.anchor_words)} anchor words")

    def _filter_anchor_words(self):
        """过滤掉与目标术语重叠的锚点词"""
        if not self.target_terms:
            return

        target_lower = {t.lower() for t in self.target_terms}

        filtered = []
        for word in self.anchor_words:
            word_lower = word.lower()
            # 检查是否与任何目标术语重叠
            is_overlap = False
            for target in target_lower:
                if word_lower in target or target in word_lower:
                    is_overlap = True
                    break
            if not is_overlap:
                filtered.append(word)

        removed = len(self.anchor_words) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered out {removed} anchor words that overlap with target terms")

        self.anchor_words = filtered

    def extract_from_articles(
        self,
        articles: List[Dict],
        country: str
    ) -> pd.DataFrame:
        """
        从文章列表中提取锚点词出现

        注意：不排除包含目标词的上下文。
        锚点词本身已确保不与目标词重叠，上下文中出现目标词不影响训练。

        Args:
            articles: 文章列表，每篇包含 'content', 'title', 'publish_date' 等
            country: 国家代码 (US/UK/AU)

        Returns:
            DataFrame，包含锚点词出现的信息
        """
        occurrences = []

        for article in progress_bar(articles, desc=f"Extracting anchors from {country}"):
            content = article.get('content', '')
            if not content:
                continue

            article_id = article.get('article_id', hash(content[:100]))

            # 对每个锚点词查找出现
            for anchor in self.anchor_words:
                # 使用词边界匹配
                pattern = r'\b' + re.escape(anchor) + r'\b'

                for match in re.finditer(pattern, content, re.IGNORECASE):
                    start = match.start()
                    end = match.end()

                    # 提取上下文
                    context_start = max(0, start - self.context_window)
                    context_end = min(len(content), end + self.context_window)
                    context = content[context_start:context_end]

                    # 计算在上下文中的位置
                    position_in_context = start - context_start

                    occurrences.append({
                        'anchor_word': anchor.lower(),
                        'matched_text': match.group(),
                        'article_id': article_id,
                        'country': country,
                        'text_block': context,
                        'start_char': position_in_context,
                        'year': article.get('year'),
                        'month': article.get('month'),
                    })

        df = pd.DataFrame(occurrences)
        logger.info(f"Extracted {len(df)} anchor occurrences from {country}")

        return df

    def balance_occurrences(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        平衡每个锚点词的出现次数

        Args:
            df: 锚点词出现DataFrame

        Returns:
            平衡后的DataFrame
        """
        balanced_dfs = []

        for anchor in df['anchor_word'].unique():
            anchor_df = df[df['anchor_word'] == anchor]

            if len(anchor_df) > self.max_occurrences:
                # 随机采样
                anchor_df = anchor_df.sample(n=self.max_occurrences, random_state=42)

            balanced_dfs.append(anchor_df)

        result = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Balanced to {len(result)} occurrences")

        return result

    def get_anchor_statistics(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        获取锚点词统计信息

        Args:
            df: 锚点词出现DataFrame

        Returns:
            统计信息字典
        """
        stats = {
            'total_occurrences': len(df),
            'unique_anchors': df['anchor_word'].nunique(),
            'by_country': df.groupby('country').size().to_dict(),
            'by_anchor': df.groupby('anchor_word').size().to_dict(),
        }

        return stats


def load_target_terms(terms_path: str = None) -> Set[str]:
    """
    加载目标术语列表

    Args:
        terms_path: 术语文件路径

    Returns:
        目标术语集合
    """
    if terms_path is None:
        terms_path = Path(__file__).parent.parent.parent / "data/intermediate/discovered_terms.csv"

    terms = set()
    try:
        df = pd.read_csv(terms_path)
        terms = set(df['term'].str.lower().tolist())
        logger.info(f"Loaded {len(terms)} target terms to exclude")
    except Exception as e:
        logger.warning(f"Failed to load target terms: {e}")

    return terms


def extract_anchors_by_country(
    articles_by_country: Dict[str, List[Dict]],
    target_terms: Optional[Set[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    按国家提取锚点词出现

    Args:
        articles_by_country: {country: [articles]} 映射
        target_terms: 要排除的目标术语

    Returns:
        {country: anchor_occurrences_df} 映射
    """
    if target_terms is None:
        target_terms = load_target_terms()

    extractor = AnchorExtractor(target_terms=target_terms)

    results = {}
    for country, articles in articles_by_country.items():
        df = extractor.extract_from_articles(articles, country)
        df = extractor.balance_occurrences(df)
        results[country] = df

    return results
