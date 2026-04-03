"""
Modal GPU 复现脚本：DeBERTa 嵌入生成管线
==========================================

在 Modal 云 GPU 上完整复现从原始术语出现数据到最终语义向量的全过程。

使用方法：
    1. 安装 Modal: pip install modal
    2. 认证: modal setup  (首次使用)
    3. 上传数据并运行:
       python final_fuxian/scripts/modal_reproduce_deberta.py

管线步骤（全部在 GPU 上执行）：
    Step 1: DeBERTa 编码目标术语出现 (37,866) → U_target (37866 × 768)
    Step 2: DeBERTa 编码锚点词出现 (~49,999) → U_anchors (49999 × 768)
    Step 3: GPT-2 嵌入查找锚点词 → V_dict {100 words: (768,)}
    Step 4: 训练全局 A 矩阵（岭回归 + Procrustes 先验）→ A (768 × 768)
    Step 5: 应用 A 矩阵 → Y_global (37866 × 768)
    Step 6: 保存并验证最终 parquet

"""

import modal
import os
import sys
from pathlib import Path

# ============================================================
# Modal App 定义
# ============================================================

app = modal.App("llm-er-deberta-reproduction")

# 定义 GPU 容器镜像
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "tqdm==4.67.1",
        "tiktoken",
        "sentencepiece",
    )
)

# 创建 Modal Volume 用于数据持久化
volume = modal.Volume.from_name("llm-er-data", create_if_missing=True)
VOLUME_PATH = "/data"


# ============================================================
# 自包含的核心模块（不依赖外部 utils）
# ============================================================

PIPELINE_CODE = '''
"""自包含的 DeBERTa 管线代码"""

import re
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy import linalg
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deberta_pipeline")


# ---- ConceptOccurrenceDataset ----

class ConceptOccurrenceDataset(Dataset):
    """概念出现数据集：将术语替换为 [MASK]，返回 tokenized 输入"""

    def __init__(self, texts, concept_terms, start_positions, tokenizer,
                 max_length=512, pos_types=None):
        self.texts = texts
        self.concept_terms = concept_terms
        self.start_positions = start_positions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pos_types = pos_types if pos_types else ["noun"] * len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        term = self.concept_terms[idx]
        start_pos = self.start_positions[idx]
        pos_type = self.pos_types[idx]

        masked_text = self._create_masked_text(text, term, start_pos, pos_type)

        encoding = self.tokenizer(
            masked_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        mask_token_id = self.tokenizer.mask_token_id
        input_ids = encoding["input_ids"].squeeze()
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        mask_pos = mask_positions[0].item() if len(mask_positions) > 0 else 1

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "mask_position": mask_pos,
            "idx": idx,
        }

    def _create_masked_text(self, text, term, start_pos, pos_type="noun"):
        term_lower = term.lower()
        text_lower = text.lower()

        if pos_type == "adjective":
            return self._mask_adjective_with_noun(text, term, start_pos)

        if 0 <= start_pos < len(text):
            end_pos = start_pos + len(term)
            if text_lower[start_pos:end_pos] == term_lower:
                return text[:start_pos] + "[MASK]" + text[end_pos:]

        pattern = re.compile(r"\\b" + re.escape(term) + r"\\b", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return text[: match.start()] + "[MASK]" + text[match.end() :]
        return text

    def _mask_adjective_with_noun(self, text, adjective, start_pos):
        adj_lower = adjective.lower()
        text_lower = text.lower()
        adj_start = adj_end = -1

        if 0 <= start_pos < len(text):
            end_pos = start_pos + len(adjective)
            if text_lower[start_pos:end_pos] == adj_lower:
                adj_start, adj_end = start_pos, end_pos

        if adj_start < 0:
            pattern = re.compile(r"\\b" + re.escape(adjective) + r"\\b", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                adj_start, adj_end = match.start(), match.end()

        if adj_start < 0:
            return text

        remaining = text[adj_end:]
        noun_match = re.match(r"^(\\s+)(\\w+)", remaining)
        if noun_match:
            mask_end = adj_end + len(noun_match.group(1)) + len(noun_match.group(2))
            return text[:adj_start] + "[MASK]" + text[mask_end:]
        return text[:adj_start] + "[MASK]" + text[adj_end:]


# ---- DeBERTa Encoder ----

class DeBERTaEncoder:
    # Model version pinned for reproducibility
    # microsoft/deberta-v3-base: HuggingFace model card revision
    # To verify: transformers.AutoModel.from_pretrained("microsoft/deberta-v3-base")
    DEFAULT_MODEL = "microsoft/deberta-v3-base"

    def __init__(self, model_name=DEFAULT_MODEL, device=None,
                 batch_size=32, max_length=512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading DeBERTa: {self.model_name} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info(f"Hidden size: {self._model.config.hidden_size}")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def encode_batch(self, df, text_column="text_block", term_column="matched_term",
                     position_column="start_char", pos_type_column="pos_type",
                     show_progress=True):
        texts = df[text_column].fillna("").tolist()
        terms = df[term_column].fillna("").tolist()
        positions = df[position_column].fillna(-1).astype(int).tolist()
        pos_types = df[pos_type_column].fillna("noun").tolist() if pos_type_column in df.columns else None

        dataset = ConceptOccurrenceDataset(
            texts, terms, positions, self.tokenizer,
            self.max_length, pos_types
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        hidden_size = self.model.config.hidden_size
        all_embeddings = np.zeros((len(dataset), hidden_size), dtype=np.float32)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DeBERTa encoding", disable=not show_progress):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                mask_positions = batch["mask_position"]
                indices = batch["idx"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state

                for i in range(len(indices)):
                    pos = mask_positions[i].item()
                    emb = hidden_states[i, pos, :].cpu().numpy()
                    all_embeddings[indices[i].item()] = emb

        logger.info(f"Encoded {len(all_embeddings)} samples, shape: {all_embeddings.shape}")
        return all_embeddings


# ---- LLM Embedder (GPT-2) ----

class LLMEmbedder:
    # GPT-2 (124M params, 768-dim embeddings, 50257 vocab)
    # Used only for word embedding layer (wte) as semantic dictionary
    DEFAULT_MODEL = "gpt2"

    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        self.model_name = model_name
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._tokenizer = None
        self._embedding_layer = None

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading LLM: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            self._embedding_layer = model.transformer.wte
        elif hasattr(model, "get_input_embeddings"):
            self._embedding_layer = model.get_input_embeddings()
        else:
            raise ValueError("Cannot find embedding layer")
        self._embedding_layer.to(self.device)
        self._embedding_layer.eval()
        del model
        logger.info(f"Embedding dim: {self._embedding_layer.weight.shape[1]}")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def embedding_layer(self):
        if self._embedding_layer is None:
            self._load_model()
        return self._embedding_layer

    def get_phrase_embedding(self, phrase, aggregation="mean"):
        token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
        if not token_ids:
            return np.zeros(self.embedding_layer.weight.shape[1])
        with torch.no_grad():
            token_tensor = torch.tensor(token_ids, device=self.device)
            embeddings = self.embedding_layer(token_tensor)
            if aggregation == "mean":
                result = embeddings.mean(dim=0)
            else:
                result = embeddings[0]
        return result.cpu().numpy().astype(np.float32)


# ---- Matrix Trainer ----

class MatrixTrainer:
    def __init__(self, regularization_lambda=0.1, use_procrustes_prior=True, whiten=True):
        self.lambda_ = regularization_lambda
        self.use_procrustes_prior = use_procrustes_prior
        self.whiten = whiten
        self.A = self.A0 = self.U_mean = self.V_mean = None

    def compute_procrustes_prior(self, U, V):
        U_c = U - U.mean(axis=0)
        V_c = V - V.mean(axis=0)
        M = V_c.T @ U_c
        W, _, Zt = linalg.svd(M, full_matrices=False)
        return W @ Zt

    def compute_whitening_transform(self, X, eps=1e-6):
        mean = X.mean(axis=0)
        X_c = X - mean
        cov = X_c.T @ X_c / (X.shape[0] - 1)
        eigenvalues, eigenvectors = linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, eps, None)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
        return W, mean

    def train_from_occurrences(self, U_matrix, word_labels, V_dict):
        V_list, valid_indices = [], []
        for i, word in enumerate(word_labels):
            wl = str(word).lower()
            if wl in V_dict:
                V_list.append(V_dict[wl])
                valid_indices.append(i)

        U_valid = U_matrix[valid_indices]
        V_valid = np.array(V_list)
        logger.info(f"Training on {len(valid_indices)} occurrences "
                     f"({len(valid_indices)/len(word_labels)*100:.1f}% matched)")

        self.U_mean = U_valid.mean(axis=0)
        self.V_mean = V_valid.mean(axis=0)
        U_c = U_valid - self.U_mean
        V_c = V_valid - self.V_mean
        d_U = U_c.shape[1]
        n = U_c.shape[0]

        W_u = None
        if self.whiten and n > d_U:
            logger.info("Applying whitening transform")
            try:
                W_u, _ = self.compute_whitening_transform(U_c)
                U_c = U_c @ W_u
            except Exception as e:
                logger.warning(f"Whitening failed: {e}")
                W_u = None

        if self.use_procrustes_prior and n > d_U:
            logger.info("Computing Procrustes prior")
            self.A0 = self.compute_procrustes_prior(U_c, V_c)
        else:
            self.A0 = np.zeros((V_c.shape[1], U_c.shape[1]))

        UTU = U_c.T @ U_c + self.lambda_ * np.eye(d_U)
        VTU = V_c.T @ U_c + self.lambda_ * self.A0
        self.A = linalg.solve(UTU.T, VTU.T).T

        V_pred = U_c @ self.A.T
        mse = np.mean((V_c - V_pred) ** 2)
        logger.info(f"Training MSE: {mse:.6f}")

        if W_u is not None:
            self.A = self.A @ W_u
        return self.A

    def transform(self, U):
        single = U.ndim == 1
        if single:
            U = U.reshape(1, -1)
        U_c = U - self.U_mean
        Y = U_c @ self.A.T
        return Y.squeeze() if single else Y

    def save(self, path):
        np.savez(path, A=self.A, A0=self.A0, U_mean=self.U_mean,
                 V_mean=self.V_mean, lambda_=self.lambda_)
        logger.info(f"Saved A matrix to {path}")


# ---- Main Pipeline ----

def run_full_pipeline(data_dir="/data", output_dir="/data/output",
                      max_anchor_samples=50000, batch_size=32,
                      regularization_lambda=0.1):
    """完整的 DeBERTa 管线"""
    import time
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    # ============================================================
    # Step 1: 加载输入数据
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading input data")
    logger.info("=" * 60)

    term_occ = pd.read_parquet(data_dir / "term_occurrences.parquet")
    anchor_occ = pd.read_parquet(data_dir / "anchor_occurrences_all.parquet")
    with open(data_dir / "anchor_words.json", "r") as f:
        anchor_words = json.load(f).get("words", [])

    logger.info(f"Term occurrences: {len(term_occ)}")
    logger.info(f"Anchor occurrences: {len(anchor_occ)}")
    logger.info(f"Anchor words: {len(anchor_words)}")

    results["n_term_occurrences"] = len(term_occ)
    results["n_anchor_occurrences_raw"] = len(anchor_occ)

    # 分层采样锚点词
    if len(anchor_occ) > max_anchor_samples:
        logger.info(f"Sampling {max_anchor_samples} anchor occurrences (stratified)")
        sampled = []
        for country in anchor_occ["country"].unique():
            cdf = anchor_occ[anchor_occ["country"] == country]
            n = int(max_anchor_samples * len(cdf) / len(anchor_occ))
            n = min(n, len(cdf))
            sampled.append(cdf.sample(n=n, random_state=42))
        anchor_occ = pd.concat(sampled, ignore_index=True)
        logger.info(f"Sampled: {len(anchor_occ)}")

    results["n_anchor_occurrences_sampled"] = len(anchor_occ)

    # ============================================================
    # Step 2: DeBERTa 编码目标术语 → U_target
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 2: DeBERTa encoding target term occurrences")
    logger.info("=" * 60)

    u_target_path = output_dir / "U_target.npy"
    if u_target_path.exists():
        logger.info(f"Loading cached U_target from {u_target_path}")
        U_target = np.load(u_target_path)
    else:
        encoder = DeBERTaEncoder(batch_size=batch_size)

        # 准备 term_occ 的列名（可能有差异）
        text_col = "text_block" if "text_block" in term_occ.columns else ("paragraph" if "paragraph" in term_occ.columns else "context")
        term_col = "matched_term" if "matched_term" in term_occ.columns else "term"
        pos_col = "start_char" if "start_char" in term_occ.columns else "char_start"

        U_target = encoder.encode_batch(
            term_occ, text_column=text_col, term_column=term_col,
            position_column=pos_col, show_progress=True
        )
        np.save(u_target_path, U_target)

    logger.info(f"U_target shape: {U_target.shape}")
    results["U_target_shape"] = list(U_target.shape)

    # ============================================================
    # Step 3: DeBERTa 编码锚点词 → U_anchors
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 3: DeBERTa encoding anchor occurrences")
    logger.info("=" * 60)

    u_anchors_path = output_dir / "U_anchors.npy"
    if u_anchors_path.exists():
        logger.info(f"Loading cached U_anchors from {u_anchors_path}")
        U_anchors = np.load(u_anchors_path)
    else:
        if 'encoder' not in dir():
            encoder = DeBERTaEncoder(batch_size=batch_size)

        anchor_occ = anchor_occ.copy()
        anchor_occ["matched_term"] = anchor_occ["anchor_word"]
        if "pos_type" not in anchor_occ.columns:
            anchor_occ["pos_type"] = "noun"

        U_anchors = encoder.encode_batch(
            anchor_occ, text_column="text_block", term_column="matched_term",
            position_column="start_char", show_progress=True
        )
        np.save(u_anchors_path, U_anchors)

    logger.info(f"U_anchors shape: {U_anchors.shape}")
    results["U_anchors_shape"] = list(U_anchors.shape)

    # ============================================================
    # Step 4: GPT-2 嵌入查找 → V_dict
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 4: GPT-2 embeddings for anchor words")
    logger.info("=" * 60)

    llm = LLMEmbedder(model_name="gpt2")
    V_dict = {}
    for word in anchor_words:
        try:
            V_dict[word.lower()] = llm.get_phrase_embedding(word)
        except Exception as e:
            logger.warning(f"Failed for '{word}': {e}")

    logger.info(f"V embeddings: {len(V_dict)} words")
    results["n_anchor_V"] = len(V_dict)

    # ============================================================
    # Step 5: 训练全局 A 矩阵
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 5: Training global A matrix (ridge + Procrustes)")
    logger.info("=" * 60)

    trainer = MatrixTrainer(regularization_lambda=regularization_lambda)
    word_labels = anchor_occ["anchor_word"].values

    trainer.train_from_occurrences(U_anchors, word_labels, V_dict)

    logger.info(f"A shape: {trainer.A.shape}")
    logger.info(f"A Frobenius norm: {np.linalg.norm(trainer.A):.4f}")
    results["A_shape"] = list(trainer.A.shape)
    results["A_frobenius_norm"] = float(np.linalg.norm(trainer.A))

    trainer.save(str(output_dir / "A_matrix_global.npz"))

    # ============================================================
    # Step 6: 应用 A → Y_global
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 6: Applying A matrix → Y_global")
    logger.info("=" * 60)

    Y_global = trainer.transform(U_target)
    logger.info(f"Y_global shape: {Y_global.shape}")
    results["Y_global_shape"] = list(Y_global.shape)

    np.save(output_dir / "Y_matrix_paragraph_global.npy", Y_global)

    # ============================================================
    # Step 7: 构建并保存最终 parquet
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 7: Building final parquet")
    logger.info("=" * 60)

    # 构建输出 DataFrame
    out_df = term_occ.copy()
    out_df["U_vector"] = list(U_target)
    out_df["Y_vector_global"] = list(Y_global)
    out_df["Y_norm_global"] = np.linalg.norm(Y_global, axis=1)

    parquet_path = output_dir / "semantic_vectors_paragraph_global_A.parquet"
    out_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")
    results["output_parquet_size_mb"] = round(parquet_path.stat().st_size / 1e6, 1)

    # ============================================================
    # Step 8: 验证
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 8: Validation")
    logger.info("=" * 60)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    for country in ["US", "UK", "AU"]:
        mask = out_df["country"] == country
        Y_mean = Y_global[mask].mean(axis=0)
        results[f"Y_mean_norm_{country}"] = float(np.linalg.norm(Y_mean))
        logger.info(f"{country}: n={mask.sum()}, Y_mean_norm={np.linalg.norm(Y_mean):.4f}")

    Y_means = {}
    for c in ["US", "UK", "AU"]:
        Y_means[c] = Y_global[out_df["country"] == c].mean(axis=0)

    for c1, c2 in [("US", "UK"), ("US", "AU"), ("UK", "AU")]:
        sim = cosine_sim(Y_means[c1], Y_means[c2])
        results[f"cosine_{c1}_{c2}"] = sim
        logger.info(f"cos({c1}, {c2}) = {sim:.4f}")

    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # 保存结果摘要
    with open(output_dir / "reproduction_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)

    return results
'''


# ============================================================
# Modal 远程函数
# ============================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,  # 1 hour max
    volumes={VOLUME_PATH: volume},
    memory=32768,  # 32 GB RAM
)
def run_deberta_pipeline():
    """在 Modal GPU 上运行 DeBERTa 管线"""
    import importlib.util

    # 将管线代码写入临时文件并导入
    pipeline_path = "/tmp/pipeline.py"
    with open(pipeline_path, "w") as f:
        f.write(PIPELINE_CODE)

    spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)

    # 运行完整管线
    results = pipeline.run_full_pipeline(
        data_dir=VOLUME_PATH,
        output_dir=f"{VOLUME_PATH}/output",
        max_anchor_samples=50000,
        batch_size=32,
        regularization_lambda=0.1,
    )

    # 提交 volume 变更
    volume.commit()

    return results


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=300)
def download_results():
    """从 Volume 下载结果文件列表"""
    import os
    output_dir = f"{VOLUME_PATH}/output"
    files = {}
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            path = os.path.join(output_dir, f)
            size = os.path.getsize(path)
            files[f] = size
    return files


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def download_file(filename: str) -> bytes:
    """从 Volume 下载单个文件"""
    path = f"{VOLUME_PATH}/output/{filename}"
    with open(path, "rb") as f:
        return f.read()


# ============================================================
# 本地入口：上传数据 → 运行 → 下载结果
# ============================================================

def upload_data():
    """上传输入数据到 Modal Volume"""
    final_fuxian_root = Path(__file__).parent.parent  # final_fuxian/
    project_root = final_fuxian_root.parent  # llm-ER/
    data_sources = [
        (final_fuxian_root / "data" / "intermediate" / "term_occurrences.parquet", "term_occurrences.parquet"),
        (final_fuxian_root / "data" / "intermediate" / "anchor_occurrences_all.parquet", "anchor_occurrences_all.parquet"),
        (final_fuxian_root / "data" / "intermediate" / "anchor_words.json", "anchor_words.json"),
    ]

    print("\n📤 上传输入数据到 Modal Volume...")
    for src, dst_name in data_sources:
        if not src.exists():
            # 尝试备用路径
            alt = project_root / "data" / "intermediate" / dst_name
            if alt.exists():
                src = alt
            else:
                print(f"  ❌ 未找到: {src}")
                sys.exit(1)

        size_mb = src.stat().st_size / 1e6
        print(f"  📁 {dst_name} ({size_mb:.1f} MB)")

        # 使用 modal volume put 上传
        os.system(f'modal volume put llm-er-data "{src}" "{dst_name}"')

    print("  ✅ 数据上传完成\n")


def download_output():
    """从 Modal Volume 下载输出"""
    project_root = Path(__file__).parent.parent  # final_fuxian/
    output_dir = project_root / "outputs" / "modal_reproduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📥 下载复现结果...")

    # 下载关键文件
    key_files = [
        "semantic_vectors_paragraph_global_A.parquet",
        "A_matrix_global.npz",
        "Y_matrix_paragraph_global.npy",
        "U_target.npy",
        "reproduction_summary.json",
    ]

    for fname in key_files:
        dst = output_dir / fname
        print(f"  📁 下载 {fname}...")
        os.system(f'modal volume get llm-er-data "output/{fname}" "{dst}"')

        if dst.exists():
            size_mb = dst.stat().st_size / 1e6
            print(f"     ✅ {size_mb:.1f} MB")
        else:
            print(f"     ⚠️  下载可能失败")

    print(f"\n✅ 所有文件保存到: {output_dir}")
    return output_dir


def verify_reproduction(output_dir: Path):
    """验证复现结果与原始数据是否一致"""
    import json

    final_fuxian_root = Path(__file__).parent.parent
    original_parquet = final_fuxian_root / "data" / "semantic_vectors_paragraph_global_A.parquet"
    reproduced_parquet = output_dir / "semantic_vectors_paragraph_global_A.parquet"

    if not original_parquet.exists() or not reproduced_parquet.exists():
        print("⚠️  无法进行对比验证（缺少文件）")
        return

    print("\n🔍 验证复现结果...")

    import numpy as np
    import pandas as pd

    orig = pd.read_parquet(original_parquet)
    repro = pd.read_parquet(reproduced_parquet)

    print(f"  原始数据: {len(orig)} rows")
    print(f"  复现数据: {len(repro)} rows")
    print(f"  行数匹配: {'✅' if len(orig) == len(repro) else '❌'}")

    # 比较 Y_vector_global
    if "Y_vector_global" in orig.columns and "Y_vector_global" in repro.columns:
        Y_orig = np.vstack(orig["Y_vector_global"].values)
        Y_repro = np.vstack(repro["Y_vector_global"].values)

        # 余弦相似度
        cos_sims = []
        for i in range(len(Y_orig)):
            dot = np.dot(Y_orig[i], Y_repro[i])
            norm = np.linalg.norm(Y_orig[i]) * np.linalg.norm(Y_repro[i]) + 1e-10
            cos_sims.append(dot / norm)
        cos_sims = np.array(cos_sims)

        print(f"  Y_vector_global 余弦相似度:")
        print(f"    均值: {cos_sims.mean():.6f}")
        print(f"    最小: {cos_sims.min():.6f}")
        print(f"    >0.99: {(cos_sims > 0.99).sum()}/{len(cos_sims)} ({(cos_sims > 0.99).mean()*100:.1f}%)")

        # 欧氏距离
        dists = np.linalg.norm(Y_orig - Y_repro, axis=1)
        print(f"  欧氏距离: 均值={dists.mean():.6f}, 最大={dists.max():.6f}")

    # 比较 A 矩阵
    orig_A_path = final_fuxian_root / "outputs" / "modal_reproduction" / "A_matrix_global.npz"
    repro_A_path = output_dir / "A_matrix_global.npz"
    if orig_A_path.exists() and repro_A_path.exists():
        A_orig = np.load(orig_A_path)["A"]
        A_repro = np.load(repro_A_path)["A"]
        A_diff = np.linalg.norm(A_orig - A_repro) / np.linalg.norm(A_orig)
        print(f"  A 矩阵相对差异: {A_diff:.6f}")
        print(f"  A 矩阵 Frobenius 范数 (原始): {np.linalg.norm(A_orig):.4f}")
        print(f"  A 矩阵 Frobenius 范数 (复现): {np.linalg.norm(A_repro):.4f}")

    print("\n✅ 验证完成")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modal DeBERTa 复现管线")
    parser.add_argument("--upload-only", action="store_true", help="仅上传数据")
    parser.add_argument("--run-only", action="store_true", help="仅运行管线（假设数据已上传）")
    parser.add_argument("--download-only", action="store_true", help="仅下载结果")
    parser.add_argument("--verify-only", action="store_true", help="仅验证结果")
    args = parser.parse_args()

    if args.upload_only:
        upload_data()
    elif args.run_only:
        with app.run():
            print("🚀 在 Modal A10G GPU 上运行 DeBERTa 管线...")
            results = run_deberta_pipeline.remote()
            print("\n📊 运行结果:")
            for k, v in results.items():
                print(f"  {k}: {v}")
    elif args.download_only:
        output_dir = download_output()
    elif args.verify_only:
        output_dir = Path(__file__).parent.parent / "outputs" / "modal_reproduction"
        verify_reproduction(output_dir)
    else:
        # 完整流程：上传 → 运行 → 下载 → 验证
        print("=" * 60)
        print("DeBERTa 嵌入管线完整复现（Modal GPU）")
        print("=" * 60)

        # Step 1: 上传
        upload_data()

        # Step 2: 运行
        with app.run():
            print("🚀 在 Modal A10G GPU 上运行 DeBERTa 管线...")
            print("  预计运行时间: 15-30 分钟")
            print("  (DeBERTa 编码 ~87,000 样本 + A 矩阵训练)")
            results = run_deberta_pipeline.remote()
            print("\n📊 运行结果:")
            for k, v in results.items():
                print(f"  {k}: {v}")

        # Step 3: 下载
        output_dir = download_output()

        # Step 4: 验证
        verify_reproduction(output_dir)
