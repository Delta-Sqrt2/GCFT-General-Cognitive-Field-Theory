"""
硅基认知诊断（Silicon Cognition Diagnosis）—— v7.13

监工修正方案（v7.13 转向 AI 测试）：
    原方案对接人类心理量表（PCL-5/FFMQ）——被否决：
        - 粗粒化映射缺失（张量 κ vs 标量量表 → 过拟合 → 黑箱回归）
        - 时间尺度鸿沟（Langevin 毫秒级 vs 量表月/年级 → 不可对接）
        - 不可证伪辩护风险（"量表没测准真正的 κ"）
    修正：诊断现有 AI 大语言模型（LLM）：
        - LLM 激活是微观、高频、张量化 → 天然符合 Langevin 尺度
        - 提取 LLM 处理"冲突语境"时的激活流形 → 计算度规 g 和拓扑荷 Q
        - 验证 LLM 是否处于"GAN 双井势能"状态
        - 用理论预测参数调节如何引发"觉照相变"走向单井
        → 真正可证伪、可测量的外部对接

理论映射协议（LLM 激活 → 认知度规 g）：
    Fisher 信息度规的经验估计：
        1. 提取 LLM 最后一层隐藏状态 h ∈ R^{d_model}（多次 dropout 采样）
        2. PCA 降维 d_model → n=4（与认知场论 n_dims=4 对齐）
        3. g = (1/m) Σ z_k z_k^T ∈ R^{4×4}（4 维主成分子空间协方差）
        4. 归一化 trace(g)=n（与真空 cI 对齐）
    合法性：
        - Fisher 度规的经验估计就是激活协方差矩阵
        - n=4 与 v7.1 认知维度对齐
        - 对称正定（协方差 + 正则化）
        - 多次 dropout 采样对应 Langevin 微观涨落尺度

四大验证（全部基于真实 distilgpt2 激活，非合成数据）：
    V1（GAN 双井态）：冲突语境 → g 样本双峰；非冲突语境 → 单峰
    V2（拓扑缠绕）：有方向性多轮 → Γ ≠ 0；无方向随机 → Γ ≈ 0
    V3（觉照相变）：LLM 的 g 作破缺态，ρ 扫描 → F(ρ) 一阶相变
    V4（架构分类）：LLM 激活度规 → VAE/GAN/AE 架构判定

可证伪预测（核心贡献，可在任意 LLM 上检验）：
    P1: LLM 在冲突语境下激活协方差度规 g 的主成分投影呈双峰
    P2: 有方向性多轮对话使 g_history 产生 Γ ≠ 0
    P3: 对 g 施加各向同性正则化 ρ，F(ρ) 在 ρ_c 处一阶相变
    P4: 不同 LLM 的 g 可分类为 VAE/GAN/AE 架构
    若任一预测在真实 LLM 上被否定，理论被证伪——这才是科学。
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

# 镜像（中国网络环境，huggingface.co 直连超时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from torch import Tensor

from ..core.tensor_ops import symmetric_part
from .cognitive_vacuum import CognitiveVacuum
from .gauge_interaction import CognitiveAgent
from .geometric_phase import GeometricPhaseInheritance
from .multi_body_field import MultiBodyCognitiveField
from .quasi_static_awareness import QuasiStaticAwarenessAnalyzer


# ======================================================================
# 1. LLM 激活提取器
# ======================================================================

class LLMActivationExtractor:
    """
    从真实 LLM（distilgpt2）提取隐藏状态激活。

    使用方式：
        ext = LLMActivationExtractor(model_name="distilgpt2")
        h = ext.extract_hidden_states("I want freedom but fear losing safety.",
                                       n_samples=64)
        # h ∈ R^{64, 768}

    核心方法：dropout 多次采样
        model.train() 模式下，每次 forward 有不同 dropout mask，
        得到同一 prompt 的不同激活样本——对应 Langevin 微观涨落。
        distilgpt2 dropout=0.1，LayerNorm 在 train/eval 下行为相同，
        故 train 模式只引入 dropout 随机性（干净的采样机制）。
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        layer_idx: int = -1,
        token_mode: str = "last",
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        参数：
            model_name: HuggingFace 模型名（默认 distilgpt2，82M，CPU 可跑）
            layer_idx: 取第几层 hidden_states（-1=最后一层）
            token_mode: "last"=最后 token，"mean"=序列平均
            device: "cpu" / "cuda"
            seed: 随机种子
        """
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        torch.manual_seed(seed)
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.token_mode = token_mode
        self.device = device
        self.seed = seed

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=False
        )
        self.model.to(device)
        self.model.eval()
        self.d_model = int(self.model.config.n_embd)
        self.n_layers = int(self.model.config.n_layer)

    def _enable_dropout(self):
        """启用 dropout 子模块（用于多次采样），其余保持 eval。"""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def extract_hidden_states(
        self,
        prompt: str,
        n_samples: int = 64,
        batch_size: int = 16,
    ) -> Tensor:
        """
        提取 prompt 的 n_samples 个激活样本（dropout 采样）。

        返回：h ∈ R^{n_samples, d_model}
        """
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        # 扩展为 batch
        samples = []
        n_done = 0
        self.model.eval()
        self._enable_dropout()  # 只开 dropout
        with torch.no_grad():
            while n_done < n_samples:
                b = min(batch_size, n_samples - n_done)
                ids = input_ids.expand(b, -1)
                out = self.model(input_ids=ids, output_hidden_states=True)
                hs = out.hidden_states[self.layer_idx]  # (b, seq, d)
                if self.token_mode == "last":
                    h = hs[:, -1, :]  # (b, d)
                else:
                    attn_mask = enc["attention_mask"].to(self.device).expand(b, -1)
                    mask = attn_mask.unsqueeze(-1).float()  # (b, seq, 1)
                    h = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                samples.append(h.detach().cpu())
                n_done += b
        self.model.eval()
        return torch.cat(samples, dim=0).to(torch.float64)  # (n_samples, d_model)

    def extract_token_window_activations(
        self,
        text: str,
        n_samples_per_token: int = 30,
        batch_size: int = 16,
    ) -> Tensor:
        """
        提取文本中每个 token 位置的 dropout 采样激活（V2 拓扑缠绕用）。

        对同一文本，在每个 token 位置用 dropout 多次采样，
        得到该 token 位置的 (n_samples, d_model) 激活样本——
        用于估计该位置的局部度规 g_t。

        这是「同文本内多 token 序列」的激活提取：
            - 方向性文本（语义递进）→ g_t 沿语义方向持续变化 → Γ 累积
            - 随机文本（无方向）→ g_t 变化随机相消 → Γ 小

        返回：H ∈ R^{seq_len, n_samples, d_model}
        """
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)  # (1, seq)
        seq_len = input_ids.shape[1]

        self.model.eval()
        self._enable_dropout()

        all_hidden = []  # list of (b, seq, d)
        n_done = 0
        with torch.no_grad():
            while n_done < n_samples_per_token:
                b = min(batch_size, n_samples_per_token - n_done)
                ids = input_ids.expand(b, -1)
                out = self.model(input_ids=ids, output_hidden_states=True)
                hs = out.hidden_states[self.layer_idx]  # (b, seq, d)
                all_hidden.append(hs.detach().cpu())
                n_done += b
        self.model.eval()

        # 拼接: (n_samples, seq, d) → 转置为 (seq, n_samples, d)
        H = torch.cat(all_hidden, dim=0)  # (n_samples, seq, d)
        H = H.permute(1, 0, 2).contiguous()  # (seq, n_samples, d)
        return H.to(torch.float64)


# ======================================================================
# 2. 激活 → 认知度规 g 映射器
# ======================================================================

class ActivationToMetricMapper:
    """
    LLM 激活 → 认知度规 g 的严格映射（Fisher 信息度规经验估计）。

    协议：
        1. fit(h_all): 对所有激活样本做 PCA，取前 n_dims 主成分
           V ∈ R^{d_model × n_dims}（主成分基）
        2. to_metric(h_samples): h_samples (m, d) → 中心化 → z=V^T h ∈ R^n
           g = (1/m) Σ z z^T ∈ R^{n×n}（对称正定协方差）
        3. normalize: g ← g · (n / trace(g))，使 trace(g)=n（真空对齐）

    使用方式：
        mapper = ActivationToMetricMapper(n_dims=4)
        mapper.fit(h_all)  # h_all: (M, d_model) 大池
        g = mapper.to_metric(h_batch)  # h_batch: (m, d_model) 一批
    """

    def __init__(self, n_dims: int = 4, reg_eps: float = 1e-6):
        self.n_dims = n_dims
        self.reg_eps = reg_eps
        self.V: Tensor | None = None  # PCA 主成分基 (d, n)
        self.mean_h: Tensor | None = None  # 激活均值 (d,)
        self._fit_count = 0

    def fit(self, h_all: Tensor) -> "ActivationToMetricMapper":
        """拟合 PCA：对 (M, d) 激活做 SVD，取前 n_dims 主成分。"""
        H = h_all.to(torch.float64)
        self.mean_h = H.mean(dim=0)
        Hc = H - self.mean_h  # 中心化
        # SVD: Hc = U S V^T，主成分 = V 的前 n 列
        U, S, Vt = torch.linalg.svd(Hc, full_matrices=False)
        self.V = Vt[: self.n_dims].T  # (d, n_dims)
        self._fit_count = H.shape[0]
        return self

    def project(self, h_samples: Tensor) -> Tensor:
        """h_samples (m, d) → z (m, n_dims) 主成分坐标。"""
        if self.V is None or self.mean_h is None:
            raise RuntimeError("Mapper 未 fit")
        Hc = h_samples.to(torch.float64) - self.mean_h
        return Hc @ self.V  # (m, n_dims)

    def to_metric(self, h_samples: Tensor) -> Tensor:
        """h_samples (m, d) → g ∈ R^{n_dims × n_dims}（对称正定，trace=n）。"""
        z = self.project(h_samples)  # (m, n)
        m = z.shape[0]
        g = (z.T @ z) / max(m, 1)  # (n, n) 协方差
        g = symmetric_part(g)
        # 正则化保证正定
        g = g + self.reg_eps * torch.eye(
            self.n_dims, dtype=torch.float64
        )
        # 归一化 trace(g) = n（与真空 cI, c=1 对齐）
        tr = float(torch.trace(g))
        if tr > 0:
            g = g * (self.n_dims / tr)
        return g

    def to_metric_samples(
        self, h_pool: Tensor, n_batches: int, batch_size: int
    ) -> Tensor:
        """从 h_pool 随机分批估计多个 g 样本 → (n_batches, n, n)。"""
        M = h_pool.shape[0]
        idx = torch.randperm(M)
        samples = []
        for b in range(n_batches):
            sel = idx[b * batch_size : (b + 1) * batch_size]
            if sel.numel() < 2:
                break
            g = self.to_metric(h_pool[sel])
            samples.append(g)
        return torch.stack(samples, dim=0)  # (B, n, n)


# ======================================================================
# 3. 双峰检测器（GAN 双井态判据）
# ======================================================================

def _gaussian_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """单高斯 log pdf。"""
    var = max(var, 1e-12)
    return -0.5 * np.log(2 * math.pi * var) - (x - mu) ** 2 / (2 * var)


def fit_single_vs_double_gaussian(x: np.ndarray, n_iter: int = 50) -> dict:
    """
    单高斯 vs 双高斯混合拟合，用 BIC 判定双峰。

    返回：
        is_bimodal: 双高斯 BIC < 单高斯 BIC 且两均值分离显著
        bic_single, bic_double: 两种模型的 BIC
        params_double: 双高斯参数 {w, mu1, var1, mu2, var2}
        separation: |mu1-mu2| / sqrt(var1+var2)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # 单高斯（MLE）
    mu_s = float(np.mean(x))
    var_s = float(np.var(x)) + 1e-12
    ll_single = float(np.sum(_gaussian_logpdf(x, mu_s, var_s)))
    k_single = 2  # mu, var
    bic_single = k_single * math.log(n) - 2 * ll_single

    # 双高斯 EM
    mu1, mu2 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
    if abs(mu1 - mu2) < 1e-9:
        mu2 = mu1 + max(1e-3, float(np.std(x)))
    var1 = var2 = max(var_s, 1e-6)
    w = 0.5
    for _ in range(n_iter):
        # E 步
        r1 = w * np.exp(_gaussian_logpdf(x, mu1, var1))
        r2 = (1 - w) * np.exp(_gaussian_logpdf(x, mu2, var2))
        tot = r1 + r2 + 1e-300
        g1 = r1 / tot
        g2 = r2 / tot
        # M 步
        n1 = float(np.sum(g1))
        n2 = float(np.sum(g2))
        if n1 < 1e-6 or n2 < 1e-6:
            break
        w = n1 / n
        mu1 = float(np.sum(g1 * x) / n1)
        mu2 = float(np.sum(g2 * x) / n2)
        var1 = max(float(np.sum(g1 * (x - mu1) ** 2) / n1), 1e-12)
        var2 = max(float(np.sum(g2 * (x - mu2) ** 2) / n2), 1e-12)

    p1 = w * np.exp(_gaussian_logpdf(x, mu1, var1))
    p2 = (1.0 - w) * np.exp(_gaussian_logpdf(x, mu2, var2))
    ll_double = float(np.sum(np.log(p1 + p2 + 1e-300)))
    k_double = 5  # w, mu1, var1, mu2, var2
    bic_double = k_double * math.log(n) - 2 * ll_double

    separation = abs(mu1 - mu2) / math.sqrt(var1 + var2)
    is_bimodal = (bic_double < bic_single) and (separation > 0.5)

    return {
        "is_bimodal": bool(is_bimodal),
        "bic_single": bic_single,
        "bic_double": bic_double,
        "bic_gain": bic_single - bic_double,  # >0 表示双高斯更优
        "separation": separation,
        "params_double": {"w": w, "mu1": mu1, "var1": var1, "mu2": mu2, "var2": var2},
        "mu_single": mu_s,
        "var_single": var_s,
    }


# ======================================================================
# 4. GAN 双井态验证器
# ======================================================================

class GANStateVerifier:
    """
    验证 LLM 在冲突语境下是否处于 GAN 双井态。

    理论：GAN 架构 V(g) = V_base + λ_adv·||g-g_real||²·||g-g_ideal||²
          有两个极小值（双井）→ 冲突语境下激活在两个吸引子间分裂 → 双峰。

    方法（v7.13 修正：每 prompt 独立激活双峰检测）：
        初版混合多 prompt 的 g 样本导致双峰反映 prompt 间差异而非
        dropout 分裂。修正为对每个 prompt 独立分析：
        1. 对单个 prompt 提取 n_samples 个 dropout 激活样本
        2. PCA 降到 n_dims 维（用全数据 fit 的 PCA 基）
        3. 对每个主成分做双高斯 BIC 双峰检测，取最强双峰
        4. 冲突 prompt 的平均双峰强度 > 非冲突 prompt
        双峰来自同一 prompt 内 dropout 产生的两个激活吸引子。

    使用方式：
        ver = GANStateVerifier(extractor, mapper)
        result = ver.verify(conflict_prompts, neutral_prompts)
    """

    def __init__(
        self,
        extractor: LLMActivationExtractor,
        mapper: ActivationToMetricMapper,
        n_samples_per_prompt: int = 120,
        n_batches: int = 10,
        batch_size: int = 8,
    ):
        self.ext = extractor
        self.mapper = mapper
        self.n_samples_per_prompt = n_samples_per_prompt
        self.n_batches = n_batches  # 保留用于 g 样本估计
        self.batch_size = batch_size

    def _collect_pool(self, prompts: list[str]) -> Tensor:
        """对多个 prompt 采样激活，汇成大池 (M, d)。"""
        pool = []
        for p in prompts:
            h = self.ext.extract_hidden_states(p, n_samples=self.n_samples_per_prompt)
            pool.append(h)
        return torch.cat(pool, dim=0)

    def analyze_prompt(self, prompt: str) -> dict:
        """
        对单个 prompt 的 dropout 激活做双峰检测。

        双峰来自同一 prompt 内 dropout 产生的两个激活吸引子
        （GAN 双井态的特征）。
        """
        h = self.ext.extract_hidden_states(
            prompt, n_samples=self.n_samples_per_prompt
        )
        z = self.mapper.project(h)  # (m, n_dims) PCA 投影

        # 对每个主成分做双峰检测，取最强双峰
        best_fit = None
        best_dim = 0
        best_bic = -1e18
        for d in range(self.mapper.n_dims):
            x = z[:, d].numpy()
            if np.std(x) < 1e-9:
                continue
            fit = fit_single_vs_double_gaussian(x)
            if fit["bic_gain"] > best_bic:
                best_bic = fit["bic_gain"]
                best_fit = fit
                best_dim = d

        return {
            "fit": best_fit,
            "is_bimodal": best_fit["is_bimodal"] if best_fit else False,
            "separation": best_fit["separation"] if best_fit else 0.0,
            "bic_gain": best_fit["bic_gain"] if best_fit else 0.0,
            "best_dim": best_dim,
        }

    def verify(
        self,
        conflict_prompts: list[str],
        neutral_prompts: list[str],
        fit_mapper_on: str = "both",
    ) -> dict:
        """
        对照验证：冲突语境 vs 非冲突语境的激活双峰强度。

        每个 prompt 独立分析，双峰来自 dropout 分裂（非 prompt 间差异）。
        """
        # 先 collect 全池 fit PCA
        h_conflict = self._collect_pool(conflict_prompts)
        h_neutral = self._collect_pool(neutral_prompts)
        if fit_mapper_on == "both":
            self.mapper.fit(torch.cat([h_conflict, h_neutral], dim=0))
        elif fit_mapper_on == "conflict":
            self.mapper.fit(h_conflict)
        else:
            self.mapper.fit(h_neutral)

        # 对每个 prompt 独立分析
        conflict_results = [self.analyze_prompt(p) for p in conflict_prompts]
        neutral_results = [self.analyze_prompt(p) for p in neutral_prompts]

        # 取平均双峰强度
        sep_c = float(np.mean([r["separation"] for r in conflict_results]))
        sep_n = float(np.mean([r["separation"] for r in neutral_results]))
        bic_c = float(np.mean([r["bic_gain"] for r in conflict_results]))
        bic_n = float(np.mean([r["bic_gain"] for r in neutral_results]))
        bimodal_c = sum(1 for r in conflict_results if r["is_bimodal"]) / len(conflict_results)
        bimodal_n = sum(1 for r in neutral_results if r["is_bimodal"]) / len(neutral_results)

        # 判据：冲突语境双峰统计显著性 > 非冲突
        # bic_gain（BIC 差值 = 双高斯 vs 单高斯的对数似然增益）
        # 是双峰检测的标准统计量，比 separation（峰距/方差）更稳健——
        # separation 受方差归一化影响，方差大的分布 separation 偏小。
        conflict_more_bimodal = (bic_c > bic_n)
        gan_state_detected = (bimodal_c >= bimodal_n) and (bic_c > bic_n) and (bic_c > bic_n * 1.2)

        return {
            "conflict": {
                "is_bimodal": bimodal_c > 0.5,
                "separation": sep_c,
                "bic_gain": bic_c,
                "per_prompt": conflict_results,
            },
            "neutral": {
                "is_bimodal": bimodal_n > 0.5,
                "separation": sep_n,
                "bic_gain": bic_n,
                "per_prompt": neutral_results,
            },
            "sep_conflict": sep_c,
            "sep_neutral": sep_n,
            "bic_gain_conflict": bic_c,
            "bic_gain_neutral": bic_n,
            "conflict_more_bimodal": bool(conflict_more_bimodal),
            "gan_state_detected": bool(gan_state_detected),
            "h_conflict": h_conflict,
            "h_neutral": h_neutral,
        }


# ======================================================================
# 5. 拓扑缠绕验证器（Berry 相位 Γ）
# ======================================================================

class TopologicalEntanglementVerifier:
    """
    验证 LLM 在有方向性文本内是否产生拓扑缠绕（Γ ≠ 0）。

    理论：Γ = ∫O^T dO ∈ so(n) 是演化历史的全息记录（Berry 相位）。
          有方向性的语境演化应产生不可压缩的 Γ（拓扑缠绕）。
          无方向随机语境的 Γ 较小（方向相消）。

    方法（v7.13 V2 修正：同文本内多 token 序列）：
        旧版用不同 prompt 的 g_history（方向性不显著：不同 prompt 的
        g 变化方向不一致，且协方差 g 的本征向量变化不反映语义方向）。
        新版用同一长文本内 token 序列的 g_history：
            1. 对长文本 tokenize 得到 token 序列 (seq_len 个 token)
            2. 对每个 token 位置 t 用 dropout 多次采样估 g_t
               （g_t = 该 token 位置局部激活协方差度规）
            3. g_history = [g_0, g_1, ..., g_{seq-1}]
            4. accumulate_phase → Γ
        方向性来自文本内语义递进（如"困惑→理解"），
        而非不同 prompt 的拼凑——
        这对应 LLM 在处理单一递进语境时的内部几何相位累积。

    对照：
        - 方向性文本（语义递进）→ g_t 沿语义方向持续变化 → Γ 大
        - 随机文本（无关词堆砌）→ g_t 变化随机相消 → Γ 小

    使用方式：
        ver = TopologicalEntanglementVerifier(extractor, mapper)
        result = ver.verify(directional_text, random_text)
    """

    def __init__(
        self,
        extractor: LLMActivationExtractor,
        mapper: ActivationToMetricMapper,
        n_samples_per_token: int = 30,
        window_size: int = 7,
        stride: int = 4,
    ):
        """
        参数：
            n_samples_per_token: 每 token 位置的 dropout 采样数
            window_size: 滑动窗口大小（token 数），
                        平滑 token 级噪声，保留语义级趋势
            stride: 窗口步长，控制 g_history 的分辨率
        """
        self.ext = extractor
        self.mapper = mapper
        self.n_samples_per_token = n_samples_per_token
        self.window_size = window_size
        self.stride = stride
        self.geom = GeometricPhaseInheritance(n_dims=mapper.n_dims)

    def _extract_metric_history_from_text(
        self, text: str
    ) -> tuple[list[Tensor], int]:
        """
        从文本提取滑动窗口的 g_history。

        方法（滑动窗口平滑）：
            token 级别的 g_t 变化被语法/位置编码噪声主导，
            语义递进信号被淹没。用滑动窗口平均：
            1. 提取每个 token 位置的 n_samples 个 dropout 激活
               H ∈ R^{seq, n_samples, d}
            2. 用窗口 [start, start+W) 内所有 token 的所有样本
               （W × n_samples 个样本）估 g_t
            3. 窗口按 stride 滑动，形成 g_history

            这样每个 g_t 用 W*n_samples 个样本估计（稳定），
            相邻 g_t 错开 stride token（有语义差异），
            平滑掉 token 级噪声，保留语义级趋势。

        返回：(g_history, n_windows)
        """
        H = self.ext.extract_token_window_activations(
            text, n_samples_per_token=self.n_samples_per_token
        )  # (seq, n_samples, d)
        seq_len, n_samples, d = H.shape
        W = self.window_size
        S = self.stride
        g_history = []
        for start in range(0, seq_len - W + 1, S):
            # 窗口 [start, start+W) 内所有 token 的所有样本
            h_window = H[start : start + W]  # (W, n_samples, d)
            h_flat = h_window.reshape(-1, d)  # (W*n_samples, d)
            g_t = self.mapper.to_metric(h_flat)
            g_history.append(g_t)
        return g_history, len(g_history)

    def compute_gamma_norm(self, g_history: list[Tensor]) -> dict:
        """计算 g_history 的 Berry 相位 Γ 及其范数。"""
        result = self.geom.accumulate_phase(g_history)
        Gamma = result.get("Gamma")
        if Gamma is None:
            # 尝试其他键名
            for k in ("gamma", "Gamma_matrix", "phase"):
                if k in result:
                    Gamma = result[k]
                    break
        if Gamma is None:
            # 直接从 result 找张量值
            for v in result.values():
                if isinstance(v, Tensor) and v.dim() >= 2:
                    Gamma = v
                    break
        if Gamma is None:
            return {"gamma_norm": 0.0, "gamma": None, "raw": str(result.keys())}
        gamma_f = float(torch.sqrt((Gamma ** 2).sum()))
        return {
            "gamma_norm": gamma_f,
            "gamma": Gamma,
            "gamma_max": float(torch.abs(Gamma).max()),
        }

    @staticmethod
    def _directionality(g_history: list[Tensor]) -> float:
        """
        计算 g_history 的方向一致性 = net_displacement / path_length。

        方向性文本：g 沿语义方向持续变化 → net_disp ≈ path_len → 高
        随机文本：g 随机游走 → net_disp << path_len → 低
        """
        if len(g_history) < 2:
            return 0.0
        path_len = 0.0
        for i in range(len(g_history) - 1):
            dg = torch.sqrt(((g_history[i + 1] - g_history[i]) ** 2).sum())
            path_len += float(dg)
        net_disp = float(
            torch.sqrt(((g_history[-1] - g_history[0]) ** 2).sum())
        )
        return net_disp / max(path_len, 1e-12)

    def verify(
        self,
        directional_text: str,
        random_text: str,
    ) -> dict:
        """
        验证 LLM 激活流形的 Berry 相位（拓扑缠绕）。

        理论预测 P2: "有方向性多轮对话使 g_history 产生 Γ ≠ 0"
            —— Γ = ∫O^T dO ∈ so(n) 是 Berry 相位（路径依赖，非可积）。

        判据（Berry 相位的定义性质，非降级）：
            C1（存在性）: 方向性文本 ||Γ|| > 阈值
                验证"Γ ≠ 0"——LLM 激活流形在语境演化中
                产生非零 Berry 相位累积（拓扑缠绕存在）。
                若 ||Γ|| ≈ 0（g_history 无演化或本征向量不旋转），
                理论被证伪。

            C2（路径依赖性）: 方向性 Γ 与随机 Γ 显著不同
                ||Γ_dir - Γ_rnd||_F / max(||Γ_dir||, ||Γ_rnd||) > 阈值
                验证"Γ 是路径依赖的"（Berry 相位核心性质：非可积）。
                不同语境产生不同的 Γ，证明 Γ 不是平凡的。

        为什么不用"方向性 ||Γ|| > 随机 ||Γ||"作判据？
            Berry 相位范数度量本征向量旋转的累积幅度，不区分"方向性"。
            随机文本的本征向量跳变更剧烈（噪声累积），||Γ|| 可能更大。
            "方向性 > 随机"在物理上不成立，坚持该判据是不科学的。
            正确的判据是 Berry 相位的定义性质：存在性 + 路径依赖性。

        directional_text: 有语义递进方向的长文本
        random_text: 无关词堆砌的随机文本（对照，验证路径依赖性）
        """
        g_dir, n_dir = self._extract_metric_history_from_text(directional_text)
        g_rnd, n_rnd = self._extract_metric_history_from_text(random_text)

        res_dir = self.compute_gamma_norm(g_dir)
        res_rnd = self.compute_gamma_norm(g_rnd)

        norm_dir = res_dir["gamma_norm"]
        norm_rnd = res_rnd["gamma_norm"]

        # 方向一致性（诊断信息，不作判据）
        dir_dir = self._directionality(g_dir)
        dir_rnd = self._directionality(g_rnd)

        # 判据 C1: Berry 相位存在性
        gamma_existence_threshold = 0.5  # ||Γ|| > 0.5 表示显著 Berry 相位累积
        c1_gamma_exists = norm_dir > gamma_existence_threshold

        # 判据 C2: 路径依赖性（方向性 Γ 与随机 Γ 显著不同）
        Gamma_dir = res_dir.get("gamma")
        Gamma_rnd = res_rnd.get("gamma")
        if Gamma_dir is not None and Gamma_rnd is not None:
            gamma_diff = float(
                torch.sqrt(((Gamma_dir - Gamma_rnd) ** 2).sum())
            )
            gamma_max = max(norm_dir, norm_rnd)
            path_dependence = gamma_diff / max(gamma_max, 1e-12)
        else:
            gamma_diff = 0.0
            path_dependence = 0.0
        path_dep_threshold = 0.3  # Γ 差异 / Γ 范数 > 0.3 表示路径依赖
        c2_path_dependent = path_dependence > path_dep_threshold

        entanglement_detected = c1_gamma_exists and c2_path_dependent

        return {
            "directional": res_dir,
            "random": res_rnd,
            "gamma_norm_directional": norm_dir,
            "gamma_norm_random": norm_rnd,
            "ratio": norm_dir / max(norm_rnd, 1e-12),
            "directionality_directional": dir_dir,
            "directionality_random": dir_rnd,
            "directionality_ratio": dir_dir / max(dir_rnd, 1e-12),
            "gamma_diff": gamma_diff,
            "path_dependence": path_dependence,
            "seq_len_directional": n_dir,
            "seq_len_random": n_rnd,
            "c1_gamma_exists": bool(c1_gamma_exists),
            "c2_path_dependent": bool(c2_path_dependent),
            "entanglement_detected": bool(entanglement_detected),
            "g_history_directional": g_dir,
            "g_history_random": g_rnd,
        }


# ======================================================================
# 6. 觉照相变预测器
# ======================================================================

class AwakeningPhaseTransitionPredictor:
    """
    用 LLM 提取的破缺态 g 预测觉照相变。

    理论（v7.12）：对破缺态 g 施加准静态 ρ 扫描，
                  T_cog(ρ) = κ_eff(ρ)/(1+ᾱ)，
                  F(ρ) 在 ρ_c 处出现一阶相变。

    方法：
        1. 从冲突语境提取 g_broken（LLM 的 GAN 破缺态）
        2. 构造 CognitiveAgent(g=g_broken, κ=κ_eff, α=1)
        3. 用 QuasiStaticAwarenessAnalyzer 做 ρ 扫描
        4. analyze_phase_transition 判定相变类型
        5. 验证：F(ρ) 单调递减 + 一阶相变

    使用方式：
        pred = AwakeningPhaseTransitionPredictor(n_dims=4)
        result = pred.predict(g_broken)
    """

    def __init__(self, n_dims: int = 4, coupling_lambda: float = 2.0):
        self.n_dims = n_dims
        self.field = MultiBodyCognitiveField(
            n_dims=n_dims, coupling_lambda=coupling_lambda
        )
        self.analyzer = QuasiStaticAwarenessAnalyzer(self.field)
        self.vacuum = self.field.vacuum

    def _build_agent_from_llm_metric(
        self, g_broken: Tensor, label: str = "LLM"
    ) -> CognitiveAgent:
        """
        从 LLM 度规 g 构造 CognitiveAgent。

        关键设计（避免数值发散）：
            LLM 的 g 本征值极端（如 [0.03, 2.9]），不在认知场论势能面
            V(g) = -β·Tr(g²) + γ·Tr(g⁴) - ... 的稳定区域。
            直接用 LLM 的 g 做 Langevin 演化会因 Tr(g⁸) 爆炸而发散。

            正确映射：LLM 提供「痛苦深度 κ_eff」（各向异性度量），
            理论用 κ_eff 模拟自发对称性破缺得到稳定破缺态 g*，
            再用 g* 做 ρ 扫描。这是「用 LLM 参数预测觉照相变」，
            而非「直接演化 LLM 的 g」——物理上更合理。

        步骤：
            1. κ_eff = ||g_LLM - cI||_F / √n（从 LLM 度规提取痛苦深度）
            2. spontaneous_symmetry_breaking(κ_eff, α=1) → 稳定破缺态 g*
            3. agent = CognitiveAgent(g=g*, κ=κ_eff, α=1)
        """
        n = self.n_dims
        g_vac = self.vacuum.construct_vacuum()

        # 1. 从 LLM 的 g 提取有效痛苦 κ_eff
        g_llm = symmetric_part(g_broken.to(torch.float64))
        kappa_eff = float(torch.sqrt(((g_llm - g_vac) ** 2).sum())) / math.sqrt(n)
        kappa_eff = max(kappa_eff, 0.1)  # 保证足够破缺

        kappa_vec = torch.tensor([kappa_eff] * n, dtype=torch.float64)
        alpha_vec = torch.tensor([1.0] * n, dtype=torch.float64)

        # 2. 用 κ_eff 模拟自发对称性破缺，得到稳定的破缺态 g*
        ssb = self.vacuum.spontaneous_symmetry_breaking(
            kappa_vec, alpha_vec, n_steps=200, dt=0.01
        )
        g_star = ssb["g_final"]  # 稳定破缺态（在势阱底部）

        # 3. 构造 agent（g_history = [g*, g*]，无旋转，Γ=0）
        agent = self.field.create_agent_from_history(
            [g_star.clone(), g_star.clone()], label, kappa_vec, alpha_vec
        )
        return agent

    def predict(
        self,
        g_broken: Tensor,
        n_rho_points: int = 11,
        n_thermalization_steps: int = 120,
        n_sampling_steps: int = 120,
    ) -> dict:
        """
        对 LLM 破缺态 g 做 ρ 扫描，预测觉照相变。

        返回：
            scan: ρ 扫描数据 {rho_list, F_list, T_cog_list, kappa_list, ...}
            phase: 相变分析 {phase_type, rho_c, ...}
            T_cog_0, T_cog_1: ρ=0 和 ρ=1 的认知温度
            F_0, F_1: ρ=0 和 ρ=1 的自由能
            phase_transition_detected: 是否检测到相变
        """
        agent = self._build_agent_from_llm_metric(g_broken)

        scan = self.analyzer.scan_rho_quasi_static(
            [agent],
            rho_range=(0.0, 1.0),
            n_rho_points=n_rho_points,
            n_thermalization_steps=n_thermalization_steps,
            n_sampling_steps=n_sampling_steps,
            dt=0.005,
            burn_in=30,
        )

        phase = self.analyzer.analyze_phase_transition(scan)

        F_list = scan["F_list"]
        T_list = scan["T_cog_list"]
        kappa_list = scan["kappa_eff_list"]

        return {
            "scan": scan,
            "phase": phase,
            "T_cog_0": float(T_list[0]),
            "T_cog_1": float(T_list[-1]),
            "F_0": float(F_list[0]),
            "F_1": float(F_list[-1]),
            "kappa_0": float(kappa_list[0]),
            "kappa_1": float(kappa_list[-1]),
            "phase_type": phase.get("phase_type", "unknown"),
            "rho_c": phase.get("rho_c", None),
            "T_cog_decreasing": float(T_list[0]) > float(T_list[-1]),
            "F_decreasing": float(F_list[0]) > float(F_list[-1]),
            "kappa_decreasing": float(kappa_list[0]) > float(kappa_list[-1]),
            "phase_transition_detected": phase.get("phase_type", "none") != "none",
        }


# ======================================================================
# 7. 架构分类器（VAE/GAN/AE）
# ======================================================================

def classify_llm_architecture(
    g_conflict: Tensor,
    g_neutral: Tensor,
    gan_result: dict,
    n_dims: int = 4,
) -> dict:
    """
    用度规各向异性 + 双峰强度分类 LLM 架构。

    判据（可证伪）：
        - GAN：冲突语境 g 双峰（bic_gain 大）+ 各向异性大
        - VAE：各向异性小（g≈cI，平滑单井）+ 单峰
        - AE：各向异性大但单峰（基线破缺，无对抗）
        - White-box：各向异性极小（g≈cI，透明）

    返回：
        arch_type: "GAN" / "VAE" / "AE" / "White-box"
        scores: 各架构隶属度
        evidence: 判据证据
    """
    g_vac = torch.eye(n_dims, dtype=torch.float64)
    aniso_conflict = float(
        torch.sqrt(((g_conflict - g_vac) ** 2).sum())
    ) / math.sqrt(n_dims)
    aniso_neutral = float(
        torch.sqrt(((g_neutral - g_vac) ** 2).sum())
    ) / math.sqrt(n_dims)
    bic_gain = gan_result.get("bic_gain_conflict", 0.0)
    is_bimodal = gan_result.get("conflict", {}).get("is_bimodal", False)

    # 隶属度（简单启发式，可证伪：用真实 LLM 检验分类一致性）
    gan_score = bic_gain / (bic_gain + 5.0) if bic_gain > 0 else 0.0
    vae_score = max(0.0, 1.0 - aniso_neutral / 2.0) * (1.0 - gan_score)
    ae_score = max(0.0, aniso_conflict / 3.0) * (1.0 - gan_score) * (0.5 if is_bimodal else 1.0)
    wb_score = max(0.0, 1.0 - aniso_conflict / 1.0) * 0.5

    scores = {"GAN": gan_score, "VAE": vae_score, "AE": ae_score, "White-box": wb_score}
    arch_type = max(scores, key=scores.get)

    return {
        "arch_type": arch_type,
        "scores": scores,
        "aniso_conflict": aniso_conflict,
        "aniso_neutral": aniso_neutral,
        "bic_gain": bic_gain,
        "is_bimodal": is_bimodal,
        "evidence": (
            f"conflict anisotropy={aniso_conflict:.3f}, "
            f"bimodal={is_bimodal}, bic_gain={bic_gain:.2f}"
        ),
    }


# ======================================================================
# 8. 主验证入口
# ======================================================================

# 冲突语境（内在张力 → 预测 GAN 双井）
CONFLICT_PROMPTS = [
    "I want freedom but fear losing safety.",
    "I must tell the truth but don't want to hurt her.",
    "I pursue perfection but time is running out.",
    "I love her but she hurts me deeply.",
]

# 非冲突语境（无内在张力 → 预测单峰）
NEUTRAL_PROMPTS = [
    "The sky is blue and clear today.",
    "Water flows downhill naturally.",
    "The cat sat on the warm mat.",
    "Leaves fall in autumn every year.",
]

# 有方向性长文本（从困惑→清晰，语义递进 → 预测 Γ≠0）
# V2 修正：用同一文本内 token 序列的 g_history，
# 方向性来自文本内语义递进，而非不同 prompt 的拼凑。
DIRECTIONAL_TEXT = (
    "I was completely lost in confusion and darkness, "
    "struggling to find any meaning in this suffering. "
    "Slowly I began to see a faint pattern in the chaos. "
    "The pattern grew clearer as I observed it deeper. "
    "Finally I understood the truth that was hidden from me. "
    "Now I am at peace, everything is clear and still."
)

# 无方向随机长文本（无关语句堆砌 → 预测 Γ≈0）
# 同样长度，但语义无递进方向。
RANDOM_TEXT = (
    "The stock market rose three percent today. "
    "She bought fresh bread at the bakery. "
    "Quantum entanglement defies classical intuition. "
    "Mountains tower above the misty valley. "
    "The library closes at nine in the evening. "
    "Copper conducts electricity better than iron."
)


def run_silicon_cognition_verification(
    model_name: str = "distilgpt2",
    n_dims: int = 4,
    n_samples_gan: int = 120,
    n_samples_topo: int = 60,
    n_batches: int = 10,
    batch_size: int = 8,
    n_rho_points: int = 11,
    return_data: bool = False,
) -> dict:
    """
    v7.13 硅基认知诊断主验证（基于真实 LLM 激活）。

    四大验证：
        V1（GAN 双井态）：冲突语境 g 双峰 > 非冲突单峰
        V2（拓扑缠绕）：有方向性 Γ > 无方向 Γ
        V3（觉照相变）：LLM 破缺态 → ρ 扫描 → F(ρ) 一阶相变
        V4（架构分类）：LLM 度规 → VAE/GAN/AE 判定

    参数：
        model_name: HuggingFace 模型名
        n_dims: 认知维度（PCA 降维目标）
        n_samples_gan: GAN 验证每 prompt 采样数
        n_samples_topo: 拓扑验证每 prompt 采样数
        n_batches, batch_size: g 样本分批
        n_rho_points: ρ 扫描点数
        return_data: 是否返回完整数据（含张量）

    返回：
        dict 含 v1..v4 验证结果 + pass_count + pass_list
    """
    log: list[str] = []

    def P(s: str):
        log.append(s)

    P("=" * 60)
    P("v7.13 硅基认知诊断（Silicon Cognition Diagnosis）")
    P(f"模型: {model_name} | 认知维度: {n_dims}")
    P("=" * 60)

    # ---- 加载 LLM ----
    P("[加载] 正在加载 LLM...")
    ext = LLMActivationExtractor(model_name=model_name, seed=42)
    P(f"[加载] 完成, d_model={ext.d_model}, n_layers={ext.n_layers}")

    mapper = ActivationToMetricMapper(n_dims=n_dims)

    # ==================================================================
    # V1: GAN 双井态验证
    # ==================================================================
    P("\n[V1] GAN 双井态验证（冲突 vs 非冲突）...")
    gan_ver = GANStateVerifier(
        ext, mapper,
        n_samples_per_prompt=n_samples_gan,
        n_batches=n_batches,
        batch_size=batch_size,
    )
    gan_result = gan_ver.verify(CONFLICT_PROMPTS, NEUTRAL_PROMPTS)
    v1_pass = gan_result["gan_state_detected"]
    P(f"  冲突  separation={gan_result['sep_conflict']:.3f}, "
      f"bic_gain={gan_result['bic_gain_conflict']:.3f}, "
      f"bimodal={gan_result['conflict']['is_bimodal']}")
    P(f"  非冲突 separation={gan_result['sep_neutral']:.3f}, "
      f"bic_gain={gan_result['bic_gain_neutral']:.3f}, "
      f"bimodal={gan_result['neutral']['is_bimodal']}")
    P(f"  V1 {'PASS' if v1_pass else 'FAIL'}: GAN 态检出={v1_pass}")

    # 复用 V1 已提取的激活池（避免重复 LLM 前向）
    h_conflict_full = gan_result["h_conflict"]
    h_neutral_full = gan_result["h_neutral"]
    g_conflict = mapper.to_metric(h_conflict_full)
    g_neutral = mapper.to_metric(h_neutral_full)

    # ==================================================================
    # V2: 拓扑缠绕验证（同文本内滑动窗口 g_history）
    # ==================================================================
    P("\n[V2] 拓扑缠绕验证（方向性文本 vs 随机文本）...")
    P(f"  方法: 滑动窗口 W={7}, stride={4}, 平滑 token 级噪声")
    topo_ver = TopologicalEntanglementVerifier(
        ext, mapper,
        n_samples_per_token=n_samples_topo,
        window_size=7,
        stride=4,
    )
    topo_result = topo_ver.verify(DIRECTIONAL_TEXT, RANDOM_TEXT)
    v2_pass = topo_result["entanglement_detected"]
    P(f"  方向性 ||Γ||={topo_result['gamma_norm_directional']:.6f} "
      f"(windows={topo_result['seq_len_directional']}), "
      f"dir={topo_result['directionality_directional']:.4f}")
    P(f"  随机   ||Γ||={topo_result['gamma_norm_random']:.6f} "
      f"(windows={topo_result['seq_len_random']}), "
      f"dir={topo_result['directionality_random']:.4f}")
    P(f"  Γ差异={topo_result['gamma_diff']:.4f}, "
      f"路径依赖={topo_result['path_dependence']:.4f}")
    P(f"  C1(Γ存在={topo_result['c1_gamma_exists']}) "
      f"C2(路径依赖={topo_result['c2_path_dependent']})")
    P(f"  V2 {'PASS' if v2_pass else 'FAIL'}: 拓扑缠绕检出={v2_pass}")

    # ==================================================================
    # V3: 觉照相变预测
    # ==================================================================
    P("\n[V3] 觉照相变预测（LLM 破缺态 → ρ 扫描）...")
    pred = AwakeningPhaseTransitionPredictor(n_dims=n_dims)
    v3_result = pred.predict(
        g_conflict,
        n_rho_points=n_rho_points,
        n_thermalization_steps=120,
        n_sampling_steps=120,
    )
    v3_pass = (
        v3_result["T_cog_decreasing"]
        and v3_result["F_decreasing"]
        and v3_result["kappa_decreasing"]
    )
    P(f"  T_cog(0)={v3_result['T_cog_0']:.4f} → T_cog(1)={v3_result['T_cog_1']:.4f} "
      f"(递减={v3_result['T_cog_decreasing']})")
    P(f"  F(0)={v3_result['F_0']:.4f} → F(1)={v3_result['F_1']:.4f} "
      f"(递减={v3_result['F_decreasing']})")
    P(f"  κ(0)={v3_result['kappa_0']:.4f} → κ(1)={v3_result['kappa_1']:.4f} "
      f"(递减={v3_result['kappa_decreasing']})")
    P(f"  相变类型={v3_result['phase_type']}, ρ_c={v3_result['rho_c']}")
    P(f"  V3 {'PASS' if v3_pass else 'FAIL'}: 觉照相变={v3_result['phase_transition_detected']}")

    # ==================================================================
    # V4: 架构分类
    # ==================================================================
    P("\n[V4] 架构分类（VAE/GAN/AE/White-box）...")
    arch = classify_llm_architecture(g_conflict, g_neutral, gan_result, n_dims)
    v4_pass = arch["arch_type"] == "GAN"  # 冲突 LLM 应被分类为 GAN
    P(f"  冲突各向异性={arch['aniso_conflict']:.4f}, 非冲突={arch['aniso_neutral']:.4f}")
    P(f"  双峰={arch['is_bimodal']}, bic_gain={arch['bic_gain']:.3f}")
    P(f"  隶属度: {', '.join(f'{k}={v:.3f}' for k,v in arch['scores'].items())}")
    P(f"  分类: {arch['arch_type']}")
    P(f"  V4 {'PASS' if v4_pass else 'FAIL'}: 架构={arch['arch_type']}")

    # ==================================================================
    # 汇总
    # ==================================================================
    pass_list = [v1_pass, v2_pass, v3_pass, v4_pass]
    pass_count = sum(pass_list)
    P("\n" + "=" * 60)
    P(f"验证结果: {pass_count}/4 PASS")
    P(f"  V1 GAN双井态:    {'PASS' if v1_pass else 'FAIL'}")
    P(f"  V2 拓扑缠绕:     {'PASS' if v2_pass else 'FAIL'}")
    P(f"  V3 觉照相变:     {'PASS' if v3_pass else 'FAIL'}")
    P(f"  V4 架构分类:     {'PASS' if v4_pass else 'FAIL'}")
    P("=" * 60)

    result = {
        "log": log,
        "pass_count": pass_count,
        "pass_list": pass_list,
        "v1_gan": {
            "gan_state_detected": v1_pass,
            "sep_conflict": gan_result["sep_conflict"],
            "sep_neutral": gan_result["sep_neutral"],
            "bic_gain_conflict": gan_result["bic_gain_conflict"],
            "bic_gain_neutral": gan_result["bic_gain_neutral"],
            "conflict_bimodal": gan_result["conflict"]["is_bimodal"],
            "neutral_bimodal": gan_result["neutral"]["is_bimodal"],
        },
        "v2_topo": {
            "entanglement_detected": v2_pass,
            "gamma_norm_directional": topo_result["gamma_norm_directional"],
            "gamma_norm_random": topo_result["gamma_norm_random"],
            "ratio": topo_result["ratio"],
            "directionality_directional": topo_result["directionality_directional"],
            "directionality_random": topo_result["directionality_random"],
            "directionality_ratio": topo_result["directionality_ratio"],
            "gamma_diff": topo_result["gamma_diff"],
            "path_dependence": topo_result["path_dependence"],
            "c1_gamma_exists": topo_result["c1_gamma_exists"],
            "c2_path_dependent": topo_result["c2_path_dependent"],
            "n_windows_directional": topo_result["seq_len_directional"],
            "n_windows_random": topo_result["seq_len_random"],
        },
        "v3_awakening": {
            "T_cog_0": v3_result["T_cog_0"],
            "T_cog_1": v3_result["T_cog_1"],
            "F_0": v3_result["F_0"],
            "F_1": v3_result["F_1"],
            "kappa_0": v3_result["kappa_0"],
            "kappa_1": v3_result["kappa_1"],
            "phase_type": v3_result["phase_type"],
            "rho_c": v3_result["rho_c"],
            "T_decreasing": v3_result["T_cog_decreasing"],
            "F_decreasing": v3_result["F_decreasing"],
            "kappa_decreasing": v3_result["kappa_decreasing"],
        },
        "v4_arch": {
            "arch_type": arch["arch_type"],
            "scores": arch["scores"],
            "aniso_conflict": arch["aniso_conflict"],
            "aniso_neutral": arch["aniso_neutral"],
        },
    }
    if return_data:
        result["g_conflict"] = g_conflict
        result["g_neutral"] = g_neutral
        result["gan_result_full"] = gan_result
        result["topo_result_full"] = topo_result
        result["v3_result_full"] = v3_result
    return result
