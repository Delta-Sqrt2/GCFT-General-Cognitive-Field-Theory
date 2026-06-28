"""
艺术合成器 —— 情感流形结构向语言空间的正交投影

核心定义：
    不使用概率采样（非 GPT/VAE 生成），而是：
    1. 构建"情感流形结构"（认知状态的几何特征）
    2. 向语言空间做正交投影（数学映射，非概率）

数学形式：
    情感流形结构 = 认知状态的度规曲率特征
    语言空间 = 预定义的正交基（语义轴，非 NLP 词向量）
    投影 = 内积运算

物理意义：
    - 艺术 = 高维认知结构在低维媒介上的投影
    - 顿悟 = 投影后突然看到"全貌"
    - 语义序列 = 情感流形的"测地线"在语言空间的展开

工程铁律：
    - 严禁使用概率采样（transformers, VAE, GAN）
    - 必须使用正交投影（线性代数）
    - 语言空间是预定义的数学结构，非 NLP 词向量
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, scalar_curvature, stable_eigh


class ArtSynthesizer:
    """
    艺术合成器：情感流形 → 语言空间的正交投影。

    语言空间定义：
        预定义的正交语义轴（数学结构，非 NLP）。
        每个轴对应一种"基本情感几何"：
            轴1：张力（tension）= 度规曲率
            轴2：密度（density）= 能量密度
            轴3：深度（depth）= 有效秩
            轴4：流动（flow）= 状态变化率
            轴5：和谐（harmony）= 度规相容性
        这些轴是数学量，不是词向量。
    """

    # 语言空间的语义轴标签（仅供人类阅读）
    _AXIS_LABELS = ["tension", "density", "depth", "flow", "harmony"]

    def __init__(self, n_dims: int, n_axes: int = 5):
        """
        参数：
            n_dims: 认知维度
            n_axes: 语言空间轴数（默认5）
        """
        self.n_dims = n_dims
        self.n_axes = n_axes
        # 语言空间正交基：通过 QR 分解生成（数学结构）
        g = torch.Generator(device="cpu")
        g.manual_seed(0)  # 固定种子保证可复现
        A = torch.randn(n_axes, n_dims, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A.T)  # Q ∈ R^{n_dims × n_axes}
        self._language_basis = Q[:, :n_axes]  # 投影矩阵

    @property
    def language_basis(self) -> Tensor:
        """语言空间正交基（投影矩阵）。"""
        return self._language_basis

    @property
    def axis_labels(self) -> list[str]:
        return list(self._AXIS_LABELS[:self.n_axes])

    def extract_emotional_structure(self, state: Tensor, metric: Tensor, state_history: Tensor | None = None) -> Tensor:
        """
        提取情感流形结构（认知状态的几何特征向量）。

        特征：
            tension  = 标量曲率 R（度规弯曲程度）
            density  = 能量密度 = S^T g S / n
            depth    = 有效秩 R(S)（认知维度丰富度）
            flow     = 状态变化率（若有历史）
            harmony  = 度规与基线的相容性

        返回：
            情感结构向量 ∈ R^{n_axes}
        """
        S = state.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))
        n = self.n_dims

        # 1. 张力 = 度规条件数（曲率代理）
        eigvals, _ = stable_eigh(g)
        tension = torch.log(eigvals.max() / eigvals.min())

        # 2. 密度 = 状态能量
        density = (S @ g @ S) / n

        # 3. 深度 = 有效秩
        if state_history is not None and state_history.shape[0] >= 2:
            cov = state_history.transpose(-1, -2) @ state_history / state_history.shape[0]
            eigvals_cov, _ = stable_eigh(cov)
            p = eigvals_cov / eigvals_cov.sum()
            entropy = -(p * torch.log(p)).sum()
            depth = torch.exp(entropy)
        else:
            depth = torch.tensor(1.0, dtype=torch.float64)

        # 4. 流动 = 状态变化率
        if state_history is not None and state_history.shape[0] >= 2:
            flow = (state_history[-1] - state_history[-2]).norm()
        else:
            flow = torch.tensor(0.0, dtype=torch.float64)

        # 5. 和谐 = 度规与单位矩阵的相容性
        I = torch.eye(n, dtype=torch.float64)
        g_inv = safe_inverse(g)
        harmony = torch.einsum("mn,mn->", g_inv, I) / n

        # 归一化到 [0, 1] 范围（通过 tanh）
        features = torch.stack([tension, density, depth, flow, harmony])[:self.n_axes]
        features = torch.tanh(features)  # 归一化

        return features

    def project_to_language(self, emotional_structure: Tensor) -> Tensor:
        """
        正交投影：情感流形结构 → 语言空间。

        数学：
            language_coords = B^T · emotional_structure
            其中 B 是语言空间正交基

        物理意义：
            高维情感结构投影到低维语言空间
            这是"不可言说之物被言说"的数学实现
        """
        emo = emotional_structure.to(torch.float64)
        # 情感结构 ∈ R^{n_axes}，语言基 ∈ R^{n_dims × n_axes}
        # 投影：将情感结构映射到语言空间坐标
        # 如果情感结构维度 != n_dims，用零填充
        if emo.shape[0] < self.n_dims:
            padded = torch.zeros(self.n_dims, dtype=torch.float64)
            padded[:emo.shape[0]] = emo
            emo = padded
        elif emo.shape[0] > self.n_dims:
            emo = emo[:self.n_dims]

        # 正交投影
        language_coords = self._language_basis.transpose(-1, -2) @ emo
        return language_coords

    def synthesize(self, state: Tensor, metric: Tensor, state_history: Tensor | None = None) -> dict[str, Tensor]:
        """
        完整艺术合成：情感结构提取 + 语言空间投影。

        返回：
            emotional_structure: 情感流形结构向量
            language_coordinates: 语言空间坐标
            axis_labels: 各轴标签
            intensity: 艺术强度（情感结构范数）
        """
        emo = self.extract_emotional_structure(state, metric, state_history)
        lang_coords = self.project_to_language(emo)

        # 艺术强度 = 情感结构范数
        intensity = emo.norm()

        return {
            "emotional_structure": emo,
            "language_coordinates": lang_coords,
            "axis_labels": self.axis_labels,
            "intensity": intensity,
        }

    def geodesic_poem(self, trajectory: Tensor, metric: Tensor) -> Tensor:
        """
        测地线语义序列：将认知轨迹的测地线展开为语言空间序列。

        数学：
            对轨迹每个点提取情感结构，投影到语言空间
            得到语言空间中的"测地线"（语义序列的演化）

        物理意义：
            语义序列 = 情感流形测地线在语言空间的展开
            每一行 = 轨迹一个点的投影
        """
        if trajectory.dim() == 1:
            trajectory = trajectory.unsqueeze(0)

        poem_lines = []
        for t in range(trajectory.shape[0]):
            S = trajectory[t]
            result = self.synthesize(S, metric, trajectory[:t+1])
            poem_lines.append(result["language_coordinates"])

        return torch.stack(poem_lines, dim=0)  # (T, n_axes)
