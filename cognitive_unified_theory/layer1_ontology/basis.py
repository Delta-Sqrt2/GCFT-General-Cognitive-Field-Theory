"""
认知基底向量 Ξ —— 希尔伯特空间 H_cog 的正交基

物理对应：
    心智的"基本粒子"不是神经元，而是认知维度。每个 ξ_i 是一个独立的
    认知测量轴，正交性保证它们不可互相替代（威胁感知≠时间贴现）。

实现：
    通过 QR 分解从初始随机矩阵生成正交基。正交性是数学要求，不是经验假设。
    基底维度对应认知理论中的基本维度：
        ξ1 威胁感知灵敏度  (threat sensitivity)
        ξ2 时间贴现率      (temporal discounting)
        ξ3 自我指涉权重    (self-reference weight)
        ξ4 意义连贯性      (meaning coherence)
        ξ5 不确定性容忍度  (uncertainty tolerance)
        ξ6 关系依恋安全度  (attachment security)
        ξ7 行动效能感      (agency efficacy)
        ξ8 价值清晰度      (value clarity)

注意：这些"标签"仅用于人类理解，运算中不出现任何标签语义。
      所有计算在抽象张量空间进行，标签不参与方程。
"""

from __future__ import annotations

import torch
from torch import Tensor

# 认知维度的语义标签（仅供人类阅读，不参与任何计算）
_COGNITIVE_DIM_LABELS = [
    "threat_sensitivity",      # ξ1 威胁感知灵敏度
    "temporal_discounting",    # ξ2 时间贴现率
    "self_reference_weight",   # ξ3 自我指涉权重
    "meaning_coherence",       # ξ4 意义连贯性
    "uncertainty_tolerance",   # ξ5 不确定性容忍度
    "attachment_security",     # ξ6 关系依恋安全度
    "agency_efficacy",         # ξ7 行动效能感
    "value_clarity",           # ξ8 价值清晰度
]


class CognitiveBasis:
    """
    认知基底 Ξ：希尔伯特空间 H_cog 的正交基。

    数学性质：
        Ξ^T Ξ = I  （正交归一）
        dim(H_cog) = n_dims

    生成方式：
        QR 分解。Q 的列向量构成正交基。
        这是数学构造，不依赖任何经验数据。
    """

    def __init__(self, n_dims: int = 8, seed: int | None = None):
        """
        参数：
            n_dims: 认知维度数（默认 8，对应八大基本认知轴）
            seed: 随机种子（可复现性，非物理参数）
        """
        if n_dims < 3:
            raise ValueError("认知维度至少为 3（威胁/时间/自我指涉）")
        self.n_dims = n_dims
        self.seed = seed
        self._basis: Tensor | None = None
        self._labels = _COGNITIVE_DIM_LABELS[:n_dims] if n_dims <= len(_COGNITIVE_DIM_LABELS) else [f"xi_{i+1}" for i in range(n_dims)]

    @property
    def labels(self) -> list[str]:
        """维度标签（仅供人类阅读）。"""
        return list(self._labels)

    def initialize(self, generator: torch.Generator | None = None) -> Tensor:
        """
        通过 QR 分解生成正交基。

        数学：
            给定随机矩阵 A ∈ R^{n×n}，QR 分解 A = QR
            Q 的列向量 {q_1, ..., q_n} 构成 H_cog 的正交基 Ξ。

        物理意义：
            正交性保证各认知维度独立测量，无冗余。
            QR 分解是构造性证明，不依赖经验。
        """
        if generator is None:
            g = torch.Generator(device="cpu")
            if self.seed is not None:
                g.manual_seed(self.seed)
        else:
            g = generator

        # 高斯随机矩阵：各向同性，无偏
        A = torch.randn(self.n_dims, self.n_dims, generator=g, dtype=torch.float64)
        # QR 分解
        Q, R = torch.linalg.qr(A)
        # 保证数值正定性：调整符号使 R 对角为正
        signs = torch.sign(torch.diagonal(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.unsqueeze(0)

        self._basis = Q
        return Q

    @property
    def basis(self) -> Tensor:
        """正交基矩阵 Ξ ∈ R^{n×n}，每列是一个基向量 ξ_i。"""
        if self._basis is None:
            self.initialize()
        return self._basis

    def project(self, state: Tensor) -> Tensor:
        """
        将状态投影到认知基底：S_basis = Ξ^T S
        物理意义：测量状态在各认知维度上的分量。
        """
        return self.basis.transpose(-1, -2) @ state.to(torch.float64)

    def reconstruct(self, coords: Tensor) -> Tensor:
        """
        从基底坐标重建状态：S = Ξ S_basis
        """
        return self.basis @ coords.to(torch.float64)

    def orthogonality_check(self) -> Tensor:
        """
        验证正交性：Ξ^T Ξ 应为单位矩阵。
        返回与单位矩阵的偏差（应为 ~0）。
        """
        I = torch.eye(self.n_dims, dtype=torch.float64)
        return ((self.basis.transpose(-1, -2) @ self.basis) - I).abs().max()

    def __repr__(self) -> str:
        return f"CognitiveBasis(n_dims={self.n_dims}, orthogonal_error={self.orthogonality_check():.2e})"
