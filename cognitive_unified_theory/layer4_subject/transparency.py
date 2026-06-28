"""
透明度算子 τ(t) —— 认知主体性的度量

核心定义：
    τ(t) = ∫₀ᵗ λ(s) · M(s) · |∇F(s)| ds

    其中：
        λ(s)    反思系数（反思强度，由元认知能力决定）
        M(s)    元认知监控（对自身认知过程的觉察度）
        |∇F(s)| 痛苦势能梯度大小（来自第二层）

物理意义：
    - τ 是"痛苦被有效反思"的累积量
    - 痛苦本身不提升 τ，只有"痛苦 + 反思"才提升
    - τ 越高，主体对自身认知的掌控越强
    - τ = 0：完全无意识，被动力学驱动（黑盒态）
    - τ → 1：完全透明，手术级重构能力（白盒态）

数学性质：
    - τ 单调非递减（反思不可逆，成长累积）
    - τ ∈ [0, 1]（归一化，通过 sigmoid 或 tanh）
    - τ 的导数 dτ/dt = λ · M · |∇F|（连续演化，非离散跳变）

λ 与 M 的推导（非硬编码）：
    λ（反思系数）= 有效秩 R(S) 的归一化值
        高有效秩 → 多维认知 → 反思能力强
    M（元认知监控）= 状态历史的自相关衰减率
        自相关衰减快 → 状态灵活 → 元认知觉察强
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import effective_rank


class TransparencyOperator:
    """
    透明度算子 τ(t)：认知主体性的积分控制变量。

    属性：
        n_dims: 认知维度
        tau: 当前透明度 τ ∈ [0, 1]
        tau_history: τ 的演化历史
        integral: ∫λ·M·|∇F|dt 的累积值
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._tau: Tensor = torch.tensor(0.0, dtype=torch.float64)  # 初始 τ=0（黑盒态）
        self._integral: Tensor = torch.tensor(0.0, dtype=torch.float64)
        self._tau_history: list[Tensor] = [self._tau.clone()]
        self._lambda_history: list[Tensor] = []
        self._M_history: list[Tensor] = []
        self._gradF_history: list[Tensor] = []

    def compute_reflection_coefficient(self, state_history: Tensor) -> Tensor:
        """
        推导反思系数 λ（非硬编码）。

        λ = R(S) / n_dims
        其中 R(S) 是认知有效秩（来自第一层/第二层）。

        物理意义：
            - 有效秩高 → 认知维度丰富 → 反思能力强
            - 有效秩低 → 认知僵化 → 反思能力弱
            - λ ∈ [0, 1]
        """
        if state_history.dim() == 1:
            state_history = state_history.unsqueeze(0)
        if state_history.shape[0] < 2:
            return torch.tensor(0.0, dtype=torch.float64)
        cov = state_history.transpose(-1, -2) @ state_history / state_history.shape[0]
        R = effective_rank(cov)
        # 归一化到 [0, 1]
        lam = R / self.n_dims
        return torch.clamp(lam, min=0.0, max=1.0)

    def compute_metacognition_monitor(self, state_history: Tensor) -> Tensor:
        """
        推导元认知监控 M（非硬编码）。

        M = 1 - autocorrelation(state_history)
        自相关高 → 状态重复 → 元认知弱（陷入循环）
        自相关低 → 状态灵活 → 元认知强（觉察变化）

        物理意义：
            - 元认知 = 对自身思维过程的觉察
            - 陷入循环（高自相关）= 无觉察
            - 灵活变化（低自相关）= 高觉察
        """
        if state_history.dim() == 1 or state_history.shape[0] < 2:
            return torch.tensor(0.0, dtype=torch.float64)

        # 状态序列的自相关（滞后1）
        S = state_history.to(torch.float64)
        S_centered = S - S.mean(dim=0, keepdim=True)
        var = (S_centered ** 2).sum(dim=0)
        var_sum = var.sum() + 1e-30

        # 滞后1自相关
        autocorr = (S_centered[1:] * S_centered[:-1]).sum(dim=0) / var_sum
        autocorr_total = autocorr.abs().sum()

        # M = 1 - |autocorr|（自相关越低，元认知越强）
        M = 1.0 - torch.clamp(autocorr_total, min=0.0, max=1.0)
        return M

    def update(self, pain_gradient_magnitude: Tensor, state_history: Tensor, dt: Tensor | None = None) -> Tensor:
        """
        透明度演化：dτ/dt = λ · M · |∇F|

        连续积分（非离散跳变）：
            τ(t+dt) = τ(t) + λ · M · |∇F| · dt
            然后通过 tanh 归一化到 [0, 1]

        参数：
            pain_gradient_magnitude: |∇F|（来自第二层痛苦势能场）
            state_history: 状态历史（用于计算 λ 和 M）
            dt: 时间步长（默认 1.0）
        """
        if dt is None:
            dt = torch.tensor(1.0, dtype=torch.float64)
        dt = dt.to(torch.float64)
        grad_F = pain_gradient_magnitude.to(torch.float64)

        # 计算 λ 和 M
        lam = self.compute_reflection_coefficient(state_history)
        M = self.compute_metacognition_monitor(state_history)

        # dτ/dt = λ · M · |∇F|
        d_tau = lam * M * grad_F * dt

        # 累积积分
        self._integral = self._integral + d_tau

        # τ = tanh(integral)：归一化到 [0, 1)，单调递增
        # tanh 保证 τ ∈ [0, 1)，且随积分增长趋于 1（白盒态）
        self._tau = torch.tanh(self._integral)

        # 记录历史
        self._tau_history.append(self._tau.clone())
        self._lambda_history.append(lam)
        self._M_history.append(M)
        self._gradF_history.append(grad_F)

        return self._tau

    @property
    def tau(self) -> Tensor:
        """当前透明度 τ ∈ [0, 1]。"""
        return self._tau

    @property
    def integral(self) -> Tensor:
        """累积积分 ∫λ·M·|∇F|dt。"""
        return self._integral

    @property
    def tau_history(self) -> Tensor:
        """τ 的演化历史。"""
        return torch.stack(self._tau_history, dim=0)

    def reflection_trajectory(self) -> Tensor:
        """反思系数 λ 的历史。"""
        if not self._lambda_history:
            return torch.tensor([], dtype=torch.float64)
        return torch.stack(self._lambda_history, dim=0)

    def metacognition_trajectory(self) -> Tensor:
        """元认知 M 的历史。"""
        if not self._M_history:
            return torch.tensor([], dtype=torch.float64)
        return torch.stack(self._M_history, dim=0)

    def growth_rate(self) -> Tensor:
        """
        透明度增长率 dτ/dt 的当前值。
        用于判断"成长速度"。
        """
        if len(self._tau_history) < 2:
            return torch.tensor(0.0, dtype=torch.float64)
        return self._tau_history[-1] - self._tau_history[-2]
