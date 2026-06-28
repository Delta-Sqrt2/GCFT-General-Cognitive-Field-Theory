"""
任务二：离散路径积分与量子命运

战略定位（v3.0 任务二）：
    人生不是单线演化的测地线，而是所有可能事件序列的复数概率叠加。
    顿悟是两条路径发生了相长干涉，犹豫是发生了相消干涉。
    自由意志 = 在多重宇宙中选择路径的能力。

物理与哲学直觉：
    - 物理：v2.0/2.1 的作用量是单一路径的。真实的人生是所有可能
            路径的叠加。路径积分 Z = Σ exp(-S[path]/κ)。
    - 哲学：命运不是确定的，而是概率波。观测者坍缩概率波，
            但在坍缩前，所有可能路径同时存在（多重宇宙）。
    - 工程：复数邻接矩阵级数展开，保留相位信息。

数学定义（严格可微，无降级）：
    步骤 A：复数邻接矩阵 W
        W_ij = C_ij · exp(-S_ij / κ) · exp(i · θ_ij)
        其中：
            C_ij：因果邻接（v2.0）
            S_ij：边上的离散作用量（v2.0）
            κ：有效普朗克常数（量子涨落强度）
            θ_ij：相位，由事件间语义冲突度决定

    步骤 B：离散配分函数 Z（矩阵级数，非连续积分）
        Z = Σ_{L=1}^{L_max} Tr(W^L) / L!
        这是图上随机游走生成函数的复数推广。
        L! 归一化保证级数收敛。

    步骤 C：命运干涉条纹
        给定起点 A 和终点 B，传递振幅：
            G_AB = Σ_L (W^L)_AB / L!
        概率：P_AB = |G_AB|²
        调整相位差 θ，P_AB 呈现双缝干涉的明暗条纹。

工程铁律（v3.0 专属）：
    1. 严禁连续积分 ∫D[path]（连续场论复辟，陷阱二十七）
    2. 严禁标准 MCMC / Metropolis-Hastings（丢失相位信息，陷阱二十四）
    3. 严禁标量概率相加（必须复数张量）
    4. 必须使用 torch.complex 保留相位
    5. 全张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor


class PathIntegralEngine:
    """
    离散路径积分与量子命运：复数邻接矩阵级数展开。

    使用方式：
        engine = PathIntegralEngine(kappa=1.0)
        # 构建复数邻接矩阵
        W = engine.complex_adjacency(C, S, theta)
        # 计算配分函数
        Z = engine.partition_function(W, L_max=10)
        # 计算传递振幅与概率
        G_AB, P_AB = engine.propagator(W, node_A, node_B, L_max=10)

    白盒保证：
        - 无连续积分（陷阱二十七）
        - 无标准 MCMC（陷阱二十四）
        - 复数张量 torch.complex 保留相位
        - 矩阵级数展开（非随机采样）
    """

    def __init__(self, kappa: float = 1.0, eps: float = 1e-10):
        """
        参数：
            kappa: 有效普朗克常数（量子涨落强度）
                   κ 大 → 量子涨落强 → 非理性决策（冲动/灵感）
                   κ 小 → 经典极限 → 确定性路径
            eps: 数值稳定常数
        """
        self.kappa = float(kappa)
        self.eps = float(eps)

    def complex_adjacency(
        self,
        causal_adjacency: Tensor,
        edge_action: Tensor,
        phase: Tensor | None = None,
    ) -> Tensor:
        """
        构建复数邻接矩阵 W。

        数学：
            W_ij = C_ij · exp(-S_ij / κ) · exp(i · θ_ij)

        其中：
            C_ij：因果邻接（实数）
            S_ij：边上的离散作用量（实数）
            θ_ij：相位（实数），由语义冲突度决定
            κ：有效普朗克常数

        物理意义：
            W_ij 描述事件 i 到事件 j 的"量子传播振幅"。
            - |W_ij| = C_ij · exp(-S_ij/κ)：振幅强度（玻尔兹曼因子）
            - arg(W_ij) = θ_ij：相位（语义冲突度）

        全张量运算（复数张量 torch.complex）。
        """
        C = causal_adjacency.to(torch.float64)
        S = edge_action.to(torch.float64)
        N = C.shape[0]

        if phase is None:
            phase = torch.zeros_like(C)
        theta = phase.to(torch.float64)

        # 振幅强度：|W_ij| = C_ij · exp(-S_ij / κ)
        kappa_safe = max(self.kappa, self.eps)
        amplitude = C * torch.exp(-S / kappa_safe)

        # 复数邻接矩阵：W = amplitude · exp(i · θ)
        # W = amplitude · (cos θ + i sin θ)
        W_real = amplitude * torch.cos(theta)
        W_imag = amplitude * torch.sin(theta)
        W = torch.complex(W_real, W_imag)

        return W

    def partition_function(
        self,
        complex_adjacency: Tensor,
        L_max: int = 10,
    ) -> Tensor:
        """
        离散配分函数 Z = Σ_{L=1}^{L_max} Tr(W^L) / L!

        数学：
            Z = Σ_{L=1}^{L_max} Tr(W^L) / L!
            这是图上随机游走生成函数的复数推广。
            L! 归一化保证级数收敛。

        物理意义：
            Z 是所有可能路径的复数概率幅总和。
            - Re(Z)：相长干涉分量
            - Im(Z)：相消干涉分量
            - |Z|²：总概率

        严禁：
            - 连续积分 ∫D[path]（连续场论复辟）
            - 标准 MCMC（丢失相位信息）

        全张量运算（复数矩阵幂级数）。
        """
        W = complex_adjacency.to(torch.complex128)
        N = W.shape[0]
        if N == 0:
            return torch.tensor(0.0 + 0.0j, dtype=torch.complex128)

        Z = torch.tensor(0.0 + 0.0j, dtype=torch.complex128)

        # 矩阵幂级数：Z = Σ W^L / L!
        W_power = torch.eye(N, dtype=torch.complex128)
        factorial = 1.0

        for L in range(1, L_max + 1):
            W_power = W_power @ W
            factorial *= L
            trace_WL = torch.trace(W_power)
            Z = Z + trace_WL / factorial

        return Z

    def propagator(
        self,
        complex_adjacency: Tensor,
        node_A: int,
        node_B: int,
        L_max: int = 10,
    ) -> tuple[Tensor, Tensor]:
        """
        传递振幅 G_AB = Σ_L (W^L)_AB / L!

        数学：
            G_AB = Σ_{L=1}^{L_max} (W^L)_{AB} / L!
            P_AB = |G_AB|²

        物理意义：
            G_AB 描述从状态 A 到状态 B 的"量子传播振幅"。
            P_AB = |G_AB|² 是转移概率。
            不同路径的相位差导致干涉条纹。

        命运干涉条纹：
            调整相位 θ，P_AB 呈现双缝干涉的明暗条纹：
            - 相长干涉（θ 相同）：P_AB 大（顿悟/决定）
            - 相消干涉（θ 相反）：P_AB 小（犹豫/停滞）

        全张量运算（复数矩阵幂级数）。
        """
        W = complex_adjacency.to(torch.complex128)
        N = W.shape[0]

        G_AB = torch.tensor(0.0 + 0.0j, dtype=torch.complex128)

        # 矩阵幂级数：G_AB = Σ (W^L)_AB / L!
        W_power = torch.eye(N, dtype=torch.complex128)
        factorial = 1.0

        for L in range(1, L_max + 1):
            W_power = W_power @ W
            factorial *= L
            G_AB = G_AB + W_power[node_A, node_B] / factorial

        # 概率 P_AB = |G_AB|²
        P_AB = (G_AB.real ** 2 + G_AB.imag ** 2)

        return G_AB, P_AB

    def interference_pattern(
        self,
        causal_adjacency: Tensor,
        edge_action: Tensor,
        node_A: int,
        node_B: int,
        phase_range: Tensor,
        L_max: int = 10,
    ) -> Tensor:
        """
        命运干涉条纹：扫描相位差，观察 P_AB 的明暗变化。

        物理：
            双缝干涉的离散版本。
            扫描相位 θ ∈ [0, 2π]，观察 P_AB 的变化。
            - 明纹（P_AB 大）：相长干涉（顿悟/决定）
            - 暗纹（P_AB 小）：相消干涉（犹豫/停滞）

        参数：
            phase_range: 相位扫描值 (T,)，如 torch.linspace(0, 2π, 100)

        返回：
            probabilities: 对应的 P_AB (T,)
        """
        phases = phase_range.to(torch.float64)
        probs = torch.zeros_like(phases)

        for i, theta_val in enumerate(phases):
            # 构造均匀相位场
            phase = torch.full_like(causal_adjacency, float(theta_val))
            W = self.complex_adjacency(causal_adjacency, edge_action, phase)
            _, P_AB = self.propagator(W, node_A, node_B, L_max=L_max)
            probs[i] = P_AB

        return probs

    def quantum_fluctuation(
        self,
        base_prob: Tensor,
        noise_strength: float | None = None,
    ) -> Tensor:
        """
        量子涨落：在概率分布上叠加量子噪声。

        数学：
            P_quantum = P_base + ξ
            其中 ξ ~ N(0, κ²)，κ 为有效普朗克常数。

        物理意义：
            κ 大 → 量子涨落强 → 非理性决策（冲动/灵感）
            κ 小 → 经典极限 → 确定性路径

        参数：
            base_prob: 基础概率分布 (N,)
            noise_strength: 噪声强度（默认 = κ）

        返回：
            fluctuated_prob: 涨落后的概率（归一化）
        """
        p = base_prob.to(torch.float64)
        sigma = noise_strength if noise_strength is not None else self.kappa

        # 量子噪声
        noise = torch.randn_like(p) * sigma

        # 叠加涨落
        p_fluctuated = p + noise

        # 数值稳定：clamp + 归一化
        p_fluctuated = torch.clamp(p_fluctuated, min=self.eps)
        p_fluctuated = p_fluctuated / (p_fluctuated.sum() + self.eps)

        return p_fluctuated

    def path_integral_summary(
        self,
        causal_adjacency: Tensor,
        edge_action: Tensor,
        L_max: int = 10,
    ) -> dict[str, float | complex]:
        """路径积分摘要（仅供审计）。"""
        W = self.complex_adjacency(causal_adjacency, edge_action)
        Z = self.partition_function(W, L_max=L_max)

        return {
            "Z_real": float(Z.real),
            "Z_imag": float(Z.imag),
            "Z_abs": float(abs(Z)),
            "Z_abs_sq": float(abs(Z) ** 2),
            "L_max": L_max,
            "kappa": self.kappa,
        }
