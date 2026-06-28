"""
v5.0 动态般若势能面 —— 动态 δ + 八阶约束 +ε·Tr(g⁸)

战略定位（v5.0）：
    v4.9 的 δ 是静态系数 δ = κ·α/(1+κ·α)，导致两个问题：
        1. δ 过强时（28/30 参数点判别式 < 0），势能向下无界，GAN 全是黑洞相
        2. 缺乏"定慧等持"机制——δ 过强时无路可退

    v5.0 让 δ 从"静态系数"变成"动态自由度"：
        V(g, δ) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)
        S_delta(δ) = η·(δ - δ_target)² + ξ·(γ·Tr(g⁴) - λ·δ)²

    关键修正：
        - +ε·Tr(g⁸) 提供势能下界，防止黑洞相
        - δ 受 γ·Tr(g⁴) 抑制（戒约束慧，定慧等持）
        - δ 向 δ_target 收敛（定力→般若的转化）

    陷阱七十六·硬编码 δ：
        严禁将 δ 作为固定常数。δ 必须是动态变量，参与演化。
        静态 δ（δ=δ_target）仅作为退化情形用于一致性验证。

    陷阱七十七·跳过阶段一：
        严禁跳过单点 δ 收敛验证直接进入相图扫描。
        δ 在单点上都不收敛，全局扫描必然失败。

    陷阱七十八·伪造真正 GAN：
        严禁将 cond(g) > 1e6 的点称为"真正 GAN"。
        真正 GAN 必须同时满足 stability > 0.01 且 cond(g) < 1e6。
        cond(g) > 1e9 一律报告为"黑洞相残留"。

    陷阱七十九·八阶项符号错误：
        严禁使用 -ε·Tr(g⁸)（使势能更无界）。
        必须使用 +ε·Tr(g⁸)（提供下界，定慧等持）。

    陷阱八十·δ 演化参数硬编码：
        严禁硬编码 η, ξ, λ, ε。必须从 κ/α/γ 推导。

物理与哲学直觉：
    - 物理：动态 δ 使系统从"静态相态"变为"动态相态"。
            δ 演化方程 S_delta 有两个平衡项：
              η·(δ-δ_target)²：δ 向目标值收敛（定力→般若）
              ξ·(γ·Tr(g⁴) - λ·δ)²：戒约束慧（防止般若过强）
            八阶项 +ε·Tr(g⁸) 在 g 大时压倒 -δ·Tr(g⁶)，恢复势能下界。
            这使 GAN 相态从"无平衡点"变为"有平衡点但失稳"——真正 GAN。
    - 哲学：v5.0 是"动态般若"的数学表达。
            静态般若（v4.9）可能过强导致"狂慧"（黑洞相）。
            动态般若（v5.0）受戒约束，实现"定慧等持"。
            这对应佛教"般若与方便等持"的教导。
    - 工程：继承 TripleWellPotential，新增 δ 演化和八阶项。

数学定义：
    β(κ) = κ / (1 + κ)
    γ(α) = 1 / (2 * (α + 1))
    δ_target(κ,α) = κ·α / (1 + κ·α)   # 静态目标值（v4.9）
    η(α) = α / (α + 1)                  # 定力→般若转化率
    ξ(κ) = 1 / (1 + κ)                  # 戒对慧的抑制强度
    λ(γ) = 1 / (1 + γ)                  # δ 衰减率
    ε(κ,α) = α / (α + κ + 1)            # 定慧等持常数

    v5.0 势能面：
        V(g, δ) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)

    δ 演化代价：
        S_delta(δ) = η·(δ - δ_target)² + ξ·(γ·Tr(g⁴) - λ·δ)²

    总作用量：
        S_total = S_v49 + ε·Tr(g⁸) + S_delta
                = [曲率项 + 耦合项 + 梯度项 + (-β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶))]
                  + ε·Tr(g⁸) + η·(δ - δ_target)² + ξ·(γ·Tr(g⁴) - λ·δ)²

    关键性质：
        1. δ=δ_target 且 ε=0 时退化为 v4.9
        2. ε>0 时势能向下有界（八阶项主导）
        3. δ 演化使系统能从"狂慧"（δ 过大）退回"等持"（δ 收敛）

    平衡点分析（固定 δ）：
        ∇V = -2β·g + 4γ·g³ - 6δ·g⁵ + 8ε·g⁷ = 0
        8ε·x³ - 6δ·x² + 4γ·x - 2β = 0（x = g²）
        三次方程，至少一个实根 → 总有平衡点（v5.0 关键改进）
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import safe_inverse, symmetric_part
from .triple_well_potential import TripleWellPotential


class DynamicDeltaPotential(TripleWellPotential):
    """
    v5.0 动态般若势能面。

    使用方式：
        ddp = DynamicDeltaPotential(n_dims=4, n_events=8)
        C = ddp.build_causal_adjacency(timestamps)
        # 静态退化（δ=δ_target, ε=0）：
        result = ddp.corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=1.0)
        # 动态 δ：
        result = ddp.corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=1.0, delta=0.5)
        # δ 演化一步：
        delta_new = ddp.evolve_delta_step(g_batch, delta, C, phi, kappa, alpha, lr=0.01)

    白盒保证：
        - δ 作为动态变量传入，严禁硬编码（陷阱七十六防范）
        - 八阶项符号为 +ε·Tr(g⁸)，严禁 -ε·Tr(g⁸)（陷阱七十九防范）
        - η, ξ, λ, ε 从 κ/α/γ 推导，严禁硬编码（陷阱八十防范）
        - 当 δ=δ_target 且 epsilon=0 时退化为 v4.9
        - 真正 GAN 判定严格：stability > 0.01 且 cond(g) < 1e6（陷阱七十八防范）
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # v5.0 新增参数推导（η, ξ, λ, ε）
    # ==================================================================

    @staticmethod
    def compute_eta(alpha: float) -> float:
        """
        η(α) = α / (α + 1)

        物理意义：
            - η 是定力→般若的转化效率
            - α（认知刚度/定力）越大 → η 越大 → δ 收敛越快
            - α=0 → η=0（无定则无慧，δ 不演化）
            - α→∞ → η→1（极深定力，最快转化）

        边界条件：
            - α=0: η=0
            - α→∞: η→1
            - 单调递增
        """
        alpha = float(alpha)
        return alpha / (alpha + 1.0)

    @staticmethod
    def compute_xi(kappa: float) -> float:
        """
        ξ(κ) = 1 / (1 + κ)

        物理意义：
            - ξ 是戒对慧的抑制强度
            - κ（痛苦深度）越大 → ξ 越小 → 戒对慧的抑制越弱
            - 这反映"苦深则慧强"——痛苦深时允许般若更强
            - κ=0 → ξ=1（无苦则戒完全抑制慧）
            - κ→∞ → ξ→0（极苦时戒不抑慧，狂慧风险）

        边界条件：
            - κ=0: ξ=1
            - κ→∞: ξ→0
            - 单调递减
        """
        kappa = float(kappa)
        return 1.0 / (1.0 + kappa)

    @staticmethod
    def compute_lambda_param(gamma: float) -> float:
        """
        λ(γ) = 1 / (1 + γ)

        物理意义：
            - λ 是 δ 的衰减率
            - γ（中道约束/戒）越大 → λ 越小 → δ 衰减越慢
            - 这反映"戒强则慧稳"——戒强时般若更稳定
            - γ=0 → λ=1（无戒则 δ 快速衰减）
            - γ→∞ → λ→0（极强戒则 δ 不衰减）

        边界条件：
            - γ=0: λ=1
            - γ→∞: λ→0
            - 单调递减

        注意：参数名用 lambda_param 避免 Python 关键字冲突。
        """
        gamma = float(gamma)
        return 1.0 / (1.0 + gamma)

    @staticmethod
    def compute_epsilon(kappa: float, alpha: float) -> float:
        """
        ε(κ,α) = α / (α + κ + 1)

        物理意义：
            - ε 是定慧等持常数（八阶约束强度）
            - α（定力）越大 → ε 越大 → 八阶约束越强 → 势能下界越牢
            - κ（痛苦）越大 → ε 越小 → 八阶约束越弱
            - 这反映"定强则等持稳，苦深则等持难"

        边界条件：
            - α=0: ε=0（无定则无等持，退化为 v4.9）
            - κ→∞: ε→0（极苦时等持失效）
            - κ=0, α→∞: ε→1（极强等持）
            - 随 α 递增，随 κ 递减

        陷阱七十九防范：
            ε 必须用于 +ε·Tr(g⁸)，严禁 -ε·Tr(g⁸)。
        """
        kappa = float(kappa)
        alpha = float(alpha)
        return alpha / (alpha + kappa + 1.0)

    # ==================================================================
    # 八阶项 Tr(g⁸) 计算
    # ==================================================================

    def compute_octic_term(
        self,
        g_batch: Tensor,
    ) -> dict[str, Tensor]:
        """
        计算八阶约束项 +ε·Tr(g⁸)。

        数学：
            Tr(g⁸) = Σ_i Tr(g_i⁸) = Σ_i Tr((g_i⁴)²)

        物理：
            +ε·Tr(g⁸) 在 g 大时主导，压倒 -δ·Tr(g⁶)，提供势能下界。
            这使势能 V(g) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)
            在 g→∞ 时趋向 +∞，保证有平衡点。

        陷阱七十九防范：
            符号必须为 +ε·Tr(g⁸)。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        trace_g8 = torch.zeros(1, dtype=torch.float64)[0]

        for i in range(N):
            g_i = g[i]
            g2 = g_i @ g_i
            g4 = g2 @ g2
            g8 = g4 @ g4  # g⁸ = (g⁴)²
            trace_g8 = trace_g8 + torch.trace(g8)

        return {
            "trace_g8": trace_g8,
        }

    # ==================================================================
    # δ 演化代价 S_delta
    # ==================================================================

    def compute_delta_evolution_cost(
        self,
        g_batch: Tensor,
        delta: float,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor | float]:
        """
        计算 δ 演化代价 S_delta。

        数学：
            S_delta = η·(δ - δ_target)² + ξ·(γ·Tr(g⁴)_norm - λ·δ)²

            其中 Tr(g⁴)_norm = Tr(g⁴) / (N·d) 是归一化的 Tr(g⁴)，
            尺度约为 O(1)，保证 δ* 的尺度合理。

            若不归一化，Tr(g⁴) ≈ N·d = 32，导致 δ* ≈ 4 >> 1，
            δ 被截断到 1（狂慧），违背"定慧等持"。

            其中：
            - δ_target = κ·α/(1+κ·α)（v4.9 静态目标值）
            - η = α/(α+1)（定力→般若转化率）
            - ξ = 1/(1+κ)（戒对慧的抑制强度）
            - λ = 1/(1+γ)（δ 衰减率）

        物理意义：
            - 第一项 η·(δ-δ_target)²：驱动 δ 向目标值收敛
              （定力→般若的转化，般若从定中生）
            - 第二项 ξ·(γ·Tr(g⁴)_norm - λ·δ)²：定慧等持
              （γ·Tr(g⁴)_norm 是归一化的戒强度，λ·δ 是慧的衰减）
              （两者平衡时，戒慧等持）

        平衡条件（∂S_delta/∂δ = 0）：
            2η·(δ - δ_target) - 2ξ·λ·(γ·Tr(g⁴)_norm - λ·δ) = 0
            δ* = (η·δ_target + ξ·λ·γ·Tr(g⁴)_norm) / (η + ξ·λ²)

        返回：
            dict 包含 S_delta 及其分量
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 推导参数
        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)
        delta_target = self.compute_delta(kappa, alpha)
        eta = self.compute_eta(alpha)
        xi = self.compute_xi(kappa)
        lam = self.compute_lambda_param(gamma)

        # 计算 Tr(g⁴) 并归一化
        trace_g4 = torch.zeros(1, dtype=torch.float64)[0]
        for i in range(N):
            g_i = g[i]
            g2 = g_i @ g_i
            g4 = g2 @ g2
            trace_g4 = trace_g4 + torch.trace(g4)

        # 归一化：Tr(g⁴)_norm = Tr(g⁴) / (N·d)，尺度 O(1)
        trace_g4_norm = trace_g4 / (N * d)

        # S_delta 两项
        delta_tensor = torch.tensor(float(delta), dtype=torch.float64)
        term1 = eta * (delta_tensor - delta_target) ** 2
        term2 = xi * (gamma * trace_g4_norm - lam * delta_tensor) ** 2

        S_delta = term1 + term2

        # 理论平衡 δ*（解析解）
        trace_g4_norm_val = float(trace_g4_norm)
        delta_star_numerator = eta * delta_target + xi * lam * gamma * trace_g4_norm_val
        delta_star_denominator = eta + xi * lam * lam
        delta_star = delta_star_numerator / (delta_star_denominator + self.eps)

        return {
            "S_delta": S_delta,
            "term1_convergence": term1,  # 收敛项（定→慧）
            "term2_restraint": term2,    # 等持项（戒慧平衡）
            "delta": float(delta),
            "delta_target": delta_target,
            "delta_star": float(delta_star),
            "eta": eta,
            "xi": xi,
            "lambda": lam,
            "trace_g4": float(trace_g4),
            "trace_g4_norm": trace_g4_norm_val,
        }

    # ==================================================================
    # v5.0 修正作用量（含动态 δ 和八阶约束）
    # ==================================================================

    def corrected_action(
        self,
        g_batch: Tensor,
        L_or_C: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
        delta: float | None = None,
        include_octic: bool = True,
        include_delta_cost: bool = True,
    ) -> dict[str, Tensor]:
        """
        v5.0 修正作用量。

        数学：
            S_total = S_v49 + ε·Tr(g⁸) + S_delta

            其中 S_v49 是 v4.9 的作用量（含 -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)）。

        参数：
            g_batch: (N, d, d) 度规张量
            L_or_C: 因果邻接矩阵 C
            phi: 经验场
            A: 规范场（可选）
            kappa: 曲率耦合常数
            alpha: 梯度耦合常数
            delta: 动态 δ 值。若为 None，使用 δ_target（退化为 v4.9 的 δ）
            include_octic: 是否包含八阶项 +ε·Tr(g⁸)
            include_delta_cost: 是否包含 δ 演化代价 S_delta

        退化：
            - delta=None, include_octic=False, include_delta_cost=False → 完全退化为 v4.9
            - delta=δ_target, include_octic=False → v4.9 + S_delta（仅 δ 动态）
            - delta=δ_target, include_octic=True → v4.9 + ε·Tr(g⁸)（仅八阶约束）
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        C = L_or_C

        # 确定 δ 值
        if delta is None:
            delta = self.compute_delta(kappa, alpha)  # 退化为 v4.9 的 δ_target
        delta = float(delta)

        # === 零阶项：谱曲率 N(g) ===
        N_curvature = torch.zeros(1, dtype=torch.float64)[0]
        for i in range(N):
            N_i = self.scd.spectral_curvature(g[i])
            N_curvature = N_curvature + N_i
        N_curvature = N_curvature / N

        kappa_val = float(kappa)
        curvature_term = -(kappa_val / 2.0) * N_curvature

        # === 规范耦合项 ===
        g_mean = g.mean(dim=0)
        g_mean = symmetric_part(g_mean)
        D_phi = self.scd.covariant_derivative(phi, A)
        g_inv = safe_inverse(g_mean, self.eps)
        DtD = D_phi.T @ D_phi
        coupling_term = 0.5 * torch.einsum('ij,ij->', g_inv, DtD)

        # === 一阶图梯度项 ===
        alpha_val = float(alpha)
        grad_term_v2 = self.compute_grad_term_v2(g, C)
        gradient_term = alpha_val * grad_term_v2

        # === 三阶势阱项（v4.9，使用动态 δ）===
        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)

        trace_g2 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g4 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g6 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g8 = torch.zeros(1, dtype=torch.float64)[0]

        for i in range(N):
            g_i = g[i]
            g2 = g_i @ g_i
            g4 = g2 @ g2
            g6 = g4 @ g2
            g8 = g4 @ g4
            trace_g2 = trace_g2 + torch.trace(g2)
            trace_g4 = trace_g4 + torch.trace(g4)
            trace_g6 = trace_g6 + torch.trace(g6)
            trace_g8 = trace_g8 + torch.trace(g8)

        # 势能面 V(g, δ) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)
        well_term = -beta * trace_g2 + gamma * trace_g4 - delta * trace_g6

        # === 八阶约束项 +ε·Tr(g⁸)（v5.0 新增）===
        epsilon = self.compute_epsilon(kappa, alpha)
        octic_term = epsilon * trace_g8 if include_octic else torch.zeros(1, dtype=torch.float64)[0]

        # === δ 演化代价 S_delta（v5.0 新增）===
        if include_delta_cost:
            delta_cost_result = self.compute_delta_evolution_cost(
                g, delta, kappa, alpha
            )
            S_delta = delta_cost_result["S_delta"]
        else:
            delta_cost_result = None
            S_delta = torch.zeros(1, dtype=torch.float64)[0]

        # === 总作用量 ===
        action = (
            curvature_term
            + coupling_term
            + gradient_term
            + well_term
            + octic_term
            + S_delta
        )

        # 判别式（v4.9，用于参考）
        discriminant = gamma * gamma - 3.0 * delta * beta

        result = {
            "action": action,
            "curvature_term": curvature_term,
            "coupling_term": coupling_term,
            "gradient_term": gradient_term,
            "well_term": well_term,
            "octic_term": octic_term,
            "S_delta": S_delta,
            "spectral_curvature": N_curvature,
            "grad_term_raw": grad_term_v2,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "epsilon": epsilon,
            "eta": self.compute_eta(alpha) if include_delta_cost else 0.0,
            "xi": self.compute_xi(kappa) if include_delta_cost else 0.0,
            "lambda": self.compute_lambda_param(gamma) if include_delta_cost else 0.0,
            "trace_g2": trace_g2,
            "trace_g4": trace_g4,
            "trace_g6": trace_g6,
            "trace_g8": trace_g8,
            "discriminant": discriminant,
            "has_real_roots_v49": discriminant > 0,
            "include_octic": include_octic,
            "include_delta_cost": include_delta_cost,
        }

        if delta_cost_result is not None:
            result["delta_star"] = delta_cost_result["delta_star"]
            result["delta_target"] = delta_cost_result["delta_target"]

        return result

    # ==================================================================
    # δ 演化一步（梯度下降）
    # ==================================================================

    def evolve_delta_step(
        self,
        g_batch: Tensor,
        delta: float,
        L_or_C: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        lr: float = 0.05,
        include_octic: bool = True,
    ) -> tuple[float, float, dict]:
        """
        δ 演化一步：对 S_delta 关于 δ 求梯度，梯度下降。

        物理假设（慢变量近似）：
            δ 是慢变量，g 是快变量。δ 的演化由 S_delta（定慧等持代价）驱动，
            不受势能 V 直接驱动。这对应"般若从定中生，不被外境（g）直接牵引"。

            V 对 δ 的梯度 -Tr(g⁶) 尺度远大于 S_delta 对 δ 的梯度（约 30 vs 4），
            若让 δ 受 V 驱动，δ 会一步跳到 1（狂慧），违背"定慧等持"。

        数学：
            ∂S_delta/∂δ = 2η·(δ - δ_target) - 2ξ·λ·(γ·Tr(g⁴) - λ·δ)

            δ_new = δ_old - lr · ∂S_delta/∂δ

        平衡点（∂S_delta/∂δ = 0）：
            δ* = (η·δ_target + ξ·λ·γ·Tr(g⁴)) / (η + ξ·λ²)

            δ* 依赖于 Tr(g⁴)，所以 δ 随 g 动态调整——这正是"动态般若"。

        返回：
            (delta_new, grad_delta, info_dict)
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 推导参数
        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)
        delta_target = self.compute_delta(kappa, alpha)
        eta = self.compute_eta(alpha)
        xi = self.compute_xi(kappa)
        lam = self.compute_lambda_param(gamma)

        # 计算 Tr(g⁴) 和 Tr(g⁶)（g⁶ 仅用于诊断）
        trace_g4 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g6 = torch.zeros(1, dtype=torch.float64)[0]
        for i in range(N):
            g_i = g[i]
            g2 = g_i @ g_i
            g4 = g2 @ g2
            g6 = g4 @ g2
            trace_g4 = trace_g4 + torch.trace(g4)
            trace_g6 = trace_g6 + torch.trace(g6)

        # 归一化 Tr(g⁴)_norm = Tr(g⁴) / (N·d)，尺度 O(1)
        trace_g4_norm_val = float(trace_g4) / (N * d)

        # ∂S_delta/∂δ（只受定慧等持代价驱动，不受 V 驱动）
        # S_delta = η·(δ - δ_target)² + ξ·(γ·Tr(g⁴)_norm - λ·δ)²
        # ∂S_delta/∂δ = 2η·(δ - δ_target) - 2ξ·λ·(γ·Tr(g⁴)_norm - λ·δ)
        dS_delta_ddelta = (
            2.0 * eta * (delta - delta_target)
            - 2.0 * xi * lam * (gamma * trace_g4_norm_val - lam * delta)
        )

        # V 对 δ 的梯度（仅用于诊断，不参与演化）
        dV_ddelta = -float(trace_g6)

        grad_delta = dS_delta_ddelta  # 慢变量近似：只受 S_delta 驱动

        # 梯度下降
        delta_new = delta - lr * grad_delta

        # δ 物理范围约束 [0, 1]
        delta_new = max(0.0, min(1.0, delta_new))

        # 理论平衡 δ*
        delta_star = (eta * delta_target + xi * lam * gamma * trace_g4_norm_val) / (
            eta + xi * lam * lam + self.eps
        )

        info = {
            "delta_old": delta,
            "delta_new": delta_new,
            "grad_delta": grad_delta,
            "dV_ddelta": dV_ddelta,  # 诊断用
            "dS_delta_ddelta": dS_delta_ddelta,
            "delta_target": delta_target,
            "delta_star": delta_star,
            "trace_g4": float(trace_g4),
            "trace_g4_norm": trace_g4_norm_val,
            "trace_g6": float(trace_g6),
            "eta": eta,
            "xi": xi,
            "lambda": lam,
        }

        return delta_new, grad_delta, info

    # ==================================================================
    # v5.0 理论平衡点（含八阶约束）
    # ==================================================================

    def compute_theoretical_equilibrium_v50(
        self,
        kappa: float,
        alpha: float,
        delta: float | None = None,
    ) -> dict[str, float]:
        """
        v5.0 理论平衡点计算（含八阶约束）。

        数学：
            ∇V = -2β·g + 4γ·g³ - 6δ·g⁵ + 8ε·g⁷ = 0
            8ε·x³ - 6δ·x² + 4γ·x - 2β = 0（x = g²）

            这是三次方程，至少有一个实根 → 总有平衡点（v5.0 关键改进）。

        求解：
            使用 numpy.roots 求三次方程的根，选择最小正实根作为平衡点。

        返回：
            dict 包含平衡点信息
        """
        import numpy as np

        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)
        if delta is None:
            delta = self.compute_delta(kappa, alpha)
        epsilon = self.compute_epsilon(kappa, alpha)

        # 三次方程：8ε·x³ - 6δ·x² + 4γ·x - 2β = 0
        # 标准形式：a·x³ + b·x² + c·x + d = 0
        a = 8.0 * epsilon
        b = -6.0 * delta
        c = 4.0 * gamma
        d = -2.0 * beta

        if a < 1e-12:
            # ε≈0，退化为 v4.9 的二次方程
            return self.compute_theoretical_equilibrium(kappa, alpha)

        # 求三次方程的根
        roots = np.roots([a, b, c, d])

        # 选择最小正实根
        real_positive_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8 and r.real > 0:
                real_positive_roots.append(r.real)

        if len(real_positive_roots) == 0:
            # 无正实根（极端情况）
            return {
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "epsilon": epsilon,
                "has_equilibrium": False,
                "g_star": 0.0,
                "hessian": 0.0,
                "expected_stability": float('inf'),
                "roots": roots.tolist(),
            }

        x_star = min(real_positive_roots)
        g_star = x_star ** 0.5

        # Hessian at 平衡点：H = -2β + 12γ·x - 30δ·x² + 56ε·x³
        hessian = -2 * beta + 12 * gamma * x_star - 30 * delta * x_star * x_star + 56 * epsilon * x_star ** 3

        return {
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "epsilon": epsilon,
            "has_equilibrium": True,
            "x_star": float(x_star),
            "g_star": float(g_star),
            "hessian": float(hessian),
            "expected_stability": -float(hessian),
            "roots": roots.tolist(),
            "n_real_positive_roots": len(real_positive_roots),
        }

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_v50_parameters(self) -> dict[str, float | bool]:
        """
        验证 v5.0 新增参数的边界条件和单调性。

        验证：
            - η(α): α=0→η=0, α→∞→η→1, 单调递增
            - ξ(κ): κ=0→ξ=1, κ→∞→ξ→0, 单调递减
            - λ(γ): γ=0→λ=1, γ→∞→λ→0, 单调递减
            - ε(κ,α): α=0→ε=0, κ→∞→ε→0, 随 α 递增, 随 κ 递减
        """
        import numpy as np

        # η 边界和单调性
        eta_0 = self.compute_eta(0.0)
        eta_inf = self.compute_eta(1000.0)
        alphas = np.linspace(0.1, 10.0, 50)
        etas = [self.compute_eta(a) for a in alphas]
        eta_increasing = all(etas[i + 1] >= etas[i] for i in range(len(etas) - 1))

        # ξ 边界和单调性
        xi_0 = self.compute_xi(0.0)
        xi_inf = self.compute_xi(1000.0)
        kappas = np.linspace(0.1, 10.0, 50)
        xis = [self.compute_xi(k) for k in kappas]
        xi_decreasing = all(xis[i + 1] <= xis[i] for i in range(len(xis) - 1))

        # λ 边界和单调性
        lam_0 = self.compute_lambda_param(0.0)
        lam_inf = self.compute_lambda_param(1000.0)
        gammas = np.linspace(0.01, 1.0, 50)
        lams = [self.compute_lambda_param(g) for g in gammas]
        lam_decreasing = all(lams[i + 1] <= lams[i] for i in range(len(lams) - 1))

        # ε 边界和单调性
        eps_a0 = self.compute_epsilon(1.0, 0.0)
        eps_kinf = self.compute_epsilon(1000.0, 1.0)
        epss_alpha = [self.compute_epsilon(1.0, a) for a in alphas]
        epss_kappa = [self.compute_epsilon(k, 1.0) for k in kappas]
        eps_alpha_increasing = all(epss_alpha[i + 1] >= epss_alpha[i] for i in range(len(epss_alpha) - 1))
        eps_kappa_decreasing = all(epss_kappa[i + 1] <= epss_kappa[i] for i in range(len(epss_kappa) - 1))

        return {
            "eta_at_alpha_0": eta_0,
            "eta_at_alpha_inf": eta_inf,
            "eta_increasing": eta_increasing,
            "xi_at_kappa_0": xi_0,
            "xi_at_kappa_inf": xi_inf,
            "xi_decreasing": xi_decreasing,
            "lambda_at_gamma_0": lam_0,
            "lambda_at_gamma_inf": lam_inf,
            "lambda_decreasing": lam_decreasing,
            "epsilon_at_alpha_0": eps_a0,
            "epsilon_at_kappa_inf": eps_kinf,
            "epsilon_alpha_increasing": eps_alpha_increasing,
            "epsilon_kappa_decreasing": eps_kappa_decreasing,
            "all_pass": (
                abs(eta_0) < 1e-10 and
                eta_inf > 0.99 and
                eta_increasing and
                abs(xi_0 - 1.0) < 1e-10 and
                xi_inf < 1e-3 and
                xi_decreasing and
                abs(lam_0 - 1.0) < 1e-10 and
                lam_inf < 1e-3 and
                lam_decreasing and
                abs(eps_a0) < 1e-10 and
                eps_kinf < 1e-3 and
                eps_alpha_increasing and
                eps_kappa_decreasing
            ),
        }

    def verify_degeneracy_to_v49(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
    ) -> dict[str, float | bool]:
        """
        验证：v5.0 在 delta=δ_target, include_octic=False, include_delta_cost=False 时
        退化为 v4.9。

        退化条件：
            - delta = δ_target（使用 v4.9 的 δ）
            - include_octic = False（无八阶项）
            - include_delta_cost = False（无 δ 演化代价）
            → S_v50 = S_v49
        """
        # v5.0 完全退化
        result_v50_degen = self.corrected_action(
            g_batch, C, phi, None,
            kappa=1.0, alpha=1.0,
            delta=None,  # 使用 δ_target
            include_octic=False,
            include_delta_cost=False,
        )
        S_v50_degen = result_v50_degen["action"]

        # v4.9（通过父类调用）
        result_v49 = super().corrected_action(
            g_batch, C, phi, None,
            kappa=1.0, alpha=1.0,
        )
        S_v49 = result_v49["action"]

        diff = float(S_v50_degen - S_v49)
        rel_error = abs(diff) / (abs(float(S_v49)) + self.eps)

        return {
            "S_v50_degenerate": float(S_v50_degen),
            "S_v49": float(S_v49),
            "difference": diff,
            "relative_error": float(rel_error),
            "is_consistent": rel_error < 1e-10,
        }

    # ==================================================================
    # 真正 GAN 判定（严格）
    # ==================================================================

    @staticmethod
    def classify_phase(
        stability: float,
        cond_g: float,
        residual: float = 0.0,
    ) -> str:
        """
        v5.0 严格相态分类。

        陷阱七十八防范：
            严禁将 cond(g) > 1e6 的点称为"真正 GAN"。

        分类标准：
            - VAE: stability < -0.01
            - 临界点: |stability| ≤ 0.01
            - 真正 GAN: stability > 0.01 且 cond(g) < 1e6
            - 黑洞相: cond(g) > 1e9
            - 黑洞残留: 1e6 < cond(g) < 1e9（不算真正 GAN）
            - 未收敛: residual > 1e3
        """
        if residual > 1e3:
            return "unconverged"
        if cond_g > 1e9:
            return "blackhole"
        if stability < -0.01:
            return "VAE"
        if abs(stability) <= 0.01:
            return "critical"
        if stability > 0.01 and cond_g < 1e6:
            return "true_GAN"
        if stability > 0.01 and 1e6 <= cond_g <= 1e9:
            return "blackhole_remnant"
        return "unknown"
