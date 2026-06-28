"""
任务一+二：三阶势阱势能面（v4.9 般若项）—— -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)

战略定位（v4.9）：
    v4.8 修正了符号错误并排除零模后，所有参数点都显示 VAE 相态（stability < 0）。
    但缺乏 GAN 相态和多态共存。
    v4.8 的哲学诊断：只有"戒"（γ·Tr(g⁴)）和"定"（α 调节），缺乏"慧"。

    v4.9 引入"般若"项 -δ·Tr(g⁶)，完成"戒定慧"三学的数学结构：
        V(g) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)
        其中 δ = κ·α / (1 + κ·α) 是般若参数。

    般若项的物理意义：
        - 在 g 大时（我执强时）产生"反稳定"效应
        - 打破"我执势阱"，使系统从 VAE（我执稳定）转向 GAN（动态解脱）
        - 对应佛教"般若波罗蜜"——智慧度到彼岸

    陷阱七十一·般若项符号错误：
        严禁使用 +δ·Tr(g⁶)（过度稳定化）。
        必须使用 -δ·Tr(g⁶)（产生反稳定）。

    陷阱七十二·般若参数硬编码：
        严禁硬编码 δ 值。必须从 κ·α 推导。

物理与哲学直觉：
    - 物理：三阶势能 V = -β·x² + γ·x⁴ - δ·x⁶ 在 δ 足够大时，
            Hessian 可从正（VAE）变为负（GAN），产生相变。
            判别式 Δ = 16(γ² - 3δβ)：
              γ² > 3δβ → 两个实根 → 多态共存可能
              γ² = 3δβ → 临界点 → 相变边界
              γ² < 3δβ → 无实根 → 只有 g=0（失稳）
    - 哲学：δ = κ·α/(1+κ·α) 表示"痛苦与觉知的耦合产生般若"。
            单纯痛苦（κ）无觉知 → 无般若 → 纯 VAE
            单纯觉知（α）无痛苦 → 无般若 → 纯 VAE
            痛苦 + 觉知 → 般若涌现 → 可能产生 GAN
    - 工程：继承 DoubleWellPotential，仅新增 δ 和 Tr(g⁶) 计算。

数学定义：
    β(κ) = κ / (1 + κ)          # 我执深度
    γ(α) = 1 / (2 * (α + 1))    # 中道约束
    δ(κ,α) = κ·α / (1 + κ·α)   # 般若参数

    三阶势阱项（v4.9）：
        V(g) = -β · Tr(g²) + γ · Tr(g⁴) - δ · Tr(g⁶)

    修正作用量：
        S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm_v2
            - β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)

    关键性质：
        1. δ=0（κ=0 或 α=0）时退化为 v4.8
        2. δ→1（κ·α→∞）时般若饱和
        3. 当 γ² < 3δβ 时，平衡点消失，系统失稳（GAN 相态）

    平衡点分析：
        ∇V = -2β·g + 4γ·g³ - 6δ·g⁵ = 0
        6δ·x² - 4γ·x + 2β = 0（x = g²）
        x = (2γ ± 2·sqrt(γ² - 3δβ)) / (6δ)

        Hessian at 平衡点：
        H = -2β + 12γ·x - 30δ·x²
        当 δ 足够大时，H < 0 → stability > 0 → GAN 相态
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part
from .double_well_potential import DoubleWellPotential


class TripleWellPotential(DoubleWellPotential):
    """
    三阶势阱势能面（v4.9 般若项）。

    使用方式：
        twp = TripleWellPotential(n_dims=4, n_events=8)
        C = twp.build_causal_adjacency(timestamps)
        result = twp.corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=1.0)
        # result 包含三阶势阱项的贡献

    白盒保证：
        - δ(κ,α) 从 κ·α 推导，严禁硬编码（陷阱七十二防范）
        - 般若项符号为 -δ·Tr(g⁶)，严禁 +δ·Tr(g⁶)（陷阱七十一防范）
        - 当 κ=0 或 α=0 时退化为 v4.8
        - 零模排除由 v4.8 修正后的 max_real_eigenvalue 保证（陷阱七十三防范）
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # δ(κ,α) 推导
    # ==================================================================

    @staticmethod
    def compute_delta(kappa: float, alpha: float) -> float:
        """
        δ(κ,α) = κ·α / (1 + κ·α)

        物理意义：
            - δ 是般若参数（打破我执势阱的强度）
            - κ（痛苦深度）与 α（认知刚度）的耦合产生般若
            - κ=0 或 α=0 → δ=0（无般若，退化为 v4.8）
            - κ·α → ∞ → δ→1（般若饱和）

        边界条件：
            - κ=0: δ=0（无痛苦，无般若）
            - α=0: δ=0（无觉知，无般若）
            - κ·α→∞: δ→1（般若饱和）

        单调性：
            - κ 增大 → δ 增大（痛苦深 → 般若强）
            - α 增大 → δ 增大（认知刚 → 般若强）
        """
        kappa = float(kappa)
        alpha = float(alpha)
        ka = kappa * alpha
        return ka / (1.0 + ka)

    # ==================================================================
    # 三阶势阱项计算
    # ==================================================================

    def compute_triple_well(
        self,
        g_batch: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor | float]:
        """
        计算三阶势阱项 V(g) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)（v4.9）。

        数学：
            Tr(g²) = Σ_i Tr(g_i · g_i)
            Tr(g⁴) = Σ_i Tr(g_i⁴)
            Tr(g⁶) = Σ_i Tr(g_i⁶)

            V(g) = -β · Tr(g²) + γ · Tr(g⁴) - δ · Tr(g⁶)

        v4.9 般若项：
            -δ·Tr(g⁶) 在 g 大时主导，产生"反稳定"效应。
            当 γ² < 3δβ 时，平衡点消失，系统失稳（GAN 相态）。

        参数：
            g_batch: (N, d, d) 度规张量
            kappa: 曲率耦合常数（决定 β 和 δ）
            alpha: 梯度耦合常数（决定 γ 和 δ）

        返回：
            dict 包含三阶势阱项及其分量
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)
        delta = self.compute_delta(kappa, alpha)

        # Tr(g²), Tr(g⁴), Tr(g⁶)
        trace_g2 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g4 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g6 = torch.zeros(1, dtype=torch.float64)[0]

        for i in range(N):
            g_i = g[i]
            g2 = g_i @ g_i  # g²
            g4 = g2 @ g2    # g⁴
            g6 = g4 @ g2    # g⁶
            trace_g2 = trace_g2 + torch.trace(g2)
            trace_g4 = trace_g4 + torch.trace(g4)
            trace_g6 = trace_g6 + torch.trace(g6)

        # 三阶势阱项（v4.9：-β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)）
        well_term = -beta * trace_g2 + gamma * trace_g4 - delta * trace_g6

        # 判别式 Δ = 16(γ² - 3δβ)
        discriminant = gamma * gamma - 3.0 * delta * beta

        return {
            "well_term": well_term,
            "trace_g2": trace_g2,
            "trace_g4": trace_g4,
            "trace_g6": trace_g6,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "discriminant": discriminant,
            "has_real_roots": discriminant > 0,  # γ² > 3δβ → 有实根
            "attractive_term": -beta * trace_g2,   # 我执项
            "repulsive_term": gamma * trace_g4,    # 中道项
            "wisdom_term": -delta * trace_g6,      # 般若项
        }

    # ==================================================================
    # 修正作用量（含三阶势阱项）
    # ==================================================================

    def corrected_action(
        self,
        g_batch: Tensor,
        L_or_C: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor]:
        """
        修正作用量（v4.9 含三阶势阱项）。

        数学：
            S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm_v2
                - β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶)

            其中：
            - β = κ / (1 + κ)
            - γ = 1 / (2 * (α + 1))
            - δ = κ·α / (1 + κ·α)

        v4.9 般若项：
            -δ·Tr(g⁶) 是"慧"的数学表达，打破我执势阱。

        参数：
            L_or_C: 因果邻接矩阵 C（v4.5+ 模式）
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        C = L_or_C

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

        # === 一阶图梯度项（v4.5 新形式）===
        alpha_val = float(alpha)
        grad_term_v2 = self.compute_grad_term_v2(g, C)
        gradient_term = alpha_val * grad_term_v2

        # === 三阶势阱项（v4.9 新增）===
        well_result = self.compute_triple_well(g, kappa, alpha)
        well_term = well_result["well_term"]

        # === 总作用量 ===
        action = curvature_term + coupling_term + gradient_term + well_term

        return {
            "action": action,
            "curvature_term": curvature_term,
            "coupling_term": coupling_term,
            "gradient_term": gradient_term,
            "well_term": well_term,
            "spectral_curvature": N_curvature,
            "grad_term_raw": grad_term_v2,
            "beta": well_result["beta"],
            "gamma": well_result["gamma"],
            "delta": well_result["delta"],
            "trace_g2": well_result["trace_g2"],
            "trace_g4": well_result["trace_g4"],
            "trace_g6": well_result["trace_g6"],
            "discriminant": well_result["discriminant"],
            "has_real_roots": well_result["has_real_roots"],
        }

    # ==================================================================
    # 理论平衡点计算
    # ==================================================================

    def compute_theoretical_equilibrium(
        self,
        kappa: float,
        alpha: float,
    ) -> dict[str, float]:
        """
        计算理论平衡点 g* = sqrt(x*) * I。

        数学：
            6δ·x² - 4γ·x + 2β = 0
            x = (2γ ± 2·sqrt(γ² - 3δβ)) / (6δ)

            较小根（稳定平衡点）：
            x* = (2γ - 2·sqrt(γ² - 3δβ)) / (6δ)
            g* = sqrt(x*) * I

        返回：
            dict 包含平衡点信息
        """
        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)
        delta = self.compute_delta(kappa, alpha)

        discriminant = gamma * gamma - 3.0 * delta * beta

        if discriminant < 0:
            # 无实根，系统失稳
            return {
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "discriminant": discriminant,
                "has_equilibrium": False,
                "g_star": 0.0,
                "hessian": 0.0,
                "expected_stability": float('inf'),  # 失稳
            }

        sqrt_disc = discriminant ** 0.5

        # 较小根（稳定平衡点）
        if delta > 1e-12:
            x_star = (2 * gamma - 2 * sqrt_disc) / (6 * delta)
        else:
            # δ=0，退化为 v4.8：x* = β/(2γ)
            x_star = beta / (2 * gamma)

        if x_star < 0:
            x_star = 0.0

        g_star = x_star ** 0.5

        # Hessian at 平衡点：H = -2β + 12γ·x - 30δ·x²
        hessian = -2 * beta + 12 * gamma * x_star - 30 * delta * x_star * x_star

        return {
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "discriminant": discriminant,
            "has_equilibrium": True,
            "x_star": x_star,
            "g_star": g_star,
            "hessian": hessian,
            "expected_stability": -hessian,  # stability = -H/τ，τ>0
        }

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_delta_boundary(self) -> dict[str, float | bool]:
        """
        验证 δ(κ,α) 的边界条件和单调性。
        """
        # 边界条件
        delta_k0 = self.compute_delta(0.0, 1.0)
        delta_a0 = self.compute_delta(1.0, 0.0)
        delta_inf = self.compute_delta(1000.0, 1000.0)

        # 单调性验证
        kappa_values = [0.1, 1.0, 5.0, 10.0]
        alpha_values = [0.1, 1.0, 10.0, 100.0]

        deltas_k = [self.compute_delta(k, 1.0) for k in kappa_values]
        deltas_a = [self.compute_delta(1.0, a) for a in alpha_values]

        delta_k_increasing = all(deltas_k[i] < deltas_k[i+1] for i in range(len(deltas_k)-1))
        delta_a_increasing = all(deltas_a[i] < deltas_a[i+1] for i in range(len(deltas_a)-1))

        return {
            "delta_at_kappa_0": delta_k0,
            "delta_at_alpha_0": delta_a0,
            "delta_at_inf": delta_inf,
            "delta_k_increasing": delta_k_increasing,
            "delta_a_increasing": delta_a_increasing,
            "all_pass": (
                abs(delta_k0) < 1e-10 and
                abs(delta_a0) < 1e-10 and
                delta_inf > 0.99 and
                delta_k_increasing and
                delta_a_increasing
            ),
        }

    def verify_degeneracy_to_v48(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
    ) -> dict[str, float | bool]:
        """
        验证：κ=0 或 α=0 时退化为 v4.8。

        v4.9 退化条件：
            κ=0 → δ=0，三阶势阱退化为双势阱
            α=0 → δ=0，三阶势阱退化为双势阱
        """
        # v4.9 在 κ=0 时
        result_v49_k0 = self.corrected_action(g_batch, C, phi, None, kappa=0.0, alpha=1.0)
        S_v49_k0 = result_v49_k0["action"]

        # v4.9 在 α=0 时
        result_v49_a0 = self.corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=0.0)
        S_v49_a0 = result_v49_a0["action"]

        # v4.8 在 κ=0 时（通过父类调用）
        result_v48_k0 = super().corrected_action(g_batch, C, phi, None, kappa=0.0, alpha=1.0)
        S_v48_k0 = result_v48_k0["action"]

        # v4.8 在 α=0 时
        result_v48_a0 = super().corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=0.0)
        S_v48_a0 = result_v48_a0["action"]

        diff_k0 = float(S_v49_k0 - S_v48_k0)
        diff_a0 = float(S_v49_a0 - S_v48_a0)

        return {
            "S_v49_k0": float(S_v49_k0),
            "S_v48_k0": float(S_v48_k0),
            "diff_k0": diff_k0,
            "S_v49_a0": float(S_v49_a0),
            "S_v48_a0": float(S_v48_a0),
            "diff_a0": diff_a0,
            "is_consistent_k0": abs(diff_k0) < 1e-10,
            "is_consistent_a0": abs(diff_a0) < 1e-10,
            "is_consistent": abs(diff_k0) < 1e-10 and abs(diff_a0) < 1e-10,
        }
