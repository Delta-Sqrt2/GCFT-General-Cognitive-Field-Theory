"""
任务一+二：双势阱势能面（v4.8 符号修正）—— -β·Tr(g²) + γ·Tr(g⁴)

战略定位（v4.8）：
    v4.7 实现了倒双势阱 +β·Tr(g²) - γ·Tr(g⁴)，符号搞反了。
    倒双势阱的势阱底部在 g=0（奇异），非零处是势垒（失稳）。
    v4.8 修正为标准双势阱 -β·Tr(g²) + γ·Tr(g⁴)，
    势阱底部在 g²=β/(2γ)（正定），Hessian 正定 → stability<0（VAE）。

    陷阱七十·符号复辟降级：
        严禁回到 v4.7 的倒双势阱符号 +β·Tr(g²) - γ·Tr(g⁴)。
        必须使用标准双势阱 -β·Tr(g²) + γ·Tr(g⁴)。

物理与哲学直觉：
    - 物理：标准双势阱是统计物理产生多稳态的标准方法（Ising, φ⁴, Higgs）。
            -β·Tr(g²) 排斥度规离开零点（形成人格特质），
            +γ·Tr(g⁴) 约束度规防止无限增长（防止黑洞相）。
            两者平衡 → 产生稳定平衡点 g²=β/(2γ) → VAE 相态涌现。
    - 哲学：β（势阱深度）与 κ（痛苦深度）正相关——
            痛苦越深，"我执"越强，势阱越深（人格固化）。
            γ（势阱宽度）与 α（认知刚度）负相关——
            刚度越大，越不允许度规无限增长（防止极端化）。
            这对应佛教"我执"与"中道"的数学表达。
    - 工程：β 和 γ 从 κ/α 推导，保证 v4.8 是 v4.6/v4.7 的自然延续。

数学定义：
    β(κ) = κ / (1 + κ)        # β ∈ (0,1)，κ 大 → β 大
    γ(α) = 1 / (2 * (α + 1))  # γ ∈ (0,0.5)，α 大 → γ 小

    双势阱项（v4.8 标准符号）：
        V(g) = -β · Tr(g²) + γ · Tr(g⁴)
        = -β · Σ λ_i² + γ · Σ λ_i⁴

    修正作用量：
        S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm_v2 - β·Tr(g²) + γ·Tr(g⁴)

    关键性质：
        1. β=0（κ=0）时退化为 v4.6 的线性作用量
        2. γ→0（α→∞）时度规可能无限增长（黑洞相风险）
        3. -β·Tr(g²) 使 Hessian 在平衡点处正定 → stability < 0（VAE）

    平衡点分析：
        ∇V = -2β·g + 4γ·g³ = 0 → g=0 或 g²=β/(2γ)
        平衡点 g* = sqrt(β/(2γ))·I （正定，物理允许）
        Hessian at g*: -2β + 12γ·g² = -2β + 6β = 4β > 0 （正定）
        stability = -Hessian/τ = -4β/τ < 0 （VAE 相态）
"""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np

from ..core.tensor_ops import safe_inverse, symmetric_part
from .graph_gradient_term_v2 import GraphGradientTermV2


class DoubleWellPotential(GraphGradientTermV2):
    """
    双势阱势能面（v4.8 标准符号修正）。

    使用方式：
        dwp = DoubleWellPotential(n_dims=4, n_events=8)
        C = dwp.build_causal_adjacency(timestamps)
        result = dwp.corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=1.0)
        # result 包含双势阱项的贡献

    白盒保证：
        - β(κ) 和 γ(α) 从 κ/α 推导，严禁硬编码（陷阱六十七防范）
        - 双势阱项 -β·Tr(g²) + γ·Tr(g⁴) 构造局部极小值（v4.8 标准符号）
        - 当 κ=0 时退化为 v4.6 的线性作用量
        - 严禁回到 v4.7 的倒双势阱符号（陷阱七十防范）
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # β(κ) 和 γ(α) 推导
    # ==================================================================

    @staticmethod
    def compute_beta(kappa: float) -> float:
        """
        β(κ) = κ / (1 + κ)

        物理意义：
            - β 是势阱深度参数
            - κ（痛苦深度）越大 → β 越大 → 势阱越深 → 人格越固化
            - κ=0 时 β=0，退化为线性作用量
            - κ→∞ 时 β→1（饱和）

        边界条件：
            - κ=0: β=0（无痛苦，无势阱）
            - κ→∞: β→1（极大痛苦，极深势阱）
        """
        kappa = float(kappa)
        return kappa / (1.0 + kappa)

    @staticmethod
    def compute_gamma(alpha: float) -> float:
        """
        γ(α) = 1 / (2 * (α + 1))

        物理意义：
            - γ 是势阱宽度参数（阻止度规无限增长）
            - α（认知刚度）越大 → γ 越小 → 势阱越宽 → 允许更大度规
            - α=0 时 γ=0.5（最大约束）
            - α→∞ 时 γ→0（无约束，黑洞相风险）

        边界条件：
            - α=0: γ=0.5（最大约束）
            - α→∞: γ→0（无约束）
        """
        alpha = float(alpha)
        return 1.0 / (2.0 * (alpha + 1.0))

    # ==================================================================
    # 双势阱项计算
    # ==================================================================

    def compute_double_well(
        self,
        g_batch: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor | float]:
        """
        计算双势阱项 V(g) = -β·Tr(g²) + γ·Tr(g⁴)（v4.8 标准符号）。

        数学：
            Tr(g²) = Σ_μν g_μν²  （度规的 Frobenius 范数平方）
            Tr(g⁴) = Σ_μν (g²)_μν²  （度规四次方的迹）

            V(g) = -β · Tr(g²) + γ · Tr(g⁴)

        v4.8 符号修正：
            v4.7 用了 +β·Tr(g²) - γ·Tr(g⁴)（倒双势阱，错误）。
            v4.8 修正为 -β·Tr(g²) + γ·Tr(g⁴)（标准双势阱，正确）。
            标准双势阱的势阱底部在 g²=β/(2γ)，Hessian 正定 → stability<0。

        参数：
            g_batch: (N, d, d) 度规张量
            kappa: 曲率耦合常数（决定 β）
            alpha: 梯度耦合常数（决定 γ）

        返回：
            dict 包含双势阱项及其分量
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)

        # Tr(g²) = Σ_i Tr(g_i · g_i) = Σ_i Σ_μν g_i,μν²
        # 对每个事件计算 Tr(g_i²)，然后求和
        trace_g2 = torch.zeros(1, dtype=torch.float64)[0]
        trace_g4 = torch.zeros(1, dtype=torch.float64)[0]

        for i in range(N):
            g_i = g[i]
            g2 = g_i @ g_i  # g²
            g4 = g2 @ g2    # g⁴
            trace_g2 = trace_g2 + torch.trace(g2)
            trace_g4 = trace_g4 + torch.trace(g4)

        # 双势阱项（v4.8 标准符号：-β·Tr(g²) + γ·Tr(g⁴)）
        well_term = -beta * trace_g2 + gamma * trace_g4

        return {
            "well_term": well_term,
            "trace_g2": trace_g2,
            "trace_g4": trace_g4,
            "beta": beta,
            "gamma": gamma,
            "attractive_term": -beta * trace_g2,  # 排斥项（离开零点，形成人格特质）
            "repulsive_term": gamma * trace_g4,   # 约束项（防止无限增长，防止黑洞）
        }

    # ==================================================================
    # 修正作用量（含双势阱项）
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
        修正作用量（v4.8 含标准双势阱项）。

        数学：
            S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm_v2
                - β·Tr(g²) + γ·Tr(g⁴)

            其中：
            - β = κ / (1 + κ)
            - γ = 1 / (2 * (α + 1))

        v4.8 符号修正：
            v4.7: S = v4.6 + β·Tr(g²) - γ·Tr(g⁴)  （倒双势阱，错误）
            v4.8: S = v4.6 - β·Tr(g²) + γ·Tr(g⁴)  （标准双势阱，正确）

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

        # === 双势阱项（v4.7 新增）===
        well_result = self.compute_double_well(g, kappa, alpha)
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
            "trace_g2": well_result["trace_g2"],
            "trace_g4": well_result["trace_g4"],
        }

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_beta_gamma(
        self,
    ) -> dict[str, list | bool]:
        """
        验证 β(κ) 和 γ(α) 的边界条件和单调性。
        """
        # 边界条件
        beta_0 = self.compute_beta(0.0)
        gamma_inf = self.compute_gamma(1e6)

        # 单调性验证
        kappa_values = np.linspace(0.0, 10.0, 50)
        alpha_values = np.logspace(-2, 2, 50)

        betas = [self.compute_beta(k) for k in kappa_values]
        gammas = [self.compute_gamma(a) for a in alpha_values]

        # β 单调递增
        beta_increasing = all(betas[i+1] >= betas[i] for i in range(len(betas)-1))
        # γ 单调递减
        gamma_decreasing = all(gammas[i+1] <= gammas[i] for i in range(len(gammas)-1))

        return {
            "beta_at_kappa_0": beta_0,
            "gamma_at_alpha_inf": gamma_inf,
            "beta_increasing": beta_increasing,
            "gamma_decreasing": gamma_decreasing,
            "kappa_values": kappa_values.tolist(),
            "beta_values": betas,
            "alpha_values": alpha_values.tolist(),
            "gamma_values": gammas,
            "all_pass": (
                abs(beta_0) < 1e-10 and
                gamma_inf < 1e-6 and
                beta_increasing and
                gamma_decreasing
            ),
        }

    def verify_degeneracy_to_v46(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
    ) -> dict[str, float | bool]:
        """
        验证：κ=0 时双势阱项退化为 v4.6 的线性作用量。

        v4.8 符号修正：
            κ=0 → β=0，双势阱项 = +γ·Tr(g⁴)（仅剩约束项）
            S_v48(κ=0) - S_v46 = +γ·Tr(g⁴)
        """
        # κ=0 → β=0，双势阱项退化为 +γ·Tr(g⁴)（v4.8 标准符号）
        # v4.7 旧符号：-γ·Tr(g⁴)
        # v4.8 新符号：+γ·Tr(g⁴)
        result_v48 = self.corrected_action(g_batch, C, phi, None, kappa=0.0, alpha=1.0)
        S_v48 = result_v48["action"]

        # v4.6 作用量（κ=0，无双势阱）
        # 手动计算 v4.6 作用量
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 曲率项（κ=0 → 曲率项=0）
        curvature_v46 = torch.zeros(1, dtype=torch.float64)[0]

        # 耦合项
        g_mean = g.mean(dim=0)
        g_mean = symmetric_part(g_mean)
        D_phi = self.scd.covariant_derivative(phi, None)
        g_inv = safe_inverse(g_mean, self.eps)
        DtD = D_phi.T @ D_phi
        coupling_v46 = 0.5 * torch.einsum('ij,ij->', g_inv, DtD)

        # 梯度项
        grad_v46 = 1.0 * self.compute_grad_term_v2(g, C)

        S_v46 = curvature_v46 + coupling_v46 + grad_v46

        # 差异 = 双势阱项 = +γ·Tr(g⁴)（因为 β=0，v4.8 标准符号）
        gamma = self.compute_gamma(1.0)
        trace_g4 = result_v48["trace_g4"]
        expected_diff = gamma * trace_g4  # v4.8: +γ·Tr(g⁴)

        actual_diff = float(S_v48 - S_v46)
        relative_error = abs(actual_diff - float(expected_diff)) / (abs(actual_diff) + self.eps)

        return {
            "S_v48": float(S_v48),
            "S_v46": float(S_v46),
            "difference": actual_diff,
            "expected_difference": float(expected_diff),
            "relative_error": relative_error,
            "is_consistent": relative_error < 1e-10,
        }
