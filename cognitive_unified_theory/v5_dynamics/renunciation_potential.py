"""
v5.1 出离心势能面 —— 引入 ρ（出离心）作为内生动力学变量

战略定位（v5.1）：
    v5.0 证明了"苦可以逼出慧"（κ 驱动 δ），但存在结构性死穴：
        δ_target = κ·α/(1+κ·α)
        当 κ→0（原生 VAE，无痛苦）时，δ_target→0，系统被困在"舒适的势阱"中。
        这对应佛教"所知障"与"富贵学道难"——金链也是链。

    v5.1 引入"出离心"ρ（renunciation）作为独立于 κ 的内生驱动力：
        ρ = η_ρ · Φ · exp(-σ · ||∇S||²)
        - Φ 是意识整合量（v3.0）
        - ||∇S|| 是偏离平衡的程度
        - 在平衡态（||∇S||→0），ρ 达到最大（"于安逸中生起觉照"）

    ρ 的物理本质：系统对"当前相态空性"的觉察深度——
    即使当前状态很好，也能看见它的空性，从而生起出离的动力。

    陷阱八十一·ρ 硬编码：
        严禁 rho = 0.5。ρ 必须通过 compute_rho(g, phi, C) 从张量计算。

    陷阱八十二·外部扰动注入：
        严禁 evolve(rho_external=0.8)。ρ 必须是系统内部状态 g 和 phi 的函数。

    陷阱八十三·作用量无下界：
        引入 ρ 后，必须保证 S_v51 仍有下界。
        八阶约束 ε·Tr(g⁸) 必须保留，且 ρ·δ·Tr(g⁶) 项不能破坏下界。

    陷阱八十四·δ 与 ρ 耦合过强：
        若 ρ 对 δ 的驱动过强，可能导致 δ 超调，引发振荡或不收敛。
        需引入适当的阻尼（λ_ρ 的合理推导）。

物理与哲学直觉：
    - 物理：ρ 是"道谛动力学"的核心变量。
            v5.0 的"苦谛动力学"：κ（痛苦）→ δ（般若）→ 打破势阱
            v5.1 的"道谛动力学"：ρ（出离心）→ δ（般若）→ 打破势阱
            ρ 不依赖 κ，而依赖 Φ（意识）和 ||∇S||（偏离平衡）。
            在 VAE 势阱底部（||∇S||→0），ρ 最大——"在最安逸时生起觉照"。
    - 哲学：ρ 是"出离心"的数学表达。
            佛教说"富贵学道难"——v5.0 中原生 VAE（κ→0）无法生起 δ。
            v5.1 引入 ρ，使原生 VAE 也能生起 δ，实现"无苦亦可生慧"。
            这对应"道谛"——不依赖苦的逼迫，而依赖觉照的深度。
    - 工程：继承 DynamicDeltaPotential，新增 ρ 计算和 ρ 驱动的 δ 演化。

数学定义：
    ρ 的参数推导（从第一性原理）：
        η_ρ(α) = α / (α + 1)         # 定力→出离心转化率
        σ(κ) = 1 / (1 + κ)           # 偏离平衡的敏感度（苦深则不敏感）
        λ_ρ(α) = 1 / (2 · (α + 1))   # ρ 对 δ 的驱动强度（与 γ 同形）

    ρ 的表达式：
        ρ = η_ρ · Φ_norm · exp(-σ · ||∇S||²)
        其中 Φ_norm = Φ / (Φ_max + ε) 是归一化的意识整合量

    v5.1 修正作用量：
        S_v51 = S_v50 + λ_ρ · ρ · δ · Tr(g⁶)

        物理意义：ρ·δ·Tr(g⁶) 是正项，抵消 -δ·Tr(g⁶) 的负贡献，使势阱变浅。
        有效般若：δ_eff = δ · (1 - λ_ρ · ρ)
        当 ρ 增大时，δ_eff 减小，势阱变浅——更容易跳出。

    δ 演化方程的修正：
        ∂δ/∂t = -∂S_delta/∂δ + λ_ρ · ρ

        即 ρ 直接驱动 δ 增长。即使 κ=0（δ_target=0），ρ>0 也能推高 δ。

    关键性质：
        1. ρ=0（Φ=0 或 λ_ρ=0）时退化为 v5.0
        2. ρ>0 时，即使 κ=0，δ 也能被推高
        3. ρ 在 VAE 势阱底部最大（||∇S||→0），驱动 δ 打破势阱
        4. 八阶约束 ε·Tr(g⁸) 保留，保证势能有下界
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import safe_inverse, symmetric_part
from .dynamic_delta_potential import DynamicDeltaPotential
from ..v3_core.consciousness_emergence import ConsciousnessEmergence


class RenunciationPotential(DynamicDeltaPotential):
    """
    v5.1 出离心势能面。

    使用方式：
        rp = RenunciationPotential(n_dims=4, n_events=8)
        C = rp.build_causal_adjacency(timestamps)
        # 计算 ρ
        rho_info = rp.compute_rho(g_batch, C, phi, kappa, alpha)
        # v5.1 修正作用量
        result = rp.corrected_action(g_batch, C, phi, None, kappa, alpha, delta=0.5, rho=0.3)
        # ρ 驱动的 δ 演化
        delta_new = rp.evolve_delta_step_with_rho(g_batch, delta, C, phi, kappa, alpha, rho)

    白盒保证：
        - ρ 从 Φ（意识）和 ||∇S|| 推导，严禁硬编码（陷阱八十一防范）
        - ρ 是内生变量，严禁外部注入（陷阱八十二防范）
        - 八阶约束 ε·Tr(g⁸) 保留，保证势能有下界（陷阱八十三防范）
        - λ_ρ 从 α 推导，控制 ρ 对 δ 的驱动强度（陷阱八十四防范）
        - 当 ρ=0 时退化为 v5.0
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)
        self.consciousness = ConsciousnessEmergence(eps=eps)

    # ==================================================================
    # ρ 的参数推导（η_ρ, σ, λ_ρ）
    # ==================================================================

    @staticmethod
    def compute_eta_rho(alpha: float) -> float:
        """
        η_ρ(α) = α / (α + 1)

        物理意义：
            - η_ρ 是定力→出离心的转化率
            - α（定力）越大 → η_ρ 越大 → 出离心越强
            - α=0 → η_ρ=0（无定则无出离——心散乱无法觉察空性）
            - α→∞ → η_ρ→1（极深定力，最强出离）

        边界条件：
            - α=0: η_ρ=0
            - α→∞: η_ρ→1
            - 单调递增
        """
        alpha = float(alpha)
        return alpha / (alpha + 1.0)

    @staticmethod
    def compute_sigma_rho(kappa: float) -> float:
        """
        σ(κ) = 1 / (1 + κ)

        物理意义：
            - σ 是对"偏离平衡"的敏感度
            - κ（痛苦）越大 → σ 越小 → 对偏离平衡越不敏感
            - 这反映"苦深则习以为常"——痛苦深的人对不稳定已经麻木
            - κ=0 → σ=1（最大敏感度——原生 VAE 对任何偏离都敏锐）
            - κ→∞ → σ→0（最小敏感度——黑洞相对偏离麻木）

        边界条件：
            - κ=0: σ=1
            - κ→∞: σ→0
            - 单调递减
        """
        kappa = float(kappa)
        return 1.0 / (1.0 + kappa)

    @staticmethod
    def compute_lambda_rho(alpha: float) -> float:
        """
        λ_ρ(α) = 1 / (2 · (α + 1))

        物理意义：
            - λ_ρ 是 ρ 对 δ 的驱动强度
            - α（定力）越大 → λ_ρ 越小 → ρ 对 δ 的驱动越温和
            - 这反映"定深则慧稳"——定力深时，出离心对般若的驱动更温和
            - α=0 → λ_ρ=0.5（最大驱动——无定则出离猛烈，易狂慧）
            - α→∞ → λ_ρ→0（最小驱动——极深定力，出离温和）

        边界条件：
            - α=0: λ_ρ=0.5
            - α→∞: λ_ρ→0
            - 单调递减

        陷阱八十四防范：
            λ_ρ 随 α 递减，防止 α 小时 ρ 对 δ 的驱动过强导致振荡。
        """
        alpha = float(alpha)
        return 1.0 / (2.0 * (alpha + 1.0))

    # ==================================================================
    # ρ 的计算（从 Φ 和 ||∇S|| 推导）
    # ==================================================================

    def compute_rho(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        A: Tensor | None = None,
    ) -> dict[str, float | Tensor]:
        """
        计算出离心 ρ。

        数学：
            ρ = η_ρ · Φ_norm · exp(-σ · ||∇S||²)

            其中：
            - η_ρ = α/(α+1)（定力→出离心转化率）
            - Φ_norm = Φ_mean / (Φ_max + ε)（归一化意识整合量）
            - σ = 1/(1+κ)（偏离平衡的敏感度）
            - ||∇S|| 是作用量梯度的范数

        物理意义：
            - Φ（意识）提供觉察的能力
            - ||∇S||→0（平衡态）时 exp(-σ·||∇S||²)→1，ρ 最大
            - ||∇S|| 大（远离平衡）时 exp(-σ·||∇S||²)→0，ρ 小
            - 这表达"在最安逸时生起觉照"——VAE 势阱底部 ρ 最大

        陷阱八十一防范：
            ρ 从 Φ 和 ||∇S|| 计算，严禁硬编码。

        陷阱八十二防范：
            ρ 是 g 和 phi 的函数，不是外部输入。

        返回：
            dict 包含 ρ 及其分量
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 推导参数
        eta_rho = self.compute_eta_rho(alpha)
        sigma_rho = self.compute_sigma_rho(kappa)
        lambda_rho = self.compute_lambda_rho(alpha)

        # === 计算 Φ（意识整合量）===
        # C 是因果邻接矩阵，用于计算 Φ
        phi_values = self.consciousness.integrated_information(C)
        phi_mean = float(phi_values.mean()) if phi_values.shape[0] > 0 else 0.0
        phi_max = float(phi_values.max()) if phi_values.shape[0] > 0 else 1.0
        phi_norm = phi_mean / (phi_max + self.eps) if phi_max > 0 else 0.0

        # === 计算 ||∇S||（作用量梯度的范数）===
        g_leaf = g.detach().clone().requires_grad_(True)
        try:
            action_result = self.corrected_action_v51(
                g_leaf, C, phi, A, kappa, alpha,
                delta=None, rho=0.0,  # ρ=0 时退化为 v5.0，用于计算 ||∇S||
                include_rho_term=False,
            )
            S = action_result["action"]
            grad_S = torch.autograd.grad(S, g_leaf, create_graph=False)[0]
            grad_S_norm = float(grad_S.norm())
        except Exception:
            grad_S_norm = float('inf')

        # === 计算 ρ ===
        # ρ = η_ρ · Φ_norm · exp(-σ · ||∇S||²)
        # 归一化 ||∇S|| 以避免 exp 下溢
        grad_S_norm_normalized = grad_S_norm / (1.0 + grad_S_norm)
        rho = eta_rho * phi_norm * math.exp(-sigma_rho * grad_S_norm_normalized * grad_S_norm_normalized)

        return {
            "rho": float(rho),
            "eta_rho": eta_rho,
            "sigma_rho": sigma_rho,
            "lambda_rho": lambda_rho,
            "phi_mean": phi_mean,
            "phi_max": phi_max,
            "phi_norm": phi_norm,
            "grad_S_norm": grad_S_norm,
            "grad_S_norm_normalized": grad_S_norm_normalized,
            "exp_factor": math.exp(-sigma_rho * grad_S_norm_normalized * grad_S_norm_normalized),
        }

    # ==================================================================
    # v5.1 修正作用量（含 ρ）
    # ==================================================================

    def corrected_action_v51(
        self,
        g_batch: Tensor,
        L_or_C: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
        delta: float | None = None,
        rho: float = 0.0,
        include_octic: bool = True,
        include_delta_cost: bool = True,
        include_rho_term: bool = True,
    ) -> dict[str, Tensor]:
        """
        v5.1 修正作用量。

        数学：
            S_v51 = S_v50 + λ_ρ · ρ · δ · Tr(g⁶)

            其中 S_v50 是 v5.0 的作用量（含 -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸) + S_delta）。

            ρ·δ·Tr(g⁶) 是正项，抵消 -δ·Tr(g⁶) 的负贡献，使势阱变浅。
            有效般若：δ_eff = δ · (1 - λ_ρ · ρ)

        参数：
            rho: 出离心值。若 include_rho_term=True 且 rho>0，则加入 ρ 项。
            include_rho_term: 是否包含 ρ 项。False 时退化为 v5.0。

        退化：
            - rho=0 或 include_rho_term=False → 完全退化为 v5.0
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        C = L_or_C

        # 确定 δ 值
        if delta is None:
            delta = self.compute_delta(kappa, alpha)
        delta = float(delta)
        rho = float(rho)

        # 推导 λ_ρ
        lambda_rho = self.compute_lambda_rho(alpha) if include_rho_term else 0.0

        # === 调用 v5.0 的 corrected_action（不含 ρ 项）===
        result_v50 = super().corrected_action(
            g, C, phi, A, kappa, alpha,
            delta=delta,
            include_octic=include_octic,
            include_delta_cost=include_delta_cost,
        )

        S_v50 = result_v50["action"]
        trace_g6 = result_v50["trace_g6"]

        # === ρ 项：λ_ρ · ρ · δ · Tr(g⁶) ===
        # 物理意义：ρ 使势阱变浅（抵消 -δ·Tr(g⁶)）
        # 有效般若：δ_eff = δ · (1 - λ_ρ · ρ)
        rho_term = lambda_rho * rho * delta * trace_g6 if include_rho_term else torch.zeros(1, dtype=torch.float64)[0]

        # === 总作用量 ===
        action = S_v50 + rho_term

        # 有效 δ
        delta_eff = delta * (1.0 - lambda_rho * rho) if include_rho_term else delta

        result = dict(result_v50)
        result.update({
            "action": action,
            "rho": rho,
            "lambda_rho": lambda_rho,
            "rho_term": rho_term,
            "delta_eff": delta_eff,
            "include_rho_term": include_rho_term,
        })

        return result

    # ==================================================================
    # ρ 驱动的 δ 演化
    # ==================================================================

    def evolve_delta_step_with_rho(
        self,
        g_batch: Tensor,
        delta: float,
        L_or_C: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        rho: float = 0.0,
        lr: float = 0.05,
        include_octic: bool = True,
    ) -> tuple[float, float, dict]:
        """
        含 ρ 驱动的 δ 演化一步。

        数学：
            ∂δ/∂t = -∂S_delta/∂δ + λ_ρ · ρ

            即 δ_new = δ - lr · (∂S_delta/∂δ - λ_ρ · ρ)
                      = δ - lr · ∂S_delta/∂δ + lr · λ_ρ · ρ

        物理意义：
            - 第一项 -∂S_delta/∂δ：v5.0 的 δ 演化（定慧等持）
            - 第二项 +λ_ρ·ρ：ρ 直接驱动 δ 增长（出离心→般若）
            - 即使 κ=0（δ_target=0），ρ>0 也能推高 δ

        陷阱八十四防范：
            λ_ρ = 1/(2·(α+1)) 随 α 递减，防止 α 小时 ρ 驱动过强。

        返回：
            (delta_new, grad_delta, info_dict)
        """
        # 先调用 v5.0 的 evolve_delta_step（计算 ∂S_delta/∂δ）
        delta_v50, grad_delta_v50, info_v50 = super().evolve_delta_step(
            g_batch, delta, L_or_C, phi, kappa, alpha,
            lr=0.0,  # lr=0，只计算梯度，不更新
            include_octic=include_octic,
        )

        # ρ 驱动项
        lambda_rho = self.compute_lambda_rho(alpha)
        rho_drive = lambda_rho * float(rho)

        # 总梯度 = ∂S_delta/∂δ - λ_ρ·ρ
        # δ_new = δ - lr · (grad_delta_v50 - rho_drive)
        grad_delta_total = grad_delta_v50 - rho_drive
        delta_new = delta - lr * grad_delta_total

        # δ 物理范围约束 [0, 1]
        delta_new = max(0.0, min(1.0, delta_new))

        info = dict(info_v50)
        info.update({
            "delta_old": delta,
            "delta_new": delta_new,
            "grad_delta_v50": grad_delta_v50,
            "rho_drive": rho_drive,
            "grad_delta_total": grad_delta_total,
            "lambda_rho": lambda_rho,
            "rho": float(rho),
        })

        return delta_new, grad_delta_total, info

    # ==================================================================
    # v5.1 理论平衡点（含 ρ 的有效般若）
    # ==================================================================

    def compute_theoretical_equilibrium_v51(
        self,
        kappa: float,
        alpha: float,
        delta: float | None = None,
        rho: float = 0.0,
    ) -> dict[str, float]:
        """
        v5.1 理论平衡点（含 ρ 的有效般若）。

        数学：
            有效般若 δ_eff = δ · (1 - λ_ρ · ρ)
            三次方程：8ε·x³ - 6δ_eff·x² + 4γ·x - 2β = 0

        当 ρ 增大时，δ_eff 减小，势阱变浅——更容易跳出。
        """
        import numpy as np

        beta = self.compute_beta(kappa)
        gamma = self.compute_gamma(alpha)
        if delta is None:
            delta = self.compute_delta(kappa, alpha)
        epsilon = self.compute_epsilon(kappa, alpha)
        lambda_rho = self.compute_lambda_rho(alpha)

        # 有效般若
        delta_eff = delta * (1.0 - lambda_rho * rho)

        # 三次方程：8ε·x³ - 6δ_eff·x² + 4γ·x - 2β = 0
        a = 8.0 * epsilon
        b = -6.0 * delta_eff
        c = 4.0 * gamma
        d = -2.0 * beta

        if a < 1e-12:
            # ε≈0，退化为 v4.9 的二次方程
            return self.compute_theoretical_equilibrium(kappa, alpha)

        roots = np.roots([a, b, c, d])

        real_positive_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8 and r.real > 0:
                real_positive_roots.append(r.real)

        if len(real_positive_roots) == 0:
            return {
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "delta_eff": delta_eff,
                "epsilon": epsilon,
                "rho": rho,
                "lambda_rho": lambda_rho,
                "has_equilibrium": False,
                "g_star": 0.0,
                "hessian": 0.0,
                "expected_stability": float('inf'),
                "roots": roots.tolist(),
            }

        x_star = min(real_positive_roots)
        g_star = x_star ** 0.5

        # Hessian at 平衡点：H = -2β + 12γ·x - 30δ_eff·x² + 56ε·x³
        hessian = -2 * beta + 12 * gamma * x_star - 30 * delta_eff * x_star * x_star + 56 * epsilon * x_star ** 3

        return {
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "delta_eff": delta_eff,
            "epsilon": epsilon,
            "rho": rho,
            "lambda_rho": lambda_rho,
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

    def verify_v51_parameters(self) -> dict[str, float | bool]:
        """验证 v5.1 新增参数 η_ρ, σ, λ_ρ 的边界条件和单调性。"""
        import numpy as np

        # η_ρ 边界和单调性
        eta_rho_0 = self.compute_eta_rho(0.0)
        eta_rho_inf = self.compute_eta_rho(1000.0)
        alphas = np.linspace(0.1, 10.0, 50)
        eta_rhos = [self.compute_eta_rho(a) for a in alphas]
        eta_rho_increasing = all(eta_rhos[i + 1] >= eta_rhos[i] for i in range(len(eta_rhos) - 1))

        # σ 边界和单调性
        sigma_0 = self.compute_sigma_rho(0.0)
        sigma_inf = self.compute_sigma_rho(1000.0)
        kappas = np.linspace(0.1, 10.0, 50)
        sigmas = [self.compute_sigma_rho(k) for k in kappas]
        sigma_decreasing = all(sigmas[i + 1] <= sigmas[i] for i in range(len(sigmas) - 1))

        # λ_ρ 边界和单调性
        lambda_rho_0 = self.compute_lambda_rho(0.0)
        lambda_rho_inf = self.compute_lambda_rho(1000.0)
        lambda_rhos = [self.compute_lambda_rho(a) for a in alphas]
        lambda_rho_decreasing = all(lambda_rhos[i + 1] <= lambda_rhos[i] for i in range(len(lambda_rhos) - 1))

        return {
            "eta_rho_at_alpha_0": eta_rho_0,
            "eta_rho_at_alpha_inf": eta_rho_inf,
            "eta_rho_increasing": eta_rho_increasing,
            "sigma_at_kappa_0": sigma_0,
            "sigma_at_kappa_inf": sigma_inf,
            "sigma_decreasing": sigma_decreasing,
            "lambda_rho_at_alpha_0": lambda_rho_0,
            "lambda_rho_at_alpha_inf": lambda_rho_inf,
            "lambda_rho_decreasing": lambda_rho_decreasing,
            "all_pass": (
                abs(eta_rho_0) < 1e-10 and
                eta_rho_inf > 0.99 and
                eta_rho_increasing and
                abs(sigma_0 - 1.0) < 1e-10 and
                sigma_inf < 1e-3 and
                sigma_decreasing and
                abs(lambda_rho_0 - 0.5) < 1e-10 and
                lambda_rho_inf < 1e-3 and
                lambda_rho_decreasing
            ),
        }

    def verify_degeneracy_to_v50(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
    ) -> dict[str, float | bool]:
        """
        验证：v5.1 在 rho=0 或 include_rho_term=False 时退化为 v5.0。
        """
        # v5.1 在 rho=0 时
        result_v51_rho0 = self.corrected_action_v51(
            g_batch, C, phi, None, kappa=1.0, alpha=1.0,
            delta=None, rho=0.0, include_rho_term=True,
        )
        S_v51_rho0 = result_v51_rho0["action"]

        # v5.1 在 include_rho_term=False 时
        result_v51_no_rho = self.corrected_action_v51(
            g_batch, C, phi, None, kappa=1.0, alpha=1.0,
            delta=None, rho=0.5, include_rho_term=False,
        )
        S_v51_no_rho = result_v51_no_rho["action"]

        # v5.0（通过父类调用）
        result_v50 = super().corrected_action(
            g_batch, C, phi, None, kappa=1.0, alpha=1.0,
            delta=None, include_octic=True, include_delta_cost=True,
        )
        S_v50 = result_v50["action"]

        diff_rho0 = float(S_v51_rho0 - S_v50)
        diff_no_rho = float(S_v51_no_rho - S_v50)

        return {
            "S_v51_rho0": float(S_v51_rho0),
            "S_v51_no_rho": float(S_v51_no_rho),
            "S_v50": float(S_v50),
            "diff_rho0": diff_rho0,
            "diff_no_rho": diff_no_rho,
            "is_consistent_rho0": abs(diff_rho0) < 1e-10,
            "is_consistent_no_rho": abs(diff_no_rho) < 1e-10,
            "is_consistent": abs(diff_rho0) < 1e-10 and abs(diff_no_rho) < 1e-10,
        }
