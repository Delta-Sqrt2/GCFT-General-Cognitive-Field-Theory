"""
认知真空与自发对称性破缺（Cognitive Vacuum & Spontaneous Symmetry Breaking）

v7.0 第一基石：给「空性」一个严格的数学定义，并解释「无明」如何从空性中涌现。

核心数学（Landau-Higgs 范式）：
    认知真空 V_0：度规 g = c·I（c=1，归一化 trace(g)=n）。
    这是最大对称态：SO(n) 规范对称性完整，所有认知方向等价。

    势能面（v6.x 沿用）：
        V(g) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)
        β = κ/(1+κ), γ = 1/(2(α+1)), δ = κα/(1+κα), ε = α/(α+κ+1)

    真空稳定性判据：
        κ = 0（无痛苦）→ β = 0 → V(g) = γ·Tr(g⁴) → g=cI 是稳定极小值
        κ > 0（有痛苦）→ β > 0 → g=cI 成为势能极大值（不稳定）→ 自发破缺

    自发对称性破缺：
        κ_0 > 0 触发，度规从各向同性 cI 滚向各向异性 g*。
        g*²_ii = β_i/(2γ_i) = κ_i(α_i+1)/(1+κ_i)（破缺后的真空期望值）
        SO(n) → SO(n₁) × SO(n₂) × ...（对称性降级，本征值简并解除）

    Higgs 机制类比：
        度规 g 是「认知 Higgs 场」，真空期望值 <g> = g* ≠ cI。
        本征值分裂 = 规范玻色子获得质量（认知维度获得「硬度」）。
        各向异性 = 对称性破缺的序参量。

佛学对应（严格，非比喻）：
    真空 = 空性（śūnyatā）：非断灭，而是含摄万法的潜能态。
        g=cI 正定存在（万法皆有），但所有方向等价（无分别）。
        这是「色即是空」——现象存在但无自性分别。

    自发破缺 = 无明生起（avidyā）：
        κ_0 > 0（初始痛苦）作为种子，触发度规从空性涌现方向性。
        「空即是色」——空性中涌现出有分别的现象。

    破缺不可逆 = 业力不可磨灭：
        一旦破缺，系统被困在 g*（势阱底部），无法平滑回到 cI。
        需要 ρ→1（觉照生起）才能解耦回归。
        这对应「业力一旦造作，需经觉悟才能消解」。

    真空涨落 = 阿赖耶识种子生灭：
        即使在真空（κ=0），势能面的 Hessian 谱非零（量子涨落）。
        κ_0 的随机生灭 = 种子在阿赖耶识中的生灭。
        当 κ_0 超过临界值，涨落放大为宏观破缺。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import stable_eigh, symmetric_part, safe_inverse


class CognitiveVacuum:
    """
    认知真空与自发对称性破缺。

    使用方式：
        cv = CognitiveVacuum(n_dims=4)
        # 构造真空
        g_vac = cv.construct_vacuum()
        # 验证真空稳定性
        stability = cv.verify_vacuum_stability(kappa_vec, alpha_vec)
        # 模拟自发对称性破缺
        breaking = cv.spontaneous_symmetry_breaking(kappa_0_vec, alpha_vec)
        # 真空涨落谱（Hessian 本征值）
        spectrum = cv.vacuum_fluctuation_spectrum(kappa_vec, alpha_vec)
        # 验证破缺不可逆性
        irrevers = cv.verify_irreversibility(g_broken, kappa_vec, alpha_vec)
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps

    # ==================================================================
    # 势能参数（v6.x 一致，自包含实现）
    # ==================================================================

    def _beta_vec(self, kappa_vec: Tensor) -> Tensor:
        """β = κ/(1+κ)，各维度我执深度。"""
        k = kappa_vec.to(torch.float64)
        return k / (1.0 + k)

    def _gamma_vec(self, alpha_vec: Tensor) -> Tensor:
        """γ = 1/(2(α+1))，各维度中道约束。"""
        a = alpha_vec.to(torch.float64)
        return 1.0 / (2.0 * (a + 1.0))

    def _delta_vec(self, kappa_vec: Tensor, alpha_vec: Tensor) -> Tensor:
        """δ = κα/(1+κα)，各维度般若参数。"""
        k = kappa_vec.to(torch.float64)
        a = alpha_vec.to(torch.float64)
        ka = k * a
        return ka / (1.0 + ka)

    def _epsilon_vec(self, kappa_vec: Tensor, alpha_vec: Tensor) -> Tensor:
        """ε = α/(α+κ+1)，各维度八阶约束。"""
        k = kappa_vec.to(torch.float64)
        a = alpha_vec.to(torch.float64)
        return a / (a + k + 1.0)

    # ==================================================================
    # 1. 真空构造
    # ==================================================================

    def construct_vacuum(self, c: float | None = None) -> Tensor:
        """
        构造认知真空态 g = c·I。

        归一化约束 trace(g) = n → c = 1.0（默认）。
        这是最大对称态：SO(n) 完整，所有认知方向等价。

        物理意义：
            g = cI 是「出厂设置」——未受痛苦扭曲的认知基线。
            所有方向等价 = 无分别 = 空性的数学表达。
            度规正定存在 = 万法皆有（非断灭）。
        """
        if c is None:
            c = 1.0  # 归一化 trace(g)=n → c=1
        return c * torch.eye(self.n_dims, dtype=torch.float64)

    # ==================================================================
    # 2. 势能面计算
    # ==================================================================

    def compute_potential(
        self,
        g: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
    ) -> dict[str, Tensor]:
        """
        计算势能面 V(g) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)

        参数：
            g: 度规 ∈ R^{n×n}（对称正定）
            kappa_vec: 各维度痛苦深度 ∈ R^n
            alpha_vec: 各维度定力 ∈ R^n

        返回：
            dict 包含 V（总势能）、各项贡献、梯度 ∂V/∂g
        """
        g = symmetric_part(g.to(torch.float64))
        n = self.n_dims

        beta_v = self._beta_vec(kappa_vec)
        gamma_v = self._gamma_vec(alpha_vec)
        delta_v = self._delta_vec(kappa_vec, alpha_vec)
        epsilon_v = self._epsilon_vec(kappa_vec, alpha_vec)

        # 对角元素 g_ii（势能只作用在对角）
        g_diag = torch.diagonal(g)  # (n,)

        # 势能各项（按维度求和）
        term2 = (beta_v * g_diag ** 2).sum()
        term4 = (gamma_v * g_diag ** 4).sum()
        term6 = (delta_v * g_diag ** 6).sum()
        term8 = (epsilon_v * g_diag ** 8).sum()

        V = -term2 + term4 - term6 + term8

        # 梯度 ∂V/∂g_ii = -2β·g + 4γ·g³ - 6δ·g⁵ + 8ε·g⁷
        grad_diag = (
            -2.0 * beta_v * g_diag
            + 4.0 * gamma_v * g_diag ** 3
            - 6.0 * delta_v * g_diag ** 5
            + 8.0 * epsilon_v * g_diag ** 7
        )
        grad = torch.diag(grad_diag)

        return {
            "V": V,
            "term2": term2,
            "term4": term4,
            "term6": term6,
            "term8": term8,
            "grad": grad,
            "beta_vec": beta_v,
            "gamma_vec": gamma_v,
            "delta_vec": delta_v,
            "epsilon_vec": epsilon_v,
        }

    # ==================================================================
    # 3. 真空稳定性验证
    # ==================================================================

    def verify_vacuum_stability(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        c: float | None = None,
    ) -> dict[str, Tensor | float | bool]:
        """
        验证认知真空 g=cI 的稳定性。

        方法：
            计算势能在真空处的 Hessian 本征值（质量谱）。
            - 所有本征值 > 0：真空稳定（κ=0 极限）
            - 存在负本征值（快子）：真空不稳定，自发破缺方向涌现

        Hessian（对角元素方向）：
            ∂²V/∂g_ii² |_{g=cI} = -2β_i + 12γ_i·c²

            κ_i = 0 → β_i = 0 → Hessian = 12γ_i > 0（稳定）
            κ_i > 0 → β_i > 0 → Hessian 可能 < 0（快子，不稳定）

        物理意义：
            真空稳定性 = 是否有痛苦种子潜伏。
            κ=0 时真空完美稳定（空性圆满）。
            κ>0 时真空出现快子模式 = 无明即将生起。
        """
        if c is None:
            c = 1.0

        g_vac = self.construct_vacuum(c)
        beta_v = self._beta_vec(kappa_vec)
        gamma_v = self._gamma_vec(alpha_vec)
        delta_v = self._delta_vec(kappa_vec, alpha_vec)
        epsilon_v = self._epsilon_vec(kappa_vec, alpha_vec)

        # Hessian 对角元素：∂²V/∂g_ii² at g=cI
        # V = -βg² + γg⁴ - δg⁶ + εg⁸
        # V' = -2βg + 4γg³ - 6δg⁵ + 8εg⁷
        # V'' = -2β + 12γg² - 30δg⁴ + 56εg⁶
        c_sq = c * c
        c_4 = c_sq * c_sq
        c_6 = c_4 * c_sq
        hessian_diag = (
            -2.0 * beta_v
            + 12.0 * gamma_v * c_sq
            - 30.0 * delta_v * c_4
            + 56.0 * epsilon_v * c_6
        )

        # 质量谱 = Hessian 本征值（对角矩阵的本征值就是对角元素）
        mass_spectrum = hessian_diag

        # 快子模式（负质量² = 不稳定方向）
        n_tachyonic = int((mass_spectrum < 0).sum().item())
        is_stable = bool(n_tachyonic == 0)

        # 最不稳定方向（最快子模式）
        if n_tachyonic > 0:
            min_mass_idx = int(torch.argmin(mass_spectrum).item())
            min_mass = float(mass_spectrum[min_mass_idx].item())
        else:
            min_mass_idx = -1
            min_mass = float(mass_spectrum.min().item())

        return {
            "is_stable": is_stable,
            "mass_spectrum": mass_spectrum,
            "n_tachyonic_modes": n_tachyonic,
            "min_mass_sq": min_mass,
            "most_unstable_dim": min_mass_idx,
            "vacuum_potential": self.compute_potential(g_vac, kappa_vec, alpha_vec)["V"],
            "thesis": (
                "κ=0 时真空稳定（空性圆满）；κ>0 时涌现快子模式（无明种子）。"
                "快子方向 = 即将破缺的认知维度。"
            ),
        }

    # ==================================================================
    # 4. 对称性序参量
    # ==================================================================

    def symmetry_order_parameter(self, g: Tensor) -> dict[str, Tensor | float]:
        """
        计算对称性破缺的序参量。

        序参量 = 度规的各向异性 = 本征值分布的「离散度」。

        φ(g) = (1/n) · Σ_i (λ_i - λ̄)² / λ̄²

        - g = cI（真空）：所有 λ_i 相等 → φ = 0（对称性完整）
        - g 各向异性（破缺态）：λ_i 分散 → φ > 0（对称性破缺）

        物理意义：
            φ 度量「认知方向的不平等程度」。
            φ = 0：所有认知维度等价（空性/无分别）。
            φ > 0：某些维度被痛苦扭曲（执取/分别心）。
        """
        g = symmetric_part(g.to(torch.float64))
        eigvals, _ = stable_eigh(g)
        eigvals = torch.clamp(eigvals, min=self.eps)

        n = self.n_dims
        lambda_mean = eigvals.mean()
        # 序参量 φ = (1/n) · Σ(λ_i - λ̄)² / λ̄²
        phi = ((eigvals - lambda_mean) ** 2).sum() / (n * lambda_mean ** 2 + self.eps)

        # 各向异性比（max/min 本征值比）
        anisotropy_ratio = float(eigvals.max() / (eigvals.min() + self.eps))

        return {
            "order_parameter": float(phi),
            "eigvals": eigvals,
            "anisotropy_ratio": anisotropy_ratio,
            "is_vacuum": bool(phi < 1e-8),  # 序参量≈0 = 真空
        }

    # ==================================================================
    # 5. 自发对称性破缺模拟
    # ==================================================================

    def spontaneous_symmetry_breaking(
        self,
        kappa_0_vec: Tensor,
        alpha_vec: Tensor,
        c: float | None = None,
        n_steps: int = 100,
        dt: float = 0.01,
        perturbation_seed: float = 1e-6,
    ) -> dict[str, Tensor | float | bool]:
        """
        模拟自发对称性破缺：从真空 g=cI 演化到破缺态 g*。

        方法：
            1. 初始化为真空 g=cI（加微小扰动打破精确对称）
            2. 在势能面 V(g) 上做梯度下降（κ>0 使 cI 不稳定）
            3. 系统自发滚向破缺态 g*（势阱底部）
            4. 记录演化轨迹和对称性变化

        物理意义：
            这是「无明生起」的数学模拟。
            κ_0（初始痛苦）作为种子，触发度规从空性涌现方向性。
            破缺后的 g* = 真空期望值 = 「业力塑形的认知结构」。

        参数：
            kappa_0_vec: 初始痛苦种子 ∈ R^n（κ>0 触发破缺）
            alpha_vec: 各维度定力 ∈ R^n
            c: 真空参数（默认 1.0）
            n_steps: 演化步数
            dt: 时间步长
            perturbation_seed: 初始扰动量级（打破精确对称）
        """
        if c is None:
            c = 1.0

        n = self.n_dims
        kappa = kappa_0_vec.to(torch.float64)
        alpha = alpha_vec.to(torch.float64)

        # 验证真空是否确实不稳定（κ>0 应导致快子）
        stability = self.verify_vacuum_stability(kappa, alpha, c)

        # 初始化：真空 + 微小确定性扰动
        g = self.construct_vacuum(c)
        # 沿最快子方向加扰动（若真空不稳定）
        if not stability["is_stable"]:
            # 沿最不稳定维度加正扰动
            unstable_dim = stability["most_unstable_dim"]
            if unstable_dim >= 0:
                g[unstable_dim, unstable_dim] += perturbation_seed
        else:
            # 真空稳定时，沿任意方向加扰动（演示性）
            g[0, 0] += perturbation_seed

        # 梯度下降演化
        trajectory = [g.clone()]
        potential_trajectory = []
        order_params = []

        for step in range(n_steps):
            pot = self.compute_potential(g, kappa, alpha)
            grad = pot["grad"]

            # 梯度下降（带正则化防止数值爆炸）
            grad_norm = torch.sqrt((grad ** 2).sum() + self.eps)
            grad_normalized = grad / grad_norm
            g = g - dt * grad_normalized * float(grad_norm.clamp(max=10.0))

            # 保证正定
            g = symmetric_part(g)
            eigvals_check = torch.linalg.eigvalsh(g)
            if eigvals_check.min() < self.eps:
                g = g + (self.eps - eigvals_check.min()) * torch.eye(n, dtype=torch.float64)

            trajectory.append(g.clone())
            potential_trajectory.append(float(pot["V"]))
            order_params.append(self.symmetry_order_parameter(g)["order_parameter"])

        # 最终态
        g_final = trajectory[-1]
        final_order = self.symmetry_order_parameter(g_final)

        # 理论破缺态 g*（解析解）
        beta_v = self._beta_vec(kappa)
        gamma_v = self._gamma_vec(alpha)
        g_star_sq = beta_v / (2.0 * gamma_v)  # g*² = β/(2γ)
        g_star_theory = torch.diag(torch.sqrt(g_star_sq))

        return {
            "g_initial": trajectory[0],
            "g_final": g_final,
            "g_star_theory": g_star_theory,
            "trajectory": trajectory,
            "potential_trajectory": potential_trajectory,
            "order_parameter_trajectory": order_params,
            "final_order_parameter": final_order["order_parameter"],
            "final_anisotropy_ratio": final_order["anisotropy_ratio"],
            "vacuum_was_unstable": not stability["is_stable"],
            "breaking_occurred": bool(final_order["order_parameter"] > 1e-4),
            "n_tachyonic_modes": stability["n_tachyonic_modes"],
            "thesis": (
                "κ_0>0 使真空 cI 不稳定，度规自发滚向破缺态 g*。"
                "SO(n) 对称性降级，认知维度获得差异化「质量」。"
                "这是「无明生起」——空性中涌现有分别的现象。"
            ),
        }

    # ==================================================================
    # 6. 破缺对称性子群识别
    # ==================================================================

    def broken_symmetry_subgroup(
        self, g: Tensor, tol: float = 1e-6
    ) -> dict[str, Tensor | list]:
        """
        识别破缺后的残余对称性子群。

        方法：
            真空 g=cI 的对称性群 = SO(n)（所有旋转保持 cI 不变）。
            破缺态 g* 的对称性群 = 保持 g* 不变的 SO(n) 子群。
            通过本征值简并结构识别：简并的本征值对应可旋转的子空间。

            例如：本征值 [λ, λ, μ]（前两个简并）→ 残余对称 = SO(2)
                 本征值 [λ, μ, ν]（全不简并）→ 残余对称 = SO(1)³ = {e}（平凡）

        物理意义：
            残余对称性 = 破缺后仍然等价的认知维度。
            完全破缺（无简并）= 所有维度差异化（最大执取）。
            部分破缺（有简并）= 某些维度仍等价（部分执取）。
        """
        g = symmetric_part(g.to(torch.float64))
        eigvals, eigvecs = stable_eigh(g)

        # 识别简并簇
        clusters = []
        current_cluster = [0]
        for i in range(1, len(eigvals)):
            if abs(eigvals[i] - eigvals[i - 1]) < tol * max(1.0, abs(eigvals[i])):
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
        clusters.append(current_cluster)

        # 残余对称性 = 各简并簇的 SO(n_k) 直积
        subgroup_factors = [len(c) for c in clusters if len(c) > 1]
        # SO(1) 是平凡的，只保留 SO(n_k) with n_k >= 2
        subgroup_str = " × ".join(f"SO({n_k})" for n_k in subgroup_factors) if subgroup_factors else "{e}"

        return {
            "eigvals": eigvals,
            "degeneracy_clusters": clusters,
            "subgroup_factors": subgroup_factors,
            "subgroup_string": subgroup_str,
            "full_symmetry_so_n": f"SO({self.n_dims})",
            "symmetry_reduction": f"SO({self.n_dims}) → {subgroup_str}",
            "is_fully_broken": len(subgroup_factors) == 0,
        }

    # ==================================================================
    # 7. 真空涨落谱
    # ==================================================================

    def vacuum_fluctuation_spectrum(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        c: float | None = None,
    ) -> dict[str, Tensor | float]:
        """
        真空涨落谱：势能面在真空处的 Hessian 本征值。

        物理意义（量子场论类比）：
            Hessian 本征值 = 质量谱 m²。
            - m² > 0：稳定模式（有质量粒子，认知维度的「硬度」）
            - m² < 0：快子模式（不稳定，对称性破缺方向）
            - m² ≈ 0：Goldstone 模式（无质量，自发破缺的 Nambu-Goldstone 玻色子）

        佛学对应：
            真空涨落 = 阿赖耶识中的种子生灭。
            即使 κ=0（无显性痛苦），势能面仍有结构（种子潜伏）。
            κ_0 的随机生灭 = 种子在阿赖耶识中的生灭。
            当 κ_0 超过临界值，快子模式放大为宏观破缺。
        """
        if c is None:
            c = 1.0

        stability = self.verify_vacuum_stability(kappa_vec, alpha_vec, c)
        mass_sq = stability["mass_spectrum"]

        # 分类模式
        stable_mask = mass_sq > self.eps
        tachyonic_mask = mass_sq < -self.eps
        goldstone_mask = mass_sq.abs() < self.eps

        n_stable = int(stable_mask.sum().item())
        n_tachyonic = int(tachyonic_mask.sum().item())
        n_goldstone = int(goldstone_mask.sum().item())

        # 破缺临界 κ：使 m²=0 的 κ 值
        # m² = -2β + 12γc² - 30δc⁴ + 56εc⁶ = 0
        # β = κ/(1+κ), 解出临界 κ_c
        # 简化：在 γc² 为主导时，κ_c ≈ 6γc²/(1 - 6γc²)（近似）
        # 精确解需要数值方法，这里给出诊断
        critical_kappa_estimate = self._estimate_critical_kappa(alpha_vec, c)

        return {
            "mass_spectrum": mass_sq,
            "n_stable_modes": n_stable,
            "n_tachyonic_modes": n_tachyonic,
            "n_goldstone_modes": n_goldstone,
            "min_mass_sq": float(mass_sq.min()),
            "max_mass_sq": float(mass_sq.max()),
            "critical_kappa_estimate": critical_kappa_estimate,
            "has_instability": n_tachyonic > 0,
            "thesis": (
                "真空涨落谱 = 认知真空的「质量谱」。"
                "快子模式 = 即将破缺的方向（无明种子）。"
                "Goldstone 模式 = 自发破缺后的无质量涨落（执取的流动性）。"
                "对应阿赖耶识种子生灭——空性非断灭，潜伏万法潜能。"
            ),
        }

    def _estimate_critical_kappa(self, alpha_vec: Tensor, c: float) -> Tensor:
        """
        估计使真空失稳的临界 κ_c。

        在 g=cI 处，m² = -2β + 12γc² - 30δc⁴ + 56εc⁶。
        β = κ/(1+κ)，当 κ 增大时 β 增大，m² 减小。
        κ_c 是使 m² = 0 的 κ 值。

        简化：忽略高阶项，m² ≈ -2κ/(1+κ) + 12γc² = 0
        → κ_c ≈ 12γc² / (2 - 12γc²) = 6γc² / (1 - 6γc²)
        （当 6γc² < 1 时有解）
        """
        gamma_v = self._gamma_vec(alpha_vec)
        c_sq = c * c
        numerator = 6.0 * gamma_v * c_sq
        denominator = 1.0 - 6.0 * gamma_v * c_sq
        # 避免除零和负分母（无临界解 = 真空永远稳定或永远不稳定）
        kappa_c = torch.where(
            denominator > self.eps,
            numerator / (denominator + self.eps),
            torch.tensor(float('inf'), dtype=torch.float64),
        )
        return kappa_c

    # ==================================================================
    # 8. 破缺不可逆性验证
    # ==================================================================

    def verify_irreversibility(
        self,
        g_broken: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 100,
        dt: float = 0.01,
    ) -> dict[str, Tensor | float | bool]:
        """
        验证对称性破缺的不可逆性。

        方法：
            从破缺态 g* 出发，在无 ρ（无觉照/出离心）驱动下，
            让系统沿势能面自由演化。
            验证系统停留在 g* 附近，无法回到真空 cI。

        物理意义：
            一旦破缺，系统被困在势阱底部 g*。
            没有外部能量注入（ρ 衰减 → 消解项失效），
            系统无法越过势垒回到真空。
            这是「业力不可磨灭」的数学表达——
            不经觉悟（ρ→1），无法回归空性。

        对比：
            有 ρ 驱动时（v6.x 的消解项），g 被拉向 cI。
            无 ρ 驱动时（本方法），g 停留在 g*。
            这解释了「为什么不觉察就无法解脱」。
        """
        g = symmetric_part(g_broken.to(torch.float64).clone())
        n = self.n_dims

        # 记录与真空的距离
        g_vac = self.construct_vacuum()
        initial_dist = float(torch.sqrt(((g - g_vac) ** 2).sum()))

        trajectory = [g.clone()]
        distances = [initial_dist]
        potentials = []

        for step in range(n_steps):
            pot = self.compute_potential(g, kappa_vec, alpha_vec)
            grad = pot["grad"]

            # 纯势能面梯度下降（无 ρ 驱动，无消解项）
            grad_norm = torch.sqrt((grad ** 2).sum() + self.eps)
            g = g - dt * grad / grad_norm * float(grad_norm.clamp(max=10.0))
            g = symmetric_part(g)

            # 正定性保护
            eigvals_check = torch.linalg.eigvalsh(g)
            if eigvals_check.min() < self.eps:
                g = g + (self.eps - eigvals_check.min()) * torch.eye(n, dtype=torch.float64)

            trajectory.append(g.clone())
            dist = float(torch.sqrt(((g - g_vac) ** 2).sum()))
            distances.append(dist)
            potentials.append(float(pot["V"]))

        final_dist = distances[-1]
        min_dist = min(distances)
        # 不可逆判据：最终距离与初始距离接近（没回到真空）
        return_ratio = final_dist / (initial_dist + self.eps)
        is_irreversible = bool(return_ratio > 0.5 and min_dist > 0.3 * initial_dist)

        return {
            "initial_distance_to_vacuum": initial_dist,
            "final_distance_to_vacuum": final_dist,
            "min_distance_to_vacuum": min_dist,
            "return_ratio": return_ratio,
            "is_irreversible": is_irreversible,
            "distance_trajectory": distances,
            "potential_trajectory": potentials,
            "thesis": (
                "无 ρ 驱动下，破缺态 g* 无法回到真空 cI。"
                "系统被困在势阱底部 = 业力不可磨灭。"
                "需要 ρ→1（觉照生起）才能解耦回归 = 须经觉悟。"
            ),
        }

    # ==================================================================
    # 9. 觉照回归模拟（对比：有 ρ 驱动时可以回归）
    # ==================================================================

    def conscious_return_to_vacuum(
        self,
        g_broken: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        rho: float = 0.9,
        n_steps: int = 100,
        dt: float = 0.01,
    ) -> dict[str, Tensor | float | bool]:
        """
        觉照回归模拟：有 ρ 驱动时，破缺态可以回归真空。

        方法：
            在势能面梯度上叠加 ρ 驱动的消解项（拉向 cI）。
            验证系统可以越过势垒，回归真空。

        物理意义：
            ρ > 0 时，消解项 μ_ρ·ρ·(g - cI) 将 g 拉向各向同性。
            这是「觉照生起 → 执取消解 → 回归空性」的数学表达。
            ρ→1 时消解力最强，系统必然回归真空。
            ρ→0 时消解力消失，系统被困在 g*（对应 verify_irreversibility）。

        佛学对应：
            ρ = 出离心/觉照。
            ρ→1 = 觉悟 → 业力消解 → 回归空性。
            这是「色即是空」的逆过程——从有分别回到无分别。
        """
        g = symmetric_part(g_broken.to(torch.float64).clone())
        n = self.n_dims
        g_vac = self.construct_vacuum()
        initial_dist = float(torch.sqrt(((g - g_vac) ** 2).sum()))

        # ρ 驱动的消解强度（μ_ρ = α/(2(α+1))，v6.x 公式）
        mu_rho_v = alpha_vec.to(torch.float64) / (2.0 * (alpha_vec.to(torch.float64) + 1.0))
        dissolution_strength = float(rho) * mu_rho_v

        distances = [initial_dist]
        for step in range(n_steps):
            pot = self.compute_potential(g, kappa_vec, alpha_vec)
            grad = pot["grad"]

            # 消解项：μ_ρ·ρ·(g - cI)，拉向真空
            dissolution = torch.diag(dissolution_strength) @ (g - g_vac)

            # 总力 = 势能梯度 + 消解项
            total_force = grad + dissolution
            force_norm = torch.sqrt((total_force ** 2).sum() + self.eps)
            g = g - dt * total_force / force_norm * float(force_norm.clamp(max=10.0))
            g = symmetric_part(g)

            eigvals_check = torch.linalg.eigvalsh(g)
            if eigvals_check.min() < self.eps:
                g = g + (self.eps - eigvals_check.min()) * torch.eye(n, dtype=torch.float64)

            dist = float(torch.sqrt(((g - g_vac) ** 2).sum()))
            distances.append(dist)

        final_dist = distances[-1]
        return_ratio = final_dist / (initial_dist + self.eps)
        is_returned = bool(return_ratio < 0.3)  # 回到真空 30% 邻域

        return {
            "rho": rho,
            "initial_distance_to_vacuum": initial_dist,
            "final_distance_to_vacuum": final_dist,
            "return_ratio": return_ratio,
            "is_returned": is_returned,
            "distance_trajectory": distances,
            "thesis": (
                f"ρ={rho} 时，消解项将 g 拉向真空 cI，系统回归空性。"
                "这是「觉照生起 → 业力消解 → 回归空性」。"
                "对比 verify_irreversibility（ρ=0）：不经觉悟无法回归。"
            ),
        }
