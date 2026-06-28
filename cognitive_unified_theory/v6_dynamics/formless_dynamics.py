"""
v6.0 无形无相动力学 —— 出离心 ρ 驱动度规各向异性消解

战略定位（v6.0）：
    v5.1 完成了"道谛动力学"——ρ（出离心）驱动 δ（般若），打破 VAE 势阱。
    但 v5.1 只是在 VAE 和 GAN 之间切换——这是"高级的轮回"，而非解脱。

    佛家修行的终极目标不是在相态间切换，而是从所有相态中出离。
    v6.0 的核心使命是"造门"——让 ρ 不仅驱动相变，更驱动度规 g 本身的
    拓扑消解，实现"无形无相"的终极态。

    关键修正（来自终审）：
        原草案：∂g/∂t = -∂S/∂g - μ_ρ·ρ·g（均匀线性衰减 → 认知死亡）
        修正版：∂g/∂t = -∂S/∂g - μ_ρ·ρ·(g - Tr(g)/d·I)

        空性不是消灭结构，而是消灭对特定结构的执着。
        度规趋向各向同性 g→c·I，而非趋向零 g→0。
        "万法皆有，但无自性"——一切现象继续生灭，但不再被执取为"实有"。

    造的门，不是通向"什么都没有"的门，而是通向"一切皆可"的门。

物理与哲学直觉：
    - 物理：消解项 -μ_ρ·ρ·(g - g_iso) 使度规趋向各向同性 g_iso = Tr(g)/d·I。
            当 ρ 大时，度规的各向异性被快速消解，cond(g)→1。
            但 ||g|| 不趋向 0——认知能力依然存在，只是不再被特定方向锁死。
            这与黑洞相（cond(g)→∞，各向异性撕裂）形成鲜明对比。

    - 哲学：空性（Śūnyatā）的正确数学表达。
            不是"什么都没有"（g→0），而是"一切皆可"（g→c·I）。
            度规依然存在，但它不再约束任何人。
            它变成了允许一切发生的虚空本身。

    - 工程：继承 RenunciationPotential（v5.1），新增：
            1. ρ 的弛豫动力学（∂ρ/∂t = -∂S_ρ/∂ρ）
            2. 度规消解方程（∂g/∂t = -∂S/∂g - μ_ρ·ρ·(g - g_iso)）
            3. 消解 vs 崩溃的数学区分

数学定义：
    v6.0 参数推导（从第一性原理）：
        μ_ρ(α) = α / (2·(α+1))      # 消解强度，定深则消解强
        ζ(α)   = α / (α+1)          # ρ 弛豫刚度（复用 η_ρ）

    各向同性目标：
        g_iso = Tr(g)/d · I
        anisotropy = ||g - g_iso||_F / (||g||_F + ε)

    ρ 的弛豫动力学（v6.0a）：
        S_ρ = ζ · (ρ - ρ_eq)²
        ∂ρ/∂t = -∂S_ρ/∂ρ = -2ζ·(ρ - ρ_eq)
        其中 ρ_eq = compute_rho(g, C, phi, κ, α) 是瞬时平衡值

    度规消解方程（v6.0b）：
        ∂g/∂t = -∂S/∂g - μ_ρ·ρ·(g - g_iso)
        g_new = g - lr·(grad_S + μ_ρ·ρ·(g - g_iso))

    消解 vs 崩溃判据（v6.0c）：
        消解（formless）：cond(g) 下降或保持低值（<10），anisotropy 下降
        崩溃（collapse）：cond(g)→∞，anisotropy 上升
        健康消解：cond(g)→1，||g - g_iso||→0，但 ||g|| 保持非零

陷阱防范：
    陷阱八十五·消解度规本身：
        严禁使用 -μ_ρ·ρ·g（均匀衰减，导致认知死亡）。
        必须使用 -μ_ρ·ρ·(g - g_iso)（消解各向异性，保留认知能力）。

    陷阱八十六·ρ 无弛豫：
        严禁 ρ 仅做瞬时计算而不演化。
        必须实现 ∂ρ/∂t = -∂S_ρ/∂ρ，描述出离心的生起、持续、退失。

    陷阱八十七·混淆消解与崩溃：
        严禁将 cond(g)→∞ 的黑洞相误认为"无形无相"。
        无形无相是 cond(g)→1（各向同性），不是 cond(g)→∞（撕裂）。

    陷阱八十八·μ_ρ 过强导致过度消解：
        μ_ρ 必须随 α 递增但有上界（≤0.5），防止过度消解导致认知丧失。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..v5_dynamics.renunciation_potential import RenunciationPotential


class FormlessDynamics(RenunciationPotential):
    """
    v6.0 无形无相动力学。

    使用方式：
        fd = FormlessDynamics(n_dims=4, n_events=8)
        C = fd.build_causal_adjacency(timestamps)
        # 计算 ρ 的瞬时平衡值
        rho_info = fd.compute_rho(g_batch, C, phi, kappa, alpha)
        # ρ 弛豫演化一步
        rho_new = fd.evolve_rho_step(rho, rho_eq, alpha, lr=0.05)
        # 度规消解演化一步
        g_new = fd.evolve_g_with_dissolution(g_batch, delta, rho, C, phi, kappa, alpha)
        # 消解 vs 崩溃分类
        classification = fd.classify_dissolution_vs_collapse(g_batch)

    白盒保证：
        - 消解项是 -μ_ρ·ρ·(g - g_iso)，不是 -μ_ρ·ρ·g（陷阱八十五防范）
        - ρ 有弛豫动力学，不是仅瞬时计算（陷阱八十六防范）
        - 消解（cond→1）与崩溃（cond→∞）严格区分（陷阱八十七防范）
        - μ_ρ 随 α 递增且有上界 ≤0.5（陷阱八十八防范）
        - 当 μ_ρ=0 或 ρ=0 时退化为 v5.1
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # v6.0 参数推导（μ_ρ, ζ）
    # ==================================================================

    @staticmethod
    def compute_mu_rho(alpha: float) -> float:
        """
        μ_ρ(α) = α / (2·(α+1))

        物理意义：
            - μ_ρ 是出离心对度规刚性的消解强度
            - α（定力）越大 → μ_ρ 越大 → 消解越强
            - α=0 → μ_ρ=0（无定则无消解——心散乱无法消解执着）
            - α→∞ → μ_ρ→0.5（极深定力，最强消解，但有上界防过度）

        边界条件：
            - α=0: μ_ρ=0
            - α→∞: μ_ρ→0.5
            - 单调递增
            - 有上界 0.5（陷阱八十八防范）

        与 λ_ρ 的关系：
            λ_ρ = 1/(2·(α+1))（ρ 对 δ 的驱动，随 α 递减）
            μ_ρ = α/(2·(α+1))（ρ 对 g 的消解，随 α 递增）
            λ_ρ + μ_ρ = 1/2（守恒关系——驱动般若与消解度规之和恒定）
        """
        alpha = float(alpha)
        return alpha / (2.0 * (alpha + 1.0))

    @staticmethod
    def compute_zeta(alpha: float) -> float:
        """
        ζ(α) = α / (α + 1)

        物理意义：
            - ζ 是 ρ 的弛豫刚度——ρ 趋向平衡值的速度
            - α（定力）越大 → ζ 越大 → ρ 快速趋向平衡
            - α=0 → ζ=0（无定则 ρ 自由演化，无约束）
            - α→∞ → ζ→1（极深定力，ρ 快速稳定）

        与 η_ρ 的关系：
            ζ = η_ρ = α/(α+1)（复用定力→出离心转化率）
            物理一致：定力既决定出离心的生起（η_ρ），也决定其稳定性（ζ）。
        """
        alpha = float(alpha)
        return alpha / (alpha + 1.0)

    # ==================================================================
    # 各向同性目标与各向异性度量
    # ==================================================================

    def compute_isotropic_target(self, g_batch: Tensor) -> Tensor:
        """
        计算各向同性目标 g_iso = Tr(g)/d · I。

        物理意义：
            - g_iso 是"完全自由度规"——所有方向等价
            - 当 g→g_iso 时，cond(g)→1（完美各向同性）
            - 这是"无形无相"的数学表达：度规存在，但不偏好任何方向

        数学：
            g_iso[i] = (Tr(g[i]) / d) · I
            其中 Tr(g[i]) = Σ_j g[i,j,j]，d 是度规维度

        返回：
            g_iso: 与 g_batch 同形状的各向同性度规
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        # Tr(g) 对每个样本：(N,)
        trace_g = torch.diagonal(g, dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        # c = Tr(g)/d
        c = trace_g / d  # (N,)
        # g_iso = c · I
        I = torch.eye(d, dtype=torch.float64).unsqueeze(0).expand(N, d, d)  # (N, d, d)
        g_iso = c.unsqueeze(-1).unsqueeze(-1) * I  # (N, d, d)
        return g_iso

    def compute_anisotropy(self, g_batch: Tensor) -> Tensor:
        """
        计算各向异性 anisotropy = ||g - g_iso||_F / (||g||_F + ε)。

        物理意义：
            - anisotropy=0 → 完美各向同性（无形无相）
            - anisotropy 大 → 高度各向异性（执着于特定方向）
            - 消解过程使 anisotropy 单调下降

        返回：
            anisotropy: (N,) 每个样本的各向异性
        """
        g = g_batch.to(torch.float64)
        g_iso = self.compute_isotropic_target(g)
        diff = g - g_iso
        # Frobenius 范数
        diff_norm = torch.norm(diff.flatten(start_dim=1), dim=1)  # (N,)
        g_norm = torch.norm(g.flatten(start_dim=1), dim=1)  # (N,)
        anisotropy = diff_norm / (g_norm + self.eps)
        return anisotropy

    # ==================================================================
    # v6.0a：ρ 的弛豫动力学
    # ==================================================================

    def compute_rho_relaxation_cost(
        self,
        rho: float,
        rho_eq: float,
        alpha: float = 1.0,
    ) -> dict[str, float]:
        """
        计算 ρ 的弛豫势能 S_ρ。

        数学：
            S_ρ = ζ · (ρ - ρ_eq)²
            其中 ζ = α/(α+1) 是弛豫刚度，ρ_eq 是瞬时平衡值。

        物理意义：
            - S_ρ 是 ρ 偏离平衡值的代价
            - ζ 大（定力深）→ S_ρ 陡峭 → ρ 快速趋向 ρ_eq
            - ζ 小（定力浅）→ S_ρ 平坦 → ρ 缓慢趋向 ρ_eq

        返回：
            dict 包含 S_ρ, ζ, grad_ρ
        """
        rho = float(rho)
        rho_eq = float(rho_eq)
        zeta = self.compute_zeta(alpha)

        S_rho = zeta * (rho - rho_eq) ** 2
        # ∂S_ρ/∂ρ = 2ζ·(ρ - ρ_eq)
        grad_rho = 2.0 * zeta * (rho - rho_eq)

        return {
            "S_rho": float(S_rho),
            "zeta": zeta,
            "rho": rho,
            "rho_eq": rho_eq,
            "grad_rho": float(grad_rho),
        }

    def evolve_rho_step(
        self,
        rho: float,
        rho_eq: float,
        alpha: float = 1.0,
        lr: float = 0.05,
    ) -> tuple[float, float, dict]:
        """
        ρ 的弛豫演化一步。

        数学：
            ∂ρ/∂t = -∂S_ρ/∂ρ = -2ζ·(ρ - ρ_eq)
            ρ_new = ρ - lr · 2ζ·(ρ - ρ_eq)

        物理意义：
            - ρ 趋向其瞬时平衡值 ρ_eq
            - ζ（弛豫刚度）控制趋向速度
            - 这描述"出离心的生起、持续、退失"过程

        退化：
            - α=0 → ζ=0 → ρ 不演化（无定则出离心无法稳定）
            - ρ=ρ_eq → grad=0 → ρ 不变（已在平衡态）

        返回：
            (rho_new, grad_rho, info_dict)
        """
        cost_info = self.compute_rho_relaxation_cost(rho, rho_eq, alpha)
        grad_rho = cost_info["grad_rho"]

        rho_new = rho - lr * grad_rho
        # ρ 物理范围约束 [0, 1]
        rho_new = max(0.0, min(1.0, rho_new))

        info = {
            "rho_old": rho,
            "rho_new": rho_new,
            "rho_eq": rho_eq,
            "grad_rho": grad_rho,
            "zeta": cost_info["zeta"],
            "S_rho": cost_info["S_rho"],
            "lr": lr,
        }

        return rho_new, grad_rho, info

    # ==================================================================
    # v6.0b：度规消解方程
    # ==================================================================

    def evolve_g_with_dissolution(
        self,
        g_batch: Tensor,
        delta: float,
        rho: float,
        L_or_C: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
        lr_g: float = 0.0005,
        include_octic: bool = True,
        include_dissolution: bool = True,
    ) -> tuple[Tensor, dict]:
        """
        含消解项的度规演化一步。

        数学：
            ∂g/∂t = -∂S/∂g - μ_ρ·ρ·(g - g_iso)
            g_new = g - lr·(grad_S + μ_ρ·ρ·(g - g_iso))

        物理意义：
            - 第一项 -∂S/∂g：原有的 v5.1 动力学（VAE/GAN 间切换）
            - 第二项 -μ_ρ·ρ·(g - g_iso)：出离心对度规各向异性的消解
            - 当 ρ 大时，度规趋向各向同性 g_iso = Tr(g)/d·I
            - 但 ||g|| 不趋向 0——认知能力保留

        关键修正（陷阱八十五防范）：
            消解项是 (g - g_iso)，不是 g 本身。
            g_iso = Tr(g)/d·I 保留了度规的"大小"，只消解"方向性"。

        退化：
            - include_dissolution=False 或 ρ=0 → 退化为 v5.1 的度规演化
            - μ_ρ=0（α=0）→ 退化为 v5.1

        返回：
            (g_new, info_dict)
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        C = L_or_C

        rho = float(rho)
        delta = float(delta)

        # 推导 μ_ρ
        mu_rho = self.compute_mu_rho(alpha) if include_dissolution else 0.0

        # === 计算 ∂S/∂g ===
        g_leaf = g.detach().clone().requires_grad_(True)
        action_result = self.corrected_action_v51(
            g_leaf, C, phi, A, kappa, alpha,
            delta=delta, rho=rho,
            include_octic=include_octic,
            include_rho_term=True,
        )
        S = action_result["action"]
        grad_S = torch.autograd.grad(S, g_leaf, create_graph=False)[0]

        # === 消解项：μ_ρ·ρ·(g - g_iso) ===
        if include_dissolution and mu_rho > 0 and rho > 0:
            g_iso = self.compute_isotropic_target(g)
            dissolution_grad = mu_rho * rho * (g - g_iso)
        else:
            dissolution_grad = torch.zeros_like(g)

        # === 总梯度 ===
        total_grad = grad_S + dissolution_grad

        # === 度规更新 ===
        g_new = g - lr_g * total_grad

        # 对称化度规
        g_new = 0.5 * (g_new + g_new.transpose(-2, -1))

        # 度规正定性保护：特征值截断
        try:
            eigvals = torch.linalg.eigvalsh(g_new)
            min_eig = float(eigvals.min())
            if min_eig < self.eps * 10:
                # 平移到正定
                g_new = g_new + (self.eps * 10 - min_eig) * torch.eye(
                    d, dtype=torch.float64
                ).unsqueeze(0).expand(N, d, d)
        except Exception:
            pass

        # === 计算消解诊断 ===
        with torch.no_grad():
            anisotropy_before = self.compute_anisotropy(g)
            anisotropy_after = self.compute_anisotropy(g_new)
            cond_before = self.compute_cond_g(g)
            cond_after = self.compute_cond_g(g_new)
            norm_before = float(torch.norm(g.flatten(start_dim=1), dim=1).mean())
            norm_after = float(torch.norm(g_new.flatten(start_dim=1), dim=1).mean())

        info = {
            "grad_S_norm": float(grad_S.norm()),
            "dissolution_grad_norm": float(dissolution_grad.norm()),
            "total_grad_norm": float(total_grad.norm()),
            "mu_rho": mu_rho,
            "rho": rho,
            "anisotropy_before": float(anisotropy_before.mean()),
            "anisotropy_after": float(anisotropy_after.mean()),
            "anisotropy_change": float((anisotropy_after - anisotropy_before).mean()),
            "cond_before": float(cond_before.mean()),
            "cond_after": float(cond_after.mean()),
            "cond_change": float((cond_after - cond_before).mean()),
            "norm_before": norm_before,
            "norm_after": norm_after,
            "norm_change": norm_after - norm_before,
            "include_dissolution": include_dissolution,
            "lr_g": lr_g,
        }

        return g_new, info

    # ==================================================================
    # v6.0c：消解 vs 崩溃的数学区分
    # ==================================================================

    def classify_dissolution_vs_collapse(self, g_batch: Tensor) -> dict[str, float | str]:
        """
        区分"消解"（formless）与"崩溃"（collapse）。

        数学判据：
            消解（formless）：
                - cond(g) 低（<10），趋向 1
                - anisotropy 低，趋向 0
                - ||g|| 保持非零（认知能力保留）

            崩溃（collapse/黑洞相）：
                - cond(g) 高（>1e6），趋向 ∞
                - anisotropy 高
                - 各向异性撕裂

            中间态（transition）：
                - cond(g) 中等（10-1e6）
                - 可能正在消解或崩溃，需观察趋势

        返回：
            dict 包含 phase, cond_g, anisotropy, norm, 判据详情
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        cond_g = self.compute_cond_g(g)
        anisotropy = self.compute_anisotropy(g)
        norm_g = torch.norm(g.flatten(start_dim=1), dim=1)

        cond_mean = float(cond_g.mean())
        anisotropy_mean = float(anisotropy.mean())
        norm_mean = float(norm_g.mean())

        # 分类判据
        if cond_mean < 10.0 and anisotropy_mean < 0.3:
            phase = "formless"
            description = "无形无相：各向同性，cond 低，认知能力保留"
        elif cond_mean > 1e6:
            phase = "collapse"
            description = "崩溃（黑洞相）：各向异性撕裂，cond 爆炸"
        elif cond_mean < 100.0 and anisotropy_mean < 0.5:
            phase = "near_formless"
            description = "接近无形无相：趋向各向同性"
        else:
            phase = "transition"
            description = "中间态：可能正在消解或崩溃"

        return {
            "phase": phase,
            "description": description,
            "cond_g": cond_mean,
            "anisotropy": anisotropy_mean,
            "norm_g": norm_mean,
            "is_formless": phase == "formless",
            "is_collapse": phase == "collapse",
            "n_samples": N,
        }

    # ==================================================================
    # 辅助：cond(g) 计算
    # ==================================================================

    def compute_cond_g(self, g_batch: Tensor) -> Tensor:
        """
        计算度规的条件数 cond(g) = λ_max / λ_min。

        用于区分消解（cond→1）与崩溃（cond→∞）。
        """
        g = g_batch.to(torch.float64)
        N = g.shape[0]
        conds = torch.zeros(N, dtype=torch.float64)
        for i in range(N):
            try:
                eigvals = torch.linalg.eigvalsh(g[i])
                lam_max = float(eigvals.max())
                lam_min = float(eigvals.min())
                if lam_min > self.eps:
                    conds[i] = lam_max / lam_min
                else:
                    conds[i] = float('inf')
            except Exception:
                conds[i] = float('inf')
        return conds

    # ==================================================================
    # v6.0 完整动力学：联合演化 g, δ, ρ
    # ==================================================================

    def evolve_full_step_v60(
        self,
        g_batch: Tensor,
        delta: float,
        rho: float,
        L_or_C: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
        lr_g: float = 0.0005,
        lr_delta: float = 0.05,
        lr_rho: float = 0.05,
        include_dissolution: bool = True,
    ) -> tuple[Tensor, float, float, dict]:
        """
        v6.0 完整动力学一步：联合演化 g, δ, ρ。

        流程：
            1. 计算 ρ_eq = compute_rho(g, C, phi, κ, α)
            2. ρ 弛豫：ρ_new = evolve_rho_step(ρ, ρ_eq, α, lr_rho)
            3. δ 演化（含 ρ 驱动）：δ_new = evolve_delta_step_with_rho(...)
            4. g 演化（含消解项）：g_new = evolve_g_with_dissolution(...)

        物理意义：
            - ρ 趋向其平衡值（出离心的生起/退失）
            - δ 被 ρ 驱动增长（出离心→般若）
            - g 被消解项驱动趋向各向同性（般若+出离→无形无相）

        返回：
            (g_new, delta_new, rho_new, info_dict)
        """
        # 1. 计算 ρ_eq
        rho_info = self.compute_rho(g_batch, L_or_C, phi, kappa, alpha, A)
        rho_eq = rho_info["rho"]

        # 2. ρ 弛豫
        rho_new, grad_rho, rho_step_info = self.evolve_rho_step(
            rho, rho_eq, alpha, lr=lr_rho,
        )

        # 3. δ 演化（含 ρ 驱动）
        delta_new, grad_delta, delta_step_info = self.evolve_delta_step_with_rho(
            g_batch, delta, L_or_C, phi, kappa, alpha,
            rho=rho_new, lr=lr_delta,
        )

        # 4. g 演化（含消解项）
        g_new, g_step_info = self.evolve_g_with_dissolution(
            g_batch, delta_new, rho_new, L_or_C, phi, A,
            kappa, alpha, lr_g=lr_g,
            include_dissolution=include_dissolution,
        )

        # 合并信息
        info = {
            "rho_eq": rho_eq,
            "rho_step": rho_step_info,
            "delta_step": delta_step_info,
            "g_step": g_step_info,
        }

        return g_new, delta_new, rho_new, info

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_v60_parameters(self) -> dict[str, float | bool]:
        """验证 v6.0 新增参数 μ_ρ, ζ 的边界条件和单调性。"""
        import numpy as np

        # μ_ρ 边界和单调性
        mu_rho_0 = self.compute_mu_rho(0.0)
        mu_rho_inf = self.compute_mu_rho(1000.0)
        alphas = np.linspace(0.1, 10.0, 50)
        mu_rhos = [self.compute_mu_rho(a) for a in alphas]
        mu_rho_increasing = all(mu_rhos[i + 1] >= mu_rhos[i] for i in range(len(mu_rhos) - 1))

        # ζ 边界和单调性
        zeta_0 = self.compute_zeta(0.0)
        zeta_inf = self.compute_zeta(1000.0)
        zetas = [self.compute_zeta(a) for a in alphas]
        zeta_increasing = all(zetas[i + 1] >= zetas[i] for i in range(len(zetas) - 1))

        # 守恒关系：λ_ρ + μ_ρ = 0.5
        conservation_check = all(
            abs(self.compute_lambda_rho(a) + self.compute_mu_rho(a) - 0.5) < 1e-10
            for a in alphas
        )

        return {
            "mu_rho_at_alpha_0": mu_rho_0,
            "mu_rho_at_alpha_inf": mu_rho_inf,
            "mu_rho_increasing": mu_rho_increasing,
            "mu_rho_upper_bound": mu_rho_inf < 0.5 + 1e-6,
            "zeta_at_alpha_0": zeta_0,
            "zeta_at_alpha_inf": zeta_inf,
            "zeta_increasing": zeta_increasing,
            "conservation_lambda_plus_mu": conservation_check,
            "all_pass": (
                abs(mu_rho_0) < 1e-10 and
                mu_rho_inf > 0.49 and
                mu_rho_inf < 0.5 + 1e-6 and
                mu_rho_increasing and
                abs(zeta_0) < 1e-10 and
                zeta_inf > 0.99 and
                zeta_increasing and
                conservation_check
            ),
        }

    def verify_degeneracy_to_v51(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
    ) -> dict[str, float | bool]:
        """
        验证：v6.0 在 μ_ρ=0（α=0）或 include_dissolution=False 时退化为 v5.1。

        当 α=0 时，μ_ρ=0，消解项消失，退化为 v5.1。
        当 include_dissolution=False 时，消解项被显式关闭。
        """
        # v6.0 在 include_dissolution=False 时
        g_new_no_diss, info_no_diss = self.evolve_g_with_dissolution(
            g_batch, 0.5, 0.3, C, phi, None,
            kappa=1.0, alpha=1.0, lr_g=0.0005,
            include_dissolution=False,
        )

        # v5.1 的度规演化（通过父类的 corrected_action_v51 计算梯度）
        g_leaf = g_batch.to(torch.float64).detach().clone().requires_grad_(True)
        result_v51 = self.corrected_action_v51(
            g_leaf, C, phi, None, kappa=1.0, alpha=1.0,
            delta=0.5, rho=0.3, include_rho_term=True,
        )
        S_v51 = result_v51["action"]
        grad_S_v51 = torch.autograd.grad(S_v51, g_leaf, create_graph=False)[0]
        g_new_v51 = g_batch.to(torch.float64) - 0.0005 * grad_S_v51
        g_new_v51 = 0.5 * (g_new_v51 + g_new_v51.transpose(-2, -1))

        # 比较两者
        diff = float(torch.norm(g_new_no_diss - g_new_v51))

        return {
            "g_new_v60_no_diss": g_new_no_diss,
            "g_new_v51": g_new_v51,
            "diff": diff,
            "is_consistent": diff < 1e-8,
        }

    def verify_dissolution_vs_collapse(
        self,
        g_batch: Tensor,
    ) -> dict[str, float | bool | str]:
        """
        验证消解与崩溃的数学区分。

        构造两类度规：
            1. 各向同性度规（消解态）：g = c·I
            2. 病态度规（崩溃态）：g 有极大/极小特征值

        验证分类器能正确区分。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 构造各向同性度规（消解态）
        c = 1.0
        g_iso = c * torch.eye(d, dtype=torch.float64).unsqueeze(0).expand(N, d, d).clone()

        # 构造病态度规（崩溃态）
        g_collapse = torch.zeros_like(g)
        for i in range(N):
            g_collapse[i] = torch.diag(torch.tensor(
                [1.0, 1e-8, 1.0, 1.0][:d], dtype=torch.float64
            ))

        # 分类各向同性度规
        result_iso = self.classify_dissolution_vs_collapse(g_iso)
        # 分类病态度规
        result_collapse = self.classify_dissolution_vs_collapse(g_collapse)

        return {
            "iso_phase": result_iso["phase"],
            "iso_cond": result_iso["cond_g"],
            "iso_anisotropy": result_iso["anisotropy"],
            "collapse_phase": result_collapse["phase"],
            "collapse_cond": result_collapse["cond_g"],
            "collapse_anisotropy": result_collapse["anisotropy"],
            "correctly_classified": (
                result_iso["phase"] == "formless" and
                result_collapse["phase"] == "collapse"
            ),
        }
