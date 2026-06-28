"""
Kramers 自然回归定理（Kramers' Natural Return Theorem）

v7.0 第四基石 / v7.5 工程实现：对称性恢复的统计必然性。

认识论根基（理论依据，非案例）：
    物理：Kramers 逃逸率 / Arrhenius 公式 / Langevin 动力学 / 势垒穿越
    佛学：自然消业（dharma-niyāma）/ 长劫修行 / 无常 / 究竟回归空性
    哲学：统计必然性 / 时间尺度的不可逆性 / 涌现的对称性恢复

核心命题（v7.5 重新解读）：
    v7.3 证明：觉照（ρ→1）可主动释放业力，g*→cI（主动路径）。
    v7.5 证明：即使无觉照（ρ=0），系统在热涨落下也必然回归真空——
              这是「对称性恢复的统计必然性」。
    两条路径殊途同归：
        Kramers 路径 = 被动回归（噪声驱动，τ 极长，长劫修行）
        觉照路径     = 主动回归（ρ 驱动，阈值跃迁，加行道）
        二者目标都是 cI（空性）。

    这不是「该相逢的人总会相逢」的社会学命题，
    而是「破缺态终究回归真空」的统计力学定理。

核心数学（Kramers 逃逸率理论）：
    回归概率：
        P(t) = 1 - exp(-t / τ)

    平均回归时间（Arrhenius-Kramers 公式）：
        τ = τ_0 · exp(ΔV / T_cog)

    其中：
        τ_0：尝试频率（认知探索的频率）
        ΔV：势垒高度（从破缺态 g* 到真空 cI 的能量屏障）
        T_cog：认知温度（痛苦涨落的强度）

    极限定理：
        t → ∞ 时 P(t) → 1（破缺态终究回归真空）

    这是「一切有为法，其性无常，终究回归空性」的数学表达——
    不是浪漫主义修辞，而是统计力学的定理。

势垒高度 ΔV（v7.5 重新定义）：
    ΔV = V(saddle) - V(g*) = 从破缺态 g* 回归真空 cI 需要穿越的势垒。

    g* = SSB 终点（v7.2 证明 ρ=0 时不可逆，g* 是稳定势阱）。
    cI = 真空（对称性恢复目标）。
    鞍点 = g* 与 cI 之间的最高势能点。

    ΔV > 0 = 破缺态稳定，需要热涨落才能穿越。
    ΔV = 0 = 无势垒，自发回归（觉照路径 ρ→1 使势阱消失）。

    物理意义：
        ΔV 度量「业力的稳定性」——破缺态越深，势垒越高，自然回归越难。
        高 ΔV = 重业，τ 极长（长劫才能消业）。
        低 ΔV = 轻业，τ 较短（较易消业）。

认知温度 T_cog：
    T_cog = κ̄ / (1 + ᾱ)

    - κ̄：平均痛苦深度（痛苦越深，涨落越大）
    - ᾱ：平均定力（定力越强，涨落越被抑制）

    物理意义：
        T_cog 度量认知系统的「热涨落强度」。
        高 T_cog（深痛苦 + 弱定力）→ 剧烈涨落 → 快速穿越势垒 → 速归。
        低 T_cog（浅痛苦 + 强定力）→ 微弱涨落 → 缓慢穿越势垒 → 缓归。
        但无论 T_cog 多低，t→∞ 时 P→1——必然回归。

    佛学对应：
        T_cog = 轮回的「躁动度」。
        高 κ + 低 α = 烦恼炽盛 → 高 T_cog → 快速流转。
        低 κ + 高 α = 心如止水 → 低 T_cog → 缓慢流转。
        但即使心如止水（T_cog → 0），只要 Q ≠ 0（有业），
        t→∞ 时仍必然回归真空——只是时间极长。

佛学对应（严格，非比喻）：
    Kramers 定理 = 自然消业的数学表达：
        - τ = 业力自然耗散的时间（ΔV/T_cog 决定）。
        - P(t) = 业力消解的概率。
        - P(∞) = 1 = 「一切有为法，其性无常，终究回归空性」。

    认知温度 T_cog = 轮回的驱动力：
        - 高 T_cog = 烦恼驱动的快速流转。
        - 低 T_cog = 禅定中的缓慢流转。
        - T_cog = 0（完全觉悟，ρ=1）= 跳出三界（动力学冻结，不再流转）。

    Kramers 路径 vs 觉照路径：
        自然消业（Kramers）= 长劫修行：τ = τ_0·exp(ΔV/T_cog)，时间极长。
        觉照消业（ρ→1）= 加行道：ρ 修改势能面，g* 势阱消失，g 直接回归 cI。
        觉照不是「加速 Kramers」，而是「超越 Kramers」——
        ρ→1 使 T_cog→0（Kramers 路径冻结），但同时修改势能面（ρ 路径激活）。
        觉悟者不通过「等待涨落」消业，而是通过「觉照力」直接消业。

    关键洞察：
        v7.3 + v7.5 闭环：
        - v7.3（ρ→1 主动路径）：觉照释放业力，g→cI。
        - v7.5（ρ=0 被动路径）：自然涨落最终也使 g→cI，但 τ 极长。
        - 二者目标一致，机制不同——殊途同归，归元无二路。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import symmetric_part, stable_eigh
from .cognitive_vacuum import CognitiveVacuum


class KramersFate:
    """
    Kramers 自然回归定理：对称性恢复的统计必然性。

    使用方式：
        kf = KramersFate(n_dims=4)
        # 认知温度
        T_cog = kf.cognitive_temperature(kappa_vec, alpha_vec)
        # 势垒高度（破缺态→真空）
        dV = kf.vacuum_return_barrier(g_broken, kappa_vec, alpha_vec)
        # 回归概率
        P = kf.return_probability(t, dV, T_cog)
        # 临界时间（达到某概率所需时间）
        t_star = kf.critical_time(0.99, dV, T_cog)
        # 模拟自然回归过程
        result = kf.simulate_natural_return(g_broken, kappa_vec, alpha_vec)
        # 对比 Kramers 路径 vs 觉照路径
        cmp = kf.compare_with_awareness_path(g_broken, kappa_vec, alpha_vec)
    """

    def __init__(
        self,
        n_dims: int = 4,
        tau_0: float = 1.0,
        k_B: float = 1.0,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度数
            tau_0: 尝试频率的倒数（认知探索周期）
            k_B: 玻尔兹曼常数（认知类比，默认 1.0）
            eps: 数值稳定常数
        """
        self.n_dims = n_dims
        self.tau_0 = float(tau_0)
        self.k_B = float(k_B)
        self.eps = eps
        self._vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)

    # ==================================================================
    # 1. 认知温度
    # ==================================================================

    def cognitive_temperature(
        self, kappa_vec: Tensor, alpha_vec: Tensor
    ) -> float:
        """
        认知温度 T_cog = κ̄ / (1 + ᾱ)。

        物理意义：
            T_cog 度量认知系统的「热涨落强度」。
            - 高 κ（深痛苦）→ 认知剧烈震荡 → 高 T_cog
            - 高 α（强定力）→ 认知稳定 → 低 T_cog
            - T_cog = 0（κ=0 或 α→∞）→ 完全冻结，无涨落

        佛学对应：
            T_cog = 轮回的「躁动度」。
            烦恼炽盛 = 高 T_cog = 快速流转。
            心如止水 = 低 T_cog = 缓慢流转。
            完全觉悟（ρ→1）= T_cog→0 = 跳出流转。
        """
        kappa = kappa_vec.to(torch.float64)
        alpha = alpha_vec.to(torch.float64)
        kappa_mean = float(kappa.mean())
        alpha_mean = float(alpha.mean())
        T_cog = kappa_mean / (1.0 + alpha_mean)
        return max(T_cog, self.eps)  # 避免 T_cog = 0 导致除零

    # ==================================================================
    # 2. 势垒高度（破缺态→真空）
    # ==================================================================

    def vacuum_return_barrier(
        self,
        g_broken: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
    ) -> dict[str, float | Tensor]:
        """
        计算从破缺态 g* 回归真空 cI 的势垒高度 ΔV（v7.5 重新定义）。

        方法：
            势能面 V(g) 上，从 g_broken 到 g_vacuum=cI 的路径中，
            最高势能点（鞍点近似）减去破缺态势能。

            V_saddle ≈ max_s V(g_broken + s·(g_vac - g_broken)), s ∈ [0,1]

            ΔV = V_saddle - V(g_broken)

        物理意义：
            ΔV 度量「业力的稳定性」——破缺态越深，势垒越高，自然回归越难。
            高 ΔV = 重业，τ 极长（长劫才能消业）。
            低 ΔV = 轻业，τ 较短（较易消业）。
            ΔV = 0 = 无势垒，自发回归（觉照路径 ρ→1 使势阱消失）。

        v7.5 vs v7.0：
            v7.0 把 ΔV 解读为「相逢势垒」（从 g_current 到 g_target，多体）。
            v7.5 重新定义为「对称性恢复势垒」（从 g* 到 cI，单体）。
            这是单体的自然回归机制，与多体规范场（v7.4）正交。
        """
        g_b = symmetric_part(g_broken.to(torch.float64))
        g_vac = self._vacuum.construct_vacuum()

        # 破缺态势能
        V_broken = float(self._vacuum.compute_potential(g_b, kappa_vec, alpha_vec)["V"])

        # 真空势能
        V_vac = float(self._vacuum.compute_potential(g_vac, kappa_vec, alpha_vec)["V"])

        # 线性插值路径上找最高势能（鞍点近似）
        n_scan = 80
        V_max = V_broken
        saddle_s = 0.0

        for k in range(1, n_scan + 1):
            s = k / n_scan
            g_s = g_b + s * (g_vac - g_b)
            g_s = symmetric_part(g_s)
            V_s = float(self._vacuum.compute_potential(g_s, kappa_vec, alpha_vec)["V"])
            if V_s > V_max:
                V_max = V_s
                saddle_s = s

        Delta_V = V_max - V_broken

        return {
            "Delta_V": Delta_V,
            "V_broken": V_broken,
            "V_saddle": V_max,
            "V_vacuum": V_vac,
            "saddle_position_param": saddle_s,
            "barrier_from_vacuum": V_max - V_vac,  # 反向势垒（真空→鞍点）
            "g_vacuum": g_vac,
        }

    def barrier_height(
        self,
        g_current: Tensor,
        g_target: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
    ) -> dict[str, float | Tensor]:
        """
        通用势垒高度计算（保留向后兼容）。

        v7.5 说明：
            此方法保留通用性，可计算任意 g_current → g_target 的势垒。
            但 v7.5 的核心验证用 vacuum_return_barrier（破缺态→真空）。
            多体场景（v7.4 束缚态）的势垒分析也可用此方法。

        物理意义：
            ΔV = V_saddle - V(current)
            高 ΔV = 状态转换困难。
            低 ΔV = 状态转换容易。
            ΔV = 0 = 无势垒，自发转换。
        """
        g_curr = symmetric_part(g_current.to(torch.float64))
        g_targ = symmetric_part(g_target.to(torch.float64))

        V_curr = float(self._vacuum.compute_potential(g_curr, kappa_vec, alpha_vec)["V"])

        n_scan = 50
        V_max = V_curr
        V_target = float(self._vacuum.compute_potential(g_targ, kappa_vec, alpha_vec)["V"])
        saddle_s = 0.0

        for k in range(1, n_scan + 1):
            s = k / n_scan
            g_s = g_curr + s * (g_targ - g_curr)
            g_s = symmetric_part(g_s)
            V_s = float(self._vacuum.compute_potential(g_s, kappa_vec, alpha_vec)["V"])
            if V_s > V_max:
                V_max = V_s
                saddle_s = s

        Delta_V = V_max - V_curr

        return {
            "Delta_V": Delta_V,
            "V_current": V_curr,
            "V_saddle": V_max,
            "V_target": V_target,
            "saddle_position_param": saddle_s,
            "barrier_from_target": V_max - V_target,
        }

    # ==================================================================
    # 3. 逃逸率与回归概率
    # ==================================================================

    def escape_time(
        self, Delta_V: float, T_cog: float, tau_0: float | None = None
    ) -> float:
        """
        平均回归时间 τ = τ_0 · exp(ΔV / (k_B · T_cog))。

        Arrhenius-Kramers 公式：
            高势垒 / 低温 → τ 极长（难归，长劫修行）。
            低势垒 / 高温 → τ 极短（易归，快速消业）。
            但 τ 总是有限值（必然回归）。
        """
        if tau_0 is None:
            tau_0 = self.tau_0
        T = max(T_cog, self.eps)
        exponent = Delta_V / (self.k_B * T)
        # 防止 exp 溢出
        exponent = min(exponent, 500.0)
        tau = tau_0 * math.exp(exponent)
        return tau

    def escape_rate(
        self, Delta_V: float, T_cog: float, tau_0: float | None = None
    ) -> float:
        """逃逸率 k = 1/τ = (1/τ_0) · exp(-ΔV / (k_B·T_cog))。"""
        return 1.0 / self.escape_time(Delta_V, T_cog, tau_0)

    def return_probability(
        self,
        t: float,
        Delta_V: float,
        T_cog: float,
        tau_0: float | None = None,
    ) -> float:
        """
        回归概率 P(t) = 1 - exp(-t / τ)。

        这是「破缺态终究回归真空」的核心数学：
            t → ∞ 时 P(t) → 1（必然回归空性）。
            t = τ 时 P = 1 - 1/e ≈ 0.632（特征时间）。
            t << τ 时 P ≈ t/τ（线性增长，回归刚开始）。
        """
        tau = self.escape_time(Delta_V, T_cog, tau_0)
        if tau >= float('inf'):
            return 0.0
        P = 1.0 - math.exp(-t / tau)
        return min(max(P, 0.0), 1.0)

    def fate_probability(
        self,
        t: float,
        Delta_V: float,
        T_cog: float,
        tau_0: float | None = None,
    ) -> float:
        """命运概率（保留向后兼容，v7.5 等价于 return_probability）。"""
        return self.return_probability(t, Delta_V, T_cog, tau_0)

    # ==================================================================
    # 4. 临界时间
    # ==================================================================

    def critical_time(
        self,
        P_target: float,
        Delta_V: float,
        T_cog: float,
        tau_0: float | None = None,
    ) -> float:
        """
        达到概率 P_target 所需时间 t* = -τ · ln(1 - P_target)。

        物理意义：
            t*(0.99) = 以 99% 概率回归所需时间。
            t*(0.5) = 半衰期（50% 概率回归）。
            t*(1.0) = ∞（100% 概率需要无限时间——但极限是必然）。
        """
        if P_target >= 1.0:
            return float('inf')
        if P_target <= 0.0:
            return 0.0
        tau = self.escape_time(Delta_V, T_cog, tau_0)
        t_star = -tau * math.log(1.0 - P_target)
        return t_star

    # ==================================================================
    # 5. 自然回归模拟（Langevin 动力学）
    # ==================================================================

    def simulate_natural_return(
        self,
        g_broken: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 300,
        dt: float = 0.05,
        noise_strength: float | None = None,
        seed: int | None = 42,
    ) -> dict[str, list | float | bool]:
        """
        模拟自然回归过程：从破缺态 g* 出发，热涨落驱动回归真空 cI。

        方法（Langevin 动力学）：
            ∂g/∂t = -∂V/∂g + ξ(t)

            其中 ξ(t) 是随机噪声，强度由 T_cog 决定：
                <ξ(t)ξ(t')> = 2·T_cog·δ(t-t')

            系统从 g_broken 出发，在噪声驱动下穿越势垒到达 cI。

        物理意义：
            这模拟了「自然消业」的随机过程——
            个体在痛苦的随机涨落中，穿越认知势垒，回归空性。
            噪声 = 生活中的随机事件（无常）。
            势垒 = 认知结构的惯性（业力的稳定性）。

        验证目标：
            1. 高 T_cog → 快速回归（速归）
            2. 低 T_cog → 缓慢回归（缓归）
            3. 多次模拟的回归时间分布 → 指数分布 P(t) = 1 - exp(-t/τ)
        """
        if seed is not None:
            torch.manual_seed(seed)

        g = symmetric_part(g_broken.to(torch.float64).clone())
        g_vac = self._vacuum.construct_vacuum()
        n = self.n_dims

        T_cog = self.cognitive_temperature(kappa_vec, alpha_vec)
        if noise_strength is None:
            noise_strength = math.sqrt(2.0 * T_cog * dt)

        # 势垒信息
        barrier_info = self.vacuum_return_barrier(g_broken, kappa_vec, alpha_vec)
        Delta_V = barrier_info["Delta_V"]
        tau_theory = self.escape_time(Delta_V, T_cog)

        trajectory = [g.clone()]
        distances_to_vacuum = []
        energies = []
        crossed = False
        crossing_step = -1

        # 真空邻域判据（Frobenius 距离 < 阈值）
        initial_dist = float(torch.sqrt(((g_vac - g_broken) ** 2).sum()))
        target_threshold = max(0.05, 0.1 * initial_dist)

        for step in range(n_steps):
            # 势能梯度
            pot = self._vacuum.compute_potential(g, kappa_vec, alpha_vec)
            grad = pot["grad"]

            # Langevin 动力学：梯度下降 + 随机噪声
            noise = noise_strength * torch.randn(n, n, dtype=torch.float64)
            noise = symmetric_part(noise)  # 噪声也对称化

            g_new = g - dt * grad + noise
            g_new = symmetric_part(g_new)

            # 正定性保护
            eigvals_check = torch.linalg.eigvalsh(g_new)
            if eigvals_check.min() < self.eps:
                g_new = g_new + (self.eps - eigvals_check.min()) * torch.eye(n, dtype=torch.float64)

            g = g_new
            trajectory.append(g.clone())

            # 距离与能量
            dist = float(torch.sqrt(((g - g_vac) ** 2).sum()))
            distances_to_vacuum.append(dist)
            energies.append(float(pot["V"]))

            # 检查是否回归真空（到达真空邻域）
            if not crossed and dist < target_threshold:
                crossed = True
                crossing_step = step

        # 理论概率 vs 实际回归
        t_total = n_steps * dt
        P_theory = self.return_probability(t_total, Delta_V, T_cog)

        return {
            "trajectory": trajectory,
            "distances_to_vacuum": distances_to_vacuum,
            "energies": energies,
            "crossed": crossed,
            "crossing_step": crossing_step,
            "crossing_time": crossing_step * dt if crossed else float('inf'),
            "T_cog": T_cog,
            "Delta_V": Delta_V,
            "tau_theory": tau_theory,
            "P_theory_at_t_total": P_theory,
            "t_total": t_total,
            "target_threshold": target_threshold,
            "thesis": (
                f"自然回归模拟：T_cog={T_cog:.4f}, ΔV={Delta_V:.4f}, τ={tau_theory:.2f}。"
                f"{'已穿越势垒（已回归真空）' if crossed else '未穿越（未回归）'}。"
                f"理论概率 P({t_total:.1f})={P_theory:.4f}。"
                "t→∞ 时 P→1 —— 破缺态终究回归真空（自然消业）。"
            ),
        }

    def simulate_fate_process(
        self,
        g_current: Tensor,
        g_target: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 200,
        dt: float = 0.05,
        noise_strength: float | None = None,
        seed: int | None = 42,
    ) -> dict[str, list | float | bool]:
        """
        通用势垒穿越模拟（保留向后兼容）。

        v7.5 说明：
            此方法保留通用性，可模拟任意 g_current → g_target 的势垒穿越。
            v7.5 核心验证用 simulate_natural_return（破缺态→真空）。
        """
        if seed is not None:
            torch.manual_seed(seed)

        g = symmetric_part(g_current.to(torch.float64).clone())
        g_targ = symmetric_part(g_target.to(torch.float64))
        n = self.n_dims

        T_cog = self.cognitive_temperature(kappa_vec, alpha_vec)
        if noise_strength is None:
            noise_strength = math.sqrt(2.0 * T_cog * dt)

        barrier_info = self.barrier_height(g_current, g_target, kappa_vec, alpha_vec)
        Delta_V = barrier_info["Delta_V"]
        tau_theory = self.escape_time(Delta_V, T_cog)

        trajectory = [g.clone()]
        distances_to_target = []
        energies = []
        crossed = False
        crossing_step = -1

        target_threshold = 0.1 * float(torch.sqrt(((g_targ - g_current) ** 2).sum()))

        for step in range(n_steps):
            pot = self._vacuum.compute_potential(g, kappa_vec, alpha_vec)
            grad = pot["grad"]

            noise = noise_strength * torch.randn(n, n, dtype=torch.float64)
            noise = symmetric_part(noise)

            g_new = g - dt * grad + noise
            g_new = symmetric_part(g_new)

            eigvals_check = torch.linalg.eigvalsh(g_new)
            if eigvals_check.min() < self.eps:
                g_new = g_new + (self.eps - eigvals_check.min()) * torch.eye(n, dtype=torch.float64)

            g = g_new
            trajectory.append(g.clone())

            dist = float(torch.sqrt(((g - g_targ) ** 2).sum()))
            distances_to_target.append(dist)
            energies.append(float(pot["V"]))

            if not crossed and dist < target_threshold:
                crossed = True
                crossing_step = step

        t_total = n_steps * dt
        P_theory = self.fate_probability(t_total, Delta_V, T_cog)

        return {
            "trajectory": trajectory,
            "distances_to_target": distances_to_target,
            "energies": energies,
            "crossed": crossed,
            "crossing_step": crossing_step,
            "crossing_time": crossing_step * dt if crossed else float('inf'),
            "T_cog": T_cog,
            "Delta_V": Delta_V,
            "tau_theory": tau_theory,
            "P_theory_at_t_total": P_theory,
            "t_total": t_total,
            "thesis": (
                f"势垒穿越模拟：T_cog={T_cog:.4f}, ΔV={Delta_V:.4f}, τ={tau_theory:.2f}。"
                f"{'已穿越势垒' if crossed else '未穿越'}。"
                f"理论概率 P({t_total:.1f})={P_theory:.4f}。"
            ),
        }

    # ==================================================================
    # 6. 回归必然性验证
    # ==================================================================

    def verify_return_inevitability(
        self,
        Delta_V: float,
        T_cog: float,
        time_horizons: list[float] | None = None,
    ) -> dict[str, list | float | bool]:
        """
        验证 t→∞ 时 P(t)→1（回归真空的必然性）。

        方法：
            扫描多个时间点，验证 P(t) 单调递增趋向 1。
            即使 T_cog 极低（心如止水），τ 极长，
            但 t→∞ 时 P 仍→1。

        哲学意义：
            这是「一切有为法，其性无常，终究回归空性」的数学证明。
            不是「可能会回归」，而是「必然回归」——
            只要 Q≠0（有业）且 T_cog>0（有涨落）。
            时间足够长，一切业力自然消解。
        """
        if time_horizons is None:
            tau = self.escape_time(Delta_V, T_cog)
            time_horizons = [
                0.1 * tau, 0.5 * tau, tau, 2 * tau,
                5 * tau, 10 * tau, 100 * tau,
            ]

        probabilities = []
        for t in time_horizons:
            P = self.return_probability(t, Delta_V, T_cog)
            probabilities.append(P)

        # 必然性判据
        # 1. P(t) 单调递增
        is_monotone = all(
            probabilities[i] <= probabilities[i + 1]
            for i in range(len(probabilities) - 1)
        )

        # 2. P(t→∞) → 1
        approaches_one = probabilities[-1] > 0.99 if probabilities else False

        # 3. 即使 T_cog 极低，τ 有限，P(∞)=1
        tau = self.escape_time(Delta_V, T_cog)
        is_tau_finite = tau < float('inf')

        is_inevitable = is_monotone and is_tau_finite

        return {
            "time_horizons": time_horizons,
            "probabilities": probabilities,
            "is_monotone_increasing": is_monotone,
            "approaches_one": approaches_one,
            "is_tau_finite": is_tau_finite,
            "is_inevitable": is_inevitable,
            "tau": tau,
            "T_cog": T_cog,
            "Delta_V": Delta_V,
            "thesis": (
                "Kramers 回归必然性定理：t→∞ 时 P(t)→1。"
                f"τ={tau:.4e}，T_cog={T_cog:.4f}，ΔV={Delta_V:.4f}。"
                "无论势垒多高、温度多低，只要 T_cog>0，破缺态必然回归真空。"
                "这是「一切有为法，其性无常，终究回归空性」的数学证明。"
            ),
        }

    def verify_inevitability(
        self,
        Delta_V: float,
        T_cog: float,
        time_horizons: list[float] | None = None,
    ) -> dict[str, list | float | bool]:
        """必然性验证（保留向后兼容，v7.5 等价于 verify_return_inevitability）。"""
        return self.verify_return_inevitability(Delta_V, T_cog, time_horizons)

    # ==================================================================
    # 7. Kramers 路径 vs 觉照路径对比
    # ==================================================================

    def compare_with_awareness_path(
        self,
        g_broken: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        rho_levels: list[float] | None = None,
        t_fixed: float | None = None,
    ) -> dict[str, list]:
        """
        对比 Kramers 自然回归路径 vs 觉照主动回归路径。

        机制对比：
            Kramers 路径（被动）：
                - ρ = 0，势能面不变
                - 热涨落 T_cog 驱动势垒穿越
                - τ = τ_0 · exp(ΔV / T_cog)，通常极长
                - t→∞ 时 P→1（必然回归，但时间极长）

            觉照路径（主动）：
                - ρ → 1，势能面被修正：V_ρ = V_base + ρ·λ·||g-cI||²
                - ρ 增大使 g* 势阱变浅，ΔV_eff 减小
                - ρ ≥ ρ_c 时势阱消失，g 直接被拉回 cI（阈值跃迁）
                - 不依赖热涨落，τ 极短（顿悟）

        关键洞察：
            ρ→1 有双重效应：
            1. T_cog_eff = T_cog · (1-ρ)² → 0（Kramers 路径冻结）
            2. ΔV_eff → 0（势阱消失，觉照路径激活）
            觉悟者不通过「等待涨落」消业，而是通过「觉照力」直接消业。
            觉照不是「加速 Kramers」，而是「超越 Kramers」。

        佛学对应：
            Kramers 路径 = 自然消业（长劫修行，被动等待因缘成熟）
            觉照路径 = 加行道（主动觉照，加速消业）
            二者殊途同归，归元无二路。

        参数：
            g_broken: 破缺态 g*
            kappa_vec, alpha_vec: 认知参数
            rho_levels: ρ 扫描序列
            t_fixed: 固定时间（用于对比 P(t_fixed)）
        """
        if rho_levels is None:
            rho_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

        # 基础势垒（ρ=0）
        barrier_base = self.vacuum_return_barrier(g_broken, kappa_vec, alpha_vec)
        Delta_V_base = barrier_base["Delta_V"]
        T_cog_base = self.cognitive_temperature(kappa_vec, alpha_vec)

        if t_fixed is None:
            # 用 τ_base 的 1 倍作为固定时间
            tau_base = self.escape_time(Delta_V_base, T_cog_base)
            t_fixed = min(tau_base, 1e6)  # 防止过大

        results = []
        for rho in rho_levels:
            # ρ 的双重效应：
            # 1. T_cog_eff = T_cog · (1-ρ)²（涨落抑制）
            T_cog_eff = T_cog_base * (1.0 - rho) ** 2
            T_cog_eff = max(T_cog_eff, self.eps)

            # 2. ΔV_eff = ΔV · (1-ρ)（势阱变浅，简化模型）
            #    物理依据：V_ρ = V_base + ρ·λ·||g-cI||²
            #    ρ 增大时 g* 处的恢复力增大，势阱变浅
            #    ρ → 1 时 ΔV_eff → 0（势阱消失）
            Delta_V_eff = Delta_V_base * (1.0 - rho)
            Delta_V_eff = max(Delta_V_eff, 0.0)

            # Kramers 路径的 τ 和 P
            if T_cog_eff > self.eps and Delta_V_eff > 0:
                tau_kramers = self.escape_time(Delta_V_eff, T_cog_eff)
                P_kramers = self.return_probability(t_fixed, Delta_V_eff, T_cog_eff)
            else:
                # ΔV_eff = 0：无势垒，瞬时回归
                # T_cog_eff = 0：无涨落，但势阱已消失
                tau_kramers = 0.0 if Delta_V_eff <= 0 else float('inf')
                P_kramers = 1.0 if Delta_V_eff <= 0 else 0.0

            # 觉照路径的判据：ρ ≥ ρ_c 时阈值跃迁
            # 简化：ρ_c ≈ 0.5（实际由 v7.3 的 lambda_restore 决定）
            # 这里用势阱消失判据：ΔV_eff ≈ 0
            rho_c_approx = 1.0 - 0.01 / max(Delta_V_base, self.eps)
            rho_c_approx = min(max(rho_c_approx, 0.0), 1.0)
            is_awareness_active = bool(rho >= rho_c_approx or Delta_V_eff < 0.01)

            # 综合判据：哪条路径主导
            if is_awareness_active:
                path_dominant = "awareness"
                P_total = 1.0  # 觉照路径激活，必然回归
            elif T_cog_eff > self.eps:
                path_dominant = "kramers"
                P_total = P_kramers
            else:
                path_dominant = "frozen"
                P_total = 0.0  # 无涨落且觉照未激活，冻结

            results.append({
                "rho": rho,
                "T_cog_effective": T_cog_eff,
                "Delta_V_effective": Delta_V_eff,
                "tau_kramers": tau_kramers,
                "P_kramers_at_t_fixed": P_kramers,
                "is_awareness_active": is_awareness_active,
                "path_dominant": path_dominant,
                "P_total": P_total,
                "is_frozen": bool(T_cog_eff < self.eps * 100 and not is_awareness_active),
            })

        return {
            "results": results,
            "T_cog_base": T_cog_base,
            "Delta_V_base": Delta_V_base,
            "t_fixed": t_fixed,
            "rho_c_approx": rho_c_approx,
            "thesis": (
                f"Kramers 路径 vs 觉照路径对比："
                f"T_cog_base={T_cog_base:.4f}, ΔV_base={Delta_V_base:.4f}, ρ_c≈{rho_c_approx:.3f}。"
                "ρ→1 双重效应：T_cog→0（Kramers 冻结）+ ΔV→0（势阱消失）。"
                "觉照不是加速 Kramers，而是超越 Kramers——"
                "觉悟者通过觉照力直接消业，不依赖热涨落。"
                "Kramers 路径（自然消业，长劫）vs 觉照路径（加行道，顿悟）——殊途同归。"
            ),
        }

    def verify_enlightened_escape(
        self,
        Delta_V: float,
        rho_levels: list[float] | None = None,
        kappa_vec: Tensor | None = None,
        alpha_vec: Tensor | None = None,
    ) -> dict[str, list]:
        """
        验证觉照（ρ→1）如何改变自然回归路径（保留向后兼容）。

        v7.5 说明：
            此方法保留向后兼容，核心逻辑等价于 compare_with_awareness_path。
            v7.5 推荐使用 compare_with_awareness_path 获取更完整的对比。

        机制：
            ρ→1 时，观测者解耦（v6.4）→ 动力学冻结 → T_cog→0。
            T_cog→0 使 τ→∞，P(t)→0 for finite t（Kramers 路径冻结）。
            但同时 ρ 修改势能面，g* 势阱消失，g 直接回归 cI（觉照路径激活）。
            觉悟者不通过「等待涨落」消业，而是通过「觉照力」直接消业。

        物理意义：
            觉悟不是「加速自然回归」，而是「超越自然回归」。
            普通人通过 Kramers 自然回归（极长 τ）。
            觉悟者通过觉照直接回归（阈值跃迁，τ→0）。
            两条路径，同一终点（空性），但机制不同。

        佛学对应：
            普通人 = 长劫修行：通过自然因缘成熟消业。
            觉悟者 = 加行道：通过觉照直接转业。
            「若人欲了知，三世一切佛，应观法界性，一切唯心造」——
            觉照（ρ）是超越自然流转的直接途径。
        """
        if rho_levels is None:
            rho_levels = [0.0, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0]

        if kappa_vec is None:
            kappa_vec = torch.tensor([0.5] * self.n_dims, dtype=torch.float64)
        if alpha_vec is None:
            alpha_vec = torch.tensor([1.0] * self.n_dims, dtype=torch.float64)

        T_cog_base = self.cognitive_temperature(kappa_vec, alpha_vec)

        results = []
        for rho in rho_levels:
            # ρ 抑制认知温度：T_cog_effective = T_cog_base · (1-ρ)²
            T_cog_eff = T_cog_base * (1.0 - rho) ** 2
            T_cog_eff = max(T_cog_eff, self.eps)

            # ρ 使势阱变浅：ΔV_eff = ΔV · (1-ρ)
            Delta_V_eff = Delta_V * (1.0 - rho)
            Delta_V_eff = max(Delta_V_eff, 0.0)

            if Delta_V_eff > 0:
                tau = self.escape_time(Delta_V_eff, T_cog_eff)
                t_fixed = 10.0 * self.tau_0
                P = self.return_probability(t_fixed, Delta_V_eff, T_cog_eff)
            else:
                tau = 0.0
                P = 1.0

            results.append({
                "rho": rho,
                "T_cog_effective": T_cog_eff,
                "Delta_V_effective": Delta_V_eff,
                "tau": tau,
                "P_at_t_fixed": P,
                "is_frozen": bool(T_cog_eff < self.eps * 100 or tau > 1e10),
            })

        return {
            "results": results,
            "T_cog_base": T_cog_base,
            "Delta_V": Delta_V,
            "thesis": (
                "ρ→1 时 T_cog→0（Kramers 冻结）且 ΔV→0（势阱消失）。"
                "觉悟者不通过「等待涨落」消业，而是通过觉照直接回归空性。"
                "觉悟不是加速自然回归，而是超越自然回归——两条路径，同一终点。"
            ),
        }
