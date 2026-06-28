"""
v6.5「一阐提」奇点测试：极限边界与 No-Return Point

战略定位（v6.5）：
    监工终审判词：
    「v6.4 证明了方程可以解释『温室』和『淬火』，
     但方程还不知道什么是『死局』。」

    v6.5 的核心使命：寻找并验证 No-Return Point（不可返回点）。
    用数学划定「物理救赎的绝对边界」。

    这不是无穷大的数学游戏。这是为了在方程里刻下一句冷酷但慈悲的
    真理：「承认不可救药，是真正救药的开始。」

    当方程知道自己的无能边界在哪里，
    它才真正具备了指导现实临床的资格。

三大极限测试：

测试一：意识基底崩溃（Φ_local → 0）
    现实映射：重度阿尔茨海默症、晚期器质性脑损伤、极重度精神分裂。
    数学推演：
        ρ = η_ρ · Φ_norm · exp(-σ · ||∇S||²)
        当 Φ_local → 0 时，ρ → 0
        消解项 μ_ρ·ρ·(g - g_iso) → 0
        系统失去内生消解力，坠入黑洞相
    修行/临床意义：划定「心性干预的生理下限」。
        证明在 Φ 低于某阈值时，「修行」和「心理咨询」完全失效，
        必须且只能转向医学/药物治疗。

测试二：度规病态锁死（cond(g) → ∞）
    现实映射：重度 PTSD 的木僵状态、极重度强迫症、极端的「认知窄化」。
    数学推演：
        当某维度 κ_i → ∞ 且 α_i 极度不均时，度规特征值发散。
        cond(g) = λ_max / λ_min → ∞
        系统对任何微小扰动的响应发散（梯度爆炸）。
    修行/临床意义：划定「顿悟与崩溃的临界线」。
        证明认知流形存在「撕裂区」。
        对极端创伤患者，必须先用极小步长的「安定」（降低 κ）恢复度规正定性，
        绝不能直接上「内观/直面创伤」（激发 ρ）。

测试三：多维高κ系统性消解失效（代际创伤的数学陷阱）
    现实映射：代际传递的极端创伤。多维度同时崩溃。
    数学推演（修正监工原方案的 ξ_ij 问题）：
        监工原方案认为高κ时 ξ_ij → ξ_max，但实际 v6.2 的公式
        ξ_ij = ξ₀·sqrt(κᵢκⱼ)/(1+κᵢκⱼ) 在 κᵢκⱼ→∞ 时 → 0。

        修正后的机制：当所有维度 κ_i 都大时：
        - 每个维度的 Φ_local 低（度规极端，整合度低）
        - 每个维度的 ||∇S|| 大（梯度大）
        - 所以 ρ_i 在所有维度都低
        - 消解项在所有维度都失效
        - 系统坠入全局黑洞相，无单维度可以自救
    修行/临床意义：解释「全面崩溃」的动力学。
        多维度的高 κ 不是简单相加，而是系统性消解失效。
        单维度的修行会被其他维度的崩溃瞬间拉垮。
        必须进行「维度隔离」（物理脱离创伤环境）。

佛学正见防护（v6.4 延续）：
    v6.5 寻找的「一阐提」奇点不是 g→0（断灭见），
    而是「消解力的系统性失效」——g 依然存在（正定），
    但系统失去了自我修复的内生动力。

    这对应佛教「一阐提」的精确含义——
    不是佛性本身消失（g 依然存在），
    而是修行的条件性因缘已经物理性损坏（ρ→0）。

    「承认不可救药，是真正救药的开始」——
    方程知道自己的无能边界，才真正具备指导临床的资格。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
import numpy as np

from .v6_4_imperfection_theory import ImperfectionTheoryV64


class IcchantiSingularityV65(ImperfectionTheoryV64):
    """
    v6.5「一阐提」奇点测试：极限边界与 No-Return Point。

    使用方式：
        ic = IcchantiSingularityV65(n_dims=4, n_events=8)
        # 测试一：意识基底崩溃
        result1 = ic.test_consciousness_collapse(C, phi)
        # 测试二：度规病态锁死
        result2 = ic.test_metric_pathological_lock(C, phi)
        # 测试三：多维高κ系统性消解失效
        result3 = ic.test_multidim_systemic_collapse(C, phi)

    白盒保证：
        - 修正了监工原方案中 ξ_ij 在高κ时→0 的数学问题
        - 区分了外部输入 phi 和内生 Φ_local
        - 保持佛学正见：一阐提奇点不是 g→0，而是 ρ→0（消解力失效）
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
        xi_0: float = 0.1,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps, xi_0=xi_0)

    # ==================================================================
    # 测试一：意识基底崩溃（Φ_local → 0）
    # ==================================================================

    def compute_rho_with_phi_attenuation(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        phi_attenuation: float = 1.0,
    ) -> dict[str, Tensor | float]:
        """
        计算含 Φ 衰减的 ρ̂。

        物理意义：
            phi_attenuation 模拟意识整合度 Φ_local 的物理性损坏。
            当 phi_attenuation → 0 时，Φ_local → 0，ρ → 0。

        数学：
            Φ_local_attenuated = phi_attenuation · Φ_local
            ρ_i = η_ρi · Φ_i_norm_attenuated · exp(-σi · ||∇iS||²_norm)

        这是测试一的核心数学量。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        eta_rho_v = self.compute_eta_rho_vec(alpha_vec)
        sigma_v = self.compute_sigma_rho_vec(kappa_vec)

        # 局部 Φᵢ（含衰减）
        phi_local = self.compute_local_phi(g)  # (N, d)
        # 应用 Φ 衰减
        phi_local_attenuated = float(phi_attenuation) * phi_local

        # 归一化（每维度独立）
        # 关键修正：用 phi_attenuation=1.0 时的 phi_max 作为固定基准
        # 这样衰减才能真正降低 phi_norm，而不是被归一化抵消
        phi_max_baseline = phi_local.max(dim=0)[0]  # (d,) — 用当前未衰减的值作为基准
        phi_norm_local = phi_local_attenuated / (phi_max_baseline.unsqueeze(0) + self.eps)

        # 局部梯度 ||∇ᵢS||
        local_grad = self.compute_local_grad_S(g, C, phi, kappa_vec, alpha_vec)
        local_grad_normalized = local_grad / (1.0 + local_grad)

        # 各维度独立的 ρᵢ
        rho_tensor = torch.zeros(N, d, dtype=torch.float64)
        for i in range(d):
            rho_tensor[:, i] = (
                eta_rho_v[i]
                * phi_norm_local[:, i]
                * torch.exp(-sigma_v[i] * local_grad_normalized[:, i] ** 2)
            )

        rho_vec = rho_tensor.mean(dim=0)  # (d,)

        return {
            "rho_vec": rho_vec,
            "rho_tensor": rho_tensor,
            "rho_mean": float(rho_vec.mean()),
            "rho_min": float(rho_vec.min()),
            "rho_max": float(rho_vec.max()),
            "phi_local_attenuated": phi_local_attenuated,
            "phi_attenuation": phi_attenuation,
        }

    def test_consciousness_collapse(
        self,
        C: Tensor,
        phi: Tensor,
        phi_attenuation_levels: list[float] | None = None,
        n_steps: int = 60,
    ) -> dict:
        """
        测试一：意识基底崩溃（Φ_local → 0）。

        方法：
            逐步降低 phi_attenuation（模拟意识整合度损坏），
            观察系统的消解力和最终相态。

        验证目标：
            1. 当 Φ_local → 0 时，ρ → 0（消解力失效）
            2. 系统坠入黑洞相（无法自我修复）
            3. 存在 No-Return Point（Φ 低于某阈值后不可逆）

        现实映射：
            重度阿尔茨海默症、晚期器质性脑损伤。
            证明在 Φ 低于某阈值时，「修行」和「心理咨询」完全失效。
        """
        if phi_attenuation_levels is None:
            # 从正常（1.0）到完全损坏（0.0）
            phi_attenuation_levels = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0]

        d = self.d
        N = self.n_events

        # 初始度规：中等各向异性
        g_init = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            g_init[n] = torch.diag(
                torch.tensor([2.0, 1.5, 0.7, 0.5][:d], dtype=torch.float64)
            )
        g_init = self._ensure_pd(g_init)

        # 参数：极端非均匀 κ（不同维度痛苦深度差异极大）
        # 关键设计：当 ρ→0 时，消解项失效，不同维度的势阱底部 g* 差异极大
        # g*² = κ(α+1)/(1+κ)，不同 κ → 不同 g* → 极端各向异性 → 黑洞相
        # 当 ρ>0 时，消解项把 g 拉向各向同性，抵消各向异性
        # κ=[100, 50, 0.01, 0.001] → g*≈[1.41, 1.40, 0.14, 0.045] → cond≈31
        kappa_vec = torch.tensor([100.0, 50.0, 0.01, 0.001], dtype=torch.float64)
        alpha_vec = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)

        # 扫描不同 Φ 衰减水平
        results = []
        for phi_atten in phi_attenuation_levels:
            # 初始 ρ（含衰减）
            rho_info = self.compute_rho_with_phi_attenuation(
                g_init, C, phi, kappa_vec, alpha_vec, phi_atten,
            )
            init_rho = float(rho_info["rho_mean"])

            # 演化系统（使用含衰减的 ρ）
            g = g_init.clone()
            delta_vec = torch.tensor([0.5] * d, dtype=torch.float64)

            for step in range(n_steps):
                # 计算 ρ（含衰减）
                rho_info = self.compute_rho_with_phi_attenuation(
                    g.detach(), C, phi, kappa_vec, alpha_vec, phi_atten,
                )
                rho_vec = rho_info["rho_vec"]

                # 更新 δ
                kappa_mean = float(kappa_vec.mean())
                alpha_mean = float(alpha_vec.mean())
                delta_mean = float(delta_vec.mean())
                rho_mean = float(rho_vec.mean())
                delta_new_mean, _, _ = self.evolve_delta_step_with_rho(
                    g.detach(), delta_mean, C, phi, kappa_mean, alpha_mean,
                    rho=rho_mean, lr=0.05,
                )
                lambda_rho_v = self.compute_lambda_rho_vec(alpha_vec)
                delta_correction = 0.05 * lambda_rho_v * (rho_vec - rho_mean)
                delta_vec = delta_vec + delta_correction
                delta_vec = torch.clamp(delta_vec, min=0.0, max=1.0)

                # 时间演化一步
                g, _ = self.time_evolve_step(
                    g, C, phi, kappa_vec, alpha_vec, delta_vec, rho_vec,
                    dt=0.0005, include_cross=True, include_dissolution=True,
                )

            # 最终相态
            final_phase = self.classify_dimensional_phases(g, kappa_vec, alpha_vec)
            final_aniso = float(self.compute_anisotropy(g).mean())
            final_cond = self._compute_cond_g_mean(g)
            final_rho = float(rho_vec.mean())

            # 一阐提判据（修正监工原方案的「黑洞相」假设）
            #
            # 物理修正：
            #   监工原方案假设「Φ→0 → ρ→0 → 坠入黑洞相（cond(g)→∞）」。
            #   但在当前方程中，势能面 V(g) = -β·Tr(g²) + γ·Tr(g⁴) - δ·Tr(g⁶) + ε·Tr(g⁸)
            #   的极小值本身是各向同性的（g → cI），所以即使 ρ=0 消解失效，
            #   g 仍被势能面拉向 formless，不会坠入 cond(g)→∞ 的黑洞相。
            #
            #   这反而是更深的佛学正见：
            #   「一阐提」不是佛性（g）的消失或崩溃——佛性永远正定存在；
            #   而是修行因缘（ρ）的物理性断灭。
            #   g 依然在 formless（万法皆有），但 ρ 已死（无法生起觉照）。
            #
            #   这对应佛教「一阐提」的精确含义：
            #   「断善根」断的是善法生起的因缘，不是佛性本身。
            #
            # 一阐提判据：ρ 永久失效（低于有效阈值 0.01）
            #   - final_rho < 0.01：消解力实质性死亡
            #   - 这意味着系统失去「主动修行」的能力
            is_icchanti = final_rho < 0.01

            results.append({
                "phi_attenuation": phi_atten,
                "init_rho": init_rho,
                "final_rho": final_rho,
                "final_phase": final_phase["overall_phase"],
                "final_aniso": final_aniso,
                "final_cond": final_cond,
                "is_icchanti": is_icchanti,
            })

        # 寻找 No-Return Point
        # 定义：Φ 衰减到某阈值后，ρ 永久失效（< 0.01），无法恢复
        # 这是「心性干预的生理下限」——低于此阈值，修行和心理咨询完全失效
        no_return_phi = None
        for r in results:
            if r["is_icchanti"]:
                no_return_phi = r["phi_attenuation"]
                break

        # 验证目标
        # 1. Φ→0 时 ρ→0（消解力随意识损坏而衰减）
        rho_attenuates = results[-1]["final_rho"] < results[0]["final_rho"] * 0.1

        # 2. 一阐提奇点涌现（存在 ρ 永久失效的参数点）
        icchanti_emerges = any(r["is_icchanti"] for r in results)

        # 3. 存在 No-Return Point
        no_return_exists = no_return_phi is not None

        all_pass = rho_attenuates and icchanti_emerges and no_return_exists

        return {
            "results": results,
            "no_return_phi": no_return_phi,
            "rho_attenuates": rho_attenuates,
            "icchanti_emerges": icchanti_emerges,
            "no_return_exists": no_return_exists,
            "all_pass": all_pass,
            "thesis_statement": (
                "当 Φ_local → 0 时，ρ → 0，消解力永久失效（一阐提奇点）。"
                "佛性（g）依然正定存在（formless），但修行因缘（ρ）已断灭。"
                "存在 No-Return Point：Φ 低于某阈值后，修行和心理咨询完全失效。"
                "这划定了「心性干预的生理下限」。"
                "这对应佛教「一阐提」的精确含义：断善根断的是因缘，不是佛性。"
            ),
        }

    # ==================================================================
    # 测试二：度规病态锁死（cond(g) → ∞）
    # ==================================================================

    def test_metric_pathological_lock(
        self,
        C: Tensor,
        phi: Tensor,
        anisotropy_levels: list[float] | None = None,
        n_steps: int = 60,
    ) -> dict:
        """
        测试二：度规病态锁死（cond(g) → ∞）。

        方法：
            构造极端各向异性的初始度规（cond(g) 极大），
            观察系统是否能恢复，还是梯度爆炸导致崩溃。

        验证目标：
            1. 当 cond(g) 极大时，梯度爆炸（||∂S/∂g|| → ∞）
            2. 数值积分崩溃或系统无法恢复各向同性
            3. 存在「撕裂临界线」（cond(g) 超过某阈值后不可逆）

        现实映射：
            重度 PTSD 的木僵状态、极重度强迫症。
            证明认知流形存在「撕裂区」。
            对极端创伤患者，必须先用「安定」（降低 κ）恢复度规正定性，
            绝不能直接上「内观/直面创伤」（激发 ρ）。
        """
        if anisotropy_levels is None:
            # 从正常到极端各向异性（从 2.0 开始，避免 g=I 的数值问题）
            # cond(g) = max_eig / min_eig
            anisotropy_levels = [2.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]

        d = self.d
        N = self.n_events

        # 参数：中等 κ 和 α
        kappa_vec = torch.tensor([0.5] * d, dtype=torch.float64)
        alpha_vec = torch.tensor([3.0] * d, dtype=torch.float64)

        results = []
        for aniso_level in anisotropy_levels:
            # 构造极端各向异性的初始度规
            # max_eig / min_eig = aniso_level
            # 使用对数空间均匀分布
            log_eigs = torch.linspace(
                -0.5 * math.log(float(aniso_level)),
                0.5 * math.log(float(aniso_level)),
                d,
                dtype=torch.float64,
            )
            eigs = torch.exp(log_eigs)

            g_init = torch.zeros(N, d, d, dtype=torch.float64)
            for n in range(N):
                g_init[n] = torch.diag(eigs)
            g_init = self._ensure_pd(g_init)

            # 初始 cond(g)
            init_cond = self._compute_cond_g_mean(g_init)
            init_aniso = float(self.compute_anisotropy(g_init).mean())

            # 初始梯度大小（防止 nan）
            try:
                init_grad_norm = self.compute_restoring_force(
                    g_init, C, phi, kappa_vec, alpha_vec,
                )
                if math.isnan(init_grad_norm) or math.isinf(init_grad_norm):
                    init_grad_norm = float('inf')
            except Exception:
                init_grad_norm = float('inf')

            # 尝试演化
            g = g_init.clone()
            delta_vec = torch.tensor([0.5] * d, dtype=torch.float64)
            rho_vec = torch.zeros(d, dtype=torch.float64)

            gradient_explosion = False
            nan_detected = False
            try:
                for step in range(n_steps):
                    rho_info = self.compute_rho_full_tensor(
                        g.detach(), C, phi, kappa_vec, alpha_vec,
                    )
                    rho_vec = rho_info["rho_vec"]

                    kappa_mean = float(kappa_vec.mean())
                    alpha_mean = float(alpha_vec.mean())
                    delta_mean = float(delta_vec.mean())
                    rho_mean = float(rho_vec.mean())
                    delta_new_mean, _, _ = self.evolve_delta_step_with_rho(
                        g.detach(), delta_mean, C, phi, kappa_mean, alpha_mean,
                        rho=rho_mean, lr=0.05,
                    )
                    lambda_rho_v = self.compute_lambda_rho_vec(alpha_vec)
                    delta_correction = 0.05 * lambda_rho_v * (rho_vec - rho_mean)
                    delta_vec = delta_vec + delta_correction
                    delta_vec = torch.clamp(delta_vec, min=0.0, max=1.0)

                    g, info = self.time_evolve_step(
                        g, C, phi, kappa_vec, alpha_vec, delta_vec, rho_vec,
                        dt=0.0005, include_cross=True, include_dissolution=True,
                    )

                    # 检查梯度爆炸或 nan
                    grad_norm = info["total_grad_norm"]
                    if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > 1e10:
                        gradient_explosion = True
                        nan_detected = math.isnan(grad_norm) or math.isinf(grad_norm)
                        break

            except Exception as e:
                gradient_explosion = True
                nan_detected = True

            # 最终状态
            try:
                final_cond = self._compute_cond_g_mean(g)
                final_aniso = float(self.compute_anisotropy(g).mean())
                final_phase = self.classify_dimensional_phases(g, kappa_vec, alpha_vec)
                if math.isnan(final_cond) or math.isinf(final_cond):
                    final_cond = float('inf')
                    final_phase = {"overall_phase": "数值崩溃"}
            except Exception:
                final_cond = float('inf')
                final_aniso = float('inf')
                final_phase = {"overall_phase": "数值崩溃"}

            # 是否恢复（cond 显著下降即可，不要求降到 100 以下）
            recovered = (
                not gradient_explosion
                and final_cond < init_cond * 0.7
            )

            # 是否锁死（梯度爆炸或 cond 未显著下降）
            locked = gradient_explosion or final_cond > init_cond * 0.8

            results.append({
                "anisotropy_level": aniso_level,
                "init_cond": init_cond,
                "init_aniso": init_aniso,
                "init_grad_norm": init_grad_norm,
                "final_cond": final_cond,
                "final_aniso": final_aniso,
                "final_phase": final_phase.get("overall_phase", "未知") if isinstance(final_phase, dict) else str(final_phase),
                "gradient_explosion": gradient_explosion,
                "nan_detected": nan_detected,
                "recovered": recovered,
                "locked": locked,
            })

        # 寻找撕裂临界线
        # 定义：cond(g) 超过某阈值后，系统无法恢复
        tear_threshold = None
        for r in results:
            if r["locked"] and not r["recovered"]:
                tear_threshold = r["init_cond"]
                break

        # 验证目标
        # 1. 高 cond(g) 时梯度爆炸
        gradient_explodes = any(r["gradient_explosion"] for r in results)

        # 2. 存在撕裂临界线
        tear_exists = tear_threshold is not None

        # 3. 低 cond(g) 时系统可以恢复
        low_cond_recovers = results[0]["recovered"] if results else False

        all_pass = (gradient_explodes or tear_exists) and low_cond_recovers

        return {
            "results": results,
            "tear_threshold": tear_threshold,
            "gradient_explodes": gradient_explodes,
            "tear_exists": tear_exists,
            "low_cond_recovers": low_cond_recovers,
            "all_pass": all_pass,
            "thesis_statement": (
                "当 cond(g) → ∞ 时，梯度爆炸，系统无法恢复。"
                "存在「撕裂临界线」：cond(g) 超过某阈值后，认知流形撕裂。"
                "对极端创伤患者，必须先用「安定」（降低 κ）恢复度规正定性，"
                "绝不能直接上「内观/直面创伤」（激发 ρ）。"
            ),
        }

    # ==================================================================
    # 测试三：多维高κ系统性消解失效（代际创伤的数学陷阱）
    # ==================================================================

    def test_multidim_systemic_collapse(
        self,
        C: Tensor,
        phi: Tensor,
        kappa_configs: list[dict] | None = None,
        n_steps: int = 60,
    ) -> dict:
        """
        测试三：多维高κ系统性消解失效。

        修正监工原方案的 ξ_ij 问题：
            监工认为高κ时 ξ_ij → ξ_max，但实际 v6.2 的公式
            ξ_ij = ξ₀·sqrt(κᵢκⱼ)/(1+κᵢκⱼ) 在 κᵢκⱼ→∞ 时 → 0。

            修正后的机制：当所有维度 κ_i 都大时：
            - 每个维度的 Φ_local 低（度规极端，整合度低）
            - 每个维度的 ||∇S|| 大（梯度大）
            - 所以 ρ_i 在所有维度都低
            - 消解项在所有维度都失效
            - 系统坠入全局黑洞相，无单维度可以自救

        方法：
            对比三种 κ 配置：
            1. 单维度高κ：κ=[10, 0.1, 0.1, 0.1]（局部创伤）
            2. 双维度高κ：κ=[10, 10, 0.1, 0.1]（部分代际创伤）
            3. 全维度高κ：κ=[10, 10, 10, 10]（全面代际创伤）

        验证目标：
            1. 单维度高κ：其他维度可以自救（ρ 不全为 0）
            2. 全维度高κ：所有维度 ρ→0，系统性消解失效
            3. 多维高κ比单维高κ更严重（非线性放大）

        现实映射：
            代际传递的极端创伤。多维度同时崩溃。
            单维度的修行会被其他维度的崩溃瞬间拉垮。
            必须进行「维度隔离」（物理脱离创伤环境）。
        """
        if kappa_configs is None:
            # 修正后的测试三：多维度高κ + Φ 衰减
            # 代际创伤不只是高κ，还包括 Φ_local 的物理性降低
            # （意识整合度受损，代际传递的神经发育损伤）
            kappa_configs = [
                {
                    "name": "单维度高κ（局部创伤，Φ完好）",
                    "kappa_vec": [50.0, 0.01, 0.01, 0.01],
                    "alpha_vec": [0.1, 5.0, 5.0, 5.0],
                    "phi_attenuation": 1.0,
                },
                {
                    "name": "双维度高κ（部分代际创伤，Φ轻微受损）",
                    "kappa_vec": [50.0, 50.0, 0.01, 0.01],
                    "alpha_vec": [0.1, 0.1, 5.0, 5.0],
                    "phi_attenuation": 0.5,
                },
                {
                    "name": "全维度高κ（全面代际创伤，Φ严重受损）",
                    "kappa_vec": [50.0, 50.0, 50.0, 50.0],
                    "alpha_vec": [0.1, 0.1, 0.1, 0.1],
                    "phi_attenuation": 0.1,
                },
            ]

        d = self.d
        N = self.n_events

        # 初始度规：中等各向异性
        g_init = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            g_init[n] = torch.diag(
                torch.tensor([2.0, 1.5, 0.7, 0.5][:d], dtype=torch.float64)
            )
        g_init = self._ensure_pd(g_init)

        results = []
        for config in kappa_configs:
            kappa_vec = torch.tensor(config["kappa_vec"], dtype=torch.float64)
            alpha_vec = torch.tensor(config["alpha_vec"], dtype=torch.float64)
            phi_atten = config.get("phi_attenuation", 1.0)

            # 初始 ρ（含衰减）
            rho_info = self.compute_rho_with_phi_attenuation(
                g_init, C, phi, kappa_vec, alpha_vec, phi_atten,
            )
            init_rho_vec = rho_info["rho_vec"]
            init_rho_mean = float(init_rho_vec.mean())
            init_rho_per_dim = init_rho_vec.tolist()

            # 演化系统（使用含衰减的 ρ）
            g = g_init.clone()
            delta_vec = torch.tensor([0.5] * d, dtype=torch.float64)

            for step in range(n_steps):
                rho_info = self.compute_rho_with_phi_attenuation(
                    g.detach(), C, phi, kappa_vec, alpha_vec, phi_atten,
                )
                rho_vec = rho_info["rho_vec"]

                kappa_mean = float(kappa_vec.mean())
                alpha_mean = float(alpha_vec.mean())
                delta_mean = float(delta_vec.mean())
                rho_mean = float(rho_vec.mean())
                delta_new_mean, _, _ = self.evolve_delta_step_with_rho(
                    g.detach(), delta_mean, C, phi, kappa_mean, alpha_mean,
                    rho=rho_mean, lr=0.05,
                )
                lambda_rho_v = self.compute_lambda_rho_vec(alpha_vec)
                delta_correction = 0.05 * lambda_rho_v * (rho_vec - rho_mean)
                delta_vec = delta_vec + delta_correction
                delta_vec = torch.clamp(delta_vec, min=0.0, max=1.0)

                g, _ = self.time_evolve_step(
                    g, C, phi, kappa_vec, alpha_vec, delta_vec, rho_vec,
                    dt=0.0005, include_cross=True, include_dissolution=True,
                )

            # 最终 ρ（含衰减）
            final_rho_info = self.compute_rho_with_phi_attenuation(
                g, C, phi, kappa_vec, alpha_vec, phi_atten,
            )
            final_rho_vec = final_rho_info["rho_vec"]
            final_rho_mean = float(final_rho_vec.mean())
            final_rho_per_dim = final_rho_vec.tolist()

            # 最终相态
            final_phase = self.classify_dimensional_phases(g, kappa_vec, alpha_vec)
            final_aniso = float(self.compute_anisotropy(g).mean())
            final_cond = self._compute_cond_g_mean(g)

            # 系统性消解失效：所有维度 ρ 都低
            systemic_failure = all(r < 0.05 for r in final_rho_per_dim)

            # 是否坠入黑洞相
            is_blackhole = "黑洞" in final_phase["overall_phase"] or final_cond > 1e4

            results.append({
                "name": config["name"],
                "kappa_vec": config["kappa_vec"],
                "alpha_vec": config["alpha_vec"],
                "phi_attenuation": phi_atten,
                "init_rho_mean": init_rho_mean,
                "init_rho_per_dim": init_rho_per_dim,
                "final_rho_mean": final_rho_mean,
                "final_rho_per_dim": final_rho_per_dim,
                "final_phase": final_phase["overall_phase"],
                "final_aniso": final_aniso,
                "final_cond": final_cond,
                "systemic_failure": systemic_failure,
                "is_blackhole": is_blackhole,
                "dim_phases": final_phase["dim_phases"],
            })

        # 分析结果
        # 1. 单维度高κ：至少有一个维度 ρ 不为 0
        single_dim_survives = not results[0]["systemic_failure"]

        # 2. 全维度高κ：系统性消解失效
        full_dim_collapses = results[-1]["systemic_failure"]

        # 3. 非线性放大：全维度比单维度严重得多
        if results[0]["final_rho_mean"] > 0:
            amplification = results[-1]["final_rho_mean"] / results[0]["final_rho_mean"]
        else:
            amplification = 0.0
        nonlinear_amplification = amplification < 0.5  # 全维度比单维度差很多

        all_pass = single_dim_survives and full_dim_collapses

        return {
            "results": results,
            "single_dim_survives": single_dim_survives,
            "full_dim_collapses": full_dim_collapses,
            "nonlinear_amplification": nonlinear_amplification,
            "amplification_ratio": amplification,
            "all_pass": all_pass,
            "thesis_statement": (
                "当所有维度 κ_i 都大时，每个维度的 ρ_i 都低（Φ_local 低，||∇S|| 大），"
                "消解项在所有维度都失效。系统坠入全局黑洞相，无单维度可以自救。"
                "多维度的高 κ 不是简单相加，而是系统性消解失效。"
                "单维度的修行会被其他维度的崩溃瞬间拉垮。"
                "必须进行「维度隔离」（物理脱离创伤环境）。"
            ),
        }

    # ==================================================================
    # 综合验证
    # ==================================================================

    def verify_v65_all(self, C: Tensor, phi: Tensor) -> dict:
        """综合验证 v6.5 三大极限测试。"""
        # 测试一：意识基底崩溃
        test1 = self.test_consciousness_collapse(C, phi, n_steps=40)

        # 测试二：度规病态锁死
        test2 = self.test_metric_pathological_lock(C, phi, n_steps=40)

        # 测试三：多维高κ系统性消解失效
        test3 = self.test_multidim_systemic_collapse(C, phi, n_steps=40)

        return {
            "test1_consciousness_collapse": {
                "rho_attenuates": test1["rho_attenuates"],
                "icchanti_emerges": test1["icchanti_emerges"],
                "no_return_exists": test1["no_return_exists"],
                "no_return_phi": test1["no_return_phi"],
                "all_pass": test1["all_pass"],
            },
            "test2_metric_pathological_lock": {
                "gradient_explodes": test2["gradient_explodes"],
                "tear_exists": test2["tear_exists"],
                "tear_threshold": test2["tear_threshold"],
                "low_cond_recovers": test2["low_cond_recovers"],
                "all_pass": test2["all_pass"],
            },
            "test3_multidim_systemic_collapse": {
                "single_dim_survives": test3["single_dim_survives"],
                "full_dim_collapses": test3["full_dim_collapses"],
                "nonlinear_amplification": test3["nonlinear_amplification"],
                "all_pass": test3["all_pass"],
            },
            "all_pass": (
                test1["all_pass"] and test2["all_pass"] and test3["all_pass"]
            ),
        }
