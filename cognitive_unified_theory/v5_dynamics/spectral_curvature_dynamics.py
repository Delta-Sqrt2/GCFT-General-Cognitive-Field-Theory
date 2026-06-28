"""
任务一：谱曲率反馈与内生非线性动力学（物理修正）

战略定位（v4.1 任务一）：
    v4.0 的 log(cond(g)) 标量近似导致系统过阻尼，无法产生内生极限环（GAN 失败）。
    v4.1 废黜标量近似，引入谱曲率：度规 Laplace-Beltrami 算子特征值谱的
    非线性反馈。只有恢复真实的二次型曲率，系统才具备产生 Hopf 分岔
    （极限环）和混沌（扩散模型）的物理基础。

    陷阱三十八·算力病态降级：
        严禁计算完整 Riemann/Ricci 张量（8^5=32768 次循环）。
        严禁退回 log(cond) 标量近似。
        必须使用谱曲率反馈：N(g) = Σ λ_i log(λ_i)
        计算复杂度 O(d³)，全可微，autograd 兼容。

物理与哲学直觉：
    - 物理：谱曲率 = 度规特征值谱的 von Neumann 熵。
            N(g) = Σ λ_i log(λ_i) 度量度规的"信息熵"。
            当度规均匀（VAE 平坦态）→ N(g) ≈ -d log(d)（最大熵）
            当度规坍缩（白盒奇点）→ N(g) → 0（最小熵）
            这提供了非线性反馈：度规越病态，反馈越强。
    - 哲学：谱曲率是"认知信息密度"的度量。
            高熵 = 认知维度丰富（VAE）
            低熵 = 认知维度锁死（白盒/RL）
    - 工程：torch.linalg.eigh 计算特征值，全可微，O(d³)。

数学定义（严格可微，无降级）：
    谱曲率反馈：
        N(g) = Σ_i λ_i log(λ_i)
        其中 λ_i 是度规 g 的特征值（归一化后）

    更新后的保守作用量：
        S[g, φ] = ∫ ( N(g)/(2κ) + (1/2) g^μν (D_μφ)^T (D_νφ) ) dV dt

    非线性动力学方程：
        ġ = -(∂S/∂g + ∂F/∂ġ) / τ
        其中 N(g) 的非线性反馈使系统具备 Hopf 分岔能力。

    陷阱四十一·奇点污染：
        当 cond(g) > 10^10 时，主动截断，标记为"黑洞相"（-1）。
        严禁 NaN 传播或掩盖。
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from ..core.tensor_ops import (
    effective_rank,
    safe_inverse,
    stable_eigh,
    symmetric_part,
)


class SpectralCurvatureDynamics:
    """
    谱曲率反馈与内生非线性动力学。

    使用方式：
        dyn = SpectralCurvatureDynamics(n_dims=8, kappa=1.0, lambda_dissip=0.1)
        # 谱曲率反馈
        N = dyn.spectral_curvature(g)
        # 度规演化（含非线性反馈）
        g_dot = dyn.metric_velocity(g, A, phi, g_prev)
        # 演化一步
        g_new = dyn.evolve_step(g, A, phi, g_prev, dt=0.01)

    白盒保证：
        - 谱曲率 N(g) = Σ λ_i log(λ_i)，全可微（陷阱三十八）
        - 非线性反馈维持 Hopf 分岔能力
        - 奇点隔离协议（陷阱四十一）
        - 无标量近似降级
    """

    # 黑洞相标记值
    BLACK_HOLE_MARKER = -1.0
    SINGULARITY_THRESHOLD = 1e10  # cond(g) > 10^10 → 黑洞相

    def __init__(
        self,
        n_dims: int = 8,
        kappa: float = 1.0,
        lambda_dissip: float = 0.1,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            kappa: 痛苦耦合常数 κ
            lambda_dissip: 耗散系数 λ
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.kappa = float(kappa)
        self.lambda_dissip = float(lambda_dissip)
        self.eps = float(eps)

    # ==================================================================
    # 谱曲率反馈 N(g) = Σ λ_i log(λ_i)
    # ==================================================================

    def spectral_curvature(self, metric: Tensor) -> Tensor:
        """
        谱曲率反馈 N(g) = Σ_i λ_i log(λ_i) —— 度规信息熵。

        数学：
            1. 对度规 g 做特征分解：g = Q Λ Q^T
            2. 归一化特征值：p_i = λ_i / Σ λ_j
            3. 谱曲率：N(g) = Σ_i p_i log(p_i)
               （这是特征值分布的 von Neumann 熵的负值）

        物理：
            - N(g) 大（负值小）→ 度规均匀 → VAE 平坦态
            - N(g) 小（负值大）→ 度规坍缩 → 白盒奇点
            - N(g) 提供非线性反馈：度规越病态，反馈越强

        严禁：
            - 完整 Riemann/Ricci 张量计算（陷阱三十八）
            - log(cond) 标量近似（过阻尼）
            - Python for 循环遍历维度（必须向量化）

        全可微：torch.linalg.eigh 支持 autograd。
        """
        g = symmetric_part(metric.to(torch.float64))
        n = g.shape[-1]

        # 特征值分解（autograd 可微）
        eigvals = torch.linalg.eigvalsh(g)
        eigvals = torch.clamp(eigvals, min=self.eps)

        # 归一化特征值（概率分布）
        p = eigvals / eigvals.sum()

        # 谱曲率：N(g) = Σ p_i log(p_i)
        # 这是 von Neumann 熵的负值
        N = (p * torch.log(p)).sum()

        return N  # 负值，越接近 0 越均匀

    def spectral_curvature_gradient(self, metric: Tensor) -> Tensor:
        """
        谱曲率对度规的梯度 ∂N/∂g（autograd 反向传播）。

        数学：
            ∂N/∂g = autograd(N(g), g)

        物理：
            梯度方向指向度规均匀化方向（减小病态）。
            非线性反馈 = -∂N/∂g 驱动度规向 VAE 平坦态演化。
        """
        g = symmetric_part(metric.to(torch.float64))
        g_leaf = g.detach().clone().requires_grad_(True)

        N = self.spectral_curvature(g_leaf)

        grad_N = torch.autograd.grad(
            N, g_leaf, create_graph=False, retain_graph=False
        )[0]
        grad_N = symmetric_part(grad_N)

        return grad_N.detach()

    # ==================================================================
    # 更新后的保守作用量 S[g, φ]
    # ==================================================================

    def covariant_derivative(
        self,
        phi: Tensor,
        A: Tensor | None,
    ) -> Tensor:
        """
        协变导数 D_μ φ = ∂_μ φ - i A_μ φ（复用 v4.0 实现）。
        """
        phi = phi.to(torch.float64)
        N_events, d = phi.shape

        if A is None:
            phi_mean = phi.mean(dim=0, keepdim=True)
            D_phi = phi - phi_mean
            return D_phi

        A_phi = torch.einsum('mnij,nj->mi', A.reshape(A.shape[0], A.shape[1], d, d), phi)
        A_phi = A_phi / N_events

        phi_mean = phi.mean(dim=0, keepdim=True)
        D_phi = (phi - phi_mean) - A_phi
        return D_phi

    def compute_action(
        self,
        metric: Tensor,
        A: Tensor | None,
        phi: Tensor,
    ) -> dict[str, Tensor]:
        """
        更新后的保守作用量（含谱曲率）：

        S[g, φ] = -N(g)/(2κ) + (1/2) g^μν (D_μφ)^T (D_νφ)

        数学：
            - 谱曲率项 -N(g)/(2κ)：非线性痛苦势能（正代价）
              N(g) = Σ p_i log(p_i) ≤ 0 是负熵（von Neumann 熵的负值）
              -N(g) ≥ 0 是正的曲率代价，与 v4.0 的 R(g) = log(cond) ≥ 0 同号
            - 规范耦合项：规范场对认知的影响
            - S 大 → 系统远离平衡（高曲率 + 高耦合）
            - S 小 → 系统接近平衡

        关键改进（vs v4.0）：
            v4.0: R(g) = log(cond(g)) → 标量，过阻尼
            v4.1: -N(g) = -Σ p_i log(p_i) ≥ 0 → 非线性正代价，可产生 Hopf 分岔
        """
        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        # 谱曲率项：-N(g) / (2κ)（正代价，N(g) ≤ 0）
        N = self.spectral_curvature(g)
        kappa_safe = max(self.kappa, self.eps)
        curvature_term = -N / (2.0 * kappa_safe)  # -N ≥ 0，正代价

        # 协变导数
        D_phi = self.covariant_derivative(phi, A)  # (N, d)

        # 规范耦合项：(1/2) g^μν (D_μ φ)^T (D_ν φ)
        g_inv = safe_inverse(g, self.eps)
        DtD = D_phi.T @ D_phi
        coupling_term = 0.5 * torch.einsum('ij,ij->', g_inv, DtD)

        # 总作用量
        action = curvature_term + coupling_term

        return {
            "action": action,
            "curvature_term": curvature_term,
            "coupling_term": coupling_term,
            "spectral_curvature": N,
        }

    # ==================================================================
    # Rayleigh 耗散函数 F
    # ==================================================================

    def entropy_flow_divergence(
        self,
        phi: Tensor,
        metric: Tensor,
    ) -> Tensor:
        """熵流散度 ∇·J（复用 v4.0 实现）。"""
        phi = phi.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))

        N_events = phi.shape[0]
        phi_mean = phi.mean(dim=0, keepdim=True)
        phi_centered = phi - phi_mean
        cov = (phi_centered.T @ phi_centered) / N_events

        g_inv = safe_inverse(g, self.eps)
        diff = cov - g
        div_J = torch.einsum('ij,ij->', g_inv, diff)
        return div_J

    def rayleigh_dissipation(
        self,
        phi: Tensor,
        metric: Tensor,
    ) -> Tensor:
        """Rayleigh 耗散函数 F = (1/2) λ (∇·J)²。"""
        div_J = self.entropy_flow_divergence(phi, metric)
        F = 0.5 * self.lambda_dissip * div_J ** 2
        return F

    # ==================================================================
    # 非线性度规演化（含谱曲率反馈）
    # ==================================================================

    def metric_velocity(
        self,
        metric: Tensor,
        A: Tensor | None,
        phi: Tensor,
        metric_prev: Tensor | None = None,
    ) -> Tensor:
        """
        度规演化速率 ġ（含谱曲率非线性反馈）。

        数学：
            广义欧拉-拉格朗日方程：
            ġ = -(∂S/∂g + ∂F/∂ġ) / τ

            其中 S 包含谱曲率 N(g)，提供非线性反馈。
            τ 是弛豫时间（由度规特征值推导）。

        关键改进（vs v4.0）：
            v4.0: ∂S/∂g 只含 log(cond) 的线性梯度 → 过阻尼
            v4.1: ∂S/∂g 含 N(g) = Σ p_i log(p_i) 的非线性梯度 → Hopf 分岔能力

        陷阱四十一·奇点污染：
            当 cond(g) > 10^10 时，返回零速度（黑洞相锁定）。
            严禁 NaN 传播。
        """
        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        # === 奇点隔离协议（陷阱四十一）===
        eigvals_check = torch.linalg.eigvalsh(g)
        eigvals_check = torch.clamp(eigvals_check, min=self.eps)
        cond_g = float(eigvals_check.max() / eigvals_check.min())

        if cond_g >= self.SINGULARITY_THRESHOLD:
            # 黑洞相：返回零速度，锁定状态
            return torch.zeros_like(g)

        # === autograd 计算作用量梯度 ===
        g_leaf = g.detach().clone().requires_grad_(True)

        action_result = self.compute_action(g_leaf, A, phi)
        S = action_result["action"]

        grad_S = torch.autograd.grad(
            S, g_leaf, create_graph=False, retain_graph=False
        )[0]
        grad_S = symmetric_part(grad_S)

        # === Rayleigh 耗散力 ===
        if metric_prev is not None:
            g_prev = symmetric_part(metric_prev.to(torch.float64))
            g_dot_prev = (g - g_prev)
            dissip_force = -self.lambda_dissip * g_dot_prev
        else:
            dissip_force = torch.zeros_like(g)

        # === 弛豫时间 τ ===
        eigvals = torch.linalg.eigvalsh(g_leaf.detach())
        eigvals = torch.clamp(eigvals, min=self.eps)
        tau = float(eigvals.mean())

        # === 度规演化速率 ===
        g_dot = -(grad_S + dissip_force) / (tau + self.eps)

        return g_dot.detach()

    def evolve_step(
        self,
        metric: Tensor,
        A: Tensor | None,
        phi: Tensor,
        metric_prev: Tensor | None = None,
        dt: float = 0.01,
    ) -> tuple[Tensor, bool]:
        """
        演化一步（含奇点隔离协议）。

        返回：
            g_new: 新度规 (d, d)
            is_black_hole: 是否进入黑洞相
        """
        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        # 奇点检测
        eigvals_check = torch.linalg.eigvalsh(g)
        eigvals_check = torch.clamp(eigvals_check, min=self.eps)
        cond_g = float(eigvals_check.max() / eigvals_check.min())

        if cond_g >= self.SINGULARITY_THRESHOLD:
            # 黑洞相：锁定状态，返回原度规
            return g, True

        # RK4 积分（单步）
        k1 = self.metric_velocity(g, A, phi, metric_prev)
        k2 = self.metric_velocity(g + 0.5 * dt * k1, A, phi, g)
        k3 = self.metric_velocity(g + 0.5 * dt * k2, A, phi, g)
        k4 = self.metric_velocity(g + dt * k3, A, phi, g)

        g_new = g + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        g_new = symmetric_part(g_new)

        # 正则化：确保度规正定
        eigvals, eigvecs = stable_eigh(g_new)
        eigvals = torch.clamp(eigvals, min=self.eps, max=1e8)
        g_new = (eigvecs * eigvals) @ eigvecs.T
        g_new = symmetric_part(g_new)

        # 再次奇点检测
        eigvals_final = torch.linalg.eigvalsh(g_new)
        eigvals_final = torch.clamp(eigvals_final, min=self.eps)
        cond_final = float(eigvals_final.max() / eigvals_final.min())

        if cond_final >= self.SINGULARITY_THRESHOLD:
            return g_new, True

        return g_new, False

    # ==================================================================
    # 相空间坐标与透明度
    # ==================================================================

    def phase_space_coordinates(self, metric: Tensor) -> dict[str, Tensor]:
        """计算相空间坐标 (R, cond(g), N(g))。"""
        g = symmetric_part(metric.to(torch.float64))

        R = effective_rank(g)

        eigvals = torch.linalg.eigvalsh(g)
        eigvals = torch.clamp(eigvals, min=self.eps)
        cond = eigvals.max() / eigvals.min()

        N = self.spectral_curvature(g)

        return {
            "effective_rank": R,
            "condition_number": cond,
            "spectral_curvature": N,
            "metric_trace": g.trace(),
            "metric_norm": g.norm(),
        }

    def transparency(self, metric: Tensor) -> Tensor:
        """
        透明度 τ（灰盒/白盒/黑盒度量）。

        τ = 1 - H(p) / log(d)
        其中 p 是度规特征值分布，H 是香农熵。
        """
        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        eigvals = torch.linalg.eigvalsh(g)
        eigvals = torch.clamp(eigvals, min=self.eps)
        p = eigvals / eigvals.sum()
        H = -(p * torch.log(p)).sum()
        log_d = torch.log(torch.tensor(float(d), dtype=torch.float64))

        tau = 1.0 - H / log_d
        return tau

    # ==================================================================
    # 雅可比矩阵与稳定性分析（为任务二提供基础）
    # ==================================================================

    def jacobian_max_eigenvalue(
        self,
        metric: Tensor,
        A: Tensor | None,
        phi: Tensor,
        perturbation_scale: float = 1e-6,
    ) -> Tensor:
        """
        计算度规演化方程雅可比矩阵的最大特征值实部。

        数学：
            J = ∂ġ/∂g
            λ_max = max(Re(eigenvalues(J)))

        物理：
            λ_max < 0 → 不动点稳定（VAE）
            λ_max > 0 → 不动点失稳，Hopf 分岔（GAN 极限环）
            λ_max = 0 → 临界点

        实现：
            用有限差分近似雅可比矩阵（autograd 的二阶导数不稳定）。
            对每个方向扰动，计算 ġ 的变化。
        """
        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        # 基准速度
        g_dot_base = self.metric_velocity(g, A, phi, g)

        # 有限差分雅可比
        # J[i,j] = ∂ġ_i / ∂g_j
        # 由于 g 是对称矩阵，只考虑上三角部分
        jacobian = torch.zeros(d * d, d * d, dtype=torch.float64)

        for i in range(d):
            for j in range(d):
                # 扰动方向
                delta = torch.zeros_like(g)
                delta[i, j] = perturbation_scale
                delta[j, i] = perturbation_scale  # 对称化

                g_perturbed = g + delta
                g_dot_perturbed = self.metric_velocity(g_perturbed, A, phi, g)

                # 数值差分
                diff = (g_dot_perturbed - g_dot_base) / perturbation_scale
                jacobian[:, i * d + j] = diff.flatten()

        # 雅可比矩阵的最大特征值实部
        eigvals = torch.linalg.eigvals(jacobian)
        max_real = eigvals.real.max()

        return max_real

    # ==================================================================
    # GAN 内生极限环验证（防伪紧箍咒·禁受迫振动）
    # ==================================================================

    def verify_endogenous_limit_cycle(
        self,
        g0: Tensor,
        A: Tensor | None,
        phi: Tensor,
        steps: int = 100,
        dt: float = 0.01,
        cutoff_step: int = 50,
    ) -> dict[str, Tensor]:
        """
        验证 GAN 相态的内生极限环（禁受迫振动）。

        物理过程：
            1. 前 cutoff_step 步：注入外部周期性驱动力
            2. 后 (steps - cutoff_step) 步：切断外部输入
            3. 检验后段是否仍保持振荡（内生极限环）

        判据：
            后段 β 方差 > 0.01 → 内生极限环成立
            后段 β 方差 < 0.01 → 受迫振动（失败）

        返回：
            dict 包含：
                trajectory: 度规轨迹
                beta_curve: β 函数曲线
                early_beta_var: 前段 β 方差
                late_beta_var: 后段 β 方差
                is_endogenous: 是否内生极限环
        """
        g = symmetric_part(g0.to(torch.float64))
        g_prev = g.clone()
        phi_current = phi.to(torch.float64)

        trajectory = []
        beta_curve = []
        phase_coords = []

        for step in range(steps):
            # 前 cutoff_step 步：注入外部周期性驱动力
            if step < cutoff_step:
                phase = step * 0.3
                perturbation = torch.sin(torch.tensor(phase, dtype=torch.float64)) * 0.5
                phi_driven = phi_current + perturbation * torch.randn_like(phi_current) * 0.1
            else:
                # 后段：切断外部输入
                phi_driven = phi_current

            # 演化一步（含奇点隔离）
            g_new, is_bh = self.evolve_step(g, A, phi_driven, g_prev, dt)

            trajectory.append(g_new.clone())

            # β 函数（度规演化速率）
            g_dot = self.metric_velocity(g_new, A, phi_driven, g)
            g_norm = float(g_new.norm()) + self.eps
            beta = float(g_dot.norm() / g_norm)
            beta_curve.append(beta)

            # 相空间坐标
            coords = self.phase_space_coordinates(g_new)
            phase_coords.append([
                float(coords['effective_rank']),
                float(coords['condition_number']),
                float(coords['spectral_curvature']),
                float(coords['metric_norm']),
            ])

            g_prev = g
            g = g_new

        trajectory_t = torch.stack(trajectory)
        beta_curve_t = torch.tensor(beta_curve, dtype=torch.float64)
        phase_coords_t = torch.tensor(phase_coords, dtype=torch.float64)

        # 前段与后段 β 方差
        early_beta_var = float(beta_curve_t[:cutoff_step].var())
        late_beta_var = float(beta_curve_t[cutoff_step:].var())

        # 内生极限环判定
        is_endogenous = late_beta_var > 0.01

        return {
            "trajectory": trajectory_t,
            "beta_curve": beta_curve_t,
            "phase_coords": phase_coords_t,
            "early_beta_var": early_beta_var,
            "late_beta_var": late_beta_var,
            "is_endogenous": is_endogenous,
        }
