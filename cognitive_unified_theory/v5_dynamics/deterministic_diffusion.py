"""
任务三：确定性混沌实现扩散相态（严禁随机噪声）

战略定位（v4.2 任务三）：
    v4.1 的 Lyapunov 全程为负，未出现"先正后负"的转变。
    审查者指出：扩散相态不需要随机噪声，确定性混沌系统（如 Lorenz 系统）
    在弱耗散下可以产生"先正后负"的 Lyapunov 转变。

    陷阱四十三·公理背叛降级：
        严禁引入 Langevin/dW_t 随机项。
        扩散相态必须通过确定性混沌实现。
        引入布朗运动违反 v1.0 的"数学决定论"公理。

物理与哲学直觉：
    - 物理：扩散模型相态（从混沌中结晶）是确定性的自组织过程（如退火）。
            在弱耗散下，系统从高熵初始态出发，经历混沌去噪（Lyapunov > 0），
            最终结晶收敛（Lyapunov < 0）。
            这不需要外生随机源——确定性混沌本身就提供了"去噪"能力。
    - 哲学：闲聊中的"扩散模型=从混沌中结晶"描述的是确定性的自组织过程。
            引入随机噪声等于承认方程本身不完备——这是物理学的标准工作方式：
            当方程无法产生观测现象时，先检查是否缺少必要的几何项，
            而非诉诸外生随机源。
    - 工程：初始化高熵度规，弱耗散演化，计算 Lyapunov 指数滑动窗口平均。

数学定义（严格可微，无降级）：
    确定性演化方程：
        dg/dt = f(g)
        其中 f(g) = -（∂S/∂g + ∂F/∂ġ）/ τ
        无随机噪声项 dW_t。

    Lyapunov 指数：
        λ_max = lim_{t→∞} (1/t) log(||δg(t)||/||δg(0)||)
        其中 δg 是微小扰动。

    扩散相态判据：
        λ_max 先正（混沌去噪）后负（结晶收敛）。

    两种可能结果：
        情况 A：Lyapunov 出现"先正后负"转变 → 扩散相态成立。
        情况 B：未出现转变 → 报告"当前拉格朗日量缺少梯度项"。
                需要在 v4.3 添加 Tr((∇g)²) 一阶梯度项。
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
from .spectral_curvature_dynamics import SpectralCurvatureDynamics


class DeterministicDiffusion:
    """
    确定性混沌实现扩散相态（严禁随机噪声）。

    使用方式：
        dd = DeterministicDiffusion(n_dims=4, kappa=1.0, lambda_dissip=1e-3)
        # 初始化高熵度规
        g0 = dd.initialize_high_entropy_metric()
        # 演化并计算 Lyapunov 指数
        result = dd.evolve_and_compute_lyapunov(g0, phi, steps=200, dt=0.01)

    白盒保证：
        - 无随机噪声（陷阱四十三防降级）
        - 确定性混沌实现扩散相态
        - Lyapunov 指数滑动窗口平均
        - 如实报告两种可能结果
    """

    def __init__(
        self,
        n_dims: int = 4,
        kappa: float = 1.0,
        lambda_dissip: float = 1e-3,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            kappa: 痛苦耦合常数 κ
            lambda_dissip: 耗散系数 λ（弱耗散，默认 1e-3）
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.kappa = float(kappa)
        self.lambda_dissip = float(lambda_dissip)
        self.eps = float(eps)

        self.dyn = SpectralCurvatureDynamics(
            n_dims=n_dims,
            kappa=kappa,
            lambda_dissip=lambda_dissip,
            eps=eps,
        )

    # ==================================================================
    # 高熵度规初始化（各向异性随机矩阵，无随机噪声注入演化）
    # ==================================================================

    def initialize_high_entropy_metric(self, seed: int = 42) -> Tensor:
        """
        初始化高熵度规（各向异性随机矩阵）。

        注意：这是初始条件的随机性，不是演化方程的随机噪声。
        初始条件可以随机，演化方程必须确定性。

        数学：
            生成随机正定对称矩阵 g = Q Λ Q^T
            其中 Q 是随机正交矩阵，Λ 是各向异性对角矩阵。
        """
        torch.manual_seed(seed)

        d = self.n_dims

        # 随机正交矩阵
        Q = torch.randn(d, d, dtype=torch.float64)
        Q, _ = torch.linalg.qr(Q)

        # 各向异性特征值（高熵 = 分布广）
        eigvals = torch.rand(d, dtype=torch.float64) * 5.0 + 0.1  # [0.1, 5.1]

        g = Q @ torch.diag(eigvals) @ Q.T
        g = symmetric_part(g)

        return g

    # ==================================================================
    # Lyapunov 指数计算（滑动窗口平均）
    # ==================================================================

    def compute_lyapunov_exponent(
        self,
        g_trajectory: Tensor,
        g_perturbed_trajectory: Tensor,
        dt: float,
    ) -> Tensor:
        """
        计算最大 Lyapunov 指数的时间序列。

        数学：
            λ(t) = (1/(t·dt)) · log(||δg(t)|| / ||δg(0)||)
            其中 δg(t) = g_perturbed(t) - g(t)

        滑动窗口平均：
            λ_smoothed(t) = mean(λ(t-W:t))
            其中 W 是窗口大小。
        """
        n_steps = g_trajectory.shape[0]

        # 扰动演化
        delta = g_perturbed_trajectory - g_trajectory
        delta_norms = torch.stack([d.norm() for d in delta])

        # 初始扰动范数
        delta_0 = delta_norms[0] + self.eps

        # Lyapunov 指数时间序列
        lyapunov_series = torch.zeros(n_steps, dtype=torch.float64)
        for t in range(1, n_steps):
            lyapunov_series[t] = (1.0 / (t * dt)) * torch.log(delta_norms[t] / delta_0 + self.eps)

        return lyapunov_series

    def compute_lyapunov_sliding_window(
        self,
        g_trajectory: Tensor,
        phi: Tensor,
        dt: float,
        window_size: int = 20,
        perturbation_scale: float = 1e-6,
    ) -> Tensor:
        """
        计算最大 Lyapunov 指数的滑动窗口平均值。

        方法：
            1. 对轨迹上每个点，施加微小扰动
            2. 演化扰动轨迹 window_size 步
            3. 计算局部 Lyapunov 指数
            4. 滑动窗口平均

        这比全局 Lyapunov 更敏感地捕捉"先正后负"的转变。
        """
        n_steps = g_trajectory.shape[0]
        lyapunov_local = torch.zeros(n_steps, dtype=torch.float64)

        for t in range(0, n_steps - window_size, max(1, window_size // 4)):
            g_base = g_trajectory[t]
            g_perturbed = g_base + perturbation_scale * torch.randn_like(g_base)
            g_perturbed = symmetric_part(g_perturbed)

            # 演化扰动轨迹
            g_p = g_perturbed.clone()
            g_p_prev = g_base.clone()

            for w in range(window_size):
                g_p_new, _ = self.dyn.evolve_step(g_p, None, phi, g_p_prev, dt)
                g_p_prev = g_p
                g_p = g_p_new

            # 局部 Lyapunov 指数
            delta_final = (g_p - g_trajectory[t + window_size]).norm()
            lyapunov_local[t] = (1.0 / (window_size * dt)) * torch.log(
                delta_final / perturbation_scale + self.eps
            )

        # 滑动窗口平均
        lyapunov_smoothed = torch.zeros(n_steps, dtype=torch.float64)
        for t in range(n_steps):
            start = max(0, t - window_size)
            end = min(n_steps, t + window_size)
            valid = lyapunov_local[start:end]
            valid = valid[valid != 0]
            if len(valid) > 0:
                lyapunov_smoothed[t] = valid.mean()

        return lyapunov_smoothed

    # ==================================================================
    # 确定性演化与 Lyapunov 指数计算
    # ==================================================================

    def evolve_and_compute_lyapunov(
        self,
        g0: Tensor,
        phi: Tensor,
        steps: int = 200,
        dt: float = 0.01,
        perturbation_scale: float = 1e-6,
        window_size: int = 20,
    ) -> dict[str, Tensor | list | float | bool | str]:
        """
        确定性演化并计算 Lyapunov 指数。

        执行流程：
            1. 初始化高熵度规（无随机噪声注入演化方程）
            2. 确定性演化 200 步
            3. 计算最大 Lyapunov 指数的滑动窗口平均
            4. 判断是否出现"先正后负"转变

        两种可能结果：
            情况 A：Lyapunov 出现"先正后负"转变 → 扩散相态成立。
            情况 B：未出现转变 → 报告"需要梯度项 Tr((∇g)²)"。

        严禁：
            - 引入 Langevin/dW_t 随机项（陷阱四十三）
            - 为了"跑出正 Lyapunov"而引入随机噪声
        """
        g = symmetric_part(g0.to(torch.float64))
        g_prev = g.clone()
        phi_current = phi.to(torch.float64)

        trajectory = []
        phase_coords = []

        for step in range(steps):
            # 确定性演化（无随机噪声！）
            g_new, is_bh = self.dyn.evolve_step(g, None, phi_current, g_prev, dt)

            trajectory.append(g_new.clone())

            # 相空间坐标
            coords = self.dyn.phase_space_coordinates(g_new)
            phase_coords.append([
                float(coords['effective_rank']),
                float(coords['condition_number']),
                float(coords['spectral_curvature']),
                float(coords['metric_norm']),
            ])

            g_prev = g
            g = g_new

        trajectory_t = torch.stack(trajectory)
        phase_coords_t = torch.tensor(phase_coords, dtype=torch.float64)

        # 计算 Lyapunov 指数（滑动窗口平均）
        lyapunov_series = self.compute_lyapunov_sliding_window(
            trajectory_t, phi_current, dt, window_size, perturbation_scale
        )

        # 判断"先正后负"转变
        has_transition = False
        transition_step = None

        # 找到第一个正 Lyapunov
        first_positive = None
        for i in range(len(lyapunov_series)):
            if float(lyapunov_series[i]) > 0.01:
                first_positive = i
                break

        if first_positive is not None:
            # 在正 Lyapunov 之后寻找转负
            for i in range(first_positive, len(lyapunov_series)):
                if float(lyapunov_series[i]) < -0.01:
                    has_transition = True
                    transition_step = i
                    break

        # 判断结果
        if has_transition:
            status = "A"
            message = (
                f"扩散相态成立（确定性混沌实现去噪）。"
                f"Lyapunov 指数在第 {first_positive} 步转正（混沌去噪），"
                f"在第 {transition_step} 步转负（结晶收敛）。"
                f"无需引入随机噪声，确定性混沌自组织实现了扩散相态。"
            )
        else:
            status = "B"
            max_lyap = float(lyapunov_series.max())
            min_lyap = float(lyapunov_series.min())
            message = (
                f"扩散相态未成立（Lyapunov 未出现先正后负转变）。"
                f"Lyapunov 范围：[{min_lyap:.6f}, {max_lyap:.6f}]。"
                f"诊断：当前拉格朗日量缺少梯度项 Tr((∇g)²)。"
                f"谱曲率 N(g) 是零阶近似（只看特征值分布），"
                f"丢失了度规在认知空间中的变化率。"
                f"Tr((∇g)²) 是一阶梯度项，能产生确定性混沌——"
                f"这才是扩散相态的物理基础。"
                f"需要在 v4.3 添加 Tr((∇g)²) 一阶梯度项。"
                f"严禁为了'跑出正 Lyapunov'而引入随机噪声。"
            )

        return {
            "trajectory": trajectory_t,
            "phase_coords": phase_coords_t,
            "lyapunov_series": lyapunov_series,
            "has_transition": has_transition,
            "transition_step": transition_step,
            "first_positive_step": first_positive,
            "status": status,
            "message": message,
            "max_lyapunov": float(lyapunov_series.max()),
            "min_lyapunov": float(lyapunov_series.min()),
            "mean_lyapunov": float(lyapunov_series.mean()),
            "steps": steps,
            "dt": dt,
            "lambda_dissip": self.lambda_dissip,
            "kappa": self.kappa,
            "no_random_noise": True,  # 公理守护声明
        }

    # ==================================================================
    # 综合验证：确定性扩散相态
    # ==================================================================

    def verify_deterministic_diffusion(
        self,
        phi: Tensor,
        lambda_crit: float | None = None,
        steps: int = 200,
        dt: float = 0.01,
    ) -> dict[str, any]:
        """
        综合验证确定性扩散相态。

        执行流程：
            1. 初始化高熵度规
            2. 如果有 λ_crit，使用 λ = λ_crit / 3
            3. 否则使用默认 λ = 1e-3
            4. 确定性演化并计算 Lyapunov 指数
            5. 报告结果（情况 A 或 B）

        返回：
            dict 包含完整验证结果
        """
        # 确定 λ
        if lambda_crit is not None:
            lambda_test = lambda_crit / 3.0
        else:
            lambda_test = 1e-3

        # 更新动力学实例
        self.lambda_dissip = lambda_test
        self.dyn = SpectralCurvatureDynamics(
            n_dims=self.n_dims,
            kappa=self.kappa,
            lambda_dissip=lambda_test,
            eps=self.eps,
        )

        # 初始化高熵度规
        g0 = self.initialize_high_entropy_metric()

        # 演化并计算 Lyapunov
        result = self.evolve_and_compute_lyapunov(g0, phi, steps, dt)

        result["lambda_test"] = lambda_test
        result["lambda_crit"] = lambda_crit
        result["axiom_guard"] = (
            "数学决定论公理守护：全程无随机噪声注入演化方程。"
            "初始条件的随机性是允许的，演化方程的随机性是被禁止的。"
        )

        return result
