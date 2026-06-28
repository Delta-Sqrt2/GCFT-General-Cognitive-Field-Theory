"""
任务二：无量纲耗散扫描与 Hopf 分岔定位

战略定位（v4.2 任务二）：
    v4.1 的 GAN 内生极限环失败是因为 λ=0.1 过大（过阻尼病）。
    v4.2 必须扫描无量纲比 λ/κ，找到 Hopf 分岔临界点。

    陷阱四十二·物理武断降级：
        严禁对 λ 设定绝对上限。必须扫描无量纲比 λ/κ。
        λ 的量纲是时间⁻¹，在没有标定时间尺度时设绝对上限无意义。

物理与哲学直觉：
    - 物理：λ 是社会熵流耗散率，对应个体与外界信息交换的开放程度。
            λ→0 是认知封闭（感觉剥夺），λ→∞ 是完全透明。
            λ/κ 的临界值有明确物理意义：个体在给定痛苦耦合下能承受的
            最大开放度。Hopf 分岔就是这个"开放度极限"的数学标志。
    - 哲学：Hopf 分岔是"自我对抗"（GAN）的诞生阈值。
            λ/κ < 临界值 → 系统封闭，无法产生内生振荡（VAE 独占）。
            λ/κ > 临界值 → 系统开放，内生振荡出现（GAN 相态涌现）。
    - 工程：固定 κ=1.0，扫描 λ ∈ [1e-4, 1e0]，计算雅可比特征值实部。

数学定义（严格可微，无降级）：
    无量纲参数：
        r = λ/κ（耗散-曲率比）

    Hopf 分岔条件：
        Re(λ_max(J)) = 0
        其中 J 是雅可比矩阵 ∂ġ/∂g

    内生极限环验证：
        在 r < r_crit 区域，运行 GAN 演化 100 步。
        前 50 步注入驱动力，后 50 步切断。
        后 50 步 β 方差 > 0.01 → 内生振荡成立。

分支逻辑（二次审查残余一）：
    情况 A：找到 λ/κ 临界值 → 进入任务三（λ = λ_crit / 3）。
    情况 B：在物理合理范围 [1e-4, 1e0] 内未找到临界值 →
            报告"当前参数空间无 Hopf 分岔"，并直接进入替代测试。
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


class HopfBifurcationScan:
    """
    无量纲耗散扫描与 Hopf 分岔定位。

    使用方式：
        hbs = HopfBifurcationScan(n_dims=4, kappa=1.0)
        # 扫描 λ/κ，找到 Hopf 分岔点
        scan_result = hbs.scan_lambda_kappa_ratio(phi, lambda_range=(1e-4, 1e0))
        # 验证内生极限环
        cycle_result = hbs.verify_endogenous_limit_cycle(g0, phi, lambda_val=1e-3)

    白盒保证：
        - 无量纲比 λ/κ（陷阱四十二防降级）
        - 雅可比特征值实部连续追踪
        - 分支逻辑处理未找到分岔的情况
    """

    def __init__(
        self,
        n_dims: int = 4,
        kappa: float = 1.0,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            kappa: 痛苦耦合常数 κ（固定，用于无量纲化）
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.kappa = float(kappa)
        self.eps = float(eps)

    # ==================================================================
    # 创建动力学实例（指定 λ）
    # ==================================================================

    def create_dynamics(self, lambda_dissip: float) -> SpectralCurvatureDynamics:
        """
        创建指定 λ 的动力学实例。

        注意：使用修正后的 κ 定义（κ 在分子）。
        """
        dyn = SpectralCurvatureDynamics(
            n_dims=self.n_dims,
            kappa=self.kappa,
            lambda_dissip=lambda_dissip,
            eps=self.eps,
        )
        return dyn

    # ==================================================================
    # 雅可比最大特征值实部（指定 λ）
    # ==================================================================

    def jacobian_max_real_part(
        self,
        metric: Tensor,
        phi: Tensor,
        lambda_dissip: float,
        perturbation_scale: float = 1e-5,
    ) -> float:
        """
        计算指定 λ 下雅可比矩阵的最大特征值实部。

        数学：
            J = ∂ġ/∂g
            λ_max = max(Re(eigenvalues(J)))

        物理：
            λ_max < 0 → 不动点稳定（VAE）
            λ_max > 0 → 不动点失稳，Hopf 分岔（GAN 极限环）
            λ_max = 0 → 临界点
        """
        dyn = self.create_dynamics(lambda_dissip)

        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        # 基准速度
        g_dot_base = dyn.metric_velocity(g, None, phi, g)

        # 有限差分雅可比
        jacobian = torch.zeros(d * d, d * d, dtype=torch.float64)

        for i in range(d):
            for j in range(i, d):  # 利用对称性
                delta = torch.zeros_like(g)
                delta[i, j] = perturbation_scale
                delta[j, i] = perturbation_scale

                g_perturbed = g + delta
                g_dot_perturbed = dyn.metric_velocity(g_perturbed, None, phi, g)

                diff = (g_dot_perturbed - g_dot_base) / perturbation_scale
                jacobian[:, i * d + j] = diff.flatten()
                if i != j:
                    jacobian[:, j * d + i] = diff.flatten()

        # 特征值
        eigvals = torch.linalg.eigvals(jacobian)
        max_real = float(eigvals.real.max())

        return max_real

    # ==================================================================
    # 无量纲 λ/κ 扫描
    # ==================================================================

    def scan_lambda_kappa_ratio(
        self,
        phi: Tensor,
        lambda_range: tuple[float, float] = (1e-4, 1e0),
        n_points: int = 10,
        g_init: Tensor | None = None,
    ) -> dict[str, Tensor | list | float | bool | str]:
        """
        扫描无量纲比 λ/κ，寻找 Hopf 分岔点。

        执行流程：
            1. 固定 κ=1.0，扫描 λ ∈ [1e-4, 1e0]
            2. 在每个 λ 下，计算雅可比最大特征值实部
            3. 检测特征值实部由负转正的临界点（Hopf 分岔）

        分支逻辑（二次审查残余一）：
            情况 A：找到临界点 → 报告 λ_crit
            情况 B：未找到 → 报告"当前参数空间无 Hopf 分岔"

        返回：
            dict 包含：
                lambda_values: λ 值序列
                ratio_values: λ/κ 比值序列
                eigenvalue_real_parts: 特征值实部序列
                has_bifurcation: 是否检测到分岔
                lambda_crit: 分岔临界点 λ_crit（若存在）
                ratio_crit: 分岔临界比值 λ/κ_crit（若存在）
                branch: 分支类型（"A" 或 "B"）
        """
        # 对数空间扫描
        lambda_values = torch.logspace(
            math.log10(lambda_range[0]),
            math.log10(lambda_range[1]),
            n_points,
            dtype=torch.float64,
        )

        # 初始度规
        if g_init is None:
            g_init = torch.eye(self.n_dims, dtype=torch.float64)
            # 加入轻微各向异性以触发非线性
            for i in range(self.n_dims):
                g_init[i, i] = 1.0 + 0.1 * (i + 1)

        eigenvalue_real_parts = []
        ratio_values = []

        for lambda_val in lambda_values:
            lambda_f = float(lambda_val)
            ratio = lambda_f / self.kappa
            ratio_values.append(ratio)

            max_real = self.jacobian_max_real_part(
                g_init, phi, lambda_dissip=lambda_f
            )
            eigenvalue_real_parts.append(max_real)

        eigenvalue_tensor = torch.tensor(eigenvalue_real_parts, dtype=torch.float64)
        ratio_tensor = torch.tensor(ratio_values, dtype=torch.float64)

        # 检测 Hopf 分岔点（特征值实部由负转正）
        has_bifurcation = False
        lambda_crit = None
        ratio_crit = None

        for i in range(len(eigenvalue_real_parts) - 1):
            if eigenvalue_real_parts[i] < 0 and eigenvalue_real_parts[i + 1] >= 0:
                # 线性插值找到精确临界点
                r0, r1 = ratio_values[i], ratio_values[i + 1]
                e0, e1 = eigenvalue_real_parts[i], eigenvalue_real_parts[i + 1]
                ratio_crit = r0 + (0 - e0) * (r1 - r0) / (e1 - e0 + self.eps)
                lambda_crit = ratio_crit * self.kappa
                has_bifurcation = True
                break

        # 分支逻辑
        if has_bifurcation:
            branch = "A"
            message = (
                f"Hopf 分岔点找到：λ_crit = {lambda_crit:.6e}, "
                f"λ/κ_crit = {ratio_crit:.6e}。"
                f"在此点，系统从 VAE 不动点（稳定）转变为 GAN 极限环（失稳）。"
            )
        else:
            branch = "B"
            # 检查所有特征值是否都为正（系统始终失稳）
            all_positive = all(e > 0 for e in eigenvalue_real_parts)
            all_negative = all(e < 0 for e in eigenvalue_real_parts)

            if all_positive:
                message = (
                    f"在 λ/κ ∈ [{ratio_values[0]:.4e}, {ratio_values[-1]:.4e}] 范围内，"
                    f"特征值实部始终 > 0（系统始终失稳）。"
                    f"可能需要扩展参数范围或调整初始度规。"
                )
            elif all_negative:
                message = (
                    f"在 λ/κ ∈ [{ratio_values[0]:.4e}, {ratio_values[-1]:.4e}] 范围内，"
                    f"特征值实部始终 < 0（系统始终稳定）。"
                    f"当前参数空间无 Hopf 分岔。"
                    f"进入替代测试：固定 λ=1e-3，观察 Lyapunov 指数。"
                )
            else:
                message = (
                    f"在 λ/κ 范围内特征值实部有波动但未检测到明确的过零点。"
                    f"可能需要更精细的扫描。"
                )

        return {
            "lambda_values": lambda_values,
            "ratio_values": ratio_tensor,
            "eigenvalue_real_parts": eigenvalue_tensor,
            "has_bifurcation": has_bifurcation,
            "lambda_crit": lambda_crit,
            "ratio_crit": ratio_crit,
            "branch": branch,
            "message": message,
            "kappa": self.kappa,
        }

    # ==================================================================
    # 内生极限环验证（后 50 步 β 方差时间序列）
    # ==================================================================

    def verify_endogenous_limit_cycle(
        self,
        g0: Tensor,
        phi: Tensor,
        lambda_val: float,
        steps: int = 100,
        dt: float = 0.01,
        cutoff_step: int = 50,
    ) -> dict[str, Tensor | list | float | bool]:
        """
        验证 GAN 相态的内生极限环（禁受迫振动）。

        物理过程：
            1. 前 cutoff_step 步：注入外部周期性驱动力
            2. 后 (steps - cutoff_step) 步：切断外部输入
            3. 检验后段是否仍保持振荡（内生极限环）

        判据：
            后段 β 方差 > 0.01 → 内生极限环成立
            后段 β 方差 < 0.01 → 受迫振动（失败）

        关键改进（vs v4.1）：
            输出后 50 步 β 方差的时间序列（非单一数字）。
        """
        dyn = self.create_dynamics(lambda_val)

        g = symmetric_part(g0.to(torch.float64))
        g_prev = g.clone()
        phi_current = phi.to(torch.float64)

        trajectory = []
        beta_curve = []
        beta_var_time_series = []  # 后段 β 方差时间序列

        for step in range(steps):
            # 前 cutoff_step 步：注入外部周期性驱动力
            if step < cutoff_step:
                phase = step * 0.3
                perturbation = torch.sin(torch.tensor(phase, dtype=torch.float64)) * 0.5
                phi_driven = phi_current + perturbation * torch.randn_like(phi_current) * 0.1
            else:
                # 后段：切断外部输入
                phi_driven = phi_current

            # 演化一步
            g_new, is_bh = dyn.evolve_step(g, None, phi_driven, g_prev, dt)

            trajectory.append(g_new.clone())

            # β 函数（度规演化速率）
            g_dot = dyn.metric_velocity(g_new, None, phi_driven, g)
            g_norm = float(g_new.norm()) + self.eps
            beta = float(g_dot.norm() / g_norm)
            beta_curve.append(beta)

            # 后段 β 方差时间序列（滑动窗口）
            if step >= cutoff_step:
                window_start = max(cutoff_step, step - 10)
                window_betas = beta_curve[window_start:step + 1]
                if len(window_betas) > 1:
                    window_var = float(torch.tensor(window_betas).var())
                    beta_var_time_series.append(window_var)

            g_prev = g
            g = g_new

        trajectory_t = torch.stack(trajectory)
        beta_curve_t = torch.tensor(beta_curve, dtype=torch.float64)
        beta_var_series_t = torch.tensor(beta_var_time_series, dtype=torch.float64)

        # 前段与后段 β 方差
        early_beta_var = float(beta_curve_t[:cutoff_step].var())
        late_beta_var = float(beta_curve_t[cutoff_step:].var())

        # 内生极限环判定
        is_endogenous = late_beta_var > 0.01

        return {
            "trajectory": trajectory_t,
            "beta_curve": beta_curve_t,
            "beta_var_time_series": beta_var_series_t,
            "early_beta_var": early_beta_var,
            "late_beta_var": late_beta_var,
            "is_endogenous": is_endogenous,
            "lambda_val": lambda_val,
            "ratio": lambda_val / self.kappa,
            "steps": steps,
            "cutoff_step": cutoff_step,
        }

    # ==================================================================
    # 多 λ 值的内生极限环扫描
    # ==================================================================

    def scan_endogenous_limit_cycle(
        self,
        g0: Tensor,
        phi: Tensor,
        lambda_values: list[float],
        steps: int = 100,
        dt: float = 0.01,
        cutoff_step: int = 50,
    ) -> dict[str, list]:
        """
        在多个 λ 值下验证内生极限环。

        输出 λ 与后段方差的相图，证明存在 λ* 使得后段方差 > 0.01。
        """
        results = []
        lambda_list = []
        late_vars = []
        is_endogenous_list = []

        for lambda_val in lambda_values:
            result = self.verify_endogenous_limit_cycle(
                g0, phi, lambda_val, steps, dt, cutoff_step
            )
            results.append(result)
            lambda_list.append(lambda_val)
            late_vars.append(result["late_beta_var"])
            is_endogenous_list.append(result["is_endogenous"])

        # 找到使后段方差 > 0.01 的 λ*
        lambda_star = None
        for i, (l, v) in enumerate(zip(lambda_list, late_vars)):
            if v > 0.01:
                lambda_star = l
                break

        return {
            "lambda_values": lambda_list,
            "late_beta_vars": late_vars,
            "is_endogenous_list": is_endogenous_list,
            "lambda_star": lambda_star,
            "results": results,
            "has_endogenous_cycle": lambda_star is not None,
        }

    # ==================================================================
    # 综合验证：Hopf 分岔与内生极限环
    # ==================================================================

    def verify_hopf_and_limit_cycle(
        self,
        phi: Tensor,
        g0: Tensor | None = None,
        lambda_range: tuple[float, float] = (1e-4, 1e0),
        n_scan_points: int = 10,
        steps: int = 100,
        dt: float = 0.01,
        cutoff_step: int = 50,
    ) -> dict[str, any]:
        """
        综合验证 Hopf 分岔与内生极限环。

        执行流程：
            1. 扫描 λ/κ，寻找 Hopf 分岔点
            2. 如果找到分岔点（分支 A）：
               在 λ < λ_crit 的区域验证内生极限环
            3. 如果未找到分岔点（分支 B）：
               固定 λ=1e-3，直接验证内生极限环

        返回：
            dict 包含完整验证结果
        """
        # 第一步：扫描 Hopf 分岔
        scan_result = self.scan_lambda_kappa_ratio(
            phi, lambda_range, n_scan_points, g0
        )

        result = {
            "scan_result": scan_result,
            "branch": scan_result["branch"],
        }

        if scan_result["branch"] == "A":
            # 分支 A：找到分岔点，在 λ < λ_crit 区域验证
            lambda_crit = scan_result["lambda_crit"]

            # 选择 λ = λ_crit / 3 进行验证
            lambda_test = lambda_crit / 3.0

            if g0 is None:
                g0 = torch.eye(self.n_dims, dtype=torch.float64)
                for i in range(self.n_dims):
                    g0[i, i] = 1.0 + 0.1 * (i + 1)

            cycle_result = self.verify_endogenous_limit_cycle(
                g0, phi, lambda_test, steps, dt, cutoff_step
            )

            result["cycle_result"] = cycle_result
            result["lambda_test"] = lambda_test
            result["status"] = "hopf_found"
            result["message"] = (
                f"Hopf 分岔点 λ_crit = {lambda_crit:.6e}。"
                f"在 λ = λ_crit/3 = {lambda_test:.6e} 下验证内生极限环。"
                f"后段 β 方差 = {cycle_result['late_beta_var']:.6f}。"
                f"内生极限环：{'成立' if cycle_result['is_endogenous'] else '未成立'}。"
            )
        else:
            # 分支 B：未找到分岔点，替代测试
            lambda_test = 1e-3

            if g0 is None:
                g0 = torch.eye(self.n_dims, dtype=torch.float64)
                for i in range(self.n_dims):
                    g0[i, i] = 1.0 + 0.1 * (i + 1)

            cycle_result = self.verify_endogenous_limit_cycle(
                g0, phi, lambda_test, steps, dt, cutoff_step
            )

            result["cycle_result"] = cycle_result
            result["lambda_test"] = lambda_test
            result["status"] = "no_hopf_alternative"
            result["message"] = (
                f"未找到 Hopf 分岔点。替代测试：固定 λ = {lambda_test:.6e}。"
                f"后段 β 方差 = {cycle_result['late_beta_var']:.6f}。"
                f"内生极限环：{'成立' if cycle_result['is_endogenous'] else '未成立'}。"
                f"如果未成立，说明当前拉格朗日量可能需要梯度项 Tr((∇g)²)。"
            )

        return result
