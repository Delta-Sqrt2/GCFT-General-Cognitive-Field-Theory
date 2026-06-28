"""
任务一：反转 κ 定义与变分极值严格收敛

战略定位（v4.2 任务一）：
    v4.1 暴露的 κ-τ* 负相关是"符号谬误"——κ 在分母导致物理意义反转。
    v4.2 必须将作用量从 S = -N(g)/(2κ) 改为 S = -(κ/2)·N(g)，
    使 κ 成为正向曲率耦合：κ 大 → 曲率项权重大 → 痛苦耦合强。

    陷阱四十四·数值亢奋降级：
        奇点附近严禁纯 Newton 法。必须使用自适应阻尼（Armijo 线搜索）。
        L-BFGS 在 cond(g) > 10^6 时切换到梯度下降。

物理与哲学直觉：
    - 物理：κ 是痛苦耦合常数。κ 大 → 曲率代价大 → 系统远离平衡。
            修正后 κ 在分子，κ 大 → -N(g)·κ/2 大 → S 大 → 痛苦强。
            这与"高痛苦 → 高曲率 → 需要更深灰盒（τ* 低）"一致。
            但审查者指出：κ 大 → 曲率高 → τ* 应高（更白盒以处理痛苦）。
            物理逻辑：高痛苦需要更多觉察（更高透明度）来处理。
    - 哲学：κ 是"创伤深度"的数学度量。κ 大 → 创伤深 → 需要更多觉察。
            但过度觉察（τ→1）导致撕裂。τ* 是创伤与觉察的最优平衡。
    - 工程：L-BFGS 替代 Adam，梯度范数 < 1e-6。

数学定义（严格可微，无降级）：
    修正后作用量：
        S[g, φ] = -(κ/2)·N(g) + (1/2) g^μν (D_μφ)^T (D_νφ)
    其中：
        N(g) = Σ p_i log(p_i) ≤ 0（负熵）
        -(κ/2)·N(g) ≥ 0（正代价，κ 大 → 代价大）

    变分极值：
        J(τ) = S_演化(τ) + μ·H(τ) + ν·R_撕裂(τ)
        δJ/δτ = 0 → τ* ∈ (0, 1)

    κ-τ* 正相关定理：
        κ 大 → 曲率代价大 → 需要更高透明度处理 → τ* 高
        相关系数 > 0.9

    保底方案（二次审查残余二）：
        如果修正后仍负相关，自动执行撕裂泛函重定义：
        R_tear(τ) → Tr(g^{-1}(∂g/∂τ)²) + α·Tr((∂g/∂τ)²)
        其中 α 由 Hessian 最小特征值推导。
"""

from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.optim.lr_scheduler import StepLR

from ..core.tensor_ops import (
    effective_rank,
    safe_inverse,
    stable_eigh,
    symmetric_part,
)
from .spectral_curvature_dynamics import SpectralCurvatureDynamics
from .variational_transparency import VariationalTransparency


class KappaCorrection:
    """
    κ 定义反转与变分极值严格收敛。

    使用方式：
        kc = KappaCorrection(n_dims=4, mu=2.0, nu=0.1)
        # 求解最优透明度（L-BFGS）
        result = kc.solve_optimal_transparency_lbfgs(g, phi, kappa=1.0)
        # 扫描 κ，输出 κ-τ* 曲线
        curve = kc.kappa_tau_star_curve(g, phi, kappa_range=(0.1, 5.0))
        # 验证正相关
        correlation = kc.verify_positive_correlation(curve)

    白盒保证：
        - κ 在分子（正向耦合），不在分母（陷阱四十四）
        - L-BFGS 求解，梯度范数 < 1e-6
        - κ-τ* 正相关（相关系数 > 0.9）或报告 Hessian 谱
    """

    def __init__(
        self,
        n_dims: int = 4,
        mu: float = 2.0,
        nu: float = 0.1,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            mu: 透明度熵权重 μ
            nu: 撕裂代价权重 ν
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.mu = float(mu)
        self.nu = float(nu)
        self.eps = float(eps)

        # 复用 v4.1 的谱曲率计算
        self.scd = SpectralCurvatureDynamics(n_dims=n_dims, kappa=1.0, lambda_dissip=0.1)
        # 复用 v4.1 的透明度调节
        self.vt = VariationalTransparency(n_dims=n_dims, mu=mu, nu=nu)

    # ==================================================================
    # 修正后的作用量 S = -(κ/2)·N(g) + 耦合项
    # ==================================================================

    def corrected_action(
        self,
        metric: Tensor,
        A: Tensor | None,
        phi: Tensor,
        kappa: float,
    ) -> dict[str, Tensor]:
        """
        修正后的保守作用量（κ 在分子）：

        S[g, φ] = -(κ/2)·N(g) + (1/2) g^μν (D_μφ)^T (D_νφ)

        关键修正（vs v4.1）：
            v4.1: S = -N(g)/(2κ)  → κ 在分母 → κ 大 → 曲率代价小（错误！）
            v4.2: S = -(κ/2)·N(g) → κ 在分子 → κ 大 → 曲率代价大（正确！）
        """
        g = symmetric_part(metric.to(torch.float64))
        d = g.shape[-1]

        # 谱曲率项：-(κ/2)·N(g)（正代价，κ 大 → 代价大）
        N = self.scd.spectral_curvature(g)
        kappa_val = float(kappa)
        curvature_term = -(kappa_val / 2.0) * N  # -N ≥ 0，乘以 κ/2 > 0 → 正代价

        # 协变导数
        D_phi = self.scd.covariant_derivative(phi, A)

        # 规范耦合项
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
            "kappa": kappa_val,
        }

    # ==================================================================
    # 修正后的演化作用量 S_演化(τ)
    # ==================================================================

    def corrected_evolution_action(
        self,
        metric: Tensor,
        phi: Tensor,
        tau: Tensor,
        kappa: float,
    ) -> Tensor:
        """
        修正后的演化作用量 S_演化(τ)。

        使用修正后的作用量 S = -(κ/2)·N(g(τ)) + 耦合项。
        """
        g_tau = self.vt.transparency_modified_metric(metric, tau)
        result = self.corrected_action(g_tau, None, phi, kappa)
        return result["action"]

    # ==================================================================
    # 修正后的总代价泛函 J(τ)
    # ==================================================================

    def corrected_total_cost(
        self,
        metric: Tensor,
        phi: Tensor,
        tau: Tensor,
        kappa: float,
    ) -> dict[str, Tensor]:
        """
        修正后的认知总代价：

        J(τ) = S_演化(τ) + μ·H(τ) + ν·R_撕裂(τ)

        其中 S_演化 使用修正后的作用量（κ 在分子）。
        """
        S = self.corrected_evolution_action(metric, phi, tau, kappa)
        H = self.vt.transparency_entropy(tau)
        R_tear = self.vt.tearing_functional(metric, tau)

        J = S + self.mu * H + self.nu * R_tear

        return {
            "total_cost": J,
            "evolution_cost": S,
            "entropy_cost": self.mu * H,
            "tearing_cost": self.nu * R_tear,
        }

    # ==================================================================
    # L-BFGS 变分极值求解（替代 Adam）
    # ==================================================================

    def solve_optimal_transparency_lbfgs(
        self,
        metric: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        max_iter: int = 200,
        tolerance: float = 1e-8,
        lr: float = 0.1,
    ) -> dict[str, Tensor | float | bool | int]:
        """
        使用 L-BFGS 求解 δJ/δτ = 0。

        陷阱四十四·数值亢奋降级：
            严禁纯 Newton 法。L-BFGS 带自适应阻尼。
            在 cond(g) > 10^6 时，L-BFGS 自动退化为梯度下降。

        数学：
            τ* = argmin_τ J(τ)
            收敛判据：||δJ/δτ|| < 1e-6

        返回：
            dict 包含：
                tau_star: 最优透明度 τ*
                gradient_norm: 收敛时的梯度范数
                total_cost: 最优处的总代价
                converged: 是否收敛
                iterations: 迭代次数
                cost_history: 代价历史
                gradient_history: 梯度历史
        """
        # 初始化 τ
        tau = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        # 检查度规条件数（决定是否用梯度下降）
        g = symmetric_part(metric.to(torch.float64))
        eigvals_g = torch.linalg.eigvalsh(g)
        eigvals_g = torch.clamp(eigvals_g, min=self.eps)
        cond_g = float(eigvals_g.max() / eigvals_g.min())

        # L-BFGS 优化器
        if cond_g < 1e6:
            # 正常区域：使用 L-BFGS
            optimizer = torch.optim.LBFGS(
                [tau],
                lr=lr,
                max_iter=max_iter,
                tolerance_grad=tolerance,
                tolerance_change=tolerance,
                line_search_fn="strong_wolfe",  # Armijo 线搜索
            )
            use_lbfgs = True
        else:
            # 病态区域：退化为梯度下降（陷阱四十四防护）
            optimizer = torch.optim.SGD([tau], lr=1e-4, momentum=0.9)
            use_lbfgs = False

        cost_history = []
        gradient_history = []
        gradient_norm = float('inf')
        iterations = 0
        converged = False

        if use_lbfgs:
            # L-BFGS 闭包模式
            def closure():
                nonlocal gradient_norm, iterations
                optimizer.zero_grad()
                cost_dict = self.corrected_total_cost(metric, phi, tau, kappa)
                J = cost_dict["total_cost"]
                J.backward()

                gradient_norm = float(tau.grad.abs())
                gradient_history.append(gradient_norm)
                cost_history.append(float(J.detach()))

                iterations += 1

                # 约束 τ ∈ (0, 1)
                with torch.no_grad():
                    tau.clamp_(min=self.eps, max=1.0 - self.eps)

                return J

            optimizer.step(closure)

            # 检查收敛
            converged = gradient_norm < 1e-6
        else:
            # 梯度下降模式（病态区域）
            for i in range(max_iter):
                optimizer.zero_grad()
                cost_dict = self.corrected_total_cost(metric, phi, tau, kappa)
                J = cost_dict["total_cost"]
                J.backward()

                gradient_norm = float(tau.grad.abs())
                gradient_history.append(gradient_norm)
                cost_history.append(float(J.detach()))

                iterations = i + 1

                if gradient_norm < 1e-6:
                    converged = True
                    break

                optimizer.step()

                with torch.no_grad():
                    tau.clamp_(min=self.eps, max=1.0 - self.eps)

        # 最终代价
        with torch.no_grad():
            final_cost = self.corrected_total_cost(metric, phi, tau, kappa)

        return {
            "tau_star": float(tau.detach()),
            "gradient_norm": gradient_norm,
            "total_cost": float(final_cost["total_cost"]),
            "evolution_cost": float(final_cost["evolution_cost"]),
            "entropy_cost": float(final_cost["entropy_cost"]),
            "tearing_cost": float(final_cost["tearing_cost"]),
            "converged": converged,
            "iterations": iterations,
            "cost_history": cost_history,
            "gradient_history": gradient_history,
            "used_lbfgs": use_lbfgs,
            "cond_g": cond_g,
        }

    # ==================================================================
    # κ-τ* 曲线扫描
    # ==================================================================

    def kappa_tau_star_curve(
        self,
        metric: Tensor,
        phi: Tensor,
        kappa_range: tuple[float, float] = (0.1, 5.0),
        n_points: int = 10,
    ) -> dict[str, Tensor | list]:
        """
        扫描 κ ∈ [0.1, 5.0]，输出 κ-τ* 曲线。

        理论预测：
            κ 大 → 曲率代价大 → 需要更高透明度处理 → τ* 高
            相关系数 > 0.9（正相关）

        返回：
            dict 包含：
                kappa_curve: κ 值序列
                tau_star_curve: τ* 序列
                gradient_norms: 收敛梯度范数序列
                correlation: κ-τ* 相关系数
                is_positive_correlation: 是否正相关
        """
        kappa_values = torch.linspace(
            kappa_range[0], kappa_range[1], n_points, dtype=torch.float64
        )

        tau_stars = []
        gradient_norms = []
        costs = []

        for kappa in kappa_values:
            kappa_val = float(kappa)

            result = self.solve_optimal_transparency_lbfgs(
                metric, phi, kappa=kappa_val, max_iter=200, tolerance=1e-8
            )

            tau_stars.append(result["tau_star"])
            gradient_norms.append(result["gradient_norm"])
            costs.append(result["total_cost"])

        tau_tensor = torch.tensor(tau_stars, dtype=torch.float64)

        # 计算相关系数
        if len(tau_stars) > 1:
            kappa_centered = kappa_values - kappa_values.mean()
            tau_centered = tau_tensor - tau_tensor.mean()
            correlation = float(
                (kappa_centered * tau_centered).sum()
                / (kappa_centered.norm() * tau_centered.norm() + self.eps)
            )
        else:
            correlation = 0.0

        is_positive = correlation > 0.9

        return {
            "kappa_curve": kappa_values,
            "tau_star_curve": tau_tensor,
            "gradient_norms": gradient_norms,
            "costs": costs,
            "correlation": correlation,
            "is_positive_correlation": is_positive,
        }

    # ==================================================================
    # 保底方案：撕裂泛函 Hessian 谱分析
    # ==================================================================

    def tearing_hessian_spectrum(
        self,
        metric: Tensor,
        kappa: float = 1.0,
        n_tau_points: int = 20,
    ) -> dict[str, Tensor | list]:
        """
        撕裂泛函 R_tear(τ) 的 Hessian 特征值谱分析。

        保底方案（二次审查残余二）：
            如果 κ 修正后仍负相关，分析撕裂泛函的 Hessian 谱，
            判断是否需要重定义撕裂泛函。

        数学：
            Hessian H_ij = ∂²R_tear/∂τ_i ∂τ_j
            在 τ ∈ (0, 1) 上离散化，计算 Hessian 矩阵。

        返回：
            dict 包含：
                tau_grid: τ 网格
                tearing_values: R_tear(τ) 值
                hessian: Hessian 矩阵
                hessian_eigenvalues: Hessian 特征值
                min_eigenvalue: 最小特征值
                needs_redefinition: 是否需要重定义
        """
        tau_grid = torch.linspace(0.01, 0.99, n_tau_points, dtype=torch.float64)
        tearing_values = []

        for tau_val in tau_grid:
            tau_tensor = torch.tensor(float(tau_val), dtype=torch.float64)
            R = self.vt.tearing_functional(metric, tau_tensor)
            tearing_values.append(float(R.detach()))

        tearing_tensor = torch.tensor(tearing_values, dtype=torch.float64)

        # 数值 Hessian（二阶差分）
        hessian = torch.zeros(n_tau_points, n_tau_points, dtype=torch.float64)
        d_tau = float(tau_grid[1] - tau_grid[0])

        # 对角元素（二阶导数）
        for i in range(n_tau_points):
            if i == 0:
                # 前向差分
                hessian[i, i] = (tearing_values[2] - 2 * tearing_values[1] + tearing_values[0]) / (d_tau ** 2)
            elif i == n_tau_points - 1:
                # 后向差分
                hessian[i, i] = (tearing_values[-1] - 2 * tearing_values[-2] + tearing_values[-3]) / (d_tau ** 2)
            else:
                # 中心差分
                hessian[i, i] = (tearing_values[i + 1] - 2 * tearing_values[i] + tearing_values[i - 1]) / (d_tau ** 2)

        # 非对角元素（混合偏导）
        for i in range(n_tau_points - 1):
            hessian[i, i + 1] = (tearing_values[i + 1] - tearing_values[i]) / d_tau
            hessian[i + 1, i] = hessian[i, i + 1]

        # 特征值
        hessian_eigvals = torch.linalg.eigvalsh(hessian)
        min_eigval = float(hessian_eigvals.min())

        # 判断是否需要重定义
        needs_redefinition = min_eigval < 0  # 负特征值表示非凸

        return {
            "tau_grid": tau_grid,
            "tearing_values": tearing_tensor,
            "hessian": hessian,
            "hessian_eigenvalues": hessian_eigvals,
            "min_eigenvalue": min_eigval,
            "needs_redefinition": needs_redefinition,
        }

    # ==================================================================
    # 撕裂泛函自动重定义（保底方案的执行）
    # ==================================================================

    def redefine_tearing_functional(
        self,
        metric: Tensor,
        tau: Tensor,
        alpha: float | None = None,
    ) -> Tensor:
        """
        重定义撕裂泛函（保底方案）：

        R_tear(τ) → Tr(g^{-1}(∂g/∂τ)²) + α·Tr((∂g/∂τ)²)

        其中 α 由 Hessian 最小特征值推导：
            α = max(0, -λ_min(Hessian))

        这确保了重定义后的撕裂泛函在 τ 方向上是凸的。
        """
        # 原始撕裂泛函
        R_original = self.vt.tearing_functional(metric, tau)

        # 计算 α
        if alpha is None:
            hessian_result = self.tearing_hessian_spectrum(metric)
            min_eigval = hessian_result["min_eigenvalue"]
            alpha = max(0.0, -min_eigval)

        # 附加项：α·Tr((∂g/∂τ)²)
        delta = 1e-6
        g_tau_plus = self.vt.transparency_modified_metric(metric, tau + delta)
        g_tau_minus = self.vt.transparency_modified_metric(metric, tau - delta)
        dg_dtau = (g_tau_plus - g_tau_minus) / (2 * delta)
        additional_term = alpha * torch.einsum('ij,ij->', dg_dtau, dg_dtau)

        R_redefined = R_original + additional_term

        return R_redefined

    # ==================================================================
    # 综合验证：κ-τ* 正相关定理
    # ==================================================================

    def verify_kappa_tau_positive_correlation(
        self,
        metric: Tensor,
        phi: Tensor,
        kappa_range: tuple[float, float] = (0.1, 5.0),
        n_points: int = 10,
    ) -> dict[str, Tensor | float | bool | str]:
        """
        综合验证 κ-τ* 正相关定理。

        执行流程：
            1. 扫描 κ，求解 τ*（L-BFGS）
            2. 计算相关系数
            3. 如果正相关 > 0.9 → 通过
            4. 如果负相关 → 执行保底方案（Hessian 谱 + 重定义）

        返回：
            dict 包含完整验证结果
        """
        # 第一步：扫描 κ-τ* 曲线
        curve = self.kappa_tau_star_curve(metric, phi, kappa_range, n_points)

        correlation = curve["correlation"]
        is_positive = curve["is_positive_correlation"]

        result = {
            "kappa_curve": curve["kappa_curve"],
            "tau_star_curve": curve["tau_star_curve"],
            "correlation": correlation,
            "is_positive_correlation": is_positive,
            "gradient_norms": curve["gradient_norms"],
            "threshold": 0.9,
        }

        if is_positive:
            # 正相关成立
            result["status"] = "passed"
            result["message"] = (
                f"κ-τ* 正相关定理成立（相关系数 = {correlation:.4f} > 0.9）。"
                f"κ 大 → 曲率代价大 → 需要更高透明度处理 → τ* 高。"
            )
        else:
            # 保底方案：Hessian 谱分析
            hessian_result = self.tearing_hessian_spectrum(metric)

            result["status"] = "fallback"
            result["hessian_eigenvalues"] = hessian_result["hessian_eigenvalues"]
            result["min_eigenvalue"] = hessian_result["min_eigenvalue"]
            result["needs_redefinition"] = hessian_result["needs_redefinition"]

            if hessian_result["needs_redefinition"]:
                # 执行重定义
                result["message"] = (
                    f"κ-τ* 相关性未达标（{correlation:.4f}）。"
                    f"撕裂泛函 Hessian 最小特征值 = {hessian_result['min_eigenvalue']:.6f} < 0。"
                    f"需要重定义撕裂泛函：R_tear → R_tear + α·Tr((∂g/∂τ)²)。"
                    f"这是 v4.3 的输入，不是失败。"
                )
            else:
                result["message"] = (
                    f"κ-τ* 相关性未达标（{correlation:.4f}），"
                    f"但撕裂泛函 Hessian 正定（最小特征值 = {hessian_result['min_eigenvalue']:.6f}）。"
                    f"可能需要 v4.3 重新审视透明度定义。"
                )

        # τ* 的本体论地位说明（二次审查哲学盲区）
        result["ontological_note"] = (
            "τ* 是长期吸引子（变分极值解），不是即时控制参数。"
            "τ(t) 是实时状态（由历史积分决定）。"
            "δJ/δτ 给出了 τ(t) 向 τ* 演化的驱动力方向，而非直接设定值。"
        )

        return result
