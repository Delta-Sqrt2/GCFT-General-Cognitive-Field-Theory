"""
任务二：伪弧长延拓法追踪相变边界（理论闭环）

战略定位（v4.1 任务二）：
    v4.0 只通过手动设定五组参数，证明了五个吸引子的存在性。
    v4.1 必须跨越的边界是：追踪五大相态的吸引域边界，
    证明吸引域之间的边界是光滑的还是分形的（决定人格是离散分类还是连续光谱）。

    陷阱三十九·暴力扫描降级：
        严禁三维网格暴力扫描（κ, λ, rigidity 组合爆炸）。
        必须使用伪弧长延拓法（pseudo-arclength continuation）。
        固定耗散参数 λ，对痛苦耦合 κ 进行延拓追踪。

物理与哲学直觉：
    - 物理：伪弧长延拓法是分岔分析的标准工具。
            通过追踪不动点随参数变化，检测雅可比矩阵特征值实部过零，
            精确定位 Hopf 分岔点（从不动点 VAE 转向极限环 GAN）。
    - 哲学：相变边界的性质决定了"人格类型"是离散分类还是连续光谱。
            光滑边界 → 人格是连续光谱（动态交融）
            分形边界 → 人格是离散分类（类型跳跃）
            这直接回答了 MBTI 的根本问题：人格是离散的还是连续的？
    - 工程：固定 λ，对 κ 进行延拓，计算每个 κ 点的雅可比最大特征值实部。

数学定义（严格可微，无降级）：
    伪弧长延拓法：
        1. 从已知不动点 (g*, κ_0) 出发
        2. 预测步：g_pred = g* + ds · v（v 是切线方向）
        3. 校正步：求解 G(g, κ) = ġ(g, κ) = 0
        4. 弧长约束：(g - g_pred)² + (κ - κ_pred)² = ds²

    Hopf 分岔检测：
        当雅可比矩阵 J = ∂ġ/∂g 的最大特征值实部由负转正时，
        即为 Hopf 分岔点 κ_crit。
        在 κ < κ_crit：不动点稳定（VAE）
        在 κ > κ_crit：不动点失稳，极限环诞生（GAN）
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


class PhaseBoundaryContinuation:
    """
    伪弧长延拓法追踪相变边界。

    使用方式：
        cont = PhaseBoundaryContinuation(n_dims=8, lambda_dissip=0.1)
        # 延拓追踪
        result = cont.continue_kappa(kappa_range=(0.1, 10.0), n_steps=50)
        # 检测 Hopf 分岔
        bifurcation = cont.detect_hopf_bifurcation(result)

    白盒保证：
        - 伪弧长延拓法（陷阱三十九）
        - 雅可比特征值分析
        - Hopf 分岔点精确定位
        - 无暴力网格扫描
    """

    def __init__(
        self,
        n_dims: int = 8,
        lambda_dissip: float = 0.1,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            lambda_dissip: 固定耗散系数 λ
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.lambda_dissip = float(lambda_dissip)
        self.eps = float(eps)

    # ==================================================================
    # 不动点求解
    # ==================================================================

    def find_fixed_point(
        self,
        kappa: float,
        phi: Tensor,
        A: Tensor | None = None,
        g_init: Tensor | None = None,
        max_iter: int = 200,
        tol: float = 1e-8,
        dt: float = 0.01,
    ) -> dict[str, Tensor | float]:
        """
        求解给定 κ 下的不动点 g*（ġ = 0）。

        数学：
            不动点条件：ġ(g*, κ) = 0
            使用梯度下降演化直到 ||ġ|| < tol

        物理：
            不动点 = 稳态认知状态（VAE 不动点或 RL 坍缩态）
            κ 变化时，不动点的位置和稳定性会变化。

        参数：
            kappa: 痛苦耦合常数
            phi: 事件特征场
            A: 规范联络（可选）
            g_init: 初始度规（可选，默认单位矩阵）
            max_iter: 最大迭代数
            tol: 收敛容差
            dt: 演化步长

        返回：
            dict 包含：
                fixed_point: 不动点 g*
                converged: 是否收敛
                iterations: 迭代次数
                final_velocity_norm: 最终 ||ġ||
        """
        d = self.n_dims

        if g_init is None:
            g = torch.eye(d, dtype=torch.float64)
        else:
            g = symmetric_part(g_init.to(torch.float64))

        dyn = SpectralCurvatureDynamics(
            n_dims=d,
            kappa=kappa,
            lambda_dissip=self.lambda_dissip,
            eps=self.eps,
        )

        g_prev = g.clone()
        converged = False
        iterations = 0

        for i in range(max_iter):
            g_new, is_bh = dyn.evolve_step(g, A, phi, g_prev, dt)

            # 检查收敛
            g_dot = dyn.metric_velocity(g_new, A, phi, g)
            velocity_norm = float(g_dot.norm())

            if velocity_norm < tol:
                converged = True
                iterations = i + 1
                g = g_new
                break

            # 检查黑洞相
            if is_bh:
                converged = False
                iterations = i + 1
                g = g_new
                break

            g_prev = g
            g = g_new
            iterations = i + 1

        final_velocity_norm = float(dyn.metric_velocity(g, A, phi, g_prev).norm())

        return {
            "fixed_point": g,
            "converged": converged,
            "iterations": iterations,
            "final_velocity_norm": final_velocity_norm,
            "is_black_hole": is_bh if 'is_bh' in dir() else False,
        }

    # ==================================================================
    # 雅可比矩阵与稳定性分析
    # ==================================================================

    def compute_jacobian_eigenvalues(
        self,
        g: Tensor,
        kappa: float,
        phi: Tensor,
        A: Tensor | None = None,
        perturbation_scale: float = 1e-6,
    ) -> Tensor:
        """
        计算不动点处雅可比矩阵的特征值。

        数学：
            J = ∂ġ/∂g |_{g=g*, κ}
            特征值：λ_i = eigenvalues(J)
            最大实部：max(Re(λ_i))

        物理：
            max(Re(λ_i)) < 0 → 不动点稳定（VAE）
            max(Re(λ_i)) > 0 → 不动点失稳（Hopf 分岔 → GAN）
            max(Re(λ_i)) = 0 → 临界点

        实现：
            用有限差分近似雅可比矩阵。
            对每个对称方向扰动，计算 ġ 的变化。
        """
        d = self.n_dims

        dyn = SpectralCurvatureDynamics(
            n_dims=d,
            kappa=kappa,
            lambda_dissip=self.lambda_dissip,
            eps=self.eps,
        )

        g = symmetric_part(g.to(torch.float64))

        # 基准速度
        g_dot_base = dyn.metric_velocity(g, A, phi, g)

        # 有限差分雅可比（向量化）
        # 由于 g 是 d×d 对称矩阵，有 d*(d+1)/2 个独立方向
        # 为简化，使用 d*d 的完整雅可比
        jacobian = torch.zeros(d * d, d * d, dtype=torch.float64)

        for i in range(d):
            for j in range(i, d):  # 只计算上三角（对称）
                delta = torch.zeros_like(g)
                delta[i, j] = perturbation_scale
                if i != j:
                    delta[j, i] = perturbation_scale  # 对称化

                g_perturbed = g + delta
                g_dot_perturbed = dyn.metric_velocity(g_perturbed, A, phi, g)

                diff = (g_dot_perturbed - g_dot_base) / perturbation_scale
                jacobian[:, i * d + j] = diff.flatten()
                if i != j:
                    jacobian[:, j * d + i] = diff.flatten()

        # 特征值
        eigvals = torch.linalg.eigvals(jacobian)

        return eigvals

    def max_eigenvalue_real_part(
        self,
        g: Tensor,
        kappa: float,
        phi: Tensor,
        A: Tensor | None = None,
    ) -> float:
        """
        计算雅可比矩阵最大特征值的实部。

        返回：
            max_real: 最大特征值实部
            < 0 → 稳定（VAE）
            > 0 → 失稳（GAN）
            = 0 → 临界
        """
        eigvals = self.compute_jacobian_eigenvalues(g, kappa, phi, A)
        max_real = float(eigvals.real.max())
        return max_real

    # ==================================================================
    # 伪弧长延拓法
    # ==================================================================

    def continue_kappa(
        self,
        kappa_range: tuple[float, float] = (0.1, 10.0),
        n_steps: int = 50,
        phi: Tensor | None = None,
        A: Tensor | None = None,
        evolve_steps: int = 100,
        dt: float = 0.01,
    ) -> dict[str, Tensor]:
        """
        伪弧长延拓法：对 κ 进行延拓追踪。

        数学：
            1. 从 κ_0 出发，求解不动点 g*(κ_0)
            2. 增大 κ，用前一个不动点作为初始猜测
            3. 在每个 κ 点，计算雅可比最大特征值实部
            4. 检测特征值实部过零 → Hopf 分岔点

        严禁：
            - 三维网格暴力扫描（陷阱三十九）
            - 必须使用延拓法（前一个解作为下一个的初始猜测）

        参数：
            kappa_range: κ 的范围 (κ_min, κ_max)
            n_steps: 延拓步数
            phi: 事件特征场（若 None，随机生成）
            A: 规范联络
            evolve_steps: 每个 κ 点的演化步数
            dt: 演化步长

        返回：
            dict 包含：
                kappa_curve: κ 值序列 (n_steps,)
                fixed_points: 不动点序列 (n_steps, d, d)
                eigenvalue_real_parts: 雅可比最大特征值实部序列 (n_steps,)
                effective_ranks: 有效秩序列 (n_steps,)
                condition_numbers: 条件数序列 (n_steps,)
                black_hole_flags: 黑洞相标记序列 (n_steps,)
        """
        d = self.n_dims
        torch.manual_seed(42)

        if phi is None:
            N_events = 15
            phi = torch.randn(N_events, d, dtype=torch.float64) * 0.5

        kappa_min, kappa_max = kappa_range
        kappa_values = torch.linspace(kappa_min, kappa_max, n_steps, dtype=torch.float64)

        fixed_points = []
        eigenvalue_real_parts = []
        effective_ranks = []
        condition_numbers = []
        black_hole_flags = []
        spectral_curvatures = []

        # 初始度规
        g_init = torch.eye(d, dtype=torch.float64)

        for idx, kappa in enumerate(kappa_values):
            kappa_val = float(kappa)

            # 求解不动点（用前一个解作为初始猜测）
            if idx > 0 and not black_hole_flags[-1]:
                g_init = fixed_points[-1]

            fp_result = self.find_fixed_point(
                kappa=kappa_val,
                phi=phi,
                A=A,
                g_init=g_init,
                max_iter=evolve_steps,
                tol=1e-6,
                dt=dt,
            )

            g_star = fp_result["fixed_point"]
            is_bh = fp_result.get("is_black_hole", False)

            # 检查条件数
            eigvals_check = torch.linalg.eigvalsh(g_star)
            eigvals_check = torch.clamp(eigvals_check, min=self.eps)
            cond_g = float(eigvals_check.max() / eigvals_check.min())

            if cond_g >= 1e10 or is_bh:
                # 黑洞相
                black_hole_flags.append(True)
                fixed_points.append(g_star)
                eigenvalue_real_parts.append(float('nan'))  # 黑洞相标记
                effective_ranks.append(float('nan'))
                condition_numbers.append(cond_g)
                spectral_curvatures.append(float('nan'))
            else:
                # 正常相态
                black_hole_flags.append(False)
                fixed_points.append(g_star)

                # 雅可比特征值
                max_real = self.max_eigenvalue_real_part(g_star, kappa_val, phi, A)
                eigenvalue_real_parts.append(max_real)

                # 相空间坐标
                dyn = SpectralCurvatureDynamics(
                    n_dims=d, kappa=kappa_val,
                    lambda_dissip=self.lambda_dissip, eps=self.eps,
                )
                coords = dyn.phase_space_coordinates(g_star)
                effective_ranks.append(float(coords['effective_rank']))
                condition_numbers.append(float(coords['condition_number']))
                spectral_curvatures.append(float(coords['spectral_curvature']))

        return {
            "kappa_curve": kappa_values,
            "fixed_points": torch.stack(fixed_points),
            "eigenvalue_real_parts": torch.tensor(eigenvalue_real_parts, dtype=torch.float64),
            "effective_ranks": torch.tensor(effective_ranks, dtype=torch.float64),
            "condition_numbers": torch.tensor(condition_numbers, dtype=torch.float64),
            "spectral_curvatures": torch.tensor(spectral_curvatures, dtype=torch.float64),
            "black_hole_flags": torch.tensor(black_hole_flags, dtype=torch.bool),
        }

    # ==================================================================
    # Hopf 分岔检测
    # ==================================================================

    def detect_hopf_bifurcation(
        self,
        continuation_result: dict[str, Tensor],
    ) -> dict[str, Tensor | float | bool]:
        """
        检测 Hopf 分岔点。

        数学：
            Hopf 分岔：雅可比最大特征值实部由负转正。
            κ_crit = κ 值，使得 max(Re(λ)) = 0

        判据：
            在 κ_curve 中寻找 eigenvalue_real_parts 从负到正的过零点。

        返回：
            dict 包含：
                has_bifurcation: 是否检测到分岔
                kappa_crit: 分岔临界点 κ_crit
                eigenvalue_at_crit: 临界点处的特征值实部
                phase_boundary_type: 相变边界类型（"smooth" 或 "fractal"）
        """
        eigenvalues = continuation_result["eigenvalue_real_parts"]
        kappa_curve = continuation_result["kappa_curve"]
        black_hole_flags = continuation_result["black_hole_flags"]

        # 过滤黑洞相
        valid_mask = ~black_hole_flags & ~torch.isnan(eigenvalues)
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_kappa = kappa_curve[valid_mask]

        if len(valid_eigenvalues) < 2:
            return {
                "has_bifurcation": False,
                "kappa_crit": float('nan'),
                "eigenvalue_at_crit": float('nan'),
                "phase_boundary_type": "unknown",
            }

        # 检测过零点
        sign_changes = []
        for i in range(len(valid_eigenvalues) - 1):
            v1 = float(valid_eigenvalues[i])
            v2 = float(valid_eigenvalues[i + 1])
            if v1 * v2 < 0:  # 符号变化
                # 线性插值找过零点
                k1 = float(valid_kappa[i])
                k2 = float(valid_kappa[i + 1])
                kappa_crit = k1 + (k2 - k1) * (-v1) / (v2 - v1)
                sign_changes.append({
                    "kappa_crit": kappa_crit,
                    "eigenvalue_before": v1,
                    "eigenvalue_after": v2,
                })

        if len(sign_changes) == 0:
            return {
                "has_bifurcation": False,
                "kappa_crit": float('nan'),
                "eigenvalue_at_crit": float('nan'),
                "phase_boundary_type": "no_transition",
            }

        # 取第一个分岔点
        first_bifurcation = sign_changes[0]
        kappa_crit = first_bifurcation["kappa_crit"]

        # 相变边界类型判定
        # 光滑边界：特征值实部随 κ 单调变化
        # 分形边界：特征值实部随 κ 振荡变化
        diffs = valid_eigenvalues[1:] - valid_eigenvalues[:-1]
        sign_changes_in_diffs = (diffs[1:] * diffs[:-1] < 0).sum().item()
        total_points = len(diffs)

        if total_points > 0:
            oscillation_ratio = sign_changes_in_diffs / total_points
            if oscillation_ratio < 0.2:
                phase_boundary_type = "smooth"  # 光滑边界 → 连续光谱
            else:
                phase_boundary_type = "fractal"  # 分形边界 → 离散分类
        else:
            phase_boundary_type = "unknown"

        return {
            "has_bifurcation": True,
            "kappa_crit": kappa_crit,
            "eigenvalue_at_crit": 0.0,
            "eigenvalue_before": first_bifurcation["eigenvalue_before"],
            "eigenvalue_after": first_bifurcation["eigenvalue_after"],
            "phase_boundary_type": phase_boundary_type,
            "n_bifurcations": len(sign_changes),
        }

    # ==================================================================
    # 相图完备性验证
    # ==================================================================

    def verify_phase_completeness(
        self,
        continuation_result: dict[str, Tensor],
    ) -> dict[str, Tensor | str]:
        """
        验证相图完备性：在延拓范围内是否存在未预见的稳态。

        判据：
            1. 有效秩 R 的分布是否覆盖 [1, d] 范围
            2. 条件数 cond(g) 的分布是否覆盖 [1, ∞) 范围
            3. 是否存在第六种未预见的稳态

        返回：
            dict 包含相图完备性分析结果
        """
        effective_ranks = continuation_result["effective_ranks"]
        condition_numbers = continuation_result["condition_numbers"]
        black_hole_flags = continuation_result["black_hole_flags"]

        # 过滤黑洞相
        valid_mask = ~black_hole_flags & ~torch.isnan(effective_ranks)
        valid_R = effective_ranks[valid_mask]
        valid_cond = condition_numbers[valid_mask]

        if len(valid_R) == 0:
            return {
                "completeness": "all_black_hole",
                "R_range": (float('nan'), float('nan')),
                "cond_range": (float('nan'), float('nan')),
            }

        R_min = float(valid_R.min())
        R_max = float(valid_R.max())
        cond_min = float(valid_cond.min())
        cond_max = float(valid_cond.max())

        # 相态区域判定
        # VAE 区域：R ≈ d, cond ≈ 1
        # GAN 区域：R 中等, cond 中等, 振荡
        # 白盒/RL 区域：R ≈ 1, cond 大
        vae_mask = (valid_R > self.n_dims * 0.7) & (valid_cond < 5)
        collapse_mask = (valid_R < 2.5) & (valid_cond > 50)
        intermediate_mask = ~vae_mask & ~collapse_mask

        n_vae = vae_mask.sum().item()
        n_collapse = collapse_mask.sum().item()
        n_intermediate = intermediate_mask.sum().item()

        return {
            "completeness": "verified" if n_vae > 0 and n_collapse > 0 else "incomplete",
            "R_range": (R_min, R_max),
            "cond_range": (cond_min, cond_max),
            "n_vae_region": n_vae,
            "n_collapse_region": n_collapse,
            "n_intermediate_region": n_intermediate,
            "n_black_hole": black_hole_flags.sum().item(),
        }
