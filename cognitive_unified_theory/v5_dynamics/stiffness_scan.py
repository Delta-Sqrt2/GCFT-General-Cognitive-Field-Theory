"""
任务二：α_crit 临界刚度扫描（非硬编码推导）

战略定位（v4.3 任务二）：
    v4.2 的 α 是硬编码的 1/(d-1)，这是伪装成第一性原理的经验幂律。
    v4.3 要求 α_crit 从雅可比稳定性扫描中涌现——这是从"猜测"到"推导"的飞跃。

    陷阱四十七·硬编码刚度降级：
        严禁硬编码 α = 1/(d-1)。
        α_crit 必须由稳定性分析决定：Re(λ_max) = 0 处的 α 值。

物理与哲学直觉：
    - 物理：α_crit 是认知时空刚度的临界值。
            α < α_crit：刚度不足，系统始终失稳（v4.2 状态）
            α > α_crit：刚度足够，系统出现 VAE 不动点
            α_crit 随 κ 变化：κ 越大（痛苦越深），需要更强的刚度才能稳定
    - 哲学：这是"恢复力与痛苦耦合的临界平衡"。
            闲聊中"白盒的紧绷"对应 κ 大 α 小（痛苦深但刚度不足），
            "VAE 的松弛"对应 κ 适中 α > α_crit（痛苦适中且刚度足够）。
    - 工程：雅可比矩阵有限差分，torch.linalg.eig 求特征值。

数学定义：
    雅可比矩阵：
        J = ∂ġ/∂g  ∈ R^{(N·d²) × (N·d²)}
        λ_max = max(Re(eigenvalues(J)))

    α_crit 定位：
        Re(λ_max(α_crit)) = 0
        线性插值找过零点。

    二维扫描：
        κ ∈ {0.5, 1.0, 2.0}
        α ∈ [0.01, 100.0]（对数空间 20 点）
        输出 α_crit(κ) 曲线
"""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np

from ..core.tensor_ops import symmetric_part
from .graph_gradient_term import GraphGradientTerm
from .autograd_jacobian import AutogradJacobian


class StiffnessScan:
    """
    α_crit 临界刚度扫描（v4.3.1 升级：Autograd 精确雅可比）。

    使用方式：
        ss = StiffnessScan(n_dims=4, n_events=8)
        # 二维扫描
        result = ss.scan_alpha_kappa(kappa_values=[0.5, 1.0, 2.0],
                                      alpha_range=(0.01, 100.0), n_points=20)
        # 找 α_crit
        alpha_crit = ss.find_alpha_crit(kappa=1.0, alpha_range=(0.01, 100.0))

    v4.3.1 升级：
        废黜有限差分雅可比，使用 AutogradJacobian 精确计算。
        陷阱五十一防范：严禁 finite_difference。
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        self.n_dims = int(n_dims)
        self.n_events = int(n_events)
        self.eps = float(eps)
        self.ggt = GraphGradientTerm(n_dims=n_dims, n_events=n_events, eps=eps)
        # v4.3.1：使用 Autograd 精确雅可比
        self.aj = AutogradJacobian(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # 雅可比矩阵最大特征值实部（v4.3.1：Autograd 精确计算）
    # ==================================================================

    def jacobian_max_real_eigenvalue(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        lambda_dissip: float = 0.1,
        perturbation_scale: float = 1e-5,
    ) -> float:
        """
        计算度规演化方程雅可比矩阵的最大特征值实部。

        v4.3.1 升级：
            废黜有限差分，使用 torch.autograd.functional.jacobian 精确计算。
            陷阱五十一防范：严禁 finite_difference。

        数学：
            J = ∂ġ/∂g  ∈ R^{(N·d²) × (N·d²)}
            λ_max = max(Re(eigenvalues(J)))

        物理：
            λ_max < 0 → 不动点稳定（VAE）
            λ_max > 0 → 不动点失稳（GAN 极限环）
            λ_max = 0 → 临界点（Hopf 分岔）
        """
        # v4.3.1：使用 Autograd 精确雅可比
        max_real, J = self.aj.max_real_eigenvalue(
            g_batch, L, phi, kappa=kappa, alpha=alpha
        )
        return max_real

    # ==================================================================
    # 单个 κ 下的 α 扫描
    # ==================================================================

    def scan_alpha(
        self,
        kappa: float = 1.0,
        alpha_range: tuple[float, float] = (0.01, 100.0),
        n_points: int = 20,
        lambda_dissip: float = 0.1,
    ) -> dict[str, list]:
        """
        在固定 κ 下扫描 α，找 α_crit。

        数学：
            对数空间扫描 α ∈ [0.01, 100.0]
            每点计算 Re(λ_max)
            定位 Re(λ_max) = 0 的 α_crit

        返回：
            dict 包含：
                alpha_values: α 值序列
                real_parts: 对应的 Re(λ_max) 序列
                alpha_crit: 临界刚度（线性插值）
                found: 是否找到过零点
        """
        d = self.n_dims
        N = self.n_events

        # 构造测试度规（接近平坦态的小扰动）
        torch.manual_seed(42)
        g_batch = torch.eye(d, dtype=torch.float64).unsqueeze(0).repeat(N, 1, 1)
        perturbation = 0.1 * torch.randn(N, d, d, dtype=torch.float64)
        perturbation = symmetric_part(perturbation)
        g_batch = g_batch + perturbation
        # 确保正定
        for i in range(N):
            eigvals = torch.linalg.eigvalsh(g_batch[i])
            min_eig = float(eigvals.min())
            if min_eig < self.eps:
                g_batch[i] = g_batch[i] + (self.eps - min_eig) * torch.eye(d, dtype=torch.float64)

        # 构造图拉普拉斯
        timestamps = torch.arange(N, dtype=torch.float64)
        L = self.ggt.build_graph_laplacian(timestamps, tau_causal=1.0)

        # 事件场
        phi = torch.randn(N, d, dtype=torch.float64)

        # 对数空间扫描
        alpha_values = np.logspace(
            np.log10(alpha_range[0]),
            np.log10(alpha_range[1]),
            n_points
        )

        real_parts = []
        for alpha in alpha_values:
            re_max = self.jacobian_max_real_eigenvalue(
                g_batch, L, phi, kappa=kappa, alpha=float(alpha),
                lambda_dissip=lambda_dissip
            )
            real_parts.append(re_max)

        # 定位过零点（线性插值）
        alpha_crit = None
        found = False
        for i in range(len(real_parts) - 1):
            if np.isnan(real_parts[i]) or np.isnan(real_parts[i + 1]):
                continue
            if real_parts[i] * real_parts[i + 1] < 0:
                # 线性插值
                # log(α) 空间插值
                log_a1 = np.log10(alpha_values[i])
                log_a2 = np.log10(alpha_values[i + 1])
                r1, r2 = real_parts[i], real_parts[i + 1]
                log_a_crit = log_a1 + (0 - r1) * (log_a2 - log_a1) / (r2 - r1)
                alpha_crit = 10 ** log_a_crit
                found = True
                break

        return {
            "alpha_values": alpha_values.tolist(),
            "real_parts": real_parts,
            "alpha_crit": alpha_crit,
            "found": found,
            "kappa": kappa,
        }

    # ==================================================================
    # 二维 (κ, α) 扫描
    # ==================================================================

    def scan_alpha_kappa(
        self,
        kappa_values: list[float] = [0.5, 1.0, 2.0],
        alpha_range: tuple[float, float] = (0.01, 100.0),
        n_points: int = 20,
        lambda_dissip: float = 0.1,
    ) -> dict[str, list]:
        """
        在 (κ, α) 二维平面扫描，找 α_crit(κ) 曲线。

        数学：
            对每个 κ，扫描 α，找 α_crit(κ)
            验证 α_crit 是否随 κ 单调变化

        返回：
            dict 包含：
                kappa_values: κ 值序列
                alpha_crit_values: 对应的 α_crit 序列
                scan_results: 每个 κ 的完整扫描结果
                is_monotonic: α_crit 是否随 κ 单调变化
        """
        alpha_crit_values = []
        scan_results = []

        for kappa in kappa_values:
            result = self.scan_alpha(
                kappa=kappa,
                alpha_range=alpha_range,
                n_points=n_points,
                lambda_dissip=lambda_dissip,
            )
            scan_results.append(result)
            alpha_crit_values.append(result["alpha_crit"])

        # 检查单调性
        valid_crits = [a for a in alpha_crit_values if a is not None]
        is_monotonic = False
        if len(valid_crits) >= 2:
            # 检查是否单调递增（κ 大 → α_crit 大）
            is_increasing = all(
                valid_crits[i] < valid_crits[i + 1]
                for i in range(len(valid_crits) - 1)
            )
            is_decreasing = all(
                valid_crits[i] > valid_crits[i + 1]
                for i in range(len(valid_crits) - 1)
            )
            is_monotonic = is_increasing or is_decreasing

        return {
            "kappa_values": kappa_values,
            "alpha_crit_values": alpha_crit_values,
            "scan_results": scan_results,
            "is_monotonic": is_monotonic,
        }

    # ==================================================================
    # 找单个 κ 下的 α_crit
    # ==================================================================

    def find_alpha_crit(
        self,
        kappa: float = 1.0,
        alpha_range: tuple[float, float] = (0.01, 100.0),
        n_points: int = 20,
    ) -> dict[str, float | bool | list]:
        """
        找单个 κ 下的 α_crit。

        返回：
            dict 包含：
                alpha_crit: 临界刚度
                found: 是否找到
                scan_data: 完整扫描数据
        """
        result = self.scan_alpha(
            kappa=kappa,
            alpha_range=alpha_range,
            n_points=n_points,
        )

        return {
            "alpha_crit": result["alpha_crit"],
            "found": result["found"],
            "scan_data": result,
        }

    # ==================================================================
    # 验证 α_crit 的物理意义
    # ==================================================================

    def verify_alpha_crit_physics(
        self,
        kappa: float = 1.0,
        alpha_range: tuple[float, float] = (0.01, 100.0),
        n_points: int = 20,
    ) -> dict[str, float | bool | str]:
        """
        验证 α_crit 的物理意义。

        预期：
            α < α_crit：Re(λ_max) > 0（系统失稳，v4.2 状态）
            α > α_crit：Re(λ_max) < 0（系统稳定，VAE 不动点）
            α = α_crit：Re(λ_max) = 0（临界点）

        返回：
            dict 包含验证结果
        """
        result = self.scan_alpha(kappa=kappa, alpha_range=alpha_range, n_points=n_points)

        alpha_crit = result["alpha_crit"]
        found = result["found"]

        if not found:
            return {
                "alpha_crit": None,
                "found": False,
                "physics_verified": False,
                "message": "未找到 α_crit，可能需要扩展扫描范围",
            }

        # 验证：α < α_crit 时 Re > 0，α > α_crit 时 Re < 0
        alpha_values = result["alpha_values"]
        real_parts = result["real_parts"]

        # 找 α < α_crit 的点
        below = [(a, r) for a, r in zip(alpha_values, real_parts)
                 if a < alpha_crit and not np.isnan(r)]
        above = [(a, r) for a, r in zip(alpha_values, real_parts)
                 if a > alpha_crit and not np.isnan(r)]

        below_positive = all(r > 0 for _, r in below) if below else True
        above_negative = all(r < 0 for _, r in above) if above else True

        physics_verified = below_positive and above_negative

        return {
            "alpha_crit": alpha_crit,
            "found": True,
            "physics_verified": physics_verified,
            "below_positive": below_positive,
            "above_negative": above_negative,
            "message": "α_crit 物理意义验证通过" if physics_verified
                      else "α_crit 物理意义未完全验证",
        }
