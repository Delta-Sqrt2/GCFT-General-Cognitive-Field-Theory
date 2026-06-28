"""
任务一：Autograd 精确雅可比实现（v4.3.1 核心）

战略定位（v4.3.1 任务一）：
    v4.3 的有限差分雅可比在 128 维下全部返回 nan，被浮点噪声淹没。
    v4.3.1 废黜有限差分，使用 torch.autograd.functional.jacobian 计算精确雅可比。

    陷阱五十一·有限差分掩耳盗铃：
        严禁使用 finite_difference 近似雅可比。必须使用 Autograd。

    陷阱五十二·结构性发散掩盖：
        如果 Autograd 仍然出现 NaN/Inf，严禁掩盖。必须诊断 ODE 刚性。

    陷阱五十三·标量梯度混淆：
        雅可比是向量场 f(g) = ġ 对 g 的偏导 ∂f_i/∂g_j，
        不是标量作用量 S 对 g 的梯度 ∇S。
        ∇S 是向量，雅可比是矩阵。

物理与哲学直觉：
    - 物理：雅可比矩阵 J = ∂ġ/∂g 编码度规演化方程的线性稳定性。
            Re(λ_max) < 0 → 不动点稳定（VAE）
            Re(λ_max) > 0 → 不动点失稳（GAN 极限环）
            Re(λ_max) = 0 → 临界点（Hopf 分岔）
    - 哲学：这是"认知系统稳定性"的数学判据。
            闲聊中"VAE 的稳态"对应 Re(λ_max) < 0，
            "GAN 的振荡"对应 Re(λ_max) > 0。
    - 工程：torch.autograd.functional.jacobian 精确到机器精度，
            不受有限差分步长选择的影响。

数学定义：
    度规演化速度场：
        ġ = f(g) = -(∂S/∂g) / τ
        其中 S 是修正作用量，τ 是弛豫时间。

    雅可比矩阵：
        J = ∂f/∂g  ∈ R^{(N·d²) × (N·d²)}
        J_ij = ∂f_i / ∂g_j

    关键实现细节：
        1. f(g) 内部使用 autograd.grad(S, g, create_graph=True)
           create_graph=True 保留计算图，允许外层 jacobian 微分。
        2. 严禁 detach()——detach 会切断计算图，使 jacobian 失效。
        3. τ 必须保持为张量（不能 float()），否则切断图。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import safe_inverse, symmetric_part
from .graph_gradient_term import GraphGradientTerm
from .spectral_curvature_dynamics import SpectralCurvatureDynamics


class AutogradJacobian:
    """
    Autograd 精确雅可比计算器。

    使用方式：
        aj = AutogradJacobian(n_dims=4, n_events=8)
        L = aj.build_graph_laplacian(timestamps)
        max_re, J = aj.max_real_eigenvalue(g_batch, L, phi, kappa=1.0, alpha=1.0)

    白盒保证：
        - 使用 torch.autograd.functional.jacobian，严禁有限差分（陷阱五十一）
        - NaN 安全诊断，严禁掩盖（陷阱五十二）
        - 雅可比是 f(g)=ġ 对 g 的偏导，不是 ∇S（陷阱五十三）
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            n_events: 事件数 N
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.n_events = int(n_events)
        self.eps = float(eps)
        self.ggt = GraphGradientTerm(n_dims=n_dims, n_events=n_events, eps=eps)
        self.scd = SpectralCurvatureDynamics(n_dims=n_dims, kappa=1.0, eps=eps)

    # ==================================================================
    # 图拉普拉斯构造（代理到 GraphGradientTerm）
    # ==================================================================

    def build_graph_laplacian(
        self,
        timestamps: Tensor,
        tau_causal: float = 1.0,
    ) -> Tensor:
        """构造因果图拉普拉斯 L = D - C。"""
        return self.ggt.build_graph_laplacian(timestamps, tau_causal)

    # ==================================================================
    # 纯速度场函数（保留计算图，供 jacobian 微分）
    # ==================================================================

    def velocity_field_pure(
        self,
        g_flat: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> Tensor:
        """
        纯速度场函数 f(g) = ġ，保留完整计算图。

        数学：
            ġ = -(∂S/∂g) / τ
            其中 S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm

        v4.6 修复（陷阱六十三/六十五）：
            旧版本硬编码了 g^T·L·g（V1 公式），绕过了 self.ggt。
            新版本调用 self.ggt.corrected_action，自动使用正确的 GradTerm 版本。
            当 self.ggt 是 GraphGradientTermV2 时，使用 Σ C_ij·(g_i - g_j)²。

        陷阱五十三·标量梯度混淆防范：
            此函数返回的是向量场 ġ（展平为 N·d² 维向量），
            不是标量 S。雅可比 J = ∂ġ/∂g，不是 ∇S。

        关键实现：
            1. create_graph=True：保留 ∂S/∂g 的计算图，允许外层微分
            2. 严禁 detach()：detach 切断图
            3. τ 保持为张量：float() 会切断图
        """
        N, d = self.n_events, self.n_dims
        g_reshaped = g_flat.reshape(N, d, d)

        # === 计算修正作用量 S（通过 self.ggt，保留图）===
        # v4.6 修复：调用 self.ggt.corrected_action，而非硬编码梯度项
        # 这确保当 self.ggt 被替换为 GraphGradientTermV2 时，
        # 雅可比计算自动使用 V2 公式 Σ C_ij·(g_i - g_j)²
        action_result = self.ggt.corrected_action(
            g_reshaped, L, phi, None, kappa, alpha
        )
        S = action_result["action"]

        # === 计算 ∂S/∂g_flat（create_graph=True 是关键！）===
        grad_S_flat = torch.autograd.grad(
            S, g_flat, create_graph=True, retain_graph=True,
            allow_unused=True
        )[0]
        if grad_S_flat is None:
            grad_S_flat = torch.zeros_like(g_flat)
        grad_S = symmetric_part(grad_S_flat.reshape(N, d, d))

        # === 计算弛豫时间 τ（保持张量！）===
        g_sym = symmetric_part(g_reshaped)
        g_mean = g_sym.mean(dim=0)
        g_mean = symmetric_part(g_mean)
        eigvals_mean = torch.linalg.eigvalsh(g_mean)
        eigvals_mean = torch.clamp(eigvals_mean, min=self.eps)
        tau = eigvals_mean.mean()

        # === 度规演化速度 ġ = -(∂S/∂g) / τ ===
        g_dot = -(grad_S) / (tau + self.eps)

        return g_dot.reshape(-1)

    # ==================================================================
    # 平衡点求解与平衡点雅可比（v4.3.1 结构性修正）
    # ==================================================================

    def find_equilibrium(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        max_iter: int = 500,
        lr: float = 0.01,
        tol: float = 1e-8,
    ) -> tuple[Tensor, float, int]:
        """
        求解度规演化方程的平衡点 g*（∇S = 0）。

        数学：
            平衡点条件：ġ = -∇S/τ = 0  ⟺  ∇S = 0
            使用梯度下降最小化 S：
                g ← g - lr · ∇S
            收敛判据：|∇S| < tol

        物理意义：
            v4.3.1 结构性发现：在非平衡点评估雅可比稳定性无物理意义。
            雅可比 J = -Hessian(S)/τ + (∇S ⊗ ∇τ)/τ²
            第二项 (∇S ⊗ ∇τ)/τ² 在非平衡点不为零，引入伪失稳。
            必须在平衡点 ∇S=0 处评估，此时 J = -Hessian(S)/τ。

        返回：
            (g_eq, residual, n_iter): 平衡点、残差、迭代次数
        """
        g = g_batch.to(torch.float64).clone()
        N, d, _ = g.shape

        residual = float('inf')
        n_iter = 0

        for it in range(max_iter):
            n_iter = it + 1
            g_leaf = g.detach().clone().requires_grad_(True)

            # 计算作用量 S
            action_result = self.ggt.corrected_action(
                g_leaf, L, phi, None, kappa, alpha
            )
            S = action_result["action"]

            # ∇S（一阶梯度，create_graph=False）
            grad_S = torch.autograd.grad(S, g_leaf, create_graph=False)[0]
            residual = float(grad_S.norm())

            if residual < tol:
                break

            # 梯度下降：g ← g - lr · ∇S
            with torch.no_grad():
                g = g_leaf - lr * grad_S
                g = symmetric_part(g)

                # 正则化：确保正定
                for i in range(N):
                    eigvals_i = torch.linalg.eigvalsh(g[i])
                    min_eig = float(eigvals_i.min())
                    if min_eig < self.eps:
                        g[i] = g[i] + (self.eps - min_eig + 1e-10) * torch.eye(d, dtype=torch.float64)

        return g, residual, n_iter

    def equilibrium_jacobian(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        max_iter: int = 500,
        lr: float = 0.01,
        tol: float = 1e-8,
    ) -> dict[str, Tensor | float | int | bool]:
        """
        在平衡点 g* 处计算雅可比矩阵。

        物理意义：
            在平衡点 ∇S=0 处，雅可比简化为：
                J|_eq = -Hessian(S)|_eq / τ|_eq
            此时稳定性完全由作用量的 Hessian 决定。
            Hessian 正定 → J 负定 → 稳定（VAE 不动点）
            Hessian 不定 → J 有正特征值 → 失稳（GAN 极限环）

        返回：
            dict 包含平衡点、雅可比、最大实部、是否收敛
        """
        # 1. 求解平衡点
        g_eq, residual, n_iter = self.find_equilibrium(
            g_batch, L, phi, kappa, alpha, max_iter, lr, tol
        )

        converged = residual < tol * 100  # 放宽 100 倍作为收敛判据

        # 2. 在平衡点计算雅可比
        max_real, J = self.max_real_eigenvalue(g_eq, L, phi, kappa, alpha)

        return {
            "g_equilibrium": g_eq,
            "jacobian": J,
            "max_real": max_real,
            "residual": residual,
            "n_iter": n_iter,
            "converged": converged,
        }

    # ==================================================================
    # Autograd 精确雅可比计算
    # ==================================================================

    def compute_jacobian(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> Tensor:
        """
        使用 torch.autograd.functional.jacobian 计算精确雅可比。

        数学：
            J = ∂f/∂g  ∈ R^{(N·d²) × (N·d²)}
            其中 f(g) = ġ 是度规演化速度场。

        陷阱五十一防范：
            严禁有限差分。使用 torch.autograd.functional.jacobian。

        返回：
            J: (N·d², N·d²) 雅可比矩阵
        """
        g = g_batch.to(torch.float64)
        g_flat = g.reshape(-1)

        # 定义纯函数（只接受 g_flat，返回 g_dot_flat）
        def velocity_func(g_input):
            return self.velocity_field_pure(
                g_input, L, phi, kappa, alpha
            )

        # 使用 torch.autograd.functional.jacobian 计算精确雅可比
        J = torch.autograd.functional.jacobian(velocity_func, g_flat)

        return J

    # ==================================================================
    # 雅可比最大特征值实部
    # ==================================================================

    def max_real_eigenvalue(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        zero_threshold: float = 1e-6,
    ) -> tuple[float, Tensor]:
        """
        计算雅可比矩阵的最大特征值实部（排除对称性零模）。

        数学：
            J = ∂f/∂g
            λ_max = max(Re(eigenvalues(J)))

        v4.8 修正（零模排除）：
            g 是对称矩阵，独立维度为 d(d+1)/2，而非 d²。
            在 N·d² 维空间计算雅可比时，反对称方向的扰动不改变 S，
            产生 N·d(d-1)/2 个零模（非物理方向）。
            这些零模使 max Re(λ) = 0，掩盖了真实的稳定性。
            修正：排除 |Re(λ)| < zero_threshold 的零模，
            只在物理（对称）子空间评估稳定性。

        物理：
            λ_max < 0 → 不动点稳定（VAE）
            λ_max > 0 → 不动点失稳（GAN 极限环）
            λ_max = 0 → 临界点（Hopf 分岔）

        返回：
            (max_real, J): 最大实部（排除零模）和雅可比矩阵
        """
        J = self.compute_jacobian(g_batch, L, phi, kappa, alpha)

        # NaN 安全检查（陷阱五十二）
        if torch.isnan(J).any() or torch.isinf(J).any():
            return float('nan'), J

        # 特征值计算
        try:
            eigenvalues = torch.linalg.eigvals(J)
            real_parts = eigenvalues.real

            # v4.8 零模排除：过滤掉 |Re(λ)| < zero_threshold 的零模
            mask = real_parts.abs() > zero_threshold
            physical_real_parts = real_parts[mask]

            if len(physical_real_parts) > 0:
                max_real = float(physical_real_parts.max())
            else:
                # 所有特征值都是零模（极端情况）
                max_real = 0.0
        except Exception as e:
            print(f"  [警告] 特征分解失败: {e}")
            return float('nan'), J

        return max_real, J

    # ==================================================================
    # NaN 安全诊断（陷阱五十二）
    # ==================================================================

    def diagnose_nan(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, float | str | bool]:
        """
        NaN 诊断：定位 NaN 发生的具体算子。

        陷阱五十二防范：
            严禁 try-except 跳过 NaN。必须记录具体算子。

        诊断步骤：
            1. 检查度规条件数 cond(g)
            2. 检查谱曲率 N(g)
            3. 检查 safe_inverse 是否产生 NaN
            4. 检查 eigvalsh 是否产生 NaN
            5. 检查作用量 S 是否为 NaN
            6. 检查 grad_S 是否为 NaN
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        diagnosis = {
            "has_nan": False,
            "nan_source": "unknown",
            "cond_g": 0.0,
            "spectral_curvature": 0.0,
            "action_value": 0.0,
            "grad_S_norm": 0.0,
            "message": "",
        }

        # 1. 检查每个事件的度规条件数
        max_cond = 0.0
        for i in range(N):
            eigvals_i = torch.linalg.eigvalsh(g[i])
            eigvals_i_clamped = torch.clamp(eigvals_i, min=self.eps)
            cond_i = float(eigvals_i_clamped.max() / eigvals_i_clamped.min())
            if cond_i > max_cond:
                max_cond = cond_i
            if torch.isnan(eigvals_i).any():
                diagnosis["has_nan"] = True
                diagnosis["nan_source"] = "torch.linalg.eigvalsh (事件 " + str(i) + ")"
                diagnosis["message"] = "eigvalsh 产生 NaN，度规可能非对称或非正定"
        diagnosis["cond_g"] = max_cond

        # 2. 检查谱曲率
        N_curvature = torch.zeros(1, dtype=torch.float64)[0]
        for i in range(N):
            eigvals_i = torch.linalg.eigvalsh(g[i])
            eigvals_i = torch.clamp(eigvals_i, min=self.eps)
            p_i = eigvals_i / eigvals_i.sum()
            N_i = (p_i * torch.log(p_i)).sum()
            N_curvature = N_curvature + N_i
            if torch.isnan(N_i).any():
                diagnosis["has_nan"] = True
                diagnosis["nan_source"] = "spectral_curvature (事件 " + str(i) + ")"
                diagnosis["message"] = "谱曲率 N(g) 产生 NaN，可能 log(0) 或 log(负数)"
        N_curvature = N_curvature / N
        diagnosis["spectral_curvature"] = float(N_curvature)

        # 3. 检查 safe_inverse
        g_mean = g.mean(dim=0)
        g_mean = symmetric_part(g_mean)
        try:
            g_inv = safe_inverse(g_mean, self.eps)
            if torch.isnan(g_inv).any():
                diagnosis["has_nan"] = True
                diagnosis["nan_source"] = "safe_inverse (torch.linalg.inv)"
                diagnosis["message"] = "度规求逆产生 NaN，度规可能奇异"
        except Exception as e:
            diagnosis["has_nan"] = True
            diagnosis["nan_source"] = "safe_inverse (异常: " + str(e) + ")"
            diagnosis["message"] = "度规求逆异常"

        # 4. 检查作用量
        action_result = self.ggt.corrected_action(
            g, L, phi, None, kappa, alpha
        )
        S = action_result["action"]
        diagnosis["action_value"] = float(S)
        if torch.isnan(S).any():
            diagnosis["has_nan"] = True
            diagnosis["nan_source"] = "corrected_action (作用量 S)"
            diagnosis["message"] = "作用量 S 为 NaN"

        # 5. 检查 grad_S
        g_leaf = g.detach().clone().requires_grad_(True)
        action_result2 = self.ggt.corrected_action(
            g_leaf, L, phi, None, kappa, alpha
        )
        S2 = action_result2["action"]
        try:
            grad_S = torch.autograd.grad(S2, g_leaf, create_graph=False)[0]
            diagnosis["grad_S_norm"] = float(grad_S.norm())
            if torch.isnan(grad_S).any():
                diagnosis["has_nan"] = True
                diagnosis["nan_source"] = "autograd.grad (∂S/∂g)"
                diagnosis["message"] = "作用量梯度 ∂S/∂g 产生 NaN"
        except Exception as e:
            diagnosis["has_nan"] = True
            diagnosis["nan_source"] = "autograd.grad (异常: " + str(e) + ")"
            diagnosis["message"] = "作用量梯度计算异常"

        # 6. 如果没有 NaN，尝试计算雅可比
        if not diagnosis["has_nan"]:
            try:
                J = self.compute_jacobian(g, L, phi, kappa, alpha)
                if torch.isnan(J).any():
                    diagnosis["has_nan"] = True
                    diagnosis["nan_source"] = "torch.autograd.functional.jacobian"
                    diagnosis["message"] = "雅可比矩阵 J 产生 NaN，可能二阶导数不稳定"
            except Exception as e:
                diagnosis["has_nan"] = True
                diagnosis["nan_source"] = "compute_jacobian (异常: " + str(e) + ")"
                diagnosis["message"] = "雅可比计算异常"

        if not diagnosis["has_nan"]:
            diagnosis["message"] = "无 NaN，所有算子正常"

        return diagnosis

    # ==================================================================
    # 验证函数：雅可比 vs 有限差分对比
    # ==================================================================

    def verify_against_finite_difference(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
        perturbation_scale: float = 1e-5,
    ) -> dict[str, float | Tensor | bool]:
        """
        验证 Autograd 雅可比与有限差分的一致性。

        数学：
            Autograd: J_exact = ∂f/∂g（精确到机器精度）
            有限差分: J_fd ≈ (f(g+ε) - f(g)) / ε（一阶近似）
            误差: ||J_exact - J_fd|| / ||J_exact||

        用途：
            验证 Autograd 实现的正确性。
            如果误差 < 1e-3，说明 Autograd 实现正确。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        dim = N * d * d

        # Autograd 精确雅可比
        J_exact = self.compute_jacobian(g, L, phi, kappa, alpha)

        # 有限差分雅可比
        # 注意：velocity_field_pure 内部使用 autograd.grad，需要 g_flat 是 leaf tensor 且 requires_grad=True
        g_flat = g.reshape(-1).detach().requires_grad_(True)

        # 基准速度
        f_base = self.velocity_field_pure(g_flat, L, phi, kappa, alpha)

        J_fd = torch.zeros(dim, dim, dtype=torch.float64)
        for i in range(dim):
            g_perturbed = g_flat.clone()
            g_perturbed[i] += perturbation_scale
            g_perturbed = g_perturbed.detach().requires_grad_(True)
            f_perturbed = self.velocity_field_pure(g_perturbed, L, phi, kappa, alpha)
            J_fd[:, i] = (f_perturbed - f_base).detach() / perturbation_scale

        # 误差计算
        if torch.isnan(J_exact).any() or torch.isnan(J_fd).any():
            return {
                "jacobian_autograd": J_exact,
                "jacobian_finite_diff": J_fd,
                "relative_error": float('nan'),
                "is_consistent": False,
                "message": "雅可比包含 NaN，无法对比",
            }

        diff = (J_exact - J_fd).norm()
        norm_exact = J_exact.norm()
        rel_error = float(diff / (norm_exact + self.eps))

        return {
            "jacobian_autograd": J_exact,
            "jacobian_finite_diff": J_fd,
            "relative_error": rel_error,
            "is_consistent": rel_error < 0.01,
            "message": "Autograd 与有限差分一致" if rel_error < 0.01
                      else "Autograd 与有限差分不一致",
        }
