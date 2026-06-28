"""
任务一：图拉普拉斯梯度项（认知时空刚度）

战略定位（v4.3 任务一）：
    v4.2 诊断出拉格朗日量缺少一阶梯度项 Tr((∇g)²)，导致系统始终失稳、
    无法形成 VAE 不动点。v4.3 给拉格朗日量补上这一项，但必须基于图拉普拉斯 L，
    而非坐标差分——因为 v2.0 的事件本体论中没有预设坐标。

    陷阱四十六·坐标幻觉降级：
        严禁使用 np.diff / torch.gradient / 有限差分。
        ∇g 必须在事件图上通过图拉普拉斯 L 定义。
        坐标差分会偷偷引入背景坐标系，违反 v2.0 事件本体论。

物理与哲学直觉：
    - 物理：图拉普拉斯 L = D - C 是 Laplace-Beltrami 算子在离散图上的标准逼近。
            GradTerm = Σ_μν (g_μν^T · L · g_μν) 等价于 Dirichlet 能量，
            提供度规在事件图上的"扩散恢复力"。
            当度规偏离平坦态时，∇g 增大，代价升高，迫使度规回归平滑。
    - 哲学：这是"认知时空刚度"的数学基础。
            闲聊中"VAE 的稳态"需要恢复力——不是被动的不动点，
            而是主动的、有弹性的稳定。梯度项提供了这种弹性。
    - 工程：einsum 批处理，O(d²·N²) 但常数因子极小。

数学定义（严格可微，无降级）：
    图梯度泛函（无坐标差分）：
        GradTerm = Σ_{μν} (g_μν^T · L · g_μν)
        其中 g_μν 是将度规的第 (μ,ν) 分量展平为长度 N 的向量。
        L = D - C 是图拉普拉斯（复用 v2.0）。

    修正作用量：
        S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm
        其中 α 是梯度耦合常数（认知时空刚度）。

    陷阱四十七·硬编码刚度降级：
        α 严禁硬编码为 1/(d-1)。α_crit 必须从雅可比扫描涌现（任务二）。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import (
    effective_rank,
    safe_inverse,
    stable_eigh,
    symmetric_part,
)
from .spectral_curvature_dynamics import SpectralCurvatureDynamics


class GraphGradientTerm:
    """
    图拉普拉斯梯度项（认知时空刚度）。

    使用方式：
        ggt = GraphGradientTerm(n_dims=4, n_events=8)
        # 构造图拉普拉斯
        L = ggt.build_graph_laplacian(timestamps, tau_causal=1.0)
        # 计算梯度项
        grad_term = ggt.compute_grad_term(g_batch, L)
        # 修正作用量
        S = ggt.corrected_action(g_batch, L, phi, A, kappa=1.0, alpha=1.0)

    白盒保证：
        - GradTerm 基于图拉普拉斯 L，严禁坐标差分（陷阱四十六）
        - α 不硬编码，由任务二扫描涌现（陷阱四十七）
        - 全张量运算，einsum 批处理
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
        # 复用 v4.1 的谱曲率动力学（零阶项）
        self.scd = SpectralCurvatureDynamics(n_dims=n_dims, kappa=1.0, eps=eps)

    # ==================================================================
    # 图拉普拉斯构造（复用 v2.0 逻辑，但独立实现避免循环依赖）
    # ==================================================================

    def build_graph_laplacian(
        self,
        timestamps: Tensor,
        tau_causal: float = 1.0,
    ) -> Tensor:
        """
        构造因果图拉普拉斯 L = D - C。

        数学：
            因果邻接 C_ij = exp(-|t_i - t_j| / τ) · H(t_j - t_i)
            其中 H 是因果方向阶跃（t_j > t_i 时为 1）。
            度矩阵 D = diag(Σ_j C_ij)
            拉普拉斯 L = D - C

        物理：
            L 编码事件图的扩散性质。
            完全图：L 的零特征值对应全局均值。
            空图：L = 0，GradTerm = 0。

        严禁：
            - 使用坐标差分（陷阱四十六）
            - 必须基于事件因果图
        """
        t = timestamps.to(torch.float64)
        N = t.shape[0]

        # 因果邻接矩阵 C_ij = exp(-|t_i - t_j| / τ) · H(t_j - t_i)
        # 使用软阶跃避免硬跳变
        DeltaT = t.unsqueeze(0) - t.unsqueeze(1)  # DeltaT[i,j] = t_j - t_i
        tau_soft = tau_causal * 0.1
        H_soft = torch.sigmoid(DeltaT / tau_soft)  # 软因果方向
        K_temporal = torch.exp(-DeltaT.abs() / tau_causal)
        C = K_temporal * H_soft

        # 度矩阵
        degrees = C.sum(dim=-1)
        D = torch.diag(degrees)
        L = D - C

        return L

    def build_complete_graph_laplacian(self, N: int) -> Tensor:
        """
        构造完全图的拉普拉斯（用于验证）。

        完全图：所有节点互相连接，C_ij = 1 (i≠j)
        L = N·I - J（J 是全 1 矩阵）
        零特征值对应全局均值。
        """
        C = torch.ones(N, N, dtype=torch.float64) - torch.eye(N, dtype=torch.float64)
        degrees = C.sum(dim=-1)
        D = torch.diag(degrees)
        L = D - C
        return L

    def build_empty_graph_laplacian(self, N: int) -> Tensor:
        """
        构造空图的拉普拉斯（用于验证）。

        空图：无连接，C = 0
        L = 0
        GradTerm = 0。
        """
        return torch.zeros(N, N, dtype=torch.float64)

    # ==================================================================
    # 图梯度泛函 GradTerm = Σ_μν (g_μν^T · L · g_μν)
    # ==================================================================

    def compute_grad_term(
        self,
        g_batch: Tensor,
        L: Tensor,
    ) -> Tensor:
        """
        图梯度泛函 GradTerm = Σ_{μν} (g_μν^T · L · g_μν)。

        数学：
            g_batch: (N, d, d) 张量，N 个事件，每个事件一个 d×d 度规。
            L: (N, N) 图拉普拉斯。
            g_μν: 将度规的第 (μ,ν) 分量展平为长度 N 的向量。

            GradTerm = Σ_{μν} g_μν^T · L · g_μν
                      = Σ_{μν} Σ_{ij} g_iμν · L_ij · g_jμν

            向量化实现：
                g_flat = g_batch.reshape(N, d*d)     # (N, d²)
                Lg = L @ g_flat                       # (N, d²)
                GradTerm = (g_flat * Lg).sum()        # 标量

        物理：
            - 完全图：GradTerm ∝ 全局方差（度规偏离全局均值的程度）
            - 空图：GradTerm = 0（无连接，无梯度代价）
            - 因果图：GradTerm 度量度规在因果邻接上的变化率

        严禁：
            - 使用 np.diff / torch.gradient / 有限差分（陷阱四十六）
            - 标量近似（必须是全张量运算）

        全可微：L @ g_flat 支持 autograd。
        """
        g = g_batch.to(torch.float64)
        N, d1, d2 = g.shape
        assert d1 == d2 == self.n_dims, f"度规维度不匹配: {d1}x{d2} vs {self.n_dims}"
        assert N == L.shape[0] == L.shape[1], f"事件数不匹配: {N} vs {L.shape}"

        # 向量化实现（einsum 批处理）
        g_flat = g.reshape(N, -1)  # (N, d²)
        Lg = L @ g_flat            # (N, d²)
        grad_term = (g_flat * Lg).sum()

        return grad_term

    # ==================================================================
    # 修正作用量 S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm
    # ==================================================================

    def corrected_action(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor]:
        """
        修正作用量（含图梯度项）。

        数学：
            S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm

            - 谱曲率项 -(κ/2)·N(g)：零阶痛苦势能（v4.2 修正）
            - 规范耦合项 (1/2)·Tr(g⁻¹ D_φ^T D_φ)：度规与事件的耦合
            - 图梯度项 α·GradTerm：一阶认知时空刚度（v4.3 新增）

        物理：
            - κ 大（痛苦深）→ 曲率代价大
            - α 大（刚度高）→ 梯度代价大，度规被迫平滑
            - GradTerm 提供恢复力，使系统能形成 VAE 不动点

        陷阱四十八·正刚度抹平降级：
            单一正刚度会抹平白盒黑洞相。
            此处实现的是"基础正刚度"，任务三的曲率门控会在此基础上
            实现 σ_soft·α 的变号刚度。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # === 零阶项：谱曲率 N(g)（对所有事件的度规取平均）===
        # 每个事件的度规独立计算谱曲率，然后取平均
        N_curvature = torch.zeros(1, dtype=torch.float64)[0]
        for i in range(N):
            N_i = self.scd.spectral_curvature(g[i])
            N_curvature = N_curvature + N_i
        N_curvature = N_curvature / N

        kappa_val = float(kappa)
        curvature_term = -(kappa_val / 2.0) * N_curvature  # -N ≥ 0，正代价

        # === 规范耦合项 ===
        # 使用平均度规计算耦合
        g_mean = g.mean(dim=0)  # (d, d)
        g_mean = symmetric_part(g_mean)
        D_phi = self.scd.covariant_derivative(phi, A)
        g_inv = safe_inverse(g_mean, self.eps)
        DtD = D_phi.T @ D_phi
        coupling_term = 0.5 * torch.einsum('ij,ij->', g_inv, DtD)

        # === 一阶图梯度项（v4.3 新增）===
        alpha_val = float(alpha)
        grad_term = self.compute_grad_term(g, L)
        gradient_term = alpha_val * grad_term

        # === 总作用量 ===
        action = curvature_term + coupling_term + gradient_term

        return {
            "action": action,
            "curvature_term": curvature_term,
            "coupling_term": coupling_term,
            "gradient_term": gradient_term,
            "spectral_curvature": N_curvature,
            "grad_term_raw": grad_term,
        }

    # ==================================================================
    # 验证函数（完全图与空图）
    # ==================================================================

    def verify_complete_graph(self, d: int = 4, N: int = 8) -> dict[str, Tensor]:
        """
        验证完全图：GradTerm ∝ 全局方差。

        数学：
            完全图的 L = N·I - J
            GradTerm = Σ_μν g_μν^T · L · g_μν
                     = N · Σ ||g_i||² - Σ_μν (Σ_i g_iμν)²
                     = N · Σ ||g_i - g_mean||²  （正比于全局方差）

        预期：
            GradTerm > 0（除非所有度规相同）
            GradTerm 正比于度规的全局方差
        """
        torch.manual_seed(42)
        g_batch = torch.randn(N, d, d, dtype=torch.float64)
        g_batch = symmetric_part(g_batch)
        # 确保正定
        for i in range(N):
            eigvals = torch.linalg.eigvalsh(g_batch[i])
            eigvals = torch.clamp(eigvals, min=self.eps)
            g_batch[i] = g_batch[i] - torch.linalg.eigvalsh(g_batch[i]).min() * torch.eye(d, dtype=torch.float64) + self.eps * torch.eye(d, dtype=torch.float64)

        L_complete = self.build_complete_graph_laplacian(N)
        grad_term = self.compute_grad_term(g_batch, L_complete)

        # 计算全局方差
        g_mean = g_batch.mean(dim=0)
        global_variance = ((g_batch - g_mean) ** 2).sum()

        return {
            "grad_term": grad_term,
            "global_variance": global_variance,
            "ratio": grad_term / (global_variance + self.eps),
            "is_positive": grad_term > 0,
        }

    def verify_empty_graph(self, d: int = 4, N: int = 8) -> dict[str, Tensor]:
        """
        验证空图：GradTerm = 0。

        数学：
            空图的 L = 0
            GradTerm = Σ_μν g_μν^T · 0 · g_μν = 0

        预期：
            GradTerm = 0（无连接，无梯度代价）
        """
        torch.manual_seed(42)
        g_batch = torch.randn(N, d, d, dtype=torch.float64)
        g_batch = symmetric_part(g_batch)

        L_empty = self.build_empty_graph_laplacian(N)
        grad_term = self.compute_grad_term(g_batch, L_empty)

        return {
            "grad_term": grad_term,
            "is_zero": abs(float(grad_term)) < 1e-10,
        }

    # ==================================================================
    # 度规演化（含图梯度项的恢复力）
    # ==================================================================

    def metric_velocity_with_gradient(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
        lambda_dissip: float = 0.1,
    ) -> Tensor:
        """
        含图梯度项的度规演化速率。

        数学：
            ġ = -(∂S/∂g + ∂F/∂ġ) / τ
            其中 S 包含图梯度项 α·GradTerm。

        物理：
            - 图梯度项提供恢复力：当度规偏离平坦态时，∇g 增大，
              代价升高，迫使度规回归平滑。
            - 这使系统能形成 VAE 不动点（稳定不动点）。

        实现：
            使用 autograd 计算 ∂S/∂g。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 奇点隔离协议
        g_mean = g.mean(dim=0)
        eigvals_check = torch.linalg.eigvalsh(g_mean)
        eigvals_check = torch.clamp(eigvals_check, min=self.eps)
        cond_g = float(eigvals_check.max() / eigvals_check.min())

        if cond_g >= self.scd.SINGULARITY_THRESHOLD:
            return torch.zeros_like(g)

        # autograd 计算作用量梯度
        g_leaf = g.detach().clone().requires_grad_(True)
        action_result = self.corrected_action(g_leaf, L, phi, A, kappa, alpha)
        S = action_result["action"]

        grad_S = torch.autograd.grad(
            S, g_leaf, create_graph=False, retain_graph=False
        )[0]
        grad_S = symmetric_part(grad_S) if grad_S.shape[-1] == grad_S.shape[-2] else grad_S

        # Rayleigh 耗散力（简化）
        dissip_force = torch.zeros_like(g)

        # 弛豫时间
        tau = float(eigvals_check.mean())

        # 度规演化速率
        g_dot = -(grad_S + dissip_force) / (tau + self.eps)

        return g_dot.detach()

    # ==================================================================
    # 振荡衰减测试（验证恢复力）
    # ==================================================================

    def oscillation_decay_test(
        self,
        d: int = 4,
        N: int = 8,
        kappa: float = 1.0,
        alpha: float = 1.0,
        n_steps: int = 100,
        dt: float = 0.01,
    ) -> dict[str, list]:
        """
        振荡衰减测试：验证梯度项提供恢复力。

        数学：
            在无规范场、无耗散的理想情况下，给定随机扰动的初始度规，
            观察其是否在梯度项作用下振荡衰减回归平坦（VAE 不动点）。

        预期：
            ||g - I|| 随时间衰减（梯度项提供恢复力）
            如果 alpha=0（无梯度项），度规不会衰减

        输出：
            度规范数 ||g - I|| 随时间的衰减曲线
        """
        torch.manual_seed(42)
        # 随机扰动的初始度规（偏离平坦态）
        g_init = torch.eye(d, dtype=torch.float64).unsqueeze(0).repeat(N, 1, 1)
        perturbation = 0.5 * torch.randn(N, d, d, dtype=torch.float64)
        perturbation = symmetric_part(perturbation)
        g_batch = g_init + perturbation

        # 确保正定
        for i in range(N):
            eigvals = torch.linalg.eigvalsh(g_batch[i])
            eigvals = torch.clamp(eigvals, min=self.eps)
            g_batch[i] = g_batch[i] - torch.linalg.eigvalsh(g_batch[i]).min() * torch.eye(d, dtype=torch.float64) + self.eps * torch.eye(d, dtype=torch.float64)

        # 构造因果图拉普拉斯
        timestamps = torch.arange(N, dtype=torch.float64)
        L = self.build_graph_laplacian(timestamps, tau_causal=1.0)

        # 简单的事件场
        phi = torch.randn(N, d, dtype=torch.float64)

        # 演化轨迹
        norm_trajectory = []
        g_trajectory = []

        g_current = g_batch.clone()
        for step in range(n_steps):
            # 计算度规偏离平坦态的范数
            g_mean = g_current.mean(dim=0)
            norm_deviation = float((g_mean - torch.eye(d, dtype=torch.float64)).norm())
            norm_trajectory.append(norm_deviation)
            g_trajectory.append(g_current.clone())

            # 演化一步（简化欧拉积分）
            g_dot = self.metric_velocity_with_gradient(
                g_current, L, phi, None, kappa, alpha, lambda_dissip=0.0
            )
            g_current = g_current + dt * g_dot
            g_current = symmetric_part(g_current)

            # 正则化
            for i in range(N):
                eigvals, eigvecs = torch.linalg.eigh(g_current[i])
                eigvals = torch.clamp(eigvals, min=self.eps, max=1e8)
                g_current[i] = (eigvecs * eigvals) @ eigvecs.T

        return {
            "norm_trajectory": norm_trajectory,
            "g_trajectory": g_trajectory,
            "alpha": alpha,
            "initial_norm": norm_trajectory[0],
            "final_norm": norm_trajectory[-1],
            "decay_ratio": norm_trajectory[-1] / (norm_trajectory[0] + self.eps),
        }
