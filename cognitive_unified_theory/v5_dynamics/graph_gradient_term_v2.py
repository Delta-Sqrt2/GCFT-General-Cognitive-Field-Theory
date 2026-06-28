"""
任务一：GradTerm 重定义（v4.5 核心）—— 从 g^T·L·g 到 Σ C_ij·(g_i - g_j)²

战略定位（v4.5 任务一）：
    v4.4 发现因果图拉普拉斯 L 非对称（||L - L^T|| = 0.368），
    导致 GradTerm = g^T·L·g 可为负，违反半正定性假设。
    v4.5 废黜 g^T·L·g 形式，改用天然非负的 Σ C_ij·(g_i - g_j)²。

    陷阱五十九·非对称二次型降级：
        严禁使用 g^T·L·g 作为梯度项（L 非对称时可为负）。
        必须改用 Σ C_ij·(g_i - g_j)²。

物理与哲学直觉：
    - 物理：新梯度项 Σ C_ij·(g_i - g_j)² 度量"度规在因果网络上的不均匀性"。
            完全图均匀度规时为 0（无梯度惩罚）；
            完全图非均匀度规时为正（有梯度惩罚）。
            这更符合"恢复力"的物理直觉——惩罚度规在相邻事件间的剧烈变化。
            当 C 对称时，它退化为 2·g^T·L·g，与 v4.3 一致。
    - 哲学：这是"认知时空刚度"的物理正确实现。
            闲聊中"VAE 的稳态"需要恢复力——不是被动的不动点，
            而是主动的、有弹性的稳定。新梯度项提供了这种弹性，
            且不依赖因果方向的对性（更接近真实认知网络）。
    - 工程：向量化实现，O(d²·N²)，常数因子与 v4.3 相同。

数学定义：
    新梯度项：
        GradTerm_v2 = Σ_{μν} Σ_{i,j} C_ij · (g_i,μν - g_j,μν)²

    其中：
        - g_i,μν 是第 i 个事件的度规的第 (μ,ν) 分量
        - C_ij 是因果邻接矩阵（可非对称）
        - 求和遍历所有事件对 (i,j) 和所有度规分量 (μ,ν)

    关键性质：
        1. 天然非负：平方项 ≥ 0，C_ij ≥ 0（因果核），故 GradTerm_v2 ≥ 0
        2. 与 C 对称性无关：无论 C 是否对称，GradTerm_v2 都有意义
        3. 退化一致性：当 C 对称时，GradTerm_v2 = 2·g^T·L·g
          （因为 Σ_ij C_ij·(g_i - g_j)² = 2·Σ_ij C_ij·g_i·g_j - 2·Σ_ij C_ij·g_i·g_j
           = 2·g^T·D·g - 2·g^T·C·g = 2·g^T·L·g，当 C 对称时）
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
from .graph_gradient_term import GraphGradientTerm


class GraphGradientTermV2(GraphGradientTerm):
    """
    图梯度项 v2（天然非负）。

    使用方式：
        ggt2 = GraphGradientTermV2(n_dims=4, n_events=8)
        C = ggt2.build_causal_adjacency(timestamps)
        grad_term = ggt2.compute_grad_term_v2(g_batch, C)
        S = ggt2.corrected_action(g_batch, C, phi, None, kappa=1.0, alpha=1.0)

    白盒保证：
        - GradTerm = Σ C_ij·(g_i - g_j)²，天然非负（陷阱五十九防范）
        - 与 C 对称性无关，适用于因果方向性图
        - 当 C 对称时，与 v4.3 的 g^T·L·g 一致（差因子 2）
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # 因果邻接矩阵构造（返回 C，而非 L）
    # ==================================================================

    def build_causal_adjacency(
        self,
        timestamps: Tensor,
        tau_causal: float = 1.0,
    ) -> Tensor:
        """
        构造因果邻接矩阵 C_ij。

        数学：
            C_ij = exp(-|t_i - t_j| / τ) · H_soft(t_j - t_i)
            其中 H_soft 是软因果方向阶跃。

        与 v4.3 的区别：
            v4.3 返回 L = D - C（拉普拉斯，非对称）
            v4.5 返回 C（邻接矩阵，可直接用于新梯度项）

        物理：
            C_ij 表示事件 j 对事件 i 的因果影响强度。
            时间越近，影响越强（指数衰减）。
            因果方向（t_j > t_i）影响更强（软阶跃）。
        """
        t = timestamps.to(torch.float64)
        N = t.shape[0]

        DeltaT = t.unsqueeze(0) - t.unsqueeze(1)  # DeltaT[i,j] = t_j - t_i
        tau_soft = tau_causal * 0.1
        H_soft = torch.sigmoid(DeltaT / tau_soft)  # 软因果方向
        K_temporal = torch.exp(-DeltaT.abs() / tau_causal)
        C = K_temporal * H_soft

        return C

    def build_complete_graph_adjacency(self, N: int) -> Tensor:
        """构造完全图邻接矩阵（用于验证）。C_ij = 1 (i≠j)。"""
        return torch.ones(N, N, dtype=torch.float64) - torch.eye(N, dtype=torch.float64)

    def build_empty_graph_adjacency(self, N: int) -> Tensor:
        """构造空图邻接矩阵（用于验证）。C = 0。"""
        return torch.zeros(N, N, dtype=torch.float64)

    # ==================================================================
    # 新梯度项 GradTerm_v2 = Σ C_ij·(g_i - g_j)²
    # ==================================================================

    def compute_grad_term_v2(
        self,
        g_batch: Tensor,
        C: Tensor,
    ) -> Tensor:
        """
        新梯度项 GradTerm_v2 = Σ_{μν} Σ_{i,j} C_ij · (g_i,μν - g_j,μν)²。

        数学：
            g_batch: (N, d, d) 张量
            C: (N, N) 因果邻接矩阵（可非对称）

            对每个度规分量 (μ,ν)：
                g_vec = g[:, μ, ν]  # (N,)
                diff = g_vec.unsqueeze(1) - g_vec.unsqueeze(0)  # (N, N), diff[i,j] = g_i - g_j
                term = (C * diff²).sum()  # 标量

            GradTerm_v2 = Σ_{μν} term

        向量化实现：
            g_flat = g_batch.reshape(N, d²)        # (N, d²)
            diff = g_flat.unsqueeze(1) - g_flat.unsqueeze(0)  # (N, N, d²)
            weighted_diff2 = C.unsqueeze(-1) * diff²  # (N, N, d²)
            GradTerm_v2 = weighted_diff2.sum()  # 标量

        关键性质：
            1. 天然非负：diff² ≥ 0，C ≥ 0，故 GradTerm_v2 ≥ 0
            2. 与 C 对称性无关
            3. 当 C 对称时，GradTerm_v2 = 2·g^T·L·g
        """
        g = g_batch.to(torch.float64)
        N, d1, d2 = g.shape
        assert d1 == d2 == self.n_dims, f"度规维度不匹配: {d1}x{d2} vs {self.n_dims}"
        assert N == C.shape[0] == C.shape[1], f"事件数不匹配: {N} vs {C.shape}"

        # 向量化实现
        g_flat = g.reshape(N, -1)  # (N, d²)
        # diff[i, j, :] = g_flat[i] - g_flat[j]
        diff = g_flat.unsqueeze(1) - g_flat.unsqueeze(0)  # (N, N, d²)
        # 加权平方和
        weighted_diff2 = C.unsqueeze(-1) * (diff ** 2)  # (N, N, d²)
        grad_term_v2 = weighted_diff2.sum()  # 标量

        return grad_term_v2

    # ==================================================================
    # 修正作用量（使用新梯度项）
    # ==================================================================

    def corrected_action(
        self,
        g_batch: Tensor,
        L_or_C: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor]:
        """
        修正作用量（使用 v4.5 新梯度项）。

        数学：
            S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α·GradTerm_v2

            其中 GradTerm_v2 = Σ C_ij·(g_i - g_j)²（天然非负）

        与 v4.3 的区别：
            v4.3: gradient_term = α · g^T·L·g（可为负）
            v4.5: gradient_term = α · Σ C_ij·(g_i - g_j)²（天然非负）

        参数：
            L_or_C: 因果邻接矩阵 C（v4.5 模式）
                    注意：此处参数名保留 L_or_C 以兼容牛顿法求解器接口，
                    但实际传入的应是 C（邻接矩阵），而非 L（拉普拉斯）。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        C = L_or_C  # v4.5 模式：传入的是 C

        # === 零阶项：谱曲率 N(g) ===
        N_curvature = torch.zeros(1, dtype=torch.float64)[0]
        for i in range(N):
            N_i = self.scd.spectral_curvature(g[i])
            N_curvature = N_curvature + N_i
        N_curvature = N_curvature / N

        kappa_val = float(kappa)
        curvature_term = -(kappa_val / 2.0) * N_curvature

        # === 规范耦合项 ===
        g_mean = g.mean(dim=0)
        g_mean = symmetric_part(g_mean)
        D_phi = self.scd.covariant_derivative(phi, A)
        g_inv = safe_inverse(g_mean, self.eps)
        DtD = D_phi.T @ D_phi
        coupling_term = 0.5 * torch.einsum('ij,ij->', g_inv, DtD)

        # === 一阶图梯度项（v4.5 新形式）===
        alpha_val = float(alpha)
        grad_term_v2 = self.compute_grad_term_v2(g, C)
        gradient_term = alpha_val * grad_term_v2

        # === 总作用量 ===
        action = curvature_term + coupling_term + gradient_term

        return {
            "action": action,
            "curvature_term": curvature_term,
            "coupling_term": coupling_term,
            "gradient_term": gradient_term,
            "spectral_curvature": N_curvature,
            "grad_term_raw": grad_term_v2,
        }

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_complete_graph_uniform(
        self, d: int = 4, N: int = 8
    ) -> dict[str, Tensor | bool | float]:
        """
        验证一：完全图 + 均匀度规 → GradTerm_v2 = 0。

        数学：
            完全图：C_ij = 1 (i≠j)
            均匀度规：所有 g_i 相同
            → diff = 0 → GradTerm_v2 = 0
        """
        # 均匀度规：所有事件相同
        g_uniform = torch.eye(d, dtype=torch.float64).unsqueeze(0).repeat(N, 1, 1)
        C_complete = self.build_complete_graph_adjacency(N)

        grad_term = self.compute_grad_term_v2(g_uniform, C_complete)

        return {
            "grad_term": grad_term,
            "is_zero": abs(float(grad_term)) < 1e-10,
            "description": "完全图+均匀度规 → GradTerm_v2 = 0",
        }

    def verify_complete_graph_nonuniform(
        self, d: int = 4, N: int = 8
    ) -> dict[str, Tensor | bool | float]:
        """
        验证二：完全图 + 非均匀度规 → GradTerm_v2 > 0。

        数学：
            完全图：C_ij = 1 (i≠j)
            非均匀度规：g_i 各不相同
            → diff ≠ 0 → GradTerm_v2 > 0
        """
        torch.manual_seed(42)
        g_nonuniform = torch.randn(N, d, d, dtype=torch.float64)
        g_nonuniform = symmetric_part(g_nonuniform)
        # 确保正定
        for i in range(N):
            eigvals = torch.linalg.eigvalsh(g_nonuniform[i])
            min_eig = float(eigvals.min())
            if min_eig < 1.0:
                g_nonuniform[i] = g_nonuniform[i] + (1.0 - min_eig + 0.1) * torch.eye(d, dtype=torch.float64)

        C_complete = self.build_complete_graph_adjacency(N)
        grad_term = self.compute_grad_term_v2(g_nonuniform, C_complete)

        return {
            "grad_term": grad_term,
            "is_positive": float(grad_term) > 0,
            "description": "完全图+非均匀度规 → GradTerm_v2 > 0",
        }

    def verify_consistency_with_v43(
        self, d: int = 4, N: int = 8
    ) -> dict[str, Tensor | bool | float]:
        """
        验证三：当 C 对称时，GradTerm_v2 = 2·g^T·L·g（与 v4.3 一致）。

        数学：
            当 C 对称时：
            Σ_ij C_ij·(g_i - g_j)²
            = Σ_ij C_ij·(g_i² - 2·g_i·g_j + g_j²)
            = 2·Σ_i (Σ_j C_ij)·g_i² - 2·Σ_ij C_ij·g_i·g_j
            = 2·g^T·D·g - 2·g^T·C·g
            = 2·g^T·(D - C)·g
            = 2·g^T·L·g

        验证方法：
            1. 构造对称 C（完全图）
            2. 计算 GradTerm_v2
            3. 计算 2·g^T·L·g（v4.3 形式）
            4. 比较两者
        """
        torch.manual_seed(42)
        g_batch = torch.randn(N, d, d, dtype=torch.float64)
        g_batch = symmetric_part(g_batch)
        for i in range(N):
            eigvals = torch.linalg.eigvalsh(g_batch[i])
            min_eig = float(eigvals.min())
            if min_eig < 1.0:
                g_batch[i] = g_batch[i] + (1.0 - min_eig + 0.1) * torch.eye(d, dtype=torch.float64)

        # 对称 C（完全图）
        C_symmetric = self.build_complete_graph_adjacency(N)
        # 对应的 L
        degrees = C_symmetric.sum(dim=-1)
        L_symmetric = torch.diag(degrees) - C_symmetric

        # v4.5 新形式
        grad_term_v2 = self.compute_grad_term_v2(g_batch, C_symmetric)

        # v4.3 旧形式（使用父类方法）
        grad_term_v1 = super().compute_grad_term(g_batch, L_symmetric)

        # 预期：grad_term_v2 ≈ 2 · grad_term_v1
        ratio = float(grad_term_v2 / (grad_term_v1 + self.eps))
        relative_error = abs(float(grad_term_v2 - 2 * grad_term_v1)) / (abs(float(grad_term_v2)) + self.eps)

        return {
            "grad_term_v2": grad_term_v2,
            "grad_term_v1": grad_term_v1,
            "expected_v2": 2 * grad_term_v1,
            "ratio": ratio,
            "relative_error": relative_error,
            "is_consistent": relative_error < 1e-6,
            "description": "C 对称时，GradTerm_v2 = 2·g^T·L·g",
        }

    def verify_nonnegativity(
        self, d: int = 4, N: int = 8, n_trials: int = 100
    ) -> dict[str, int | float | bool]:
        """
        验证四：GradTerm_v2 天然非负（100 次随机试验）。

        数学：
            对任意 g 和任意 C ≥ 0：
            GradTerm_v2 = Σ C_ij·(g_i - g_j)² ≥ 0
            （因为 C_ij ≥ 0 且 (g_i - g_j)² ≥ 0）
        """
        n_negative = 0
        min_value = float('inf')

        for trial in range(n_trials):
            torch.manual_seed(trial)
            g_random = torch.randn(N, d, d, dtype=torch.float64)
            g_random = symmetric_part(g_random)

            # 随机非对称 C（模拟因果图）
            C_random = torch.rand(N, N, dtype=torch.float64)
            # 确保非负
            C_random = torch.clamp(C_random, min=0)

            grad_term = self.compute_grad_term_v2(g_random, C_random)

            if float(grad_term) < 0:
                n_negative += 1
            min_value = min(min_value, float(grad_term))

        return {
            "n_trials": n_trials,
            "n_negative": n_negative,
            "min_value": min_value,
            "is_always_nonnegative": n_negative == 0,
            "description": "GradTerm_v2 天然非负（100 次随机试验）",
        }
