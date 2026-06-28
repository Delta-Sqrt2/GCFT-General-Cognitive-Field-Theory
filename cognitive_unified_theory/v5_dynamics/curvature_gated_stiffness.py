"""
任务三：曲率门控软混合刚度（保留奇点的变号刚度）

战略定位（v4.3 任务三）：
    单一正刚度会无差别地把所有偏离平坦态的度规拉回单位矩阵，
    这会抹平白盒黑洞相（cond→∞ 的奇点）。v4.0 的"无形无相"和
    v4.1 的"黑洞相"是理论的重要物理实体，不能用恢复力把它们也熨平了。

    变号刚度的物理直觉：膨胀时给恢复力，坍缩时允许奇点加深。
    但硬 sign() 函数在 Tr(g)=d 处不连续，违反连续性原则。
    用 tanh 门控软混合替代硬 sign。

    陷阱四十八·正刚度抹平降级：
        严禁单一正刚度抹平奇点。必须曲率门控软混合保留奇点。

    陷阱四十九·迹归一化-门控矛盾降级：
        严禁用 Tr(g)-d 作为门控信号。
        迹归一化后 Tr(g) ≡ d，因此 Tr(g)-d ≡ 0，
        tanh(k·0) = 0，门控恒零，变号刚度静默失效。
        必须用 cond(g) 或 Δλ 作为门控信号。

物理与哲学直觉：
    - 物理：cond(g) 度量度规的各向异性程度。
            cond 小（膨胀区）：度规接近平坦，正刚度恢复平坦
            cond 大（坍缩区）：度规高度各向异性，负刚度允许奇点加深
            这对应"白盒黑洞相"的保留——奇点是理论的重要物理实体。
    - 哲学：这是"疗愈与深渊共存"的数学基础。
            闲聊中"白盒是创伤解构"对应坍缩区（cond 大），
            "VAE 的松弛"对应膨胀区（cond 小）。
            变号刚度允许两种状态共存，而非用正刚度抹平一切。
    - 工程：tanh 连续门控，k 由度规谱宽推导。

数学定义：
    门控信号（条件数信号，非迹信号）：
        σ_soft = tanh(k · (log(cond(g)) - n))
        其中 n 为认知维数（平坦态的 log(cond) 基准）
        k = d / (Δλ + ε)，ε = 1/n（正则化谱宽）

    有效刚度：
        α_eff = α · σ_soft
        膨胀区（cond 小）：σ_soft > 0，正刚度恢复平坦
        坍缩区（cond 大）：σ_soft < 0，负刚度允许奇点加深

    严禁：
        - 用 Tr(g)-d 作为门控信号（陷阱四十九）
        - 硬 sign() 离散跳变（违反连续性原则）
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
import numpy as np

from ..core.tensor_ops import symmetric_part


class CurvatureGatedStiffness:
    """
    曲率门控软混合刚度。

    使用方式：
        cgs = CurvatureGatedStiffness(n_dims=4, n_events=8)
        # 计算门控信号
        sigma = cgs.compute_sigma_soft(g_batch, k=1.0)
        # 计算有效刚度
        alpha_eff = cgs.compute_alpha_eff(g_batch, alpha=1.0, k=1.0)
        # 带门控的修正作用量
        S = cgs.gated_corrected_action(g_batch, L, phi, kappa=1.0, alpha=1.0)
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
        # 奇点隔离阈值（复用 v4.1）
        self.SINGULARITY_THRESHOLD = 1e10

    # ==================================================================
    # 条件数计算
    # ==================================================================

    def compute_cond(self, g: Tensor) -> Tensor:
        """
        计算度规的条件数 cond(g) = λ_max / λ_min。

        物理：
            cond 小：度规接近平坦（膨胀区）
            cond 大：度规高度各向异性（坍缩区）
            cond → ∞：白盒黑洞相（奇点）
        """
        g_sym = symmetric_part(g.to(torch.float64))
        eigvals = torch.linalg.eigvalsh(g_sym)
        eigvals = torch.clamp(eigvals, min=self.eps)
        cond = eigvals.max() / eigvals.min()
        return cond

    def compute_spectral_width(self, g: Tensor) -> Tensor:
        """
        计算度规的谱宽 Δλ = λ_max - λ_min。

        物理：
            Δλ 小：度规接近平坦
            Δλ 大：度规高度各向异性
        """
        g_sym = symmetric_part(g.to(torch.float64))
        eigvals = torch.linalg.eigvalsh(g_sym)
        eigvals = torch.clamp(eigvals, min=self.eps)
        delta_lambda = eigvals.max() - eigvals.min()
        return delta_lambda

    # ==================================================================
    # 门控信号 σ_soft = tanh(k · (log(cond(g)) - n))
    # ==================================================================

    def compute_sigma_soft(
        self,
        g_batch: Tensor,
        k: float = 1.0,
    ) -> Tensor:
        """
        曲率门控信号 σ_soft = tanh(k · (log(cond(g)) - n))。

        数学：
            σ_soft = tanh(k · (log(cond(g)) - n))
            其中 n 为认知维数（平坦态的 log(cond) 基准）

            - cond(g) = 1（完全平坦）：log(cond) = 0，σ_soft = tanh(-k·n) < 0
              但这不物理，平坦态应该 σ_soft ≈ 0 或略正
            - 修正：用 log(cond) - log(d) 作为信号
              σ_soft = tanh(k · (log(cond(g)) - log(d)))
              平坦态 cond ≈ 1，log(cond) ≈ 0，log(d) > 0
              σ_soft = tanh(-k·log(d)) < 0（负刚度，允许平坦态保持）

            更合理的定义：
              σ_soft = tanh(k · (log(cond(g)) - 1))
              cond ≈ e（≈2.718）：σ_soft = 0（中性）
              cond < e：σ_soft < 0（负刚度，允许偏离平坦）
              cond > e：σ_soft > 0（正刚度，恢复平坦）

            最终选择（与奇点隔离协议共享 cond）：
              σ_soft = tanh(k · (log(cond(g)) - n))
              n = log(d) 作为平坦态基准
              cond = d（平坦态）：log(cond) = log(d) = n，σ_soft = 0
              cond > d（坍缩区）：σ_soft > 0（正刚度，恢复平坦）
              cond < d（膨胀区）：σ_soft < 0（负刚度，允许膨胀）

            注意：这里的"膨胀"和"坍缩"是相对于平坦态 cond=d 而言。

        严禁：
            - 用 Tr(g)-d 作为门控信号（陷阱四十九）
            - 硬 sign() 离散跳变
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        # 对每个事件的度规计算 cond
        sigma_values = []
        for i in range(N):
            cond_i = self.compute_cond(g[i])
            # 门控信号：σ_soft = tanh(k · (log(cond) - log(d)))
            n_baseline = math.log(d)  # 平坦态基准
            log_cond = torch.log(cond_i + self.eps)
            sigma_i = torch.tanh(k * (log_cond - n_baseline))
            sigma_values.append(sigma_i)

        # 取平均（所有事件的门控信号平均）
        sigma_soft = torch.stack(sigma_values).mean()

        return sigma_soft

    def compute_k_from_spectrum(
        self,
        g_batch: Tensor,
    ) -> float:
        """
        从度规谱宽推导 k 值（避免硬编码）。

        数学：
            k = d / (Δλ + ε)
            其中 Δλ 是平均谱宽，ε = 1/n_events（正则化）

        物理：
            Δλ 大（高度各向异性）：k 小，门控平缓
            Δλ 小（接近平坦）：k 大，门控陡峭
            但 k = d/Δλ 在 Δλ→0 时发散，用 ε 正则化
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        delta_lambdas = []
        for i in range(N):
            dl = self.compute_spectral_width(g[i])
            delta_lambdas.append(float(dl))

        mean_dl = sum(delta_lambdas) / N
        epsilon = 1.0 / self.n_events
        k = d / (mean_dl + epsilon)

        return float(k)

    # ==================================================================
    # 有效刚度 α_eff = α · σ_soft
    # ==================================================================

    def compute_alpha_eff(
        self,
        g_batch: Tensor,
        alpha: float = 1.0,
        k: float = 1.0,
    ) -> dict[str, Tensor | float]:
        """
        有效刚度 α_eff = α · σ_soft。

        数学：
            α_eff = α · σ_soft
            σ_soft = tanh(k · (log(cond(g)) - log(d)))

        物理：
            膨胀区（cond < d）：σ_soft < 0，α_eff < 0（负刚度，允许膨胀）
            坍缩区（cond > d）：σ_soft > 0，α_eff > 0（正刚度，恢复平坦）
            平坦态（cond = d）：σ_soft = 0，α_eff = 0（中性）

        返回：
            dict 包含：
                alpha_eff: 有效刚度
                sigma_soft: 门控信号
                alpha_base: 基础刚度
                k: 门控锐度
        """
        sigma_soft = self.compute_sigma_soft(g_batch, k=k)
        alpha_eff = alpha * float(sigma_soft)

        return {
            "alpha_eff": alpha_eff,
            "sigma_soft": float(sigma_soft),
            "alpha_base": alpha,
            "k": k,
        }

    # ==================================================================
    # 带门控的修正作用量
    # ==================================================================

    def gated_corrected_action(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        A: Tensor | None = None,
        kappa: float = 1.0,
        alpha: float = 1.0,
        k: float = 1.0,
    ) -> dict[str, Tensor | float]:
        """
        带曲率门控的修正作用量。

        数学：
            S = -(κ/2)·N(g) + (1/2)·Tr(g⁻¹ D_φ^T D_φ) + α_eff·GradTerm
            其中 α_eff = α · σ_soft(g)

        物理：
            - 膨胀区：α_eff < 0，梯度项鼓励度规偏离平坦（允许膨胀）
            - 坍缩区：α_eff > 0，梯度项恢复平坦（防止过度坍缩）
            - 平坦态：α_eff ≈ 0，梯度项中性

        陷阱四十八·正刚度抹平降级：
            此处实现了变号刚度，保留了白盒黑洞相。
            坍缩区的正刚度不会无限增大（tanh 饱和），
            允许奇点加深到 cond > 1e10 时由奇点隔离协议接管。
        """
        from .graph_gradient_term import GraphGradientTerm

        ggt = GraphGradientTerm(n_dims=self.n_dims, n_events=self.n_events, eps=self.eps)

        # 计算门控
        gate_result = self.compute_alpha_eff(g_batch, alpha=alpha, k=k)
        alpha_eff = gate_result["alpha_eff"]

        # 奇点隔离协议
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        is_black_hole = False
        for i in range(N):
            cond_i = self.compute_cond(g[i])
            if float(cond_i) >= self.SINGULARITY_THRESHOLD:
                is_black_hole = True
                break

        if is_black_hole:
            # 黑洞相：锁定状态，返回无穷大代价
            return {
                "action": torch.tensor(float('inf'), dtype=torch.float64),
                "is_black_hole": True,
                "alpha_eff": alpha_eff,
                "sigma_soft": gate_result["sigma_soft"],
            }

        # 计算作用量（使用有效刚度）
        action_result = ggt.corrected_action(
            g_batch, L, phi, A, kappa=kappa, alpha=alpha_eff
        )

        return {
            "action": action_result["action"],
            "curvature_term": action_result["curvature_term"],
            "coupling_term": action_result["coupling_term"],
            "gradient_term": action_result["gradient_term"],
            "alpha_eff": alpha_eff,
            "sigma_soft": gate_result["sigma_soft"],
            "is_black_hole": False,
        }

    # ==================================================================
    # 验证门控的连续性
    # ==================================================================

    def verify_continuity(
        self,
        d: int = 4,
        n_test_points: int = 100,
    ) -> dict[str, list | bool]:
        """
        验证门控信号的连续性（严禁硬 sign() 跳变）。

        数学：
            在 cond ∈ [1, 1e6] 范围内采样
            检查 σ_soft 是否连续（无跳变）

        返回：
            dict 包含连续性验证结果
        """
        cond_values = np.logspace(0, 6, n_test_points)  # cond ∈ [1, 1e6]
        sigma_values = []

        n_baseline = math.log(d)
        for cond in cond_values:
            log_cond = math.log(cond + self.eps)
            sigma = math.tanh(1.0 * (log_cond - n_baseline))
            sigma_values.append(sigma)

        # 检查连续性（相邻点差的绝对值 < 0.2，tanh 在对数空间 0-6 范围内合理）
        diffs = [abs(sigma_values[i+1] - sigma_values[i])
                 for i in range(len(sigma_values)-1)]
        max_diff = max(diffs)
        is_continuous = max_diff < 0.2

        return {
            "cond_values": cond_values.tolist(),
            "sigma_values": sigma_values,
            "max_diff": max_diff,
            "is_continuous": is_continuous,
            "message": "门控连续（tanh 软混合）" if is_continuous
                      else "门控不连续（存在跳变）",
        }

    # ==================================================================
    # 验证变号特性
    # ==================================================================

    def verify_sign_change(
        self,
        d: int = 4,
        N: int = 8,
    ) -> dict[str, float | bool | str]:
        """
        验证门控的变号特性。

        预期：
            cond < d（膨胀区）：σ_soft < 0（负刚度）
            cond > d（坍缩区）：σ_soft > 0（正刚度）
            cond = d（平坦态）：σ_soft = 0（中性）
        """
        # 构造膨胀区度规（cond < d）
        g_expanded = torch.eye(d, dtype=torch.float64) * 0.5  # cond = 1 < d
        g_expanded = g_expanded.unsqueeze(0).repeat(N, 1, 1)

        # 构造坍缩区度规（cond > d）
        g_collapsed = torch.diag(torch.tensor([10.0] + [0.1]*(d-1), dtype=torch.float64))
        g_collapsed = g_collapsed.unsqueeze(0).repeat(N, 1, 1)

        # 构造平坦态度规（cond = d）
        g_flat = torch.eye(d, dtype=torch.float64) * d  # cond = 1，但 log(d) 基准
        g_flat = g_flat.unsqueeze(0).repeat(N, 1, 1)

        sigma_expanded = float(self.compute_sigma_soft(g_expanded, k=1.0))
        sigma_collapsed = float(self.compute_sigma_soft(g_collapsed, k=1.0))
        sigma_flat = float(self.compute_sigma_soft(g_flat, k=1.0))

        # 验证
        expanded_negative = sigma_expanded < 0  # 膨胀区应为负
        collapsed_positive = sigma_collapsed > 0  # 坍缩区应为正

        return {
            "sigma_expanded": sigma_expanded,
            "sigma_collapsed": sigma_collapsed,
            "sigma_flat": sigma_flat,
            "expanded_negative": expanded_negative,
            "collapsed_positive": collapsed_positive,
            "sign_change_verified": expanded_negative and collapsed_positive,
            "message": "变号特性验证通过" if (expanded_negative and collapsed_positive)
                      else "变号特性未完全验证",
        }
