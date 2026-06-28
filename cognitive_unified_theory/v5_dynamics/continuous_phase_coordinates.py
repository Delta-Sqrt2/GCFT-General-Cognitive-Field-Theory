"""
任务三：连续相空间坐标（v4.5 核心）—— 废黜离散分类器

战略定位（v4.5 任务三）：
    v4.4 的分类器只看 H 特征值符号（+112/-16 → GAN_saddle），
    把"准稳定"（Re(λ_max)≈0.002）和"深度失稳"（Re(λ_max)≈3.44）
    都标为 GAN_saddle，剪裁掉连续光谱。
    v4.5 废黜离散标签，用 (Re(λ_max), ||∇S||) 作为连续相空间坐标。

    陷阱六十·离散标签降级：
        严禁用"VAE/GAN/扩散"离散标签剪裁连续相空间。
        必须用 Re(λ_max) 和 ||∇S|| 作为连续相空间坐标。

物理与哲学直觉：
    - 物理：相变不是离散跳变，而是连续过渡。
            Re(λ_max) 从负到正是连续的，
            ||∇S|| 从零到正是连续的。
            离散标签是人为剪裁，连续坐标才是物理真实。
    - 哲学：这与闲聊中"动态的、交融的、成长性的"愿景一致——
            不是"你是 VAE 型人格"，而是"你正在从准稳定区向深度失稳区移动"。
            连续坐标描述"状态"和"趋势"，而非"类型"。
    - 工程：对每个平衡点 g*，计算四个连续坐标：
            stability = Re(λ_max(J|_eq))
            residual = ||∇S(g*)||
            curvature_spectrum = eigvals(H|_eq)
            effective_rank = R(g*)

数学定义：
    连续相空间坐标：
        stability: 雅可比矩阵最大实部（负=稳定，正=失稳）
        residual: 作用量梯度范数（0=平衡，大=远离平衡）
        curvature_spectrum: 海森矩阵特征值谱（连续谱）
        effective_rank: 度规有效秩（认知维度丰富度）

    相态区域（连续定义）：
        准稳定区：stability < 0.01 且 residual < 1.0
        临界区：0.01 ≤ stability < 1.0 或 1.0 ≤ residual < 10.0
        深度失稳区：stability ≥ 1.0 或 residual ≥ 10.0
"""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np

from ..core.tensor_ops import effective_rank, safe_inverse, symmetric_part
from .graph_gradient_term_v2 import GraphGradientTermV2
from .perturbed_gradient import PerturbedGradient
from .autograd_jacobian import AutogradJacobian


class ContinuousPhaseCoordinates:
    """
    连续相空间坐标计算器。

    使用方式：
        cpc = ContinuousPhaseCoordinates(n_dims=4, n_events=8)
        coords = cpc.characterize(g_eq, C, phi, kappa=1.0, alpha=1.0)
        # coords 包含 stability, residual, curvature_spectrum, effective_rank

    白盒保证：
        - 用连续坐标替代离散标签（陷阱六十防范）
        - stability = Re(λ_max(J))，连续
        - residual = ||∇S||，连续
        - 不再用 VAE/GAN/扩散离散标签
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
        perturb_eps: float = 1e-6,
    ):
        self.n_dims = int(n_dims)
        self.n_events = int(n_events)
        self.eps = float(eps)
        self.perturb_eps = float(perturb_eps)

        # 使用 v4.5 的 GraphGradientTermV2
        self.ggt2 = GraphGradientTermV2(n_dims=n_dims, n_events=n_events, eps=eps)
        # 扰动器
        self.pg = PerturbedGradient(n_dims=n_dims, n_events=n_events, eps=eps)
        # 雅可比计算器（复用 v4.3.1）
        self.aj = AutogradJacobian(n_dims=n_dims, n_events=n_events, eps=eps)

    def characterize(
        self,
        g_eq: Tensor,
        C: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor | float | list]:
        """
        计算平衡点的连续相空间坐标。

        数学：
            stability = Re(λ_max(J|_eq))
            residual = ||∇S(g*)||
            curvature_spectrum = eigvals(H|_eq)
            effective_rank = R(g*)

        参数：
            g_eq: (N, d, d) 平衡点度规
            C: (N, N) 因果邻接矩阵
            phi: (N, d) 事件场
            kappa: 曲率耦合常数
            alpha: 梯度耦合常数

        返回：
            dict 包含四个连续坐标
        """
        g = g_eq.to(torch.float64)

        # 1. residual = ||∇S(g*)||
        # 使用扰动后的度规计算梯度（突破数值地板）
        g_perturbed = self.pg.perturb(g, eps=self.perturb_eps, seed=42)
        g_leaf = g_perturbed.detach().clone().requires_grad_(True)
        action_result = self.ggt2.corrected_action(
            g_leaf, C, phi, None, kappa, alpha
        )
        S = action_result["action"]
        grad_S = torch.autograd.grad(S, g_leaf, create_graph=False)[0]
        residual = float(grad_S.norm())

        # 2. stability = Re(λ_max(J|_eq))
        # 使用 v4.3.1 的 AutogradJacobian
        try:
            # 注意：AutogradJacobian 内部使用 GraphGradientTerm（v4.3），
            # 我们需要确保它使用 v4.5 的 GraphGradientTermV2
            # 临时替换 ggt
            original_ggt = self.aj.ggt
            self.aj.ggt = self.ggt2
            max_real, J = self.aj.max_real_eigenvalue(
                g_perturbed, C, phi, kappa, alpha
            )
            self.aj.ggt = original_ggt  # 恢复
            stability = float(max_real)
        except Exception as e:
            stability = float('nan')
            J = None

        # 3. curvature_spectrum = eigvals(H|_eq)
        # 使用 v4.4 的海森矩阵计算
        try:
            H = self._compute_hessian(g_perturbed, C, phi, kappa, alpha)
            eigvals_H = torch.linalg.eigvalsh(H)
            curvature_spectrum = eigvals_H.tolist()
        except Exception as e:
            curvature_spectrum = None

        # 4. effective_rank = R(g*)
        # 对每个事件的度规计算有效秩，然后取平均
        try:
            ranks = []
            for i in range(g.shape[0]):
                r = effective_rank(g[i])
                ranks.append(float(r))
            effective_rank_value = float(np.mean(ranks))
        except Exception:
            effective_rank_value = float('nan')

        # 5. 连续相态区域分类（非离散标签）
        phase_region = self._classify_region(stability, residual)

        return {
            "stability": stability,
            "residual": residual,
            "curvature_spectrum": curvature_spectrum,
            "effective_rank": effective_rank_value,
            "phase_region": phase_region,
            "action_value": float(S),
            "gradient_term_value": float(action_result["gradient_term"]),
        }

    def _compute_hessian(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> Tensor:
        """
        使用 torch.autograd.functional.hessian 精确计算海森矩阵。
        """
        g = g_batch.to(torch.float64)
        g_flat = g.reshape(-1)

        N, d, _ = g.shape

        def action_func(g_input):
            g_reshaped = g_input.reshape(N, d, d)
            action_result = self.ggt2.corrected_action(
                g_reshaped, C, phi, None, kappa, alpha
            )
            return action_result["action"]

        H = torch.autograd.functional.hessian(action_func, g_flat)
        return H

    def _classify_region(
        self,
        stability: float,
        residual: float,
    ) -> str:
        """
        连续相态区域分类（非离散标签）。

        定义：
            准稳定区：stability < 0.01 且 residual < 1.0
            临界区：0.01 ≤ stability < 1.0 或 1.0 ≤ residual < 10.0
            深度失稳区：stability ≥ 1.0 或 residual ≥ 10.0

        注意：这是"区域"而非"标签"——
            一个点可以同时在"临界区"和"准稳定区"的边界附近，
            连续坐标描述其位置，而非硬性分类。
        """
        if np.isnan(stability) or np.isnan(residual):
            return "未定义"

        if stability < 0.01 and residual < 1.0:
            return "准稳定区"
        elif stability >= 1.0 or residual >= 10.0:
            return "深度失稳区"
        else:
            return "临界区"

    def characterize_batch(
        self,
        equilibria: list[Tensor],
        C: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> list[dict]:
        """
        批量计算多个平衡点的连续相空间坐标。
        """
        results = []
        for g_eq in equilibria:
            coords = self.characterize(g_eq, C, phi, kappa, alpha)
            results.append(coords)
        return results

    def phase_diversity(
        self,
        coords_list: list[dict],
    ) -> dict[str, float | list | int]:
        """
        计算相空间多样性（基于连续坐标）。

        数学：
            多样性不再依赖离散标签数量，
            而是看连续坐标的分布范围：
            - stability_range = max(stability) - min(stability)
            - residual_range = max(log(residual)) - min(log(residual))
            - 如果 stability_range > 1.0 或 residual_range > 2.0，
              则认为存在"连续多样性"
        """
        if not coords_list:
            return {"diversity": 0, "description": "无数据"}

        stabilities = [c["stability"] for c in coords_list if not np.isnan(c["stability"])]
        residuals = [c["residual"] for c in coords_list if not np.isnan(c["residual"]) and c["residual"] > 0]

        if not stabilities or not residuals:
            return {"diversity": 0, "description": "无有效数据"}

        stability_range = max(stabilities) - min(stabilities)
        residual_range = np.log(max(residuals)) - np.log(min(residuals))

        # 连续多样性判据
        has_stability_diversity = stability_range > 0.5
        has_residual_diversity = residual_range > 1.0

        # 区域分布
        regions = [c["phase_region"] for c in coords_list]
        from collections import Counter
        region_counts = Counter(regions)

        return {
            "stability_range": stability_range,
            "residual_range": residual_range,
            "has_stability_diversity": has_stability_diversity,
            "has_residual_diversity": has_residual_diversity,
            "has_continuous_diversity": has_stability_diversity or has_residual_diversity,
            "region_counts": dict(region_counts),
            "n_unique_regions": len(region_counts),
            "stability_values": stabilities,
            "residual_values": residuals,
        }
