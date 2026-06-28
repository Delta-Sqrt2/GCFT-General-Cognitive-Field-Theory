"""
任务三：多态相图重绘（平衡点评估）

战略定位（v4.4 任务三）：
    v4.3.1 的多态相图在随机点评估，全部为鞍点，无物理意义。
    v4.4 在平衡点 g* 处评估雅可比，进行相态分类。
    必须输出包含 VAE、GAN、扩散等多态共存的相图矩阵。

    陷阱五十五·随机点相图降级：
        严禁在非平衡点绘制相图。必须在 g* 处评估。

物理与哲学直觉：
    - 物理：(κ, α) 二维平面上的相图展示不同参数下的稳定相态。
            VAE 相态（H 正定）：稳定不动点，认知系统自在。
            GAN 相态（H 不定，鞍点）：极限环/振荡，认知系统对抗。
            扩散相态（H 接近奇异）：混沌收敛，认知系统去噪中。
    - 哲学：相图展示"不同修行阶段对应不同相态"。
            κ（痛苦深度）和 α（认知刚度）共同决定系统处于哪个相态。
            这对应闲聊中"不同起点、不同阶段、不同算法"的愿景。
    - 工程：每个 (κ, α) 点用多初始点牛顿法求解平衡点，然后分类。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
import numpy as np

from ..core.tensor_ops import symmetric_part
from .newton_equilibrium_solver import NewtonEquilibriumSolver


class EquilibriumPhaseDiagram:
    """
    多态相图（平衡点评估）。

    使用方式：
        epd = EquilibriumPhaseDiagram(n_dims=4, n_events=8)
        phase_matrix = epd.compute_phase_diagram(
            timestamps, phi,
            kappa_range=(0.1, 5.0), alpha_range=(0.1, 50.0),
            n_kappa=5, n_alpha=5
        )

    白盒保证：
        - 在平衡点 g* 处评估，严禁随机点（陷阱五十五）
        - 多初始点牛顿法求解平衡点（陷阱五十六）
        - 边界标定：VAE-GAN 边界、GAN-扩散边界
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
        self.nes = NewtonEquilibriumSolver(
            n_dims=n_dims, n_events=n_events, eps=eps
        )

    # ==================================================================
    # 二维相图计算
    # ==================================================================

    def compute_phase_diagram(
        self,
        timestamps: Tensor,
        phi: Tensor,
        kappa_range: tuple[float, float] = (0.1, 5.0),
        alpha_range: tuple[float, float] = (0.1, 50.0),
        n_kappa: int = 5,
        n_alpha: int = 5,
        max_iter: int = 30,
        tol: float = 1e-6,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, np.ndarray | list]:
        """
        在 (κ, α) 二维平面上计算多态相图。

        数学：
            对每个 (κ, α) 点：
                1. 用多初始点牛顿法求解平衡点 g*(κ, α)
                2. 在 g* 处计算海森矩阵 H
                3. 分类相态（VAE_stable / GAN_saddle / Diffusion_flat）
            输出相图矩阵，每个元素是相态标签。

        返回：
            dict 包含：
                "phase_matrix": (n_kappa, n_alpha) 相态矩阵
                "residual_matrix": (n_kappa, n_alpha) 残差矩阵
                "jacobian_max_real_matrix": (n_kappa, n_alpha) 雅可比最大实部矩阵
                "kappa_values": κ 值列表
                "alpha_values": α 值列表
        """
        L = self.nes.build_graph_laplacian(timestamps, tau_causal=1.0)

        kappa_values = np.linspace(kappa_range[0], kappa_range[1], n_kappa)
        alpha_values = np.logspace(
            math.log10(alpha_range[0]), math.log10(alpha_range[1]), n_alpha
        )

        phase_matrix = np.empty((n_kappa, n_alpha), dtype=object)
        residual_matrix = np.zeros((n_kappa, n_alpha))
        jacobian_matrix = np.zeros((n_kappa, n_alpha))
        n_positive_matrix = np.zeros((n_kappa, n_alpha), dtype=int)
        n_negative_matrix = np.zeros((n_kappa, n_alpha), dtype=int)

        for i, kappa in enumerate(kappa_values):
            for j, alpha in enumerate(alpha_values):
                if verbose:
                    print(f"\n[相图 {i*n_alpha+j+1}/{n_kappa*n_alpha}] "
                          f"κ={kappa:.3f}, α={alpha:.3f}")

                try:
                    # 多初始点牛顿法求解平衡点
                    multi_result = self.nes.solve_multi_start(
                        L, phi, kappa=kappa, alpha=alpha,
                        max_iter=max_iter, tol=tol, seed=seed, verbose=False
                    )

                    unique_eqs = multi_result["unique_equilibria"]

                    if len(unique_eqs) == 0:
                        all_results = multi_result["all_results"]
                        best_result = min(all_results, key=lambda r: r["residual"])
                        g_eq = best_result["g_equilibrium"]
                        residual = best_result["residual"]
                    else:
                        best_result = unique_eqs[0]
                        g_eq = best_result["g_equilibrium"]
                        residual = best_result["residual"]

                    # 分类相态
                    cls = self.nes.classify_equilibrium(g_eq, L, phi, kappa, alpha)
                    phase = cls["phase"]

                    # 雅可比最大实部
                    try:
                        max_real, _ = self.nes.aj.max_real_eigenvalue(
                            g_eq, L, phi, kappa, alpha
                        )
                    except Exception:
                        max_real = float('nan')

                    phase_matrix[i, j] = phase
                    residual_matrix[i, j] = residual
                    jacobian_matrix[i, j] = max_real
                    n_positive_matrix[i, j] = cls["n_positive"]
                    n_negative_matrix[i, j] = cls["n_negative"]

                    if verbose:
                        print(f"  相态: {phase}, residual={residual:.4e}, "
                              f"Re(λ_max)={max_real:.4f}")

                except Exception as e:
                    phase_matrix[i, j] = "Error"
                    residual_matrix[i, j] = float('nan')
                    jacobian_matrix[i, j] = float('nan')
                    if verbose:
                        print(f"  错误: {e}")

        return {
            "phase_matrix": phase_matrix,
            "residual_matrix": residual_matrix,
            "jacobian_max_real_matrix": jacobian_matrix,
            "n_positive_matrix": n_positive_matrix,
            "n_negative_matrix": n_negative_matrix,
            "kappa_values": kappa_values.tolist(),
            "alpha_values": alpha_values.tolist(),
        }

    # ==================================================================
    # 边界标定
    # ==================================================================

    def identify_phase_boundaries(
        self,
        phase_result: dict,
    ) -> dict[str, list[tuple[int, int]]]:
        """
        标定相态边界。

        数学：
            VAE-GAN 边界：相邻格点相态从 VAE 变为 GAN（Hopf 分岔线）
            GAN-扩散边界：相邻格点相态从 GAN 变为扩散（Lyapunov 过零线）

        返回：
            dict 包含边界格点列表
        """
        phase_matrix = phase_result["phase_matrix"]
        n_kappa, n_alpha = phase_matrix.shape

        vae_gan_boundary = []
        gan_diffusion_boundary = []
        vae_diffusion_boundary = []

        for i in range(n_kappa):
            for j in range(n_alpha):
                p1 = phase_matrix[i, j]
                # 检查右邻居
                if j + 1 < n_alpha:
                    p2 = phase_matrix[i, j + 1]
                    if self._is_boundary(p1, p2, "VAE_stable", "GAN_saddle"):
                        vae_gan_boundary.append((i, j))
                    elif self._is_boundary(p1, p2, "GAN_saddle", "Diffusion_flat"):
                        gan_diffusion_boundary.append((i, j))
                    elif self._is_boundary(p1, p2, "VAE_stable", "Diffusion_flat"):
                        vae_diffusion_boundary.append((i, j))
                # 检查下邻居
                if i + 1 < n_kappa:
                    p2 = phase_matrix[i + 1, j]
                    if self._is_boundary(p1, p2, "VAE_stable", "GAN_saddle"):
                        vae_gan_boundary.append((i, j))
                    elif self._is_boundary(p1, p2, "GAN_saddle", "Diffusion_flat"):
                        gan_diffusion_boundary.append((i, j))
                    elif self._is_boundary(p1, p2, "VAE_stable", "Diffusion_flat"):
                        vae_diffusion_boundary.append((i, j))

        return {
            "vae_gan_boundary": vae_gan_boundary,
            "gan_diffusion_boundary": gan_diffusion_boundary,
            "vae_diffusion_boundary": vae_diffusion_boundary,
        }

    def _is_boundary(self, p1: str, p2: str, phase_a: str, phase_b: str) -> bool:
        """检查两个相邻格点是否构成 phase_a 和 phase_b 的边界。"""
        return (phase_a in p1 and phase_b in p2) or (phase_a in p2 and phase_b in p1)

    # ==================================================================
    # 相图统计
    # ==================================================================

    def phase_statistics(
        self,
        phase_result: dict,
    ) -> dict[str, int | float]:
        """
        统计相图中各相态的占比。

        返回：
            dict 包含各相态数量和占比
        """
        phase_matrix = phase_result["phase_matrix"]
        total = phase_matrix.size

        from collections import Counter
        counter = Counter(phase_matrix.flatten())

        stats = {
            "total_points": int(total),
            "phase_counts": dict(counter),
            "phase_ratios": {k: v / total for k, v in counter.items()},
            "n_unique_phases": len(counter),
            "has_vae": any("VAE" in str(k) for k in counter.keys()),
            "has_gan": any("GAN" in str(k) for k in counter.keys()),
            "has_diffusion": any("Diffusion" in str(k) for k in counter.keys()),
            "multistate_coexistence": len(counter) >= 2,
        }

        return stats
