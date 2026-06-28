"""
任务二：非重特征值扰动（v4.5 核心）—— 打破 eigvalsh 重特征值简并

战略定位（v4.5 任务二）：
    v4.4 发现 eigvalsh 在重特征值处梯度不稳定：
    g=c·I 时 S 与 c 无关，但 ||∇S||=1.76·c 与 c 成正比，违反链式法则。
    这导致 ||∇S|| 存在 ~4 的数值地板，无法达到 1e-6。
    v4.5 通过给 g 加微小对称扰动打破特征值简并，使梯度唯一且稳定。

    陷阱五十八·重特征值掩盖：
        严禁在重特征值状态下计算梯度。必须用扰动打破简并。

物理与哲学直觉：
    - 物理：重特征值对应"高度对称态"（如 g=c·I 的各向同性态）。
            在量子力学中，简并态的微扰理论要求先打破简并，
            再计算物理量。v4.5 采用相同策略。
    - 哲学：这是"正则化"而非"掩盖"。
            简并是零测集（勒贝格测度为零），正则化后的梯度在物理上良定义。
            类似量子场论的维度正则化——不是修改理论，而是让理论可计算。
    - 工程：扰动 ε·对称随机矩阵，ε ∈ [1e-6, 1e-4]。
            保证 S 变化 < 1e-6（物理不变），但 ||∇S|| 可突破数值地板。

数学定义：
    扰动后的度规：
        g_perturbed = g + ε · R
        其中 R 是对称随机矩阵，ε 是扰动幅度。

    关键性质：
        1. 扰动打破特征值简并：g_perturbed 的特征值几乎必然非重
        2. 扰动保持对称性：R 对称，故 g_perturbed 对称
        3. 扰动微小：||g_perturbed - g|| = ε·||R|| ≈ ε·√d
        4. 作用量变化小：|S(g_perturbed) - S(g)| = O(ε)
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import symmetric_part


class PerturbedGradient:
    """
    非重特征值扰动器。

    使用方式：
        pg = PerturbedGradient(n_dims=4, n_events=8)
        g_perturbed = pg.perturb(g_batch, eps=1e-6)
        # 然后用 g_perturbed 计算作用量和梯度

    白盒保证：
        - 扰动是对称随机矩阵，打破特征值简并（陷阱五十八防范）
        - 扰动幅度可调，保证 S 变化 < 1e-6
        - 扰动是确定性的（给定 seed），可复现
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

    def perturb(
        self,
        g_batch: Tensor,
        eps: float = 1e-6,
        seed: int | None = None,
    ) -> Tensor:
        """
        给度规加微小对称扰动，打破特征值简并。

        数学：
            g_perturbed = g + ε · R
            其中 R 是对称随机矩阵，R = (Q + Q^T) / 2，Q ~ N(0, 1)。

        参数：
            g_batch: (N, d, d) 度规张量
            eps: 扰动幅度，默认 1e-6
            seed: 随机种子（可复现）

        返回：
            g_perturbed: (N, d, d) 扰动后的度规
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        if seed is not None:
            torch.manual_seed(seed)

        # 生成对称随机矩阵
        Q = torch.randn(N, d, d, dtype=torch.float64)
        R = symmetric_part(Q)  # R = (Q + Q^T) / 2

        # 扰动
        g_perturbed = g + eps * R

        # 保持正定性（如果原度规正定，扰动后仍正定，但确保数值稳定）
        g_perturbed = symmetric_part(g_perturbed)

        return g_perturbed

    def perturb_single(
        self,
        g: Tensor,
        eps: float = 1e-6,
        seed: int | None = None,
    ) -> Tensor:
        """
        给单个度规矩阵加微小对称扰动。

        参数：
            g: (d, d) 度规矩阵
            eps: 扰动幅度
            seed: 随机种子
        """
        if seed is not None:
            torch.manual_seed(seed)

        g = g.to(torch.float64)
        d = g.shape[0]

        Q = torch.randn(d, d, dtype=torch.float64)
        R = symmetric_part(Q)

        g_perturbed = g + eps * R
        g_perturbed = symmetric_part(g_perturbed)

        return g_perturbed

    def verify_perturbation(
        self,
        g_batch: Tensor,
        action_func,
        eps: float = 1e-6,
    ) -> dict[str, float | bool]:
        """
        验证扰动的影响：S 变化 < 1e-6。

        参数：
            g_batch: (N, d, d) 度规张量
            action_func: 计算作用量的函数，输入 g，返回 S
            eps: 扰动幅度

        返回：
            dict 包含扰动前后的 S 值和变化量
        """
        g = g_batch.to(torch.float64)

        # 扰动前
        S_before = float(action_func(g))

        # 扰动后
        g_perturbed = self.perturb(g, eps=eps, seed=42)
        S_after = float(action_func(g_perturbed))

        delta_S = abs(S_after - S_before)

        return {
            "S_before": S_before,
            "S_after": S_after,
            "delta_S": delta_S,
            "eps": eps,
            "is_physical_unchanged": delta_S < 1e-6,
            "description": f"扰动 ε={eps} 后 S 变化 {delta_S:.2e}",
        }

    def verify_gradient_breakthrough(
        self,
        g_batch: Tensor,
        gradient_func,
        eps_values: list[float] | None = None,
    ) -> dict[str, list | float]:
        """
        验证扰动后 ||∇S|| 可突破数值地板。

        参数：
            g_batch: (N, d, d) 度规张量（建议用重特征值度规 g=c·I）
            gradient_func: 计算梯度范数的函数，输入 g，返回 ||∇S||
            eps_values: 扰动幅度列表

        返回：
            dict 包含不同 ε 下的 ||∇S|| 值
        """
        if eps_values is None:
            eps_values = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

        g = g_batch.to(torch.float64)
        results = []

        for eps in eps_values:
            if eps == 0.0:
                g_test = g.clone()
            else:
                g_test = self.perturb(g, eps=eps, seed=42)

            grad_norm = float(gradient_func(g_test))
            results.append({
                "eps": eps,
                "grad_norm": grad_norm,
            })

        # 找到突破数值地板的最小 ε
        baseline = results[0]["grad_norm"]
        breakthrough_eps = None
        for r in results[1:]:
            if r["grad_norm"] < baseline * 0.5:
                breakthrough_eps = r["eps"]
                break

        return {
            "results": results,
            "baseline_grad_norm": baseline,
            "breakthrough_eps": breakthrough_eps,
            "description": "扰动后 ||∇S|| 突破数值地板",
        }

    def check_eigenvalue_degeneracy(
        self,
        g: Tensor,
        tol: float = 1e-8,
    ) -> dict[str, int | float | bool]:
        """
        检查度规矩阵的特征值简并情况。

        参数：
            g: (d, d) 或 (N, d, d) 度规矩阵
            tol: 简并判定容差

        返回：
            dict 包含特征值和简并信息
        """
        g = g.to(torch.float64)

        if g.dim() == 2:
            g = g.unsqueeze(0)

        N = g.shape[0]
        all_degeneracies = []
        max_degeneracy = 1

        for i in range(N):
            eigvals = torch.linalg.eigvalsh(g[i])
            # 检查相邻特征值的差异
            diffs = eigvals[1:] - eigvals[:-1]
            n_degenerate = int((diffs < tol).sum())
            all_degeneracies.append(n_degenerate)
            max_degeneracy = max(max_degeneracy, n_degenerate + 1)

        return {
            "n_events": N,
            "degeneracies": all_degeneracies,
            "max_degeneracy": max_degeneracy,
            "has_degeneracy": max_degeneracy > 1,
            "tol": tol,
        }
