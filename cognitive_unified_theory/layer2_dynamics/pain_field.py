"""
痛苦势能场 F_trauma —— 环境输入与内在模型的残差梯度

物理对应：
    在自由能原理（Friston）中，认知系统通过最小化"惊讶"（surprisal）演化。
    痛苦势能 F_trauma = -log p(sensory | model) 的梯度。
    梯度方向指向"最陡下降"，系统沿此方向调整以减少痛苦。

数学形式：
    F(S) = (1/2) ||event - S||²_g  （度规加权的预测误差）
    ∇F = -g^{-1} (event - S)       （度规逆加权的残差）

    度规 g 由第一层给出：痛苦使度规弯曲，弯曲的度规又改变梯度的方向与大小。
    这是"痛苦让空间弯曲，弯曲的空间改变成长路径"的数学闭环。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part


class PainPotentialField:
    """
    痛苦势能场 F_trauma(S)。

    属性：
        n_dims: 认知维度
        event: 当前事件张量（环境输入）
        metric: 度规张量 g_μν（来自第一层）
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._event: Tensor | None = None
        self._metric: Tensor | None = None
        self._cached_potential: Tensor | None = None

    def set_event(self, event: Tensor, metric: Tensor) -> None:
        """
        设置当前事件与度规。
        事件是形式化张量，非自然语言。
        """
        self._event = event.to(torch.float64)
        self._metric = symmetric_part(metric.to(torch.float64))
        self._cached_potential = None

    def potential(self, state: Tensor) -> Tensor:
        """
        痛苦势能 F(S) = (1/2) (event - S)^T g (event - S)

        物理意义：
            - F 越大，状态与事件（环境要求）的失配越严重
            - 度规 g 加权：在痛苦弯曲的区域，同样的失配产生更大的势能
            - 这就是"创伤区域跨越更难"的数学表达
        """
        if self._event is None or self._metric is None:
            raise RuntimeError("未设置事件与度规")
        S = state.to(torch.float64)
        residual = self._event - S
        F = 0.5 * residual @ self._metric @ residual
        return F

    def gradient(self, state: Tensor) -> Tensor:
        """
        痛苦势能梯度 ∇F = -g (event - S)

        物理意义：
            - 梯度指向痛苦增加最快的方向
            - 系统沿 -∇F 演化（减少痛苦）
            - 度规加权：弯曲区域的梯度方向被"折射"
        """
        if self._event is None or self._metric is None:
            raise RuntimeError("未设置事件与度规")
        S = state.to(torch.float64)
        residual = self._event - S
        # ∇F = -g (event - S) = g (S - event)
        grad = self._metric @ residual
        # 注意：势能 F = 0.5 r^T g r，∂F/∂S = -g r，所以梯度 = -g r
        # 但演化方程用 -∇F = g r（下降方向）
        return -grad  # 返回 ∇F（上升方向），演化层用 -∇F

    def hessian(self, state: Tensor) -> Tensor:
        """
        痛苦势能 Hessian = g_μν
        度规本身就是势能的二阶导数。这是场与几何耦合的体现。
        """
        if self._metric is None:
            raise RuntimeError("未设置度规")
        return self._metric.clone()

    def gradient_magnitude(self, state: Tensor) -> Tensor:
        """
        梯度大小 |∇F| = sqrt(∇F^T g^{-1} ∇F)
        度规逆加权的范数。用于负熵转化判据 ∫|∇F|。
        """
        grad = self.gradient(state)
        g_inv = safe_inverse(self._metric)
        return torch.sqrt(torch.clamp(grad @ g_inv @ grad, min=1e-30))

    def accumulate_gradient_integral(self, trajectory: Tensor, metric_trajectory: list[Tensor] | None = None) -> Tensor:
        """
        计算梯度积分 ∫|∇F| dt 沿轨迹。
        用于负熵转化判据 ΔR ≥ ∫|∇F|。

        参数：
            trajectory: 状态轨迹 (T, n)
            metric_trajectory: 各时刻度规（若 None，用当前度规）
        """
        T_len = trajectory.shape[0]
        integral = torch.tensor(0.0, dtype=torch.float64)
        for t in range(T_len - 1):
            S = trajectory[t]
            if metric_trajectory is not None:
                self._metric = symmetric_part(metric_trajectory[t])
            mag = self.gradient_magnitude(S)
            dt = 1.0  # 离散时间步（由演化层决定实际 dt）
            integral = integral + mag * dt
        return integral
