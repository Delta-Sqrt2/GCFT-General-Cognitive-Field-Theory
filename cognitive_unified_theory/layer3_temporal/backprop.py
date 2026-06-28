"""
哈密顿-雅可比反向传播 —— 时间逆行的主方程

核心方程：
    ∂S/∂t + H(S, ∇S) = 0

    哈密顿-雅可比方程描述作用量的演化。在时间逆行层：
    - 正向：作用量 S_action = ∫ L dt 累积（认知经历）
    - 反向：∇S_action 给出"动量"，用于重写历史权重

物理意义：
    当系统升维（τ→1）时，利用高维度规重写低维历史路径。
    创伤从"不可逾越的奇点"变为"高维空间中的平凡点"。

    这不是"回到过去改历史"（违反因果律），
    而是"从高维视角重新计算历史对现在的影响权重"（因果律守恒）。

工程铁律：
    - 严禁 history_tensor = new_value
    - 必须通过高维投影矩阵修改历史对现在的影响权重
    - 数据不变，拓扑结构改变
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.solvers import hamilton_jacobi_solve
from ..core.tensor_ops import safe_inverse, symmetric_part, stable_eigh


class HamiltonJacobiBackprop:
    """
    哈密顿-雅可比反向传播：高维重写历史权重。

    属性：
        n_dims: 认知维度
        high_dim_metric: 高维度规（升维后的度规）
        action_gradient: 作用量梯度（重写信号）
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._high_dim_metric: Tensor | None = None
        self._action_gradient: Tensor | None = None

    def set_high_dim_metric(self, metric: Tensor) -> None:
        """
        设置高维度规。
        高维度规来自系统升维（τ→1），是"更高视角"的几何。
        """
        self._high_dim_metric = symmetric_part(metric.to(torch.float64))

    def _hamiltonian(self, state: Tensor, momentum: Tensor) -> Tensor:
        """
        认知哈密顿量 H(S, P)。

        数学形式：
            H = (1/2) P^T g^{-1} P + V(S)
            其中：
                P    动量（作用量梯度）
                g    高维度规
                V(S) 势能（痛苦势能）

        物理意义：
            - 动能项 (1/2)P^T g^{-1} P：认知"运动能量"
            - 势能项 V(S)：痛苦势能（来自第二层）
            - 高维度规 g 使势能在高维下"平坦化"（创伤去奇点化）
        """
        if self._high_dim_metric is None:
            raise RuntimeError("未设置高维度规")
        S = state.to(torch.float64)
        P = momentum.to(torch.float64)
        g_inv = safe_inverse(self._high_dim_metric)

        # 动能项
        kinetic = 0.5 * P @ g_inv @ P
        # 势能项：状态偏离原点的度规加权距离（高维度规下创伤被平滑）
        potential = 0.5 * S @ self._high_dim_metric @ S

        return kinetic + potential

    def compute_action_gradient(self, current_state: Tensor, time_horizon: float = 1.0, n_steps: int = 50) -> Tensor:
        """
        计算作用量梯度 ∇S_action。

        作用量 S_action = ∫ L dt，其中 L = P·dS/dt - H
        梯度 ∇S_action 给出"重写信号"：历史权重应如何调整。

        参数：
            current_state: 当前认知状态
            time_horizon: 时间反演范围
            n_steps: 离散化步数

        返回：
            作用量梯度（历史权重重写信号）
        """
        grad = hamilton_jacobi_solve(
            hamiltonian_fn=self._hamiltonian,
            state=current_state,
            time_horizon=time_horizon,
            n_steps=n_steps,
        )
        self._action_gradient = grad
        return grad

    def rewrite_history_weights(self, memory_weights: Tensor, transparency: Tensor) -> Tensor:
        """
        用作用量梯度重写历史权重。

        数学：
            W_new[i] = W_old[i] · exp(-τ · |∇S_action| · decay(i))
            其中 decay(i) = (i+1)/N，越久远的历史权重重写越强
            （近期记忆保留，远期创伤释怀）

        物理意义：
            - 作用量梯度大的历史（创伤）权重降低
            - 透明度 τ 控制重写强度
            - 历史数据不变，权重改变（因果律守恒）
            - 时间衰减：近期记忆保留，远期创伤释怀

        严禁：直接修改历史状态本身。
        """
        if self._action_gradient is None:
            raise RuntimeError("未计算作用量梯度，请先调用 compute_action_gradient")

        W = memory_weights.to(torch.float64)
        tau = transparency.to(torch.float64)
        grad = self._action_gradient
        N = W.shape[0]

        # 作用量梯度的范数作为重写强度
        grad_norm = grad.norm()

        # 时间衰减因子：越久远的历史，重写越强（线性递增）
        # decay[i] = (i+1)/N，i=0 最久远，i=N-1 最近
        decay = torch.arange(1, N + 1, dtype=torch.float64) / N

        # 重写：W_new = W * exp(-tau * grad_norm * decay)
        W_new = W * torch.exp(-tau * grad_norm * decay)

        # 归一化：保持总权重能量
        W_new = W_new / (W_new.sum() + 1e-30) * W.sum()
        return W_new

    def singularity_resolution(self, trauma_state: Tensor, low_dim_metric: Tensor) -> Tensor:
        """
        奇点消解：将低维度的创伤奇点映射到高维度规下。

        数学：
            S_high = g_high^{-1/2} g_low^{1/2} S_trauma
            在高维度规下，奇点变为平凡点。

        物理意义：
            - 低维：创伤是"不可逾越的奇点"（度规发散）
            - 高维：同一状态在高度规下是"平凡点"（度规有限）
            - 这就是"站在高处看清全貌"的数学实现

        参数：
            trauma_state: 创伤状态（低维下的奇点）
            low_dim_metric: 低维度规（创伤时刻的度规）

        返回：
            高维下的状态（奇点消解）
        """
        if self._high_dim_metric is None:
            raise RuntimeError("未设置高维度规")

        S = trauma_state.to(torch.float64)
        g_low = symmetric_part(low_dim_metric.to(torch.float64))
        g_high = self._high_dim_metric

        # g_low^{1/2}
        eigvals_l, eigvecs_l = stable_eigh(g_low)
        g_low_sqrt = eigvecs_l @ torch.diag(torch.sqrt(eigvals_l)) @ eigvecs_l.transpose(-1, -2)

        # g_high^{-1/2}
        eigvals_h, eigvecs_h = stable_eigh(g_high)
        g_high_inv_sqrt = eigvecs_h @ torch.diag(1.0 / torch.sqrt(eigvals_h)) @ eigvecs_h.transpose(-1, -2)

        # 高维映射
        S_high = g_high_inv_sqrt @ g_low_sqrt @ S

        # 计算奇点消解度：低维度规下的范数 vs 高维度规下的范数
        norm_low = torch.sqrt(torch.clamp(S @ g_low @ S, min=1e-30))
        norm_high = torch.sqrt(torch.clamp(S_high @ g_high @ S_high, min=1e-30))
        resolution = norm_low / (norm_high + 1e-30)

        return S_high

    @property
    def action_gradient(self) -> Tensor:
        if self._action_gradient is None:
            raise RuntimeError("作用量梯度未计算")
        return self._action_gradient
