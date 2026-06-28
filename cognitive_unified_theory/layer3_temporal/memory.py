"""
全息记忆权重 W(t_past, t_now) —— 记忆是当前状态对历史的投影

核心定义：
    W(t_past, t_now) = <ψ(t_now) | P(t_now, t_past) | ψ(t_past)>

    其中：
        ψ(t)         时刻 t 的认知状态
        P(t_now, t_past) 平行移动算子（从过去到现在的度规变换）

物理意义：
    - 记忆不是"录像"，是"当前状态对历史的相干投影"
    - 投影权重 W 越大，该历史对当前影响越大
    - 释怀 = W → 0（历史不再影响现在），但历史数据不变
    - 这就是"数据不变，拓扑结构改变"的数学实现

工程铁律：
    - 严禁 history_tensor = new_value
    - 只修改 W（权重），不修改历史状态本身
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, stable_eigh


class HolographicMemory:
    """
    全息记忆：当前状态对历史的投影权重系统。

    属性：
        n_dims: 认知维度
        history: 历史状态（不可篡改）
        weights: 记忆权重 W(t_past, t_now)，可修改
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._history: list[Tensor] = []  # 不可篡改
        self._timestamps: list[float] = []
        self._weights: list[Tensor] = []  # 可修改的权重
        self._metric_history: list[Tensor] = []

    def record(self, state: Tensor, timestamp: float, metric: Tensor) -> None:
        """
        记录新状态（追加历史，不可篡改已有记录）。
        初始权重由投影计算得出。
        """
        S = state.to(torch.float64).clone().detach()
        self._history.append(S)
        self._timestamps.append(timestamp)
        self._metric_history.append(symmetric_part(metric.to(torch.float64)).clone().detach())

        # 计算初始权重：当前状态与所有历史的投影
        if len(self._history) > 1:
            W = self._compute_projection(S, timestamp, self._metric_history[-1])
        else:
            W = torch.tensor(1.0, dtype=torch.float64)
        self._weights.append(W)

    def _compute_projection(self, current_state: Tensor, current_time: float, current_metric: Tensor) -> Tensor:
        """
        计算当前状态对所有历史的投影权重。
        W = <S_now | P | S_past> / (|S_now| |S_past|)
        """
        S_now = current_state.to(torch.float64)
        g_now = symmetric_part(current_metric)
        weights = []
        for i, S_past in enumerate(self._history):
            g_past = self._metric_history[i]
            # 平行移动算子：将过去状态移动到现在
            S_past_transported = self._parallel_transport(S_past, g_past, g_now)
            # 投影权重 = 度规内积
            inner = S_now @ g_now @ S_past_transported
            norm_now = torch.sqrt(torch.clamp(S_now @ g_now @ S_now, min=1e-30))
            norm_past = torch.sqrt(torch.clamp(S_past_transported @ g_now @ S_past_transported, min=1e-30))
            w = inner / (norm_now * norm_past)
            weights.append(w)
        # 返回最近一次的权重（当前状态对自身的投影为1，对历史递减）
        return weights[-1] if weights else torch.tensor(1.0, dtype=torch.float64)

    def _parallel_transport(self, state: Tensor, metric_from: Tensor, metric_to: Tensor) -> Tensor:
        """平行移动算子（与纤维丛一致）。"""
        S = state.to(torch.float64)
        g_from = symmetric_part(metric_from)
        g_to = symmetric_part(metric_to)

        eigvals_f, eigvecs_f = stable_eigh(g_from)
        g_from_sqrt = eigvecs_f @ torch.diag(torch.sqrt(eigvals_f)) @ eigvecs_f.transpose(-1, -2)

        eigvals_t, eigvecs_t = stable_eigh(g_to)
        g_to_inv_sqrt = eigvecs_t @ torch.diag(1.0 / torch.sqrt(eigvals_t)) @ eigvecs_t.transpose(-1, -2)

        return g_to_inv_sqrt @ g_from_sqrt @ S

    def get_weights(self) -> Tensor:
        """获取所有记忆权重（可修改的副本）。"""
        return torch.stack(self._weights, dim=0) if self._weights else torch.tensor([], dtype=torch.float64)

    def get_history(self) -> Tensor:
        """获取历史状态（只读）。"""
        return torch.stack(self._history, dim=0) if self._history else torch.tensor([], dtype=torch.float64)

    def reweight(self, index: int, new_weight: Tensor) -> None:
        """
        修改记忆权重（释怀操作）。
        严禁修改历史状态本身，只修改权重。
        """
        if index < 0 or index >= len(self._weights):
            raise IndexError(f"权重索引越界: {index}")
        self._weights[index] = new_weight.to(torch.float64)

    def reweight_by_projection(self, high_dim_metric: Tensor, transparency: Tensor) -> Tensor:
        """
        高维重投影：用高维度规重新计算所有历史权重。

        这是"站在高处看清全貌"的数学实现：
            - 高维度规 high_dim_metric 来自系统升维（τ→1）
            - 透明度 transparency 控制重投影强度
            - 创伤在高维度规下从"奇点"变为"平凡点"

        物理意义：
            原谅 = 用高维视角重新计算历史权重，使创伤权重降低。
            历史数据不变，权重改变。

        参数：
            high_dim_metric: 高维度规张量（来自升维后的认知流形）
            transparency: 透明度 τ ∈ [0, 1]

        返回：
            新的权重序列
        """
        g_high = symmetric_part(high_dim_metric.to(torch.float64))
        tau = transparency.to(torch.float64)

        new_weights = []
        for i, S_past in enumerate(self._history):
            g_past = self._metric_history[i]
            # 平行移动到高维度规
            S_transported = self._parallel_transport(S_past, g_past, g_high)
            # 高维投影权重
            if self._history:
                S_now = self._history[-1]  # 当前状态
                S_now_transported = self._parallel_transport(S_now, self._metric_history[-1], g_high)
                inner = S_now_transported @ g_high @ S_transported
                norm_now = torch.sqrt(torch.clamp(S_now_transported @ g_high @ S_now_transported, min=1e-30))
                norm_past = torch.sqrt(torch.clamp(S_transported @ g_high @ S_transported, min=1e-30))
                w_high = inner / (norm_now * norm_past)
            else:
                w_high = torch.tensor(1.0, dtype=torch.float64)

            # 透明度控制重投影强度：τ=0 不重投影，τ=1 完全重投影
            w_old = self._weights[i]
            w_new = (1 - tau) * w_old + tau * w_high
            new_weights.append(w_new)
            self._weights[i] = w_new

        return torch.stack(new_weights, dim=0)

    def trauma_intensity(self) -> Tensor:
        """
        创伤强度：权重方差的高估。
        权重方差大 → 某些历史被过度放大（创伤固着）。
        权重均匀 → 历史均衡影响（释怀）。
        """
        if len(self._weights) < 2:
            return torch.tensor(0.0, dtype=torch.float64)
        W = torch.stack(self._weights, dim=0)
        return W.var()

    def release_degree(self) -> Tensor:
        """
        释怀度：1 - 创伤强度（归一化）。
        """
        trauma = self.trauma_intensity()
        return 1.0 / (1.0 + trauma)
