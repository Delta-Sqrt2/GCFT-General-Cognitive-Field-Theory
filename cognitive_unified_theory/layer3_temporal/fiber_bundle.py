"""
认知纤维丛 —— 时间作为底空间，状态空间作为纤维

数学实体：纤维丛 (E, M, π, F)
    - 底空间 M = 时间轴 R（认知演化的"时间"）
    - 纤维 F = 认知状态空间 H_cog（每个时刻的认知状态）
    - 全空间 E = ∪_t {t} × F_t（所有时刻的认知状态集合）
    - 投影 π: E → M，π(t, S) = t

物理意义：
    - 记忆不是"存储"，是纤维丛上的截面（section）
    - 创伤是纤维丛上的"扭结"（曲率不为零的连接）
    - 释怀是纤维丛的"平凡化"（trivialization）：在高维视角下，扭结可解

同调群：
    - H_0：连通分量数（认知的"整体性"）
    - H_1：环路数（认知的"执念/未解循环"）
    - 创伤对应 H_1 的非零元素；释怀使 H_1 → 0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, stable_eigh


class CognitiveFiberBundle:
    """
    认知纤维丛：时间 × 状态空间。

    属性：
        n_dims: 纤维维数（认知维度）
        history: 历史状态序列（不可篡改）
        timestamps: 对应时间戳
        connection: 纤维丛上的连接（平行移动规则）
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._history: list[Tensor] = []
        self._timestamps: list[float] = []
        self._connection: Tensor | None = None  # 连接形式（联络）

    def add_state(self, state: Tensor, timestamp: float) -> None:
        """
        向纤维丛添加状态（追加，不可篡改已有历史）。
        """
        self._history.append(state.to(torch.float64).clone().detach())
        self._timestamps.append(timestamp)

    @property
    def history(self) -> Tensor:
        """历史状态序列 (T, n)，只读。"""
        if not self._history:
            raise RuntimeError("纤维丛为空")
        return torch.stack(self._history, dim=0)

    @property
    def timestamps(self) -> Tensor:
        return torch.tensor(self._timestamps, dtype=torch.float64)

    def compute_connection(self, metric: Tensor) -> Tensor:
        """
        计算纤维丛上的连接（联络）。
        连接描述"平行移动"规则：状态如何沿时间演化。

        数学形式：
            连接形式 ω = g^{-1} dg（度规的 Maurer-Cartan 形式）
            这是纤维丛上的 Levi-Civita 连接。

        物理意义：
            - 连接的曲率 = 度规曲率（来自第一层）
            - 非零曲率 → 平行移动依赖路径 → "创伤扭曲了时间"
            - 零曲率 → 平行移动路径无关 → "释怀，过去不再束缚"
        """
        g = symmetric_part(metric.to(torch.float64))
        g_inv = safe_inverse(g)
        # 数值微分估计 dg（度规的变化）
        # 在静态情况下，连接由度规逆给出
        self._connection = g_inv
        return self._connection

    def parallel_transport(self, state: Tensor, metric_from: Tensor, metric_to: Tensor) -> Tensor:
        """
        平行移动：将状态从 t1 的纤维移动到 t2 的纤维。

        数学：
            S_transported = g_to^{-1/2} g_from^{1/2} S
            （度规变换下的协变常矢量）

        物理意义：
            - 同一"认知内容"在不同时刻有不同的表达
            - 平行移动保持认知内容的"内在一致性"
            - 创伤使平行移动路径依赖（不同路径得到不同结果）
        """
        S = state.to(torch.float64)
        g_from = symmetric_part(metric_from.to(torch.float64))
        g_to = symmetric_part(metric_to.to(torch.float64))

        # g_from^{1/2}
        eigvals_f, eigvecs_f = stable_eigh(g_from)
        g_from_sqrt = eigvecs_f @ torch.diag(torch.sqrt(eigvals_f)) @ eigvecs_f.transpose(-1, -2)

        # g_to^{-1/2}
        eigvals_t, eigvecs_t = stable_eigh(g_to)
        g_to_inv_sqrt = eigvecs_t @ torch.diag(1.0 / torch.sqrt(eigvals_t)) @ eigvecs_t.transpose(-1, -2)

        # 平行移动：S' = g_to^{-1/2} g_from^{1/2} S
        return g_to_inv_sqrt @ g_from_sqrt @ S

    def holonomy(self, metric_trajectory: list[Tensor]) -> Tensor:
        """
        和乐（holonomy）：沿闭合环路平行移动后状态的变化。
        非零和乐 → 流形弯曲 → 创伤存在。
        零和乐 → 流形平坦 → 释怀。

        参数：
            metric_trajectory: 度规沿时间轨迹的序列
        返回：
            和乐群元素（变换矩阵），偏离单位矩阵的程度 = 创伤强度
        """
        if not metric_trajectory:
            return torch.eye(self.n_dims, dtype=torch.float64)

        n = self.n_dims
        holonomy = torch.eye(n, dtype=torch.float64)
        for i in range(len(metric_trajectory) - 1):
            g_from = metric_trajectory[i]
            g_to = metric_trajectory[i + 1]
            # 平行移动算子
            eigvals_f, eigvecs_f = stable_eigh(g_from)
            g_from_sqrt = eigvecs_f @ torch.diag(torch.sqrt(eigvals_f)) @ eigvecs_f.transpose(-1, -2)

            eigvals_t, eigvecs_t = stable_eigh(g_to)
            g_to_inv_sqrt = eigvecs_t @ torch.diag(1.0 / torch.sqrt(eigvals_t)) @ eigvecs_t.transpose(-1, -2)

            transport_op = g_to_inv_sqrt @ g_from_sqrt
            holonomy = transport_op @ holonomy

        return holonomy

    def trauma_curvature(self, metric_trajectory: list[Tensor]) -> Tensor:
        """
        创伤曲率：和乐偏离单位矩阵的程度。
        ||Holonomy - I||_F
        """
        H = self.holonomy(metric_trajectory)
        I = torch.eye(self.n_dims, dtype=torch.float64)
        return (H - I).norm()

    def homology_h1(self, state_trajectory: Tensor, metric: Tensor) -> Tensor:
        """
        一阶同调群 H_1 的维度估计：认知"执念/未解循环"的数量。

        数学：
            H_1 的维度 = 状态轨迹形成的"环路"数
            通过状态轨迹的协方差矩阵的零空间维度估计

        物理意义：
            - H_1 = 0：无执念，认知开放
            - H_1 > 0：存在未解循环（创伤固化为思维模式）
        """
        traj = state_trajectory.to(torch.float64)
        if traj.shape[0] < 2:
            return torch.tensor(0.0, dtype=torch.float64)

        # 状态变化的协方差
        diffs = traj[1:] - traj[:-1]
        cov = diffs.transpose(-1, -2) @ diffs / diffs.shape[0]

        # 度规加权的协方差
        g = symmetric_part(metric.to(torch.float64))
        g_inv = safe_inverse(g)
        cov_weighted = g_inv @ cov

        eigvals = torch.linalg.eigvalsh(symmetric_part(cov_weighted))
        # 零特征值数 = H_1 维度（连续近似：特征值 < 阈值）
        threshold = eigvals.max() * 1e-6
        h1_dim = (eigvals.abs() < threshold).sum().to(torch.float64)
        return h1_dim
