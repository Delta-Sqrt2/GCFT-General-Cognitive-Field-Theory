"""
实时战术地图 —— 心智的雷达界面

核心定义：
    将高维认知流形状态投影到 2D/3D 可视化空间。
    可视化必须对应真实的数学状态，拒绝装饰性图表。

数学形式：
    使用流形学习（MDS / PCA / t-SNE 的数学核心）降维。
    此处使用度规加权的 PCA（特征分解），保证投影对应真实几何。

投影维度：
    - 2D 投影：主平面（前两个主成分）
    - 3D 投影：主空间（前三个主成分）
    - 颜色映射：度规曲率（痛苦强度）
    - 大小映射：有效秩（认知丰富度）

工程铁律：
    - 可视化必须对应真实数学状态
    - 严禁装饰性图表（无意义的动画/特效）
    - 投影必须保持度规信息
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, effective_rank, stable_eigh


class TacticalMap:
    """
    实时战术地图：认知流形的可视化投影。

    属性：
        n_dims: 认知维度
        projection_matrix: 降维投影矩阵（度规加权 PCA）
    """

    def __init__(self, n_dims: int, n_components: int = 2):
        """
        参数：
            n_dims: 认知维度
            n_components: 投影维度（2 或 3）
        """
        self.n_dims = n_dims
        self.n_components = n_components
        self._projection_matrix: Tensor | None = None
        self._explained_variance: Tensor | None = None

    def compute_projection(self, state_history: Tensor, metric: Tensor) -> Tensor:
        """
        计算度规加权的 PCA 投影矩阵。

        数学：
            1. 度规加权中心化：S_c = S - mean(S)
            2. 度规加权协方差：C = g^{-1/2} S_c^T S_c g^{-1/2}
            3. 特征分解：C = V Λ V^T
            4. 投影矩阵：P = g^{-1/2} V[:, :k]

        物理意义：
            投影保持度规信息（弯曲空间中的主成分）
            前k个主成分对应认知流形的"主平面"
        """
        history = state_history.to(torch.float64)
        if history.dim() == 1:
            history = history.unsqueeze(0)
        g = symmetric_part(metric.to(torch.float64))

        # 中心化
        mean = history.mean(dim=0, keepdim=True)
        S_c = history - mean

        # 度规加权：g^{-1/2}
        eigvals_g, eigvecs_g = stable_eigh(g)
        g_inv_sqrt = eigvecs_g @ torch.diag(1.0 / torch.sqrt(eigvals_g)) @ eigvecs_g.transpose(-1, -2)

        # 度规加权协方差
        C = g_inv_sqrt @ S_c.transpose(-1, -2) @ S_c @ g_inv_sqrt / history.shape[0]
        C = symmetric_part(C)

        # 特征分解
        eigvals, eigvecs = stable_eigh(C)

        # 降序排列
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 投影矩阵：P = g^{-1/2} V[:, :k]
        k = min(self.n_components, self.n_dims)
        self._projection_matrix = g_inv_sqrt @ eigvecs[:, :k]
        self._explained_variance = eigvals[:k] / (eigvals.sum() + 1e-30)

        return self._projection_matrix

    def project(self, state: Tensor) -> Tensor:
        """
        将状态投影到低维战术地图。

        数学：
            projected = P^T · state
        """
        if self._projection_matrix is None:
            raise RuntimeError("未计算投影矩阵，请先调用 compute_projection")
        S = state.to(torch.float64)
        return self._projection_matrix.transpose(-1, -2) @ S

    def project_trajectory(self, trajectory: Tensor) -> Tensor:
        """投影整个轨迹。"""
        if self._projection_matrix is None:
            raise RuntimeError("未计算投影矩阵")
        traj = trajectory.to(torch.float64)
        return (self._projection_matrix.transpose(-1, -2) @ traj.transpose(-1, -2)).transpose(-1, -2)

    def render_map(self, state: Tensor, metric: Tensor, state_history: Tensor | None = None) -> dict[str, Tensor]:
        """
        渲染战术地图（返回数学状态，非像素图像）。

        返回：
            position: 2D/3D 位置（投影坐标）
            curvature_color: 曲率颜色（度规弯曲强度）
            rank_size: 有效秩大小（认知丰富度）
            explained_variance: 各主成分解释方差
            trajectory: 投影轨迹（若有历史）
        """
        if self._projection_matrix is None:
            if state_history is not None:
                self.compute_projection(state_history, metric)
            else:
                raise RuntimeError("未计算投影矩阵，且无历史数据")

        # 位置
        pos = self.project(state)

        # 曲率颜色 = 度规条件数（对数压缩）
        eigvals, _ = stable_eigh(metric)
        curvature_color = torch.log(eigvals.max() / eigvals.min())

        # 有效秩大小
        if state_history is not None and state_history.shape[0] >= 2:
            cov = state_history.transpose(-1, -2) @ state_history / state_history.shape[0]
            rank_size = effective_rank(cov)
        else:
            rank_size = torch.tensor(1.0, dtype=torch.float64)

        # 轨迹投影
        traj_proj = None
        if state_history is not None and state_history.shape[0] >= 2:
            traj_proj = self.project_trajectory(state_history)

        return {
            "position": pos,
            "curvature_color": curvature_color,
            "rank_size": rank_size,
            "explained_variance": self._explained_variance,
            "trajectory": traj_proj,
        }

    def summary_report(self, state: Tensor, metric: Tensor, tau: Tensor, state_history: Tensor | None = None) -> dict[str, Tensor]:
        """
        生成战术地图摘要报告。

        整合所有层的数学状态：
            - 位置（第五层投影）
            - 曲率（第一层度规）
            - 有效秩（第二层负熵）
            - 透明度（第四层 τ）
            - 解释方差（投影质量）
        """
        render = self.render_map(state, metric, state_history)

        return {
            "position_2d": render["position"],
            "curvature": render["curvature_color"],
            "effective_rank": render["rank_size"],
            "transparency": tau.to(torch.float64),
            "explained_variance": render["explained_variance"],
            "trajectory": render["trajectory"],
        }

    @property
    def projection_matrix(self) -> Tensor:
        if self._projection_matrix is None:
            raise RuntimeError("未计算投影矩阵")
        return self._projection_matrix

    @property
    def explained_variance(self) -> Tensor:
        if self._explained_variance is None:
            raise RuntimeError("未计算解释方差")
        return self._explained_variance
