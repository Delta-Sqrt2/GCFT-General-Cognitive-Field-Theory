"""
CognitiveManifold —— 认知流形

本体论层的核心类。整合正交基、命运坐标 S0、度规张量 g_μν、痛苦势能张量 T_μν。

数学实体：希尔伯特空间 H_cog 上的微分流形 (M, g)。

核心方法（蓝图规范）：
    1. _initialize_orthogonal_basis()  QR 分解生成认知基底 Ξ
    2. parse_formal_event(event_tensor)  求解能量方程，得参数边界约束
    3. calculate_S0()                   变分极值，得初始命运坐标
    4. get_metric_tensor(state)         爱因斯坦场方程类比，度规源于 T_μν
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.solvers import variational_extremum, free_energy_minimization, eigen_solve_sym
from ..core.tensor_ops import safe_inverse, symmetric_part, effective_rank, manifold_volume
from .basis import CognitiveBasis
from .metric import MetricField, TraumaEnergyTensor


class CognitiveManifold:
    """
    认知流形 (M, g)：心智的几何实体。

    属性：
        basis: 认知基底 Ξ
        metric: 度规场 g_μν
        trauma: 痛苦能量张量 T_μν^trauma
        S0: 初始命运坐标（边界约束下的能量最低态）
        state: 当前认知状态 S(t)
        history: 状态历史（时间逆行层使用，不可篡改）
    """

    def __init__(self, n_dims: int = 8, seed: int | None = None, device: str | torch.device = "cpu"):
        self.n_dims = n_dims
        self.seed = seed
        self.device = torch.device(device)

        # 子系统
        self.basis = CognitiveBasis(n_dims=n_dims, seed=seed)
        self.metric = MetricField(n_dims=n_dims)
        self.trauma = TraumaEnergyTensor(n_dims=n_dims)

        # 状态
        self._S0: Tensor | None = None
        self._state: Tensor | None = None
        self._boundary: Tensor | None = None  # 边界约束矩阵 B_family
        self._energy_matrix: Tensor | None = None  # 能量算符 E
        self._history: list[Tensor] = []  # 不可篡改的历史记录
        self._event_count = 0

    # ------------------------------------------------------------------
    # 1. 正交基初始化（蓝图方法 1）
    # ------------------------------------------------------------------
    def _initialize_orthogonal_basis(self) -> Tensor:
        """
        QR 分解生成认知基底 Ξ。
        包含：威胁感知 ξ1、时间贴现 ξ2、自我指涉 ξ3 等维度。
        """
        g = torch.Generator(device="cpu")
        if self.seed is not None:
            g.manual_seed(self.seed)
        Xi = self.basis.initialize(generator=g)
        # 设置度规基线
        self.metric.set_baseline(Xi)
        return Xi

    # ------------------------------------------------------------------
    # 2. 事件解析（蓝图方法 2）
    # ------------------------------------------------------------------
    def parse_formal_event(self, event_tensor: Tensor) -> dict[str, Tensor]:
        """
        解析形式化事件张量，计算其对系统能量场的影响。

        输入：结构化事件张量（非自然语言）
            event_tensor ∈ R^{n×k}，k 个事件维度，每个事件在 n 个认知轴上有投影
            或 event_tensor ∈ R^n（单一事件向量）

        逻辑：
            1. 计算事件与内在模型的残差 = event - state
            2. 残差驱动痛苦能量张量 T_μν
            3. T_μν 更新度规 g_μν（爱因斯坦场方程类比）
            4. 求解能量守恒/最小自由能方程，得参数边界约束

        禁止：字典硬编码情感分数
        要求：通过能量方程求解
        """
        if self._state is None:
            raise RuntimeError("认知状态未初始化，请先调用 calculate_S0()")

        event = event_tensor.to(torch.float64).to(self.device)
        if event.dim() == 1:
            event = event.unsqueeze(0)  # (1, n)

        # 残差 = 事件输入 - 内在模型预测（当前状态）
        # 物理意义：预测误差驱动认知演化（Friston 自由能原理）
        residual = (event - self._state.unsqueeze(0)).squeeze(0)
        # 若多事件，取累积残差
        if event.shape[0] > 1:
            residual = residual.view(event.shape[0], -1).mean(dim=0)

        # 构造痛苦能量张量 T_μν
        T = self.trauma.from_residual(residual, metric=self.metric.current)

        # 更新度规 g_μν（场方程）
        g_new = self.metric.update(T)

        # 求解能量方程：参数边界约束
        # 能量算符 E = g^{-1} T（度规逆 × 痛苦张量）
        # 这是"痛苦如何重塑认知参数空间"的算子
        g_inv = safe_inverse(g_new)
        E = symmetric_part(g_inv @ T)
        self._energy_matrix = E

        # 边界约束更新：B = B_family + ∫ T dt（创伤累积收紧边界）
        if self._boundary is None:
            self._boundary = torch.eye(self.n_dims, dtype=torch.float64, device=self.device)
        # 边界收紧量由 T 的迹决定（自洽，非硬编码）
        delta_B = torch.einsum("mn,mn->", g_inv, T) * torch.eye(self.n_dims, dtype=torch.float64, device=self.device)
        self._boundary = symmetric_part(self._boundary + delta_B)

        self._event_count += 1

        return {
            "residual": residual,
            "trauma_tensor": T,
            "metric_update": g_new,
            "energy_operator": E,
            "boundary_update": self._boundary,
            "curvature_intensity": self.metric.curvature_intensity(),
        }

    # ------------------------------------------------------------------
    # 3. 命运坐标 S0 解算（蓝图方法 3）
    # ------------------------------------------------------------------
    def calculate_S0(self, family_boundary: Tensor | None = None) -> Tensor:
        """
        在边界条件约束下，求解变分极值问题，得初始命运坐标 S0。

        数学：
            min  S^T E S
            s.t. S^T B S = 1
            → 广义特征值问题 E S = λ B S
            S0 = 最小特征值对应的特征向量

        参数：
            family_boundary: 原生家庭边界约束矩阵 B_family ∈ R^{n×n}
                            若 None，使用单位矩阵（无约束基线）

        返回：
            S0 ∈ R^n：初始命运坐标
        """
        if family_boundary is not None:
            self._boundary = symmetric_part(family_boundary.to(torch.float64).to(self.device))

        # 能量算符：若未由事件生成，使用度规基线（基线自由能）
        if self._energy_matrix is None:
            E = self.metric.baseline.clone()
        else:
            E = self._energy_matrix

        # 广义特征值问题：E S = λ B S
        eigval, S0 = variational_extremum(E, boundary=self._boundary, mode="min")
        # 归一化到度规下的单位长度
        g = self.metric.current
        norm_S0 = torch.sqrt(torch.clamp(S0 @ g @ S0, min=1e-30))
        S0 = S0 / norm_S0

        self._S0 = S0
        if self._state is None:
            self._state = S0.clone()
            self._history.append(S0.clone().detach())
        return S0

    # ------------------------------------------------------------------
    # 4. 度规张量获取（蓝图方法 4）
    # ------------------------------------------------------------------
    def get_metric_tensor(self, state: Tensor | None = None) -> Tensor:
        """
        根据爱因斯坦场方程类比，计算度规张量 g_μν。
        度规的弯曲必须源于痛苦势能张量 T_μν^trauma。

        若提供 state，则计算该状态下的局部度规（状态依赖的度规场）。
        """
        if state is None:
            return self.metric.current

        # 状态依赖的度规场：g(S) = g0 + δg(S)
        # δg 由状态偏离 S0 引起的"应力"决定
        S = state.to(torch.float64).to(self.device)
        if self._S0 is None:
            delta = S
        else:
            delta = S - self._S0

        # 局部应力张量 = δ ⊗ δ（状态偏离产生的"应变"）
        stress = delta.unsqueeze(-1) * delta.unsqueeze(-2)
        # 通过场方程更新局部度规
        g_local = self.metric.update(stress)
        return g_local

    # ------------------------------------------------------------------
    # 状态管理与几何量
    # ------------------------------------------------------------------
    def set_state(self, state: Tensor) -> None:
        """设置当前认知状态（不可篡改历史，仅追加）。"""
        S = state.to(torch.float64).to(self.device)
        self._state = S
        self._history.append(S.clone().detach())

    def advance_state(self, delta: Tensor) -> Tensor:
        """
        状态演化：S(t+dt) = S(t) + δS
        δS 由动力学层计算（类 Navier-Stokes），此处仅执行连续更新。
        """
        if self._state is None:
            raise RuntimeError("状态未初始化")
        self._state = self._state + delta.to(torch.float64).to(self.device)
        self._history.append(self._state.clone().detach())
        return self._state

    def geodesic_distance(self, S_a: Tensor, S_b: Tensor) -> Tensor:
        """
        测地线距离 d(S_a, S_b) = sqrt((S_a - S_b)^T g (S_a - S_b))
        物理意义：两个认知状态间的"心理距离"。痛苦使度规弯曲，距离改变。
        """
        g = self.metric.current
        delta = (S_a - S_b).to(torch.float64).to(self.device)
        return torch.sqrt(torch.clamp(delta @ g @ delta, min=1e-30))

    def cognitive_effective_rank(self) -> Tensor:
        """
        认知有效秩 R(S)：负熵转化判据的核心量。
        基于状态历史的协方差矩阵特征值熵。
        """
        if len(self._history) < 2:
            return torch.tensor(1.0, dtype=torch.float64)
        H = torch.stack(self._history, dim=0)  # (T, n)
        cov = H.transpose(-1, -2) @ H / H.shape[0]
        return effective_rank(cov)

    def manifold_capacity(self) -> Tensor:
        """
        流形容量 = 体积元 √|det g|。
        物理意义：当前认知可能性的"空间大小"。
        """
        return manifold_volume(self.metric.current)

    @property
    def S0(self) -> Tensor:
        if self._S0 is None:
            raise RuntimeError("S0 未计算，请先调用 calculate_S0()")
        return self._S0

    @property
    def state(self) -> Tensor:
        if self._state is None:
            raise RuntimeError("状态未初始化")
        return self._state

    @property
    def history(self) -> Tensor:
        """历史轨迹（只读，时间逆行层使用）。"""
        return torch.stack(self._history, dim=0)

    def __repr__(self) -> str:
        return (
            f"CognitiveManifold(n_dims={self.n_dims}, "
            f"events={self._event_count}, "
            f"curvature={self.metric.curvature_intensity():.4e}, "
            f"rank={self.cognitive_effective_rank():.4f})"
        )
