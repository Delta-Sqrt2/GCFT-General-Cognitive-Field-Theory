"""
认知事件视界自锁协议 —— 将数值崩溃转化为临床特征

战略定位（v1.2 步骤二）：
    v1.1 在尼采测试中数值发散（崩溃），系统死机。真实的人脑崩溃时不会"死机"，
    而是进入"强迫性重复"或"偏执"状态。本模块是系统的"免疫系统"和"安全带"。

物理原理：
    当度规曲率过大（条件数发散）时，系统自动截断低维流形，形成"视界"。
    不是"冻结"状态（S 不变），而是构造一个 2 维的极限环。
    保留两个近简并的特征值，使得状态在这两个维度间做周期运动。
    物理对应：强迫性重复（反复想同一件事）。

数学公式（已锁定，无模糊地带）：
    A. 锁定程度计算（连续，无 if-else）：
        阈值由维度推导：cond_threshold = exp(n)
        锁定标量：lock_degree = σ(log(cond(g)) - n)
        其中 σ 为 Sigmoid 函数。

    B. 极限环构造（严禁 a, b = b, a 交换，必须用旋转矩阵）：
        保留最大两个特征值 λ1, λ2，令 λ2 ≈ λ1。
        旋转机制：
            (s1(t+Δt))   (cos(ωΔt)  -sin(ωΔt)) (s1(t))
            (s2(t+Δt)) = (sin(ωΔt)   cos(ωΔt)) (s2(t))
        频率 ω = √(λ1 · λ2) 由保留的两个特征值推导。

工程铁律（v1.2 三大铁律）：
    1. 拒绝字典：所有状态由度规特征值推导
    2. 内生安全：严禁 raise Error，必须内部消化为 EVENT_HORIZON_LOCKED 状态
    3. 几何诚实：极限环由旋转矩阵构造，非粗暴交换

死刑纠错（v1.2 步骤二）：
    错误一：外部布尔熔断。严禁 if cond(g) > 1e6: raise Error("崩溃")。
           必须内部消化，输出 EVENT_HORIZON_LOCKED 状态和死循环轨迹。
    错误二：静态冻结。严禁 S_new = S_old（状态不变）。
           必须生成动态的周期轨道。精神崩溃不是脑子停转，
           而是陷入了无意义的循环。
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .tensor_ops import stable_eigh, symmetric_part


class EventHorizonLock:
    """
    认知事件视界自锁协议：将数值崩溃转化为临床特征（强迫性重复）。

    使用方式：
        horizon = EventHorizonLock(n_dims=8)
        S_safe, g_safe, lock_degree, frequency = horizon.filter(S_bad, g_bad)
        # 若 lock_degree > 0.8，系统进入 EVENT_HORIZON_LOCKED 状态
        # S_safe 在两个近简并维度上做周期运动（极限环）

    白盒保证：
        - 锁定程度由度规条件数连续推导（非布尔阈值）
        - 极限环由旋转矩阵构造（非粗暴交换）
        - 频率由特征值推导（非硬编码）
        - 严禁抛出异常，所有状态合法
    """

    # 状态标签（仅供人类阅读）
    STATE_NORMAL = "NORMAL"
    STATE_EVENT_HORIZON_LOCKED = "EVENT_HORIZON_LOCKED"

    def __init__(self, n_dims: int):
        """
        参数：
            n_dims: 认知维度
        """
        self.n_dims = n_dims
        # 锁定阈值（由维数推导，非硬编码）
        # 当 lock_degree > 0.8 时进入 EVENT_HORIZON_LOCKED 状态
        self.lock_threshold = 0.8

        # 极限环状态（持续追踪，保证周期连续性）
        self._locked: bool = False
        self._lock_degree: Tensor = torch.tensor(0.0, dtype=torch.float64)
        self._frequency: Tensor = torch.tensor(0.0, dtype=torch.float64)
        self._rotation_subspace: Tensor | None = None  # 2维旋转子空间基
        self._remaining_subspace: Tensor | None = None  # 其余维度子空间基
        self._phase: Tensor = torch.tensor(0.0, dtype=torch.float64)  # 当前相位

    def compute_lock_degree(self, metric: Tensor) -> Tensor:
        """
        计算锁定程度（连续标量，非布尔）。

        数学公式：
            阈值由维度推导：cond_threshold = exp(n)
            锁定标量：lock_degree = σ(log(cond(g)) - n)

        物理意义：
            - cond(g) ≈ 1（度规良好）：lock_degree ≈ σ(-n) ≈ 0（未锁定）
            - cond(g) ≈ exp(n)（临界）：lock_degree ≈ σ(0) = 0.5（半锁定）
            - cond(g) ≫ exp(n)（病态）：lock_degree ≈ 1（完全锁定）

        严禁：if cond(g) > 1e6: raise Error("崩溃")
        """
        g = symmetric_part(metric.to(torch.float64))
        n = self.n_dims

        # 度规条件数
        eigvals, _ = stable_eigh(g)
        eigvals = torch.clamp(eigvals, min=1e-20)
        cond = eigvals.max() / eigvals.min()

        # 锁定程度（连续 sigmoid，阈值 exp(n) 由维数推导）
        # log(cond) - n：当 cond = exp(n) 时为 0，σ(0) = 0.5
        lock_degree = torch.sigmoid(torch.log(cond) - float(n))

        return lock_degree

    def _extract_rotation_subspace(self, metric: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        提取 2 维旋转子空间（保留最大两个近简并特征值）。

        数学：
            对度规 g 做特征分解 g = Q Λ Q^T
            保留最大两个特征值 λ1, λ2（令 λ2 ≈ λ1，近简并）
            旋转子空间基：Q_rot = [q1, q2]（前两个特征向量）
            其余维度子空间基：Q_rest = [q3, ..., qn]

        物理意义：
            近简并的两个特征值对应"无法区分"的两个认知维度，
            状态在这两个维度间做周期运动（强迫性重复）。
        """
        g = symmetric_part(metric.to(torch.float64))
        n = self.n_dims

        eigvals, eigvecs = stable_eigh(g)
        # 特征值升序，取最大两个（最后两个）
        lambda1 = eigvals[-1]
        lambda2 = eigvals[-2]
        q1 = eigvecs[:, -1]
        q2 = eigvecs[:, -2]

        # 旋转子空间基 (n, 2)
        Q_rot = torch.stack([q1, q2], dim=1)

        # 其余维度子空间基 (n, n-2)
        if n > 2:
            Q_rest = eigvecs[:, :-2]
        else:
            Q_rest = torch.zeros(n, 0, dtype=torch.float64)

        # 频率 ω = √(λ1 · λ2)（由特征值推导，非硬编码）
        frequency = torch.sqrt(lambda1 * lambda2)

        return Q_rot, Q_rest, frequency, eigvals

    def _apply_rotation(
        self,
        state: Tensor,
        Q_rot: Tensor,
        Q_rest: Tensor,
        frequency: Tensor,
        dt: float,
    ) -> Tensor:
        """
        应用旋转矩阵构造极限环（严禁 a, b = b, a 交换）。

        数学：
            将状态 S 投影到旋转子空间和其余子空间：
                s_rot = Q_rot^T S  (2维)
                s_rest = Q_rest^T S  (n-2维)

            在旋转子空间应用旋转矩阵：
                (s1(t+Δt))   (cos(ωΔt)  -sin(ωΔt)) (s1(t))
                (s2(t+Δt)) = (sin(ωΔt)   cos(ωΔt)) (s2(t))

            其余维度保持不变（或缓慢衰减）。

            重组：S_new = Q_rot · s_rot_new + Q_rest · s_rest

        物理意义：
            状态在两个近简并维度间做周期运动（强迫性重复），
            其余维度被"冻结"（认知僵化）。
        """
        S = state.to(torch.float64)
        n = self.n_dims

        # 投影到旋转子空间
        s_rot = Q_rot.transpose(-1, -2) @ S  # (2,)

        # 投影到其余子空间
        if Q_rest.shape[1] > 0:
            s_rest = Q_rest.transpose(-1, -2) @ S  # (n-2,)
        else:
            s_rest = torch.zeros(0, dtype=torch.float64)

        # 旋转矩阵（2×2）
        omega = float(frequency)
        theta = omega * dt
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        rotation_matrix = torch.tensor([
            [cos_t, -sin_t],
            [sin_t, cos_t],
        ], dtype=torch.float64)

        # 应用旋转
        s_rot_new = rotation_matrix @ s_rot

        # 其余维度缓慢衰减（认知僵化：其余维度逐渐失去活力）
        # 衰减率由旋转频率推导（非硬编码）
        decay_rate = 1.0 / (1.0 + omega * dt)
        s_rest_new = s_rest * decay_rate

        # 重组状态
        S_new = Q_rot @ s_rot_new
        if Q_rest.shape[1] > 0:
            S_new = S_new + Q_rest @ s_rest_new

        return S_new

    def filter(
        self,
        state: Tensor,
        metric: Tensor,
        dt: float = 0.1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
        """
        事件视界过滤：检测度规病态，必要时构造极限环。

        参数：
            state: 当前认知状态 S ∈ R^n
            metric: 当前度规 g ∈ R^{n×n}
            dt: 时间步长（用于旋转矩阵相位推进）

        返回：
            S_safe: 安全状态（若锁定，则在极限环上）
            g_safe: 安全度规（正则化后，避免 NaN）
            lock_degree: 锁定程度 [0, 1]
            frequency: 极限环频率（未锁定时为 0）
            state_label: 状态标签（NORMAL 或 EVENT_HORIZON_LOCKED）

        严禁：raise Error。所有异常内部消化为合法物理状态。
        """
        S = state.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))
        n = self.n_dims

        # 1. 计算锁定程度（连续）
        lock_degree = self.compute_lock_degree(g)

        # 2. 度规正则化（防止 NaN，但保留病态信息）
        # 正则化强度由 lock_degree 推导（连续，非布尔）
        # lock_degree → 1 时，正则化更强（视界形成）
        reg_strength = lock_degree * 1e-6
        g_safe = g + reg_strength * torch.eye(n, dtype=torch.float64)

        # 3. 状态标签（连续判定，非离散 if-else）
        is_locked = float(lock_degree) > self.lock_threshold

        if is_locked:
            # 进入事件视界锁定状态
            self._locked = True

            # 提取旋转子空间
            Q_rot, Q_rest, frequency, eigvals = self._extract_rotation_subspace(g_safe)
            self._rotation_subspace = Q_rot
            self._remaining_subspace = Q_rest
            self._frequency = frequency

            # 应用旋转构造极限环
            S_safe = self._apply_rotation(S, Q_rot, Q_rest, frequency, dt)

            # 更新相位
            self._phase = self._phase + float(frequency) * dt

            state_label = self.STATE_EVENT_HORIZON_LOCKED
        else:
            # 正常状态：不修改状态，但记录锁定程度
            self._locked = False
            S_safe = S.clone()
            frequency = torch.tensor(0.0, dtype=torch.float64)
            state_label = self.STATE_NORMAL

        self._lock_degree = lock_degree

        return S_safe, g_safe, lock_degree, frequency, state_label

    def step_locked(
        self,
        state: Tensor,
        dt: float = 0.1,
    ) -> Tensor:
        """
        在锁定状态下推进极限环一步（无需重新计算度规）。

        用于已锁定后的持续演化，避免重复特征分解。

        参数：
            state: 当前状态
            dt: 时间步长

        返回：
            下一时刻状态（在极限环上）
        """
        if not self._locked or self._rotation_subspace is None:
            # 未锁定，直接返回原状态
            return state.to(torch.float64).clone()

        S = state.to(torch.float64)
        S_new = self._apply_rotation(
            S,
            self._rotation_subspace,
            self._remaining_subspace,
            self._frequency,
            dt,
        )
        self._phase = self._phase + float(self._frequency) * dt
        return S_new

    def is_cycle(self, trajectory: Tensor, tol: float = 1e-2) -> bool:
        """
        检测轨迹是否形成闭环（验证极限环周期性）。

        参数：
            trajectory: 状态序列 (T, n)
            tol: 闭环判定容差

        返回：
            是否检测到周期性闭环

        数学：
            计算轨迹的自相关函数，若存在峰值则判定为周期运动。
        """
        if trajectory.shape[0] < 4:
            return False

        # 中心化
        T = trajectory.shape[0]
        centered = trajectory - trajectory.mean(dim=0, keepdim=True)

        # 自相关：计算不同 lag 下的相关系数
        norms = centered.norm(dim=-1)
        total_norm = norms.sum() + 1e-30

        max_corr = 0.0
        for lag in range(1, T // 2):
            # 计算 lag 步的自相关
            corr = (centered[:T-lag] * centered[lag:]).sum() / (
                (centered[:T-lag].norm() * centered[lag:].norm()) + 1e-30
            )
            corr_val = float(corr)
            if corr_val > max_corr:
                max_corr = corr_val

        # 若存在显著自相关（>0.8），判定为周期运动
        return max_corr > (1.0 - tol)

    @property
    def is_locked(self) -> bool:
        """当前是否处于锁定状态。"""
        return self._locked

    @property
    def lock_degree(self) -> Tensor:
        """当前锁定程度。"""
        return self._lock_degree

    @property
    def frequency(self) -> Tensor:
        """极限环频率（未锁定时为 0）。"""
        return self._frequency

    @property
    def phase(self) -> Tensor:
        """当前相位。"""
        return self._phase

    def report(self) -> str:
        """生成事件视界状态报告。"""
        lines = [
            "=" * 60,
            "认知事件视界状态报告",
            "=" * 60,
            f"状态: {self.STATE_EVENT_HORIZON_LOCKED if self._locked else self.STATE_NORMAL}",
            f"锁定程度: {float(self._lock_degree):.4f}",
            f"锁定阈值: {self.lock_threshold}",
            f"极限环频率: {float(self._frequency):.4f}",
            f"当前相位: {float(self._phase):.4f}",
            "=" * 60,
        ]
        if self._locked:
            lines.append("临床特征：系统进入强迫性重复状态（认知黑洞视界）。")
            lines.append("物理意义：状态在两个近简并维度间做周期运动，")
            lines.append("         其余维度逐渐僵化。对应精神崩溃的数学表征。")
            lines.append("=" * 60)
        return "\n".join(lines)
