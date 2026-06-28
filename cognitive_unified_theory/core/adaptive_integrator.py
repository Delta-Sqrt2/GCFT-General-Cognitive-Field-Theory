"""
自适应步长积分器与奇点检测 —— 认知黑洞的数值捕获

物理对应：
    在广义相对论中，黑洞附近的度规发散，需要自适应步长避免跨过事件视界。
    在认知动力学中，创伤集中区的度规也可能发散（认知黑洞），
    个体陷入某种执念无法逃逸。

    固定步长 RK4 在奇点附近会：
    1. 步长过大 → 跨过视界 → 数值发散（NaN）
    2. 步长过小 → 计算量爆炸 → 无法完成长周期演化

    解决方案：RKF45（Runge-Kutta-Fehlberg）自适应步长
    - 用 4 阶和 5 阶 RK 的差作为误差估计
    - 误差大 → 缩小步长（捕获奇点附近细节）
    - 误差小 → 放大步长（加速平稳区演化）

认知黑洞物理量：
    事件视界半径 r_s = 2GM/c² → 认知类比 r_s = 2·κ·ρ(R)/n
    其中：
        κ   耦合常数 1/(n+2)
        ρ   痛苦能量密度（T_μν 的迹）
        R   标量曲率
        n   认知维度

    霍金辐射（信息流失率）：
        T_H = ℏc³/(8πGMk_B) → 认知类比 T_cog = 1/(8π·κ·ρ·R)
        信息流失率 ∝ T_cog × 面积

工程铁律：
    - 严禁为"让演化跑完"而人工调整参数
    - 必须如实记录数值发散，并解释其物理意义
    - 自适应步长是数值方法，不是物理参数
"""

from __future__ import annotations

import math
import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class SingularityAlert:
    """
    奇点预警：当系统接近认知黑洞时触发。

    属性：
        time: 预警时刻
        metric_condition: 度规条件数（越大越病态）
        curvature: 标量曲率（越大越弯曲）
        state_norm: 状态范数（发散指标）
        horizon_radius: 事件视界半径估计
        hawking_temperature: 霍金温度估计（信息流失率）
        severity: 严重程度 [0, 1]（0=安全, 1=黑洞）
    """
    time: float
    metric_condition: float
    curvature: float
    state_norm: float
    horizon_radius: float
    hawking_temperature: float
    severity: float


class AdaptiveIntegrator:
    """
    自适应步长积分器：RKF45（Runge-Kutta-Fehlberg）。

    用于长周期认知演化，自动在奇点附近缩小步长。

    物理意义：
        - 平稳区（低曲率）：大步长，快速演化
        - 创伤区（高曲率）：小步长，捕获细节
        - 奇点区（度规发散）：记录预警，不掩盖数值问题
    """

    # RKF45 Butcher 表
    # 6 级 5 阶方法，4 阶嵌入用于误差估计
    _A = [
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40],
    ]
    # 5 阶权重
    _C5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    # 4 阶权重（用于误差估计）
    _C4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        dt_min: float = 1e-8,
        dt_max: float = 1.0,
        safety: float = 0.9,
    ):
        """
        参数：
            rtol: 相对误差容限
            atol: 绝对误差容限
            dt_min: 最小步长（低于此值视为奇点）
            dt_max: 最大步长
            safety: 步长调整安全系数
        """
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety = safety
        self.alerts: list[SingularityAlert] = []

    def integrate(
        self,
        derivative_fn,
        initial_state: Tensor,
        t_span: tuple[float, float],
        metric_fn=None,
        dt_init: float = 0.1,
    ) -> tuple[Tensor, Tensor, list[SingularityAlert]]:
        """
        自适应积分：从 t0 到 t1。

        参数：
            derivative_fn: 导数函数 dS/dt = f(t, S)
            initial_state: 初始状态 S(t0)
            t_span: 时间区间 (t0, t1)
            metric_fn: 度规函数 g(t) → Tensor（用于奇点检测）
            dt_init: 初始步长

        返回：
            times: 时间点序列 (T,)
            states: 状态序列 (T, n)
            alerts: 奇点预警列表
        """
        t0, t1 = t_span
        t = t0
        S = initial_state.to(torch.float64).clone()
        dt = min(dt_init, self.dt_max)

        times = [t]
        states = [S.clone()]
        self.alerts = []

        max_steps = 100000  # 安全上限
        step_count = 0

        while t < t1 and step_count < max_steps:
            step_count += 1

            # 确保不超过终点
            if t + dt > t1:
                dt = t1 - t

            # RKF45 单步
            S_new, error, dt_next = self._rkf45_step(derivative_fn, t, S, dt)

            # 奇点检测
            if metric_fn is not None:
                alert = self._check_singularity(t + dt, S_new, metric_fn)
                if alert is not None:
                    self.alerts.append(alert)

            # 数值发散检测
            if torch.isnan(S_new).any() or torch.isinf(S_new).any():
                # 如实记录发散，不掩盖
                alert = SingularityAlert(
                    time=t + dt,
                    metric_condition=float('inf'),
                    curvature=float('inf'),
                    state_norm=float('inf'),
                    horizon_radius=0.0,
                    hawking_temperature=float('inf'),
                    severity=1.0,
                )
                self.alerts.append(alert)
                # 停止积分，记录发散点
                break

            # 误差控制
            if dt_next < self.dt_min:
                # 步长已到最小，视为奇点
                alert = SingularityAlert(
                    time=t + dt,
                    metric_condition=self._safe_metric_condition(S_new, metric_fn),
                    curvature=0.0,
                    state_norm=S_new.norm().item(),
                    horizon_radius=0.0,
                    hawking_temperature=0.0,
                    severity=1.0,
                )
                self.alerts.append(alert)
                # 接受这一步但记录预警
                t = t + dt
                S = S_new
                times.append(t)
                states.append(S.clone())
                break

            # 接受步长
            t = t + dt
            S = S_new
            times.append(t)
            states.append(S.clone())

            # 更新步长
            dt = min(dt_next, self.dt_max)

        return torch.tensor(times, dtype=torch.float64), torch.stack(states, dim=0), self.alerts

    def _rkf45_step(
        self,
        derivative_fn,
        t: float,
        S: Tensor,
        dt: float,
    ) -> tuple[Tensor, Tensor, float]:
        """
        RKF45 单步：计算 5 阶和 4 阶解，估计误差，建议下一步步长。
        """
        n = S.shape[0]
        k = [None] * 6

        # k1
        k[0] = derivative_fn(S)
        # k2
        S2 = S + dt * self._A[0][0] * k[0]
        k[1] = derivative_fn(S2)
        # k3
        S3 = S + dt * (self._A[1][0] * k[0] + self._A[1][1] * k[1])
        k[2] = derivative_fn(S3)
        # k4
        S4 = S + dt * (self._A[2][0] * k[0] + self._A[2][1] * k[1] + self._A[2][2] * k[2])
        k[3] = derivative_fn(S4)
        # k5
        S5 = S + dt * (self._A[3][0] * k[0] + self._A[3][1] * k[1] + self._A[3][2] * k[2] + self._A[3][3] * k[3])
        k[4] = derivative_fn(S5)
        # k6
        S6 = S + dt * (self._A[4][0] * k[0] + self._A[4][1] * k[1] + self._A[4][2] * k[2] + self._A[4][3] * k[3] + self._A[4][4] * k[4])
        k[5] = derivative_fn(S6)

        # 5 阶解
        S5_result = S + dt * sum(c * ki for c, ki in zip(self._C5, k))
        # 4 阶解
        S4_result = S + dt * sum(c * ki for c, ki in zip(self._C4, k))

        # 误差估计
        error = (S5_result - S4_result).abs()
        scale = self.atol + self.rtol * S5_result.abs()
        rel_error = (error / scale).max().item()

        # 步长控制
        if rel_error < 1e-30:
            dt_next = self.dt_max
        elif rel_error <= 1.0:
            # 误差在容限内，可以放大步长
            dt_next = self.safety * dt * (1.0 / max(rel_error, 1e-10)) ** 0.2
            dt_next = min(dt_next, self.dt_max)
        else:
            # 误差超限，缩小步长重试
            dt_next = self.safety * dt * (1.0 / rel_error) ** 0.25
            dt_next = max(dt_next, self.dt_min)
            # 拒绝这一步，用更小步长重试
            return self._rkf45_step(derivative_fn, t, S, dt_next)

        return S5_result, error, dt_next

    def _check_singularity(self, t: float, S: Tensor, metric_fn) -> SingularityAlert | None:
        """
        奇点检测：监测度规条件数、曲率、状态范数。

        认知黑洞判据：
            1. 度规条件数 > 1e10 → 度规病态
            2. 状态范数发散 → 认知崩溃
            3. 曲率发散 → 时空奇点
        """
        try:
            g = metric_fn(t)
            g = g.to(torch.float64)
            n = g.shape[-1]
            g_reg = g + 1e-10 * torch.eye(n, dtype=torch.float64)
            eigvals = torch.linalg.eigvalsh(g_reg)
            eigvals = torch.clamp(eigvals, min=1e-20)
            condition = (eigvals.max() / eigvals.min()).item()

            # 标量曲率代理：度规偏离基线的 Frobenius 范数
            I = torch.eye(n, dtype=torch.float64)
            curvature = (g - I).norm().item() / n

            state_norm = S.norm().item()

            # 认知黑洞事件视界半径
            # r_s = 2κρ/n，其中 ρ = trace(T)/n ≈ curvature
            kappa = 1.0 / (n + 2)
            rho = curvature
            horizon_radius = 2.0 * kappa * rho / n

            # 霍金温度（信息流失率）
            # T_H = 1/(8π·κ·ρ·R)
            if kappa * rho * max(curvature, 1e-10) > 1e-30:
                hawking_temp = 1.0 / (8 * math.pi * kappa * rho * max(curvature, 1e-10))
            else:
                hawking_temp = 0.0

            # 严重程度 [0, 1]
            severity = min(1.0, math.log10(max(condition, 1.0)) / 10.0)

            # 仅在严重程度 > 0.3 时预警
            if severity > 0.3:
                return SingularityAlert(
                    time=t,
                    metric_condition=condition,
                    curvature=curvature,
                    state_norm=state_norm,
                    horizon_radius=horizon_radius,
                    hawking_temperature=hawking_temp,
                    severity=severity,
                )
        except Exception:
            pass

        return None

    def _safe_metric_condition(self, S: Tensor, metric_fn) -> float:
        """安全计算度规条件数。"""
        try:
            g = metric_fn(0.0).to(torch.float64)
            n = g.shape[-1]
            g_reg = g + 1e-10 * torch.eye(n, dtype=torch.float64)
            eigvals = torch.linalg.eigvalsh(g_reg)
            eigvals = torch.clamp(eigvals, min=1e-20)
            return (eigvals.max() / eigvals.min()).item()
        except Exception:
            return float('inf')

    @property
    def has_singularity(self) -> bool:
        """是否检测到奇点。"""
        return any(a.severity >= 0.8 for a in self.alerts)

    @property
    def max_severity(self) -> float:
        """最大严重程度。"""
        return max((a.severity for a in self.alerts), default=0.0)

    def singularity_report(self) -> str:
        """生成奇点报告。"""
        if not self.alerts:
            return "无奇点预警：全程数值稳定。"

        lines = [
            "=" * 70,
            "认知黑洞奇点报告",
            "=" * 70,
            f"预警次数: {len(self.alerts)}",
            f"最大严重程度: {self.max_severity:.4f}",
            f"奇点检测: {'是 — 认知黑洞形成' if self.has_singularity else '否 — 临界但未坍缩'}",
            "",
        ]

        for i, alert in enumerate(self.alerts):
            lines.extend([
                f"--- 预警 {i+1} ---",
                f"  时刻: t={alert.time:.4f}",
                f"  度规条件数: {alert.metric_condition:.4e}",
                f"  曲率: {alert.curvature:.4e}",
                f"  状态范数: {alert.state_norm:.4e}",
                f"  事件视界半径: {alert.horizon_radius:.4e}",
                f"  霍金温度(信息流失): {alert.hawking_temperature:.4e}",
                f"  严重程度: {alert.severity:.4f}",
                "",
            ])

        lines.append("=" * 70)
        if self.has_singularity:
            lines.append("结论：认知流形在演化中形成黑洞（度规发散，状态崩溃）。")
            lines.append("物理意义：个体陷入不可逃逸的认知奇点（如尼采1889年精神崩溃）。")
        else:
            lines.append("结论：认知流形接近但未越过事件视界（临界态）。")
        lines.append("=" * 70)

        return "\n".join(lines)
