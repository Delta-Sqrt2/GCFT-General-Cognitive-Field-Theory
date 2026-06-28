"""
认知演化场 —— 类 Navier-Stokes 方程

物理对应：
    流体力学 Navier-Stokes: ∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v + f
    认知演化类比:           ∂S/∂t = -∇H(S) + η ∇²S + ξ(S)

    其中：
        S      认知状态向量（流体的"速度场"）
        H      自由能泛函（"压力"势）
        η      认知粘度（扩散项，对应"思维惯性/习惯"）
        ∇²S   拉普拉斯算子（状态在流形上的扩散）
        ξ      外部驱动力（新事件输入）

    度规 g_μν 进入拉普拉斯算子（Laplace-Beltrami 算子）：
        Δ_g S = (1/√|g|) ∂_μ (√|g| g^{μν} ∂_ν S)

工程铁律：
    - 严禁 if-else 离散化心理过程
    - 必须使用数值求解器（RK4 / 辛积分子）
    - 粘度 η 由度规曲率推导，非硬编码
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, manifold_volume
from ..core.solvers import ode_integrate


class CognitiveEvolutionField:
    """
    认知演化场：类 Navier-Stokes 的连续时间动力学。

    核心方程：
        ∂S/∂t = -∇H(S) + η(S) Δ_g S + ξ(t)

    其中：
        -∇H(S)    梯度下降（减少自由能/痛苦）
        η Δ_g S   扩散项（认知惯性，习惯性思维）
        ξ(t)      外部驱动（新事件）
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._metric: Tensor | None = None
        self._hamiltonian = None  # 自由能函数
        self._external_force: Tensor | None = None
        self._trajectory: list[Tensor] = []

    def set_metric(self, metric: Tensor) -> None:
        """设置当前度规（来自第一层）。"""
        self._metric = symmetric_part(metric.to(torch.float64))

    def set_hamiltonian(self, hamiltonian_fn) -> None:
        """
        设置自由能泛函 H(S) → scalar。
        通常由痛苦势能场 F_trauma 给出。
        """
        self._hamiltonian = hamiltonian_fn

    def set_external_force(self, force: Tensor) -> None:
        """设置外部驱动力 ξ(t)（新事件输入的连续形式）。"""
        self._external_force = force.to(torch.float64)

    def _compute_viscosity(self, state: Tensor) -> Tensor:
        """
        推导认知粘度 η(S)（非硬编码）。

        物理推导：
            在弯曲流形上，扩散系数由度规曲率决定。
            高曲率区（创伤区）→ 低粘度（思维僵化，难以扩散）
            低曲率区（开放区）→ 高粘度（思维灵活，易扩散）

            η = 1 / (1 + R_scalar)
            其中 R_scalar 是标量曲率（来自第一层）。
            这是 Riemann 几何中"热扩散系数与曲率反相关"的标准结果。

        数值稳定：
            重创伤后度规可能高度病态，通过正则化保证数值稳定。
            正则化不改变物理意义，仅消除数值奇异性。
        """
        if self._metric is None:
            return torch.tensor(1.0, dtype=torch.float64)
        # 度规的"曲率代理"：度规偏离基线的程度
        # 完整标量曲率计算昂贵，用度规条件数作为曲率代理
        # 数值稳定：正则化度规
        n = self._metric.shape[-1]
        g_reg = self._metric + 1e-10 * torch.eye(n, dtype=torch.float64)
        try:
            eigvals = torch.linalg.eigvalsh(g_reg)
        except RuntimeError:
            # 极端病态情况：使用对角元素作为近似
            eigvals = torch.diagonal(g_reg)
        eigvals = torch.clamp(eigvals, min=1e-20)
        condition = eigvals.max() / eigvals.min()
        # η = 1 / (1 + log(1 + condition))
        # 对数压缩避免极端值，结构由度规谱决定
        eta = 1.0 / (1.0 + torch.log(1.0 + condition))
        return eta

    def _laplace_beltrami(self, state: Tensor) -> Tensor:
        """
        Laplace-Beltrami 算子 Δ_g S：认知稳态项。

        物理意义：
            在流体力学中，粘性项 ν∇²v 是扩散（平滑）项。
            在认知演化中，对应"认知稳态"——状态向历史均值回归。
            这是"习惯性思维"的数学表达：认知倾向于回到熟悉的状态。

        实现：
            Δ_g S ≈ S_mean - S（向历史均值回归）
            S_mean 是状态历史的均值，代表"认知基线"
            度规加权：g (S_mean - S)，弯曲区域的回归力更强

        这防止状态发散，保证数值稳定性。
        """
        if self._metric is None:
            return torch.zeros_like(state)
        S = state.to(torch.float64)
        # 认知基线：状态历史的均值（若有），否则原点
        # 使用 S0（初始命运坐标）作为基线，代表"认知舒适区"
        if hasattr(self, '_baseline_state') and self._baseline_state is not None:
            S_baseline = self._baseline_state
        else:
            S_baseline = torch.zeros_like(S)
        # 稳态回归力：g (S_baseline - S)
        # 度规加权：弯曲区域的回归力更强（创伤区更难改变）
        return self._metric @ (S_baseline - S)

    def set_baseline_state(self, baseline: Tensor) -> None:
        """设置认知基线（通常是 S0，初始命运坐标）。"""
        self._baseline_state = baseline.to(torch.float64)

    def derivative(self, state: Tensor) -> Tensor:
        """
        演化方程右端：dS/dt = -∇H + η Δ_g S + ξ

        严禁离散逻辑；全程张量运算。
        """
        S = state.to(torch.float64)
        dS = torch.zeros_like(S)

        # 1. 梯度下降项 -∇H(S)
        if self._hamiltonian is not None:
            S_req = S.detach().clone().requires_grad_(True)
            H = self._hamiltonian(S_req)
            grad_H = torch.autograd.grad(H, S_req, create_graph=True)[0]
            dS = dS - grad_H

        # 2. 扩散项 η Δ_g S
        eta = self._compute_viscosity(S)
        laplacian = self._laplace_beltrami(S)
        dS = dS + eta * laplacian

        # 3. 外部驱动 ξ
        if self._external_force is not None:
            dS = dS + self._external_force

        return dS

    def evolve(self, initial_state: Tensor, t_span: tuple[float, float], n_steps: int = 100, method: str = "rk4") -> Tensor:
        """
        演化认知状态：数值积分类 Navier-Stokes 方程。

        参数：
            initial_state: 初始状态 S(t0)
            t_span: 时间区间 (t0, t1)
            n_steps: 积分步数
            method: 积分方法 ("euler" / "rk4" / "leapfrog")

        返回：
            轨迹 (T, n)，T = n_steps + 1
        """
        trajectory = ode_integrate(
            derivative_fn=self.derivative,
            initial_state=initial_state,
            t_span=t_span,
            n_steps=n_steps,
            method=method,
        )
        self._trajectory = [t for t in trajectory]
        return trajectory

    def geodesic_deviation(self, trajectory: Tensor) -> Tensor:
        """
        测地线偏离：轨迹偏离测地线的程度。
        物理意义：认知"惯性"与"外力"的较量。
        测地线是自由演化（无外力）的轨迹，偏离代表外部驱动的影响。
        """
        if self._metric is None:
            return torch.zeros(trajectory.shape[0], dtype=torch.float64)
        g = self._metric
        deviations = []
        for t in range(trajectory.shape[0]):
            S = trajectory[t]
            # 加速度 = d²S/dt² ≈ 二阶差分
            if t > 0 and t < trajectory.shape[0] - 1:
                accel = trajectory[t+1] - 2 * S + trajectory[t-1]
                dev = accel @ g @ accel
                deviations.append(torch.sqrt(torch.clamp(dev, min=1e-30)))
            else:
                deviations.append(torch.tensor(0.0, dtype=torch.float64))
        return torch.stack(deviations)

    @property
    def trajectory(self) -> Tensor:
        if not self._trajectory:
            raise RuntimeError("未演化")
        return torch.stack(self._trajectory, dim=0)
