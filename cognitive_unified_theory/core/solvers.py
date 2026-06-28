"""
数值求解器：变分极值 / 自由能最小化 / ODE 积分 / 哈密顿-雅可比 / 特征值问题

物理对应：
    - 变分极值        → 命运坐标 S0 是边界约束下的能量最低态
    - 自由能最小化    → 最小自由能原理（Friston），认知系统趋近最小惊讶
    - ODE 积分        → 类 Navier-Stokes 演化，连续时间动力学
    - 哈密顿-雅可比   → 时间逆行层的主方程，高维重构低维历史
    - 特征值问题      → S0 解算，度规本征模
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = torch.finfo(torch.float64).eps


def eigen_solve_sym(mat: Tensor) -> tuple[Tensor, Tensor]:
    """
    对称矩阵特征值分解（eigh）。
    用于：
        - S0 解算：边界约束矩阵的最小特征向量
        - 度规本征模：认知流形的主轴
        - 协方差有效秩：特征值分布的熵
    """
    mat = mat.to(torch.float64)
    mat = 0.5 * (mat + mat.transpose(-1, -2))  # 强制对称，消除数值噪声
    eigvals, eigvecs = torch.linalg.eigh(mat)
    return eigvals, eigvecs


def variational_extremum(
    energy_matrix: Tensor,
    boundary: Tensor | None = None,
    mode: str = "min",
) -> tuple[Tensor, Tensor]:
    """
    变分极值问题：在边界约束下求解能量的极值态。

    数学形式：
        min/max  S^T E S
        s.t.     S^T B S = 1   （边界约束归一化）

    通过广义特征值问题 E S = λ B S 求解。
    最小特征值对应"命运坐标"S0——边界约束下的能量最低态。

    参数：
        energy_matrix: 能量算符 E（如自由能 Hessian）
        boundary: 边界约束矩阵 B（如原生家庭 B_family）
        mode: "min" 取最低能态（基态），"max" 取最高能态

    返回：
        eigenvalue: 极值能量
        eigenvector: 对应的状态向量 S0
    """
    E = energy_matrix.to(torch.float64)
    E = 0.5 * (E + E.transpose(-1, -2))

    if boundary is None:
        eigvals, eigvecs = torch.linalg.eigh(E)
    else:
        B = boundary.to(torch.float64)
        B = 0.5 * (B + B.transpose(-1, -2))
        # 广义特征值问题 E S = λ B S
        # 数值稳定：B = L L^T (Cholesky)，转化为标准问题
        n_b = B.shape[-1]
        L = torch.linalg.cholesky(B + _EPS * torch.eye(n_b, dtype=torch.float64, device=B.device))
        L_inv = torch.linalg.solve_triangular(L, torch.eye(n_b, dtype=torch.float64, device=B.device), upper=False)
        E_tilde = L_inv @ E @ L_inv.transpose(-1, -2)
        E_tilde = 0.5 * (E_tilde + E_tilde.transpose(-1, -2))
        eigvals, eigvecs_tilde = torch.linalg.eigh(E_tilde)
        eigvecs = L_inv.transpose(-1, -2) @ eigvecs_tilde

    idx = 0 if mode == "min" else -1
    return eigvals[idx], eigvecs[..., idx]


def free_energy_minimization(
    initial_state: Tensor,
    energy_fn,
    n_steps: int = 200,
    lr: float = 0.05,
) -> Tensor:
    """
    最小自由能原理（Friston Free Energy Principle）：
        认知系统演化以最小化自由能 F = -log p(sensory|model) + KL(model||prior)

    实现：梯度下降求解能量泛函的极小值。
    所有"心理常数"必须通过此过程涌现，而非硬编码。

    参数：
        initial_state: 初始认知状态 S
        energy_fn: 可微能量函数 E(S) → scalar
        n_steps: 优化步数
        lr: 学习率（仅数值参数，非物理常数）

    返回：
        最小自由能态 S*
    """
    S = initial_state.to(torch.float64).clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([S], lr=lr, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        F = energy_fn(S)
        F.backward()
        return F

    for _ in range(n_steps):
        opt.step(closure)
    return S.detach()


def ode_integrate(
    derivative_fn,
    initial_state: Tensor,
    t_span: tuple[float, float],
    n_steps: int = 100,
    method: str = "rk4",
) -> Tensor:
    """
    ODE 积分器：连续时间动力学演化。
    用于类 Navier-Stokes 认知演化场 ∂S/∂t = -∇H(S) + η∇²S。

    方法：
        - "euler":  前向欧拉（一阶，快速）
        - "rk4":    经典四阶 Runge-Kutta（精度与稳定性平衡）
        - "leapfrog": 辛积分子（保守系统，能量守恒）

    严禁用 if-else 离散化心理过程；此函数是唯一的演化通道。
    """
    S0 = initial_state.to(torch.float64)
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    trajectory = [S0]
    S = S0

    if method == "euler":
        for _ in range(n_steps):
            dS = derivative_fn(S)
            S = S + dt * dS
            trajectory.append(S)

    elif method == "rk4":
        for _ in range(n_steps):
            k1 = derivative_fn(S)
            k2 = derivative_fn(S + 0.5 * dt * k1)
            k3 = derivative_fn(S + 0.5 * dt * k2)
            k4 = derivative_fn(S + dt * k3)
            S = S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(S)

    elif method == "leapfrog":
        # 辛积分子：保守哈密顿系统
        S = S0
        P = -derivative_fn(S) * 0.5 * dt  # 初始半步动量
        for _ in range(n_steps):
            S = S + P * dt
            F = -derivative_fn(S)
            P = P + F * dt
        S = S + P * 0.5 * dt
        trajectory.append(S)
    else:
        raise ValueError(f"未知积分方法: {method}")

    return torch.stack(trajectory, dim=0)


def hamilton_jacobi_solve(
    hamiltonian_fn,
    state: Tensor,
    time_horizon: float = 1.0,
    n_steps: int = 50,
) -> Tensor:
    """
    哈密顿-雅可比方程：∂S/∂t + H(S, ∇S) = 0

    时间逆行层的核心：当系统升维（τ→1）时，利用高维度规重写
    低维历史路径。创伤从"奇点"变为"高维平凡点"。

    实现：通过作用量泛函 S_action = ∫ L dt 的稳定化
    作用量对状态的梯度给出"动量"，反向传播修改历史权重。

    参数：
        hamiltonian_fn: H(S, P) → scalar，认知哈密顿量
        state: 当前认知状态
        time_horizon: 时间反演范围
        n_steps: 离散化步数

    返回：
        作用量梯度 ∇S_action，用于重写历史权重
    """
    S = state.to(torch.float64).clone().detach().requires_grad_(True)
    dt = time_horizon / n_steps

    action = torch.zeros((), dtype=torch.float64, device=S.device)
    current = S.clone()

    for _ in range(n_steps):
        # 动量 P = ∇S（作用量梯度）
        if current.requires_grad:
            P = torch.autograd.grad(
                current.sum(), S, create_graph=True, allow_unused=True
            )[0]
            if P is None:
                P = torch.zeros_like(current)
        else:
            P = torch.zeros_like(current)

        H = hamiltonian_fn(current, P)
        # 拉格朗量 L = P·dS/dt - H；在 HJ 框架下 dS/dt = -H
        L = (P * P).sum() * 0.5 - H
        action = action + L * dt
        current = current - dt * P  # 演化

    # 作用量对状态的梯度：历史权重重写信号
    grad_action = torch.autograd.grad(action, S, allow_unused=True)[0]
    if grad_action is None:
        grad_action = torch.zeros_like(S)
    return grad_action.detach()
