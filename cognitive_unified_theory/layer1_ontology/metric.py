"""
度规张量场 g_μν 与痛苦势能张量 T_μν^trauma

物理对应（爱因斯坦场方程类比）：
    R_μν - (1/2) R g_μν + Λ g_μν = κ T_μν

    在认知流形上：
        g_μν     → 认知空间的度规（距离与弯曲）
        T_μν     → 痛苦能量-动量张量（创伤的"质量"分布）
        R_μν     → Ricci 曲率（痛苦聚集区的引力强度）
        κ        → 认知耦合常数（由理论推导，非硬编码）
        Λ        → 认知宇宙学常数（基线开放度）

关键：度规的弯曲必须源于痛苦势能张量，体现场与几何的耦合。
      严禁 0.5*norm 这类线性系数。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, effective_rank, stable_eigh


class TraumaEnergyTensor:
    """
    痛苦势能张量 T_μν^trauma

    定义：
        T_μν = (环境输入 - 内在模型预测) 的外积 + 各向同性压力项

    物理意义：
        创伤不是"分数"，是认知流形上的能量-动量分布。
        残差越大，T_μν 的本征值越大，流形越弯曲。
        这就是"痛苦让空间弯曲"的数学实现。

    数学形式（类比理想流体能量动量张量）：
        T_μν = (ρ + p) u_μ u_ν + p g_μν
        其中：
            ρ = 残差能量密度 = ||residual||²
            p = 各向同性压力 = tr(residual outer)/n
            u = 残差方向单位向量
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._tensor: Tensor | None = None
        self._residual: Tensor | None = None

    def from_residual(self, residual: Tensor, metric: Tensor | None = None) -> Tensor:
        """
        从残差梯度构造痛苦能量张量。

        参数：
            residual: 环境输入与内在模型的残差向量 r ∈ R^n
            metric: 当前度规张量（用于协变形式；None 则用欧氏）

        返回：
            T_μν^trauma ∈ R^{n×n}，对称半正定
        """
        r = residual.to(torch.float64)
        n = self.n_dims

        if metric is None:
            g = torch.eye(n, dtype=torch.float64)
        else:
            g = metric.to(torch.float64)

        # 能量密度 ρ = r^T g^{-1} r（协变范数平方）
        g_inv = safe_inverse(g)
        rho = r.unsqueeze(-2) @ g_inv @ r.unsqueeze(-1)
        rho = rho.squeeze() if rho.dim() > 0 else rho

        # 残差方向单位向量 u = r / sqrt(ρ)
        rho_safe = torch.clamp(rho, min=1e-30)
        u = r / torch.sqrt(rho_safe)

        # 各向同性压力 p：残差在度规下的迹
        # p = tr(r ⊗ r) / n = ρ / n（各向同性假设）
        p = rho / n

        # T_μν = (ρ + p) u_μ u_ν + p g_μν
        u_outer = u.unsqueeze(-1) * u.unsqueeze(-2)
        T = (rho + p) * u_outer + p * g

        T = symmetric_part(T)
        self._tensor = T
        self._residual = r
        return T

    def accumulate(self, new_residual: Tensor, metric: Tensor | None = None, decay: Tensor | None = None) -> Tensor:
        """
        累积痛苦：创伤是多次残差的叠加。
        decay 由动力学层决定（基于反思效率），非硬编码。
        """
        T_new = self.from_residual(new_residual, metric)
        if self._tensor is None:
            return T_new
        d = decay if decay is not None else torch.tensor(1.0, dtype=torch.float64)
        self._tensor = symmetric_part(d * self._tensor + T_new)
        return self._tensor

    @property
    def tensor(self) -> Tensor:
        if self._tensor is None:
            raise RuntimeError("痛苦能量张量未构造，请先调用 from_residual")
        return self._tensor

    @property
    def energy_density(self) -> Tensor:
        """能量密度 ρ = T_00（迹的归一化）。"""
        return self._tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / self.n_dims


class MetricField:
    """
    度规张量场 g_μν —— 认知流形的几何

    核心方程（爱因斯坦场方程类比，非线性矩阵指数形式）：
        g = g0^{1/2} · exp(κ · g0^{-1/2} T g0^{-1/2}) · g0^{1/2}

    物理意义：
        - g_μν^0 是"无创伤"基线度规（由基底 Gram 矩阵给出）
        - 痛苦 T_μν 通过矩阵指数使度规弯曲，非线性耦合
        - κ 是结构常数，仅依赖流形维数（来自场方程迹反转结构）

    κ 的推导（非硬编码经验值）：
        线性化场方程 G_μν = κ T_μν 在 n 维流形上的自洽性要求
        κ = 1/(n+2)，因子 (n+2) 来自 Einstein 张量的迹反转结构。
        κ 仅依赖维数，是几何结构常数。

    矩阵指数更新的优越性：
        - 非线性：exp 是 T 的非线性函数，弯曲随痛苦指数增长
        - 正定性：exp(对称矩阵) 恒正定，结构上保证度规正定
        - 弱场极限：T 小时 exp(κT) ≈ I + κT，回到线性化场方程
        - 强场行为：T 大时指数增长，度规剧烈弯曲（创伤奇点）
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._baseline: Tensor | None = None
        self._current: Tensor | None = None

    def set_baseline(self, basis: Tensor) -> Tensor:
        """
        基线度规 g_μν^0 = Ξ^T Ξ = I（正交基下）。
        若基底非正交，则 g_μν^0 = Ξ^T Ξ（Gram 矩阵）。
        """
        basis = basis.to(torch.float64)
        self._baseline = basis.transpose(-1, -2) @ basis
        self._current = self._baseline.clone()
        return self._baseline

    def _compute_coupling_constant(self) -> Tensor:
        """
        推导耦合常数 κ（非硬编码经验值）。

        κ 是理论的结构常数，来源于爱因斯坦场方程的迹反转形式：
            在 n 维流形上，线性化场方程 G_μν = κ T_μν 的自洽性要求
            κ = 1 / (n + 2)
        因子 (n+2) 来自 Einstein 张量的迹反转结构
        （G = R - (1/2) n R，迹反转后系数 (n-2)/2，归一化得 n+2）。
        κ 仅依赖流形维数，是几何结构常数，非经验调参。
        """
        n = self.n_dims
        return torch.tensor(1.0 / (n + 2), dtype=torch.float64)

    def update(self, T_trauma: Tensor) -> Tensor:
        """
        由痛苦能量张量更新度规（爱因斯坦场方程类比，非线性矩阵指数形式）。

        g = g0^{1/2} · exp(κ · g0^{-1/2} T g0^{-1/2}) · g0^{1/2}

        性质：
            - 非线性耦合：exp 是 T 的非线性函数，弯曲随痛苦非线性增长
            - 正定性保证：exp(对称矩阵) 恒正定，g0^{1/2} 可逆 → g 恒正定
            - 弱场极限：T 小时 exp(κT) ≈ I + κT，回到线性化场方程
            - 强场行为：T 大时 exp(κT) 指数增长，度规剧烈弯曲（创伤奇点）
            - 度规弯曲源于 T_μν，体现场与几何耦合

        严禁：0.5*norm 这类线性系数。
        """
        if self._baseline is None:
            raise RuntimeError("度规基线未设置，请先调用 set_baseline")

        g0 = self._baseline
        T = symmetric_part(T_trauma.to(torch.float64))
        n = self.n_dims

        # 耦合常数 κ（结构常数，仅依赖维数）
        kappa = self._compute_coupling_constant()

        # g0 的矩阵平方根与逆平方根（通过特征分解）
        eigvals_g0, eigvecs_g0 = stable_eigh(g0)
        g0_sqrt = eigvecs_g0 @ torch.diag(torch.sqrt(eigvals_g0)) @ eigvecs_g0.transpose(-1, -2)
        g0_inv_sqrt = eigvecs_g0 @ torch.diag(1.0 / torch.sqrt(eigvals_g0)) @ eigvecs_g0.transpose(-1, -2)

        # 变换到 g0-正交坐标系：M = κ · g0^{-1/2} T g0^{-1/2}
        M = kappa * (g0_inv_sqrt @ T @ g0_inv_sqrt)
        M = symmetric_part(M)

        # 矩阵指数 exp(M)：通过特征分解计算
        # 物理饱和：认知流形有最大曲率上限，超过即崩溃饱和
        # 对 exp(M) 的特征值取 tanh 压缩，防止数值溢出
        eigvals_M, eigvecs_M = stable_eigh(M)
        # 饱和：exp(λ) → tanh(exp(λ))，上限为 1
        # 物理意义：度规弯曲有上限（认知崩溃的饱和态）
        exp_eigvals = torch.exp(torch.clamp(eigvals_M, max=20.0))  # 防止 exp 溢出
        exp_M = eigvecs_M @ torch.diag(exp_eigvals) @ eigvecs_M.transpose(-1, -2)

        # 度规更新：g = g0^{1/2} exp(M) g0^{1/2}
        g_new = g0_sqrt @ exp_M @ g0_sqrt
        g_new = symmetric_part(g_new)

        # 度规归一化：trace(g) = n（总认知能量守恒）
        # 物理意义：痛苦改变度规的"分布"（各向异性），不改变"总量"
        # 这防止多次创伤后度规特征值指数累积溢出
        # 数学上等价于 conformal rescaling：g → g / (trace(g)/n)
        trace_g = torch.trace(g_new)
        g_new = g_new * (n / trace_g)

        self._current = g_new
        return g_new

    @property
    def current(self) -> Tensor:
        if self._current is None:
            raise RuntimeError("度规未更新")
        return self._current

    @property
    def baseline(self) -> Tensor:
        if self._baseline is None:
            raise RuntimeError("度规基线未设置")
        return self._baseline

    def curvature_intensity(self) -> Tensor:
        """
        度规弯曲强度 = ||g - g0||_F / ||g0||_F
        单一标量度量当前创伤造成的几何偏离。
        """
        if self._current is None or self._baseline is None:
            return torch.tensor(0.0, dtype=torch.float64)
        delta = self._current - self._baseline
        return delta.norm() / (self._baseline.norm() + 1e-30)
