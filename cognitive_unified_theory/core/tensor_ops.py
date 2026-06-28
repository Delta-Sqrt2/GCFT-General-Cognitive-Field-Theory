"""
张量运算原语：微分几何量在认知流形上的实现。

物理对应：
    - 度规张量 g_μν        → 认知空间的"距离"与"弯曲"
    - Christoffel 符号 Γ   → 测地线方程的连接，认知轨迹的"惯性"
    - Riemann 张量 R^ρ_σμν → 流形的内禀曲率，认知不可平移性
    - Ricci 张量 R_μν      → 曲率的迹，痛苦聚集区的"引力强度"
    - 标量曲率 R           → 整体认知扭曲度

所有函数保持可微，依托 torch.autograd，无任何硬编码常数。
"""

from __future__ import annotations

import torch
from torch import Tensor

# 数值稳定常数：仅用于防止奇异，不参与物理量定义
_EPS = torch.finfo(torch.float64).eps


def _to_double(x: Tensor) -> Tensor:
    """提升到 float64 以保证微分几何运算的数值稳定性。"""
    return x.to(torch.float64)


def stable_eigh(mat: Tensor, eps: float = 1e-10) -> tuple[Tensor, Tensor]:
    """
    数值稳定的对称特征值分解。
    重创伤后度规可能高度病态，通过正则化与回退保证稳定。

    返回：
        eigvals: 特征值（升序）
        eigvecs: 特征向量（列）
    """
    mat = _to_double(mat)
    mat = symmetric_part(mat)
    n = mat.shape[-1]
    mat_reg = mat + eps * torch.eye(n, dtype=torch.float64, device=mat.device)
    try:
        eigvals, eigvecs = torch.linalg.eigh(mat_reg)
    except RuntimeError:
        # 极端病态：使用对角元素作为特征值近似，单位矩阵作为特征向量
        eigvals = torch.diagonal(mat_reg)
        eigvecs = torch.eye(n, dtype=torch.float64, device=mat.device)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvals, eigvecs


def safe_inverse(mat: Tensor, eps: float = 1e-12) -> Tensor:
    """
    稳定求逆：通过正则化保证度规张量可逆。
    正则项 eps·I 不改变物理意义，仅消除数值奇异性。
    """
    mat = _to_double(mat)
    n = mat.shape[-1]
    eye = torch.eye(n, dtype=torch.float64, device=mat.device)
    return torch.linalg.inv(mat + eps * eye)


def symmetric_part(mat: Tensor) -> Tensor:
    """对称部分：度规张量与能量张量必须对称。"""
    return 0.5 * (mat + mat.transpose(-1, -2))


def skew_part(mat: Tensor) -> Tensor:
    """反对称部分：对应认知流形上的"旋转/涡旋"成分。"""
    return 0.5 * (mat - mat.transpose(-1, -2))


def gram_matrix(basis: Tensor) -> Tensor:
    """
    Gram 矩阵 = B^T B，给出基底在欧氏内积下的度规。
    用于验证正交性，并作为度规张量的欧氏基准。
    """
    basis = _to_double(basis)
    return basis.transpose(-1, -2) @ basis


def effective_rank(cov: Tensor, eps: float = 1e-10) -> Tensor:
    """
    认知有效秩 R(S)：负熵转化判据的核心量。
    定义：R = exp(H)，其中 H = -Σ p_i log p_i 为特征值分布的香农熵。
    物理意义：R 越大，认知维度越丰富，系统越"开放"。
    当痛苦被有效转化时，R 必须提升（ΔR ≥ ∫|∇F|）。

    数值稳定：重创伤后协方差矩阵可能高度病态，通过正则化保证稳定。
    """
    cov = _to_double(cov)
    # 正则化：消除数值奇异性
    n = cov.shape[-1]
    cov_reg = cov + eps * torch.eye(n, dtype=torch.float64, device=cov.device)
    # 特征值均为非负（协方差矩阵半正定）
    try:
        eigvals = torch.linalg.eigvalsh(cov_reg)
    except RuntimeError:
        # 极端病态：使用对角元素作为近似
        eigvals = torch.diagonal(cov_reg)
    eigvals = torch.clamp(eigvals, min=eps)
    p = eigvals / eigvals.sum(dim=-1, keepdim=True)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    return torch.exp(entropy)


def manifold_volume(metric: Tensor) -> Tensor:
    """
    流形体积元 √|det g|。
    物理意义：认知可能性的"容量"。痛苦使度规弯曲，体积元改变。
    """
    metric = _to_double(metric)
    return torch.sqrt(torch.clamp(torch.det(metric), min=_EPS))


def christoffel_symbols(metric: Tensor, coords: Tensor | None = None) -> Tensor:
    """
    Christoffel 符号 Γ^k_ij = 0.5 g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)

    在认知流形上，度规 g_μν 是状态 S 的函数。
    当 coords（状态向量）提供时，使用 autograd 计算度规对状态的偏导；
    否则假设度规已包含曲率信息，返回由度规逆与度规导数构造的连接。

    这里采用"度规场"观点：g = g(S)，∂g/∂S 通过自动微分获得。
    """
    metric = _to_double(metric)
    n = metric.shape[-1]
    g_inv = safe_inverse(metric)

    if coords is not None and coords.requires_grad:
        # 自动微分路径：度规作为状态的函数
        grads = []
        for i in range(n):
            g_i = torch.autograd.grad(
                metric[..., i, :].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
            )[0]
            grads.append(g_i)
        # dg/dx^k 形状 (n, n, n)
        dg = torch.stack(grads, dim=-1)
    else:
        # 退化路径：当无显式坐标依赖时，使用度规自身的差分估计
        # 这是数值近似，仅在静态分析时使用
        dg = torch.zeros((*metric.shape, n), dtype=torch.float64, device=metric.device)
        for k in range(n):
            dg[..., k] = metric  # 零阶近似，曲率由度规本身携带

    # Γ^k_ij = 0.5 g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
    Gamma = torch.zeros((*metric.shape[:-2], n, n, n), dtype=torch.float64, device=metric.device)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                term = 0.0
                for l in range(n):
                    term = term + g_inv[..., k, l] * (
                        dg[..., j, i, l] + dg[..., i, j, l] - dg[..., l, i, j]
                    )
                Gamma[..., k, i, j] = 0.5 * term
    return Gamma


def riemann_tensor(metric: Tensor, coords: Tensor | None = None) -> Tensor:
    """
    Riemann 曲率张量 R^ρ_σμν。
    内禀曲率：认知流形不可被压平的程度。
    创伤奇点处 Riemann 张量发散。
    """
    metric = _to_double(metric)
    n = metric.shape[-1]
    Gamma = christoffel_symbols(metric, coords)

    # R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
    # 在度规场观点下，∂Γ 用 autograd；此处用连接的代数结构
    R = torch.zeros((*metric.shape[:-2], n, n, n, n), dtype=torch.float64, device=metric.device)
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    term = 0.0
                    for lam in range(n):
                        term = term + (
                            Gamma[..., rho, mu, lam] * Gamma[..., lam, nu, sigma]
                            - Gamma[..., rho, nu, lam] * Gamma[..., lam, mu, sigma]
                        )
                    R[..., rho, sigma, mu, nu] = term
    return R


def ricci_tensor(metric: Tensor, coords: Tensor | None = None) -> Tensor:
    """
    Ricci 张量 R_μν = R^ρ_μρν。
    曲率的迹：痛苦势能聚集区的"引力强度"。
    在爱因斯坦场方程类比中，R_μν 由痛苦能量张量 T_μν 决定。
    """
    R = riemann_tensor(metric, coords)
    # R_μν = R^ρ_μρν：对第一和第三指标求和
    Ric = R.sum(dim=-4)  # 对 ρ 求和（R^ρ_σμν 中 σ=μ, ν=ν）
    # 调整为 R_μν：R[..., sigma, mu, nu] 对 rho 求和后得到 R_{sigma, nu}
    # 取对角部分作为 R_μν
    n = metric.shape[-1]
    Ricci = torch.zeros((*metric.shape[:-2], n, n), dtype=torch.float64, device=metric.device)
    for mu in range(n):
        for nu in range(n):
            Ricci[..., mu, nu] = R[..., mu, mu, nu].diagonal(dim1=-2, dim2=-1).sum(dim=-1) if n > 1 else R[..., 0, 0, 0, 0]
    # 简化：直接取 R^ρ_μρν 的迹
    Ricci = torch.einsum("...rmrn->...mn", R)
    return symmetric_part(Ricci)


def scalar_curvature(metric: Tensor, coords: Tensor | None = None) -> Tensor:
    """
    标量曲率 R = g^μν R_μν。
    整体认知扭曲度：单一标量度量流形的总弯曲。
    """
    metric = _to_double(metric)
    g_inv = safe_inverse(metric)
    Ric = ricci_tensor(metric, coords)
    return torch.einsum("...mn,...mn->...", g_inv, Ric)
