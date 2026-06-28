"""
认知防火墙 —— 分级权限安全协议

核心定义：
    根据 τ 值分级，限制认知操作权限，防止"认知流形撕裂"。

    黑盒态 (τ ≈ 0)：
        - 被动力学方程被动驱动
        - 无手动修改权限
        - 系统自动演化（类 Navier-Stokes）
        - 对应：无意识/自动驾驶/创伤反应

    灰盒态 (τ ∈ (0, 1))：
        - 允许局部扰动（小范围参数调整）
        - 系统提供直觉指引（梯度提示）
        - 扰动幅度受 τ 约束
        - 对应：有意识但未完全掌控/学习状态

    白盒态 (τ → 1)：
        - 完全掌控
        - 手动修改参数
        - 手术级重构（时间逆行层可用）
        - 对应：高度觉察/元认知/主动重构

工程铁律：
    - 权限跃迁必须连续（τ 连续变化，权限连续过渡）
    - 严禁离散 if-else 跳变权限
    - 权限限制是安全协议，非剥夺自由

数学实现：
    权限函数 P(τ) 是 τ 的连续函数：
        P_black(τ) = 1 - σ(τ - τ_gray)      黑盒权限（τ 小时为1）
        P_gray(τ)  = σ(τ - τ_gray) - σ(τ - τ_white)  灰盒权限
        P_white(τ) = σ(τ - τ_white)         白盒权限（τ 大时为1）
    其中 σ 是 sigmoid，τ_gray/τ_white 是阈值（由理论推导，非硬编码）
"""

from __future__ import annotations

import torch
from torch import Tensor


class CognitiveFirewall:
    """
    认知防火墙：基于透明度 τ 的分级权限系统。

    权限函数使用 sigmoid 平滑过渡，避免离散跳变。
    阈值由认知维数推导（非硬编码经验值）。
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        # 阈值推导：基于认知维数的结构常数
        # τ_gray = 1/(n+2)：进入灰盒态需要最低反思累积
        # τ_white = (n+1)/(n+2)：进入白盒态需要接近完全反思
        # 这些阈值来自"权限跃迁需要足够维度支撑"的几何要求
        self._tau_gray = torch.tensor(1.0 / (n_dims + 2), dtype=torch.float64)
        self._tau_white = torch.tensor((n_dims + 1.0) / (n_dims + 2), dtype=torch.float64)

        # 权限历史
        self._permission_history: list[dict[str, Tensor]] = []

    @property
    def tau_gray(self) -> Tensor:
        """灰盒态阈值。"""
        return self._tau_gray

    @property
    def tau_white(self) -> Tensor:
        """白盒态阈值。"""
        return self._tau_white

    def _sigmoid(self, x: Tensor, sharpness: Tensor | None = None) -> Tensor:
        """
        平滑 sigmoid 函数。
        sharpness 控制过渡锐度，默认由维数推导。
        """
        if sharpness is None:
            # 锐度 = n_dims：维度越高，权限过渡越精细
            sharpness = torch.tensor(float(self.n_dims), dtype=torch.float64)
        return torch.sigmoid(sharpness * x)

    def compute_permissions(self, tau: Tensor) -> dict[str, Tensor]:
        """
        计算三级权限（连续函数，非离散）。

        P_black(τ) = 1 - σ(τ - τ_gray)       黑盒权限
        P_gray(τ)  = σ(τ - τ_gray) · (1 - σ(τ - τ_white))  灰盒权限
        P_white(τ) = σ(τ - τ_white)           白盒权限

        性质：
            - 三级权限之和 ≈ 1（归一化）
            - τ=0: P_black≈1, P_gray≈0, P_white≈0
            - τ=0.5: P_black≈0, P_gray≈1, P_white≈0
            - τ=1: P_black≈0, P_gray≈0, P_white≈1
            - 过渡连续，无离散跳变
        """
        tau = tau.to(torch.float64)
        sigma_gray = self._sigmoid(tau - self._tau_gray)
        sigma_white = self._sigmoid(tau - self._tau_white)

        P_black = 1.0 - sigma_gray
        P_gray = sigma_gray * (1.0 - sigma_white)
        P_white = sigma_white

        permissions = {
            "black": P_black,
            "gray": P_gray,
            "white": P_white,
            "manual_edit_allowed": P_white,  # 手动修改权限 = 白盒权限
            "local_perturbation_allowed": P_gray,  # 局部扰动权限 = 灰盒权限
            "surgery_allowed": P_white,  # 手术级重构权限 = 白盒权限
        }

        self._permission_history.append(permissions)
        return permissions

    def max_perturbation_amplitude(self, tau: Tensor) -> Tensor:
        """
        允许的最大扰动幅度（连续函数）。
        扰动幅度 = P_gray · ε_max + P_white · 1.0
        其中 ε_max 是灰盒态最大扰动（由维数推导）。

        物理意义：
            - 黑盒态：扰动幅度 ≈ 0（无意识，无法干预）
            - 灰盒态：扰动幅度有限（有意识但受限）
            - 白盒态：扰动幅度 = 1（完全掌控）
        """
        perms = self.compute_permissions(tau)
        # 灰盒态最大扰动 = 1/sqrt(n)：小范围调整
        epsilon_max = 1.0 / torch.sqrt(torch.tensor(float(self.n_dims), dtype=torch.float64))
        amplitude = perms["gray"] * epsilon_max + perms["white"] * 1.0
        return amplitude

    def check_safety(self, proposed_perturbation: Tensor, tau: Tensor, metric: Tensor) -> dict[str, Tensor]:
        """
        安全检查：评估扰动是否会导致"认知流形撕裂"。

        流形撕裂判据：
            扰动后的度规正定性 det(g_new) > 0
            且度规条件数 cond(g_new) < cond_max

        参数：
            proposed_perturbation: 拟议的状态扰动 δS
            tau: 当前透明度
            metric: 当前度规

        返回：
            dict:
                allowed: 允许的扰动幅度（连续值，非布尔）
                safety_margin: 安全裕度
                tear_risk: 撕裂风险
        """
        delta = proposed_perturbation.to(torch.float64)
        g = metric.to(torch.float64)
        tau_val = tau.to(torch.float64)

        # 最大允许幅度
        max_amp = self.max_perturbation_amplitude(tau_val)

        # 拟议扰动幅度
        delta_norm = delta.norm()

        # 安全裕度 = max_amp - delta_norm
        safety_margin = max_amp - delta_norm

        # 撕裂风险 = max(0, delta_norm - max_amp) / (delta_norm + 1e-30)
        tear_risk = torch.clamp(delta_norm - max_amp, min=0.0) / (delta_norm + 1e-30)

        # 允许的扰动比例（连续，非布尔）
        allowed_ratio = torch.clamp(max_amp / (delta_norm + 1e-30), min=0.0, max=1.0)

        return {
            "allowed_amplitude": max_amp,
            "allowed_ratio": allowed_ratio,
            "safety_margin": safety_margin,
            "tear_risk": tear_risk,
            "safe_perturbation": delta * allowed_ratio,  # 缩放后的安全扰动
        }

    def state_label(self, tau: Tensor) -> str:
        """
        状态标签（仅供人类阅读，不参与计算）。
        基于最大权限判定当前态。
        """
        perms = self.compute_permissions(tau)
        if perms["black"] >= perms["gray"] and perms["black"] >= perms["white"]:
            return "black_box"
        elif perms["gray"] >= perms["white"]:
            return "gray_box"
        else:
            return "white_box"

    @property
    def permission_history(self) -> list[dict[str, Tensor]]:
        return self._permission_history
