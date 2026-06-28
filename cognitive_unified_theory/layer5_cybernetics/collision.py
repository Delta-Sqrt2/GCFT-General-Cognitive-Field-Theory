"""
人际流形碰撞解算 —— 两个认知流形的相互作用

物理对应：
    两个认知流形 (M_1, g_1) 和 (M_2, g_2) 的碰撞。
    类比广义相对论中的双星系统：两个时空度规的相互作用。

关键量：
    - 认知夹角 θ：cos θ = <S_1, S_2>_g / (|S_1|_g |S_2|_g)
    - 度规相容性 C：两个度规的"重叠"程度
    - 碰撞能量 E：碰撞产生的"冲击"（残差能量）
    - 引力/斥力：度规曲率梯度决定的相互作用方向

物理意义：
    - θ 小 → 认知相近 → 共振（理解）
    - θ 大 → 认知相远 → 摩擦（冲突）
    - C 高 → 度规相容 → 可交流
    - C 低 → 度规不相容 → 误解
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part


class ManifoldCollisionSolver:
    """
    人际流形碰撞解算器。
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims

    def cognitive_angle(self, S_1: Tensor, S_2: Tensor, metric: Tensor) -> Tensor:
        """
        认知夹角 θ：两个状态在度规下的夹角。
        cos θ = <S_1, S_2>_g / (|S_1|_g |S_2|_g)

        物理意义：
            θ → 0：认知同向（深度共鸣）
            θ → π/2：认知正交（互不相关）
            θ → π：认知反向（根本冲突）
        """
        S1 = S_1.to(torch.float64)
        S2 = S_2.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))

        inner = S1 @ g @ S2
        norm1 = torch.sqrt(torch.clamp(S1 @ g @ S1, min=1e-30))
        norm2 = torch.sqrt(torch.clamp(S2 @ g @ S2, min=1e-30))

        cos_theta = inner / (norm1 * norm2)
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
        return torch.arccos(cos_theta)

    def metric_compatibility(self, g_1: Tensor, g_2: Tensor) -> Tensor:
        """
        度规相容性 C：两个度规的"重叠"程度。

        数学：
            C = tr(g_1^{-1} g_2) / n
            归一化到 [0, ∞)，C=1 表示完全相容。

        物理意义：
            C ≈ 1：两个认知空间的"几何"一致，可无失真交流
            C >> 1 或 C << 1：度规失配，交流产生"折射"
        """
        g1 = symmetric_part(g_1.to(torch.float64))
        g2 = symmetric_part(g_2.to(torch.float64))
        g1_inv = safe_inverse(g1)
        # 相容性 = tr(g1^{-1} g2) / n
        C = torch.einsum("mn,mn->", g1_inv, g2) / self.n_dims
        return C

    def collision_energy(self, S_1: Tensor, S_2: Tensor, g_1: Tensor, g_2: Tensor) -> Tensor:
        """
        碰撞能量 E：两个流形碰撞产生的"冲击"。

        数学：
            E = (1/2) ||S_1 - S_2||²_{g_avg}
            其中 g_avg = (g_1 + g_2) / 2 是平均度规

        物理意义：
            E 越大，碰撞越剧烈（冲突越强）
            E → 0，两个认知状态一致（无碰撞）
        """
        S1 = S_1.to(torch.float64)
        S2 = S_2.to(torch.float64)
        g_avg = symmetric_part((g_1.to(torch.float64) + g_2.to(torch.float64)) / 2.0)

        delta = S1 - S2
        E = 0.5 * delta @ g_avg @ delta
        return E

    def interaction_force(self, S_1: Tensor, S_2: Tensor, g_1: Tensor, g_2: Tensor) -> dict[str, Tensor]:
        """
        相互作用力：碰撞产生的"引力"或"斥力"。

        数学：
            F = -∇_S1 E = g_avg (S_2 - S_1)
            F > 0（沿 S_2 方向）：引力（吸引）
            F < 0（沿 -S_2 方向）：斥力（排斥）

        物理意义：
            引力：认知趋同（学习/认同）
            斥力：认知分化（冲突/防御）
        """
        S1 = S_1.to(torch.float64)
        S2 = S_2.to(torch.float64)
        g_avg = symmetric_part((g_1.to(torch.float64) + g_2.to(torch.float64)) / 2.0)

        # 力 = g_avg (S_2 - S_1)
        F = g_avg @ (S2 - S1)
        # 力的大小
        F_mag = torch.sqrt(torch.clamp(F @ g_avg @ F, min=1e-30))
        # 力的方向（相对 S_1）：正=引力，负=斥力
        direction = (F @ S1) / (F_mag * torch.sqrt(torch.clamp(S1 @ g_avg @ S1, min=1e-30)) + 1e-30)

        return {
            "force_vector": F,
            "force_magnitude": F_mag,
            "direction": direction,  # >0 引力，<0 斥力
            "is_attractive": torch.sigmoid(direction * 10),  # 连续值，非布尔
        }

    def solve_collision(self, S_1: Tensor, S_2: Tensor, g_1: Tensor, g_2: Tensor) -> dict[str, Tensor]:
        """
        完整碰撞解算：综合所有碰撞量。

        返回：
            angle: 认知夹角 θ
            compatibility: 度规相容性 C
            energy: 碰撞能量 E
            force: 相互作用力
            resonance: 共振度（综合指标）
        """
        theta = self.cognitive_angle(S_1, S_2, g_1)
        C = self.metric_compatibility(g_1, g_2)
        E = self.collision_energy(S_1, S_2, g_1, g_2)
        force = self.interaction_force(S_1, S_2, g_1, g_2)

        # 共振度：夹角小 + 相容性高 → 共振强
        # resonance = exp(-θ) · exp(-|log C|)
        resonance = torch.exp(-theta) * torch.exp(-torch.log(C + 1e-30).abs())

        return {
            "angle": theta,
            "compatibility": C,
            "energy": E,
            "force_vector": force["force_vector"],
            "force_magnitude": force["force_magnitude"],
            "direction": force["direction"],
            "is_attractive": force["is_attractive"],
            "resonance": resonance,
        }
