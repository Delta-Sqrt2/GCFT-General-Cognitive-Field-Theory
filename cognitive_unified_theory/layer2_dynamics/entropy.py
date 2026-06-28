"""
负熵转化判据 —— "杀不死我的必使我强大"的数学证明

核心判据：
    ΔR ≥ ∫|∇F| dt

    其中：
        R(S)    认知有效秩（负熵，来自第一层）
        ∫|∇F|   痛苦势能梯度的积分（来自第二层 pain_field）
        ΔR      有效秩的提升量

物理意义：
    - 痛苦（∫|∇F|）是"输入"
    - 成长（ΔR）是"输出"
    - 判据成立时：痛苦被有效转化为认知维度的扩展（负熵增加）
    - 判据不成立时：痛苦超过转化容量，导致认知坍缩（创伤后应激）

    这就是"杀不死我的必使我强大"的数学前提：
    只有当 ΔR ≥ ∫|∇F| 时，痛苦才转化为成长。
    否则痛苦摧毁系统（流形撕裂）。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import effective_rank


class NegativeEntropyCriterion:
    """
    负熵转化判据：ΔR ≥ ∫|∇F|

    判定痛苦是否被有效转化为认知成长。
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self._rank_history: list[Tensor] = []
        self._pain_integral_history: list[Tensor] = []

    def compute_rank(self, state_history: Tensor) -> Tensor:
        """
        计算认知有效秩 R(S)。
        基于状态历史的协方差矩阵特征值熵。

        R = exp(H)，H = -Σ p_i log p_i，p_i = λ_i / Σλ_j
        """
        if state_history.dim() == 1:
            # 单状态：用状态的自相关
            S = state_history.to(torch.float64).unsqueeze(0)
        else:
            S = state_history.to(torch.float64)
        # 协方差矩阵
        if S.shape[0] < 2:
            return torch.tensor(1.0, dtype=torch.float64)
        cov = S.transpose(-1, -2) @ S / S.shape[0]
        return effective_rank(cov)

    def evaluate(self, state_before: Tensor, state_after: Tensor, pain_integral: Tensor) -> dict[str, Tensor]:
        """
        评估负熵转化判据。

        参数：
            state_before: 痛苦事件前的状态历史 (T1, n)
            state_after:  痛苦事件后的状态历史 (T2, n)
            pain_integral: ∫|∇F| dt 痛苦梯度积分

        返回：
            dict:
                rank_before: R(S_before)
                rank_after: R(S_after)
                delta_R: ΔR = R_after - R_before
                pain_integral: ∫|∇F|
                criterion_met: ΔR ≥ ∫|∇F|（连续值，非布尔）
                growth_efficiency: ΔR / ∫|∇F|（成长效率）
                collapse_risk: max(0, ∫|∇F| - ΔR)（坍缩风险）
        """
        R_before = self.compute_rank(state_before)
        R_after = self.compute_rank(state_after)
        delta_R = R_after - R_before

        pain = pain_integral.to(torch.float64)

        # 判据满足度（连续值，非布尔）：(ΔR - ∫|∇F|) / (|ΔR| + |∫|∇F||)
        # > 0 表示判据满足，< 0 表示不满足
        denom = delta_R.abs() + pain.abs() + 1e-30
        criterion_met = (delta_R - pain) / denom

        # 成长效率
        growth_efficiency = delta_R / (pain + 1e-30)

        # 坍缩风险：痛苦超过转化容量的部分
        collapse_risk = torch.clamp(pain - delta_R, min=0.0)

        self._rank_history.append(R_after)
        self._pain_integral_history.append(pain)

        return {
            "rank_before": R_before,
            "rank_after": R_after,
            "delta_R": delta_R,
            "pain_integral": pain,
            "criterion_met": criterion_met,
            "growth_efficiency": growth_efficiency,
            "collapse_risk": collapse_risk,
        }

    def growth_trajectory(self) -> Tensor:
        """成长轨迹：有效秩随时间的演化。"""
        if not self._rank_history:
            return torch.tensor([], dtype=torch.float64)
        return torch.stack(self._rank_history, dim=0)

    def pain_trajectory(self) -> Tensor:
        """痛苦累积轨迹。"""
        if not self._pain_integral_history:
            return torch.tensor([], dtype=torch.float64)
        return torch.stack(self._pain_integral_history, dim=0)

    def is_growth_sustainable(self, window: int = 5) -> Tensor:
        """
        评估成长是否可持续。
        连续判据：最近 window 次的平均成长效率 > 0。
        """
        if len(self._rank_history) < window:
            return torch.tensor(0.0, dtype=torch.float64)
        recent_ranks = torch.stack(self._rank_history[-window:])
        recent_pains = torch.stack(self._pain_integral_history[-window:])
        # 趋势：有效秩是否在增长
        trend = (recent_ranks[-1] - recent_ranks[0]) / window
        # 痛苦是否在可控范围
        pain_avg = recent_pains.mean()
        return trend / (pain_avg + 1e-30)
