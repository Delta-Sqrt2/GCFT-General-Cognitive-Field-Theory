"""
能量方程 —— 从语言结构性质到认知能量注入的物理推导

核心公理：
    能量值必须由物理方程从结构性质推导，严禁硬编码字典查找。

物理框架：
    事件能量 = 振幅 × 时间剖面 × 方向向量

    E_event = A × s(t) × d

    其中：
        A  振幅（来自结构复杂度，信息论推导）
        s(t)  时间剖面（来自力类型，物理学标准函数）
        d  方向向量（来自力类型+宾语类别，认知物理学推导）

振幅推导（信息论）：
    A = log(1 + depth × valency × (1 + n_modifiers)) × negation_factor

    物理意义：
        - depth（依存树深度）：事件的结构复杂度
        - valency（配价）：事件的论元数量，信息容量
        - n_modifiers（修饰语数）：事件的特化程度
        - log：Shannon 信息量 I = -log(p)，结构越复杂信息量越大
        - negation_factor：否定改变能量符号（物理：反向力）

    这不是字典查找，是信息论公式应用于语法结构。

积分步长推导：
    dt = temporal_period / temporal_count

    物理意义：
        - temporal_period：事件周期（如"每周"→7天）
        - temporal_count：周期内发生次数（如"三次"→3）
        - dt：单次事件的时间间隔

    "父亲每周严厉批评我三次" → dt = 7/3 ≈ 2.33 天/次
    高频（小 dt）→ 单位时间内更多能量注入 → 持续压力
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .force_ontology import ForceType, ForceOntology, StructuralFeatures

class EnergyEquation:
    """
    能量方程：结构特征 → 事件能量张量。

    所有数值由物理方程推导，无硬编码情感分数。
    """

    def __init__(self, n_dims: int = 8):
        self.n_dims = n_dims
        self.force_ontology = ForceOntology()

    def compute_amplitude(self, features: StructuralFeatures) -> Tensor:
        """
        计算能量振幅 A（信息论推导）。

        A = log(1 + depth × valency × (1 + n_modifiers)) × negation_factor

        物理意义：
            - 结构复杂度越高 → 信息量越大 → 能量振幅越大
            - 否定改变力的方向（物理：反向力），取负号
            - log 来自 Shannon 信息论：I = log(1 + complexity)

        严禁：A = {"严厉": 0.9} 这类字典查找。
        """
        depth = max(features.tree_depth, 1)
        valency = max(features.valency, 1)
        n_mod = features.n_modifiers

        # 信息论振幅：结构复杂度的对数度量
        complexity = depth * valency * (1 + n_mod)
        amplitude = math.log(1 + complexity)

        # 否定因子：物理上反向力（能量符号反转）
        negation_factor = -1.0 if features.has_negation else 1.0

        return torch.tensor(amplitude * negation_factor, dtype=torch.float64)

    def compute_direction(self, force_type: ForceType, features: StructuralFeatures) -> Tensor:
        """
        计算能量方向向量 d（认知物理学推导）。

        d = α × d_primary(force_type) + β × d_secondary(object_category, subject_person)

        其中：
            d_primary:   力类型决定的主激活轴（物理因果链）
            d_secondary: 宾语类别决定的次激活轴（认知语言学）
            α, β:        混合系数，由结构推导

        物理推导 α, β：
            α = valency / (valency + 1)  —— 配价越高，主轴（力类型）越主导
            β = 1 / (valency + 1)       —— 宾语影响随配价递减
            这来自信息论：论元越多，单个论元的信息权重越低。
        """
        n = self.n_dims

        d_primary = self.force_ontology.primary_axes(force_type, n)
        d_secondary = self.force_ontology.secondary_axes(
            features.object_category, features.subject_person, n
        )

        # 混合系数（信息论推导，非硬编码）
        valency = max(features.valency, 1)
        alpha = valency / (valency + 1)
        beta = 1.0 / (valency + 1)

        d = alpha * d_primary + beta * d_secondary
        norm = d.norm()
        if norm > 1e-30:
            d = d / norm
        return d

    def compute_step_size(self, features: StructuralFeatures) -> float:
        """
        计算积分步长 dt（时间频率推导）。

        dt = temporal_period / temporal_count

        物理意义：
            - 高频事件（小 dt）→ 单位时间更多能量注入 → 持续压力
            - 低频事件（大 dt）→ 间歇性能量注入 → 恢复空间

        "每周三次" → dt = 7/3 ≈ 2.33
        "每天一次" → dt = 1/1 = 1.0
        "偶尔"     → dt = 30/1 = 30.0（低频）
        """
        count = max(features.temporal_count, 1e-10)
        period = max(features.temporal_period, 1e-10)
        return period / count

    def compute_force_vector(self, features: StructuralFeatures) -> Tensor:
        """
        计算事件的力向量 F ∈ R^n_dims。

        F = A × d

        其中：
            A: 振幅（信息论）
            d: 方向（认知物理学）

        这是注入认知流形的能量向量，将作为 engine.process_event() 的输入。
        """
        force_type = self.force_ontology.classify(features)
        amplitude = self.compute_amplitude(features)
        direction = self.compute_direction(force_type, features)
        return amplitude * direction

    def compute_continuous_energy(self, features: StructuralFeatures, t_array: Tensor) -> Tensor:
        """
        计算连续能量注入 E(t) = A × s(t) × d。

        用于长周期演化中的连续力场建模。

        参数：
            features: 结构特征
            t_array: 时间点序列

        返回：
            能量张量 (T, n_dims)
        """
        force_type = self.force_ontology.classify(features)
        amplitude = self.compute_amplitude(features)
        direction = self.compute_direction(force_type, features)

        # 时间剖面
        tau = self.compute_step_size(features)  # 特征时间常数
        s_t = self.force_ontology.temporal_profile(force_type, t_array, t0=0.0, tau=tau)

        # 能量 = 振幅 × 时间剖面 × 方向
        # (T,) × (n,) → (T, n)
        energy = amplitude * s_t.unsqueeze(-1) * direction.unsqueeze(0)
        return energy


class EventTensor:
    """
    事件张量：解析器的输出单元。

    格式：[time_step, force_vector]

    属性：
        time_step: 事件发生的时间步（由时间频率推导）
        force_vector: 力向量 ∈ R^n_dims（由能量方程推导）
        force_type: 力类型（由动词语法性质分类）
        amplitude: 能量振幅（由结构复杂度推导）
        step_size: 积分步长（由时间频率推导）
        source_text: 原始文本（仅日志，不参与计算）
        derivation: 推导过程（白盒审计用）
    """
    def __init__(
        self,
        time_step: float,
        force_vector: Tensor,
        force_type: ForceType,
        amplitude: Tensor,
        step_size: float,
        source_text: str = "",
        derivation: str = "",
    ):
        self.time_step = time_step
        self.force_vector = force_vector
        self.force_type = force_type
        self.amplitude = amplitude
        self.step_size = step_size
        self.source_text = source_text
        self.derivation = derivation

    def to_tensor(self) -> Tensor:
        """输出 [time_step, force_vector] 格式张量。"""
        return torch.cat([
            torch.tensor([self.time_step], dtype=torch.float64),
            self.force_vector
        ])

    def __repr__(self) -> str:
        return (
            f"EventTensor(t={self.time_step:.2f}, "
            f"type={self.force_type.name}, "
            f"|F|={self.amplitude:.4f}, "
            f"dt={self.step_size:.2f})"
        )
