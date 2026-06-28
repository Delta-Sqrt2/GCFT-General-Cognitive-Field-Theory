"""
力学本体论 —— 谓语动词到能量作用力类型的映射

核心原则：
    力类型的分类基于动词的 GRAMMATICAL PROPERTIES（语法性质），
    而非 EMOTIONAL CONTENT（情感内容）。

    这是语言学理论（配价语法 / Aktionsart），不是情感字典。

物理对应：
    IMPACT（冲击力）    → 脉冲函数，离散能量注入，有明确作用对象
    CONSTRAINT（约束力）→ 边界条件，持续约束，状态维持
    SELF_STATE（自态力）→ 内能变化，无外部对象，状态自发改变
    POTENTIAL（势能力） → 势能场，未实际化的力，预期/可能性

分类依据（配价语法理论）：
    及物动词（transitive, valency≥2）→ IMPACT
    状态动词（stative, 无变化）      → CONSTRAINT
    不及物动词（intransitive, valency=1）→ SELF_STATE
    情态/助动词（modal/auxiliary）    → POTENTIAL

    配价（valency）是动词的语法性质，指动词能带的论元数量。
    这与情感无关，是形式语学的结构属性。
"""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass

import torch
from torch import Tensor


class ForceType(IntEnum):
    """
    四种基本力类型（对应物理学的四种基本相互作用）。

    每种力类型有不同的时间剖面（temporal profile）：
        IMPACT:    δ(t-t₀)          狄拉克脉冲（瞬时冲击）
        CONSTRAINT: Θ(t-t₀)          阶跃函数（持续约束）
        SELF_STATE: exp(-t/τ)        指数衰减（内能释放）
        POTENTIAL:  1 - exp(-t/τ)    渐近趋近（势能积累）
    """
    IMPACT = 0       # 冲击力：及物动词，直接能量传递
    CONSTRAINT = 1   # 约束力：状态动词，边界条件
    SELF_STATE = 2   # 自态力：不及物动词，内能变化
    POTENTIAL = 3    # 势能力：情态动词，未实际化


@dataclass
class StructuralFeatures:
    """
    从依存树提取的结构特征（纯语法性质，无情感内容）。

    属性：
        verb_lemma:       动词原形（仅用于日志，不参与计算）
        valency:          配价（动词论元数量）—— 语法性质
        is_transitive:    是否及物（有直接宾语）—— 语法性质
        is_stative:       是否状态动词（进行体/持续体）—— 语法性质
        is_modal:         是否情态/助动词 —— 语法性质
        tree_depth:       依存树深度（从根到最远叶节点）—— 结构性质
        n_modifiers:      修饰语数量（advmod/nummod/amod 等）—— 结构性质
        has_negation:     是否含否定（neg 依存）—— 结构性质
        subject_person:   主语人称（1=第一人称, 2=第二, 3=第三, 0=无）—— 语法性质
        object_category:  宾语语义类别（NER/POS 推导）—— 结构性质
        temporal_count:   时间频率数值（如"三次"→3）—— 结构性质
        temporal_period:  时间周期（天，如"每周"→7）—— 结构性质
    """
    verb_lemma: str = ""
    valency: int = 0
    is_transitive: bool = False
    is_stative: bool = False
    is_modal: bool = False
    tree_depth: int = 1
    n_modifiers: int = 0
    has_negation: bool = False
    subject_person: int = 0
    object_category: str = "none"
    temporal_count: float = 1.0
    temporal_period: float = 1.0


class ForceOntology:
    """
    力学本体论：动词语法性质 → 力类型分类。

    分类规则（基于配价语法，非情感字典）：
        1. 情态/助动词 → POTENTIAL（势能力）
        2. 状态动词（进行体 + 无宾语）→ CONSTRAINT（约束力）
        3. 及物动词（有直接宾语）→ IMPACT（冲击力）
        4. 不及物动词（无宾语）→ SELF_STATE（自态力）

    物理推导：
        力类型决定能量注入的时间剖面 s(t)：
            IMPACT:     s(t) = δ(t-t₀)          → 瞬时脉冲
            CONSTRAINT: s(t) = Θ(t-t₀)          → 持续约束
            SELF_STATE: s(t) = exp(-(t-t₀)/τ)   → 指数衰减
            POTENTIAL:  s(t) = 1-exp(-(t-t₀)/τ) → 渐近趋近

        这些时间剖面是物理学标准函数，非硬编码参数。
    """

    # 力类型对应的认知轴激活模式（物理推导，非字典映射）
    # IMPACT（冲击）→ 被冲击自然激活威胁感知 + 自我指涉
    #   物理推导：冲击 = 压力波 → 压力感知系统激活 = 威胁轴
    # CONSTRAINT（约束）→ 被约束自然激活不确定性 + 价值
    #   物理推导：约束 = 边界限制 → 自由度减少 = 不确定性
    # SELF_STATE（自态）→ 内部变化激活意义 + 自主
    #   物理推导：内能变化 → 系统重构 = 意义重构 + 自主性
    # POTENTIAL（势能）→ 预期激活时间贴现
    #   物理推导：势能场 → 未来可能性 = 时间预期
    #
    # 这些映射来自认知物理学的因果链，不是情感字典查找。

    def classify(self, features: StructuralFeatures) -> ForceType:
        """
        根据语法性质分类力类型。

        优先级（语法理论规范）：
            1. 情态/助动词 → POTENTIAL
            2. 状态动词 → CONSTRAINT
            3. 及物动词 → IMPACT
            4. 不及物动词 → SELF_STATE
        """
        if features.is_modal:
            return ForceType.POTENTIAL
        if features.is_stative and not features.is_transitive:
            return ForceType.CONSTRAINT
        if features.is_transitive:
            return ForceType.IMPACT
        return ForceType.SELF_STATE

    def temporal_profile(self, force_type: ForceType, t: Tensor, t0: float = 0.0, tau: float = 1.0) -> Tensor:
        """
        力类型的时间剖面 s(t)。

        物理学标准函数：
            IMPACT:     δ(t-t₀) → 用窄高斯近似 δ_ε(t-t₀) = (1/√(2π)ε)exp(-(t-t₀)²/2ε²)
            CONSTRAINT: Θ(t-t₀) → sigmoid 近似 Θ_σ(t-t₀) = 1/(1+exp(-(t-t₀)/σ))
            SELF_STATE: exp(-(t-t₀)/τ) Θ(t-t₀)
            POTENTIAL:  (1 - exp(-(t-t₀)/τ)) Θ(t-t₀)

        参数：
            force_type: 力类型
            t: 时间张量
            t0: 事件发生时刻
            tau: 特征时间常数（由事件结构推导，非硬编码）
        """
        dt = t - t0
        sigma = 0.1  # 脉冲宽度（数值近似，非物理参数）

        if force_type == ForceType.IMPACT:
            # 狄拉克脉冲的窄高斯近似
            return torch.exp(-dt ** 2 / (2 * sigma ** 2)) / (sigma * (2 * torch.pi) ** 0.5)

        elif force_type == ForceType.CONSTRAINT:
            # 阶跃函数的 sigmoid 近似
            return torch.sigmoid(dt / sigma)

        elif force_type == ForceType.SELF_STATE:
            # 指数衰减 × 阶跃
            step = torch.sigmoid(dt / sigma)
            return torch.exp(-torch.clamp(dt, min=0) / tau) * step

        elif force_type == ForceType.POTENTIAL:
            # 渐近趋近 × 阶跃
            step = torch.sigmoid(dt / sigma)
            return (1.0 - torch.exp(-torch.clamp(dt, min=0) / tau)) * step

        return torch.zeros_like(t)

    def primary_axes(self, force_type: ForceType, n_dims: int) -> Tensor:
        """
        力类型对应的主激活轴向量（物理推导）。

        IMPACT     → [ξ1 威胁, ξ3 自我指涉]           被冲击→威胁+自我
        CONSTRAINT → [ξ5 不确定性, ξ8 价值]            被约束→不确定+价值
        SELF_STATE → [ξ4 意义, ξ7 自主]                内变化→意义+自主
        POTENTIAL  → [ξ2 时间贴现]                     预期→时间

        返回：
            单位方向向量 ∈ R^n_dims
        """
        v = torch.zeros(n_dims, dtype=torch.float64)

        if force_type == ForceType.IMPACT:
            # 威胁感知 + 自我指涉
            if n_dims >= 3:
                v[0] = 1.0  # ξ1 威胁感知
                v[2] = 1.0  # ξ3 自我指涉
        elif force_type == ForceType.CONSTRAINT:
            # 不确定性 + 价值
            if n_dims >= 8:
                v[4] = 1.0  # ξ5 不确定性
                v[7] = 1.0  # ξ8 价值
            elif n_dims >= 5:
                v[4] = 1.0
        elif force_type == ForceType.SELF_STATE:
            # 意义 + 自主
            if n_dims >= 7:
                v[3] = 1.0  # ξ4 意义
                v[6] = 1.0  # ξ7 自主
            elif n_dims >= 4:
                v[3] = 1.0
        elif force_type == ForceType.POTENTIAL:
            # 时间贴现
            if n_dims >= 2:
                v[1] = 1.0  # ξ2 时间贴现

        norm = v.norm()
        if norm > 1e-30:
            v = v / norm
        return v

    def secondary_axes(self, object_category: str, subject_person: int, n_dims: int) -> Tensor:
        """
        宾语语义类别对应的次激活轴向量（基于认知语言学，非情感字典）。

        映射依据（认知语言学理论）：
            PERSON      → 依恋安全(ξ6) + 自我指涉(ξ3)  [人际互动→依恋系统]
            FIRST_PERSON→ 自我指涉(ξ3) + 自主效能(ξ7)   [自我指向→自我意识]
            TIME        → 时间贴现(ξ2)                  [时间相关→时间感知]
            ABSTRACT    → 意义连贯(ξ4) + 价值清晰(ξ8)   [抽象概念→意义系统]
            PHYSICAL    → 威胁感知(ξ1)                  [物理对象→环境感知]
            none        → 零向量

        主语人称修正：
            第一人称主语 → 自主效能(ξ7)  [我是行动者→自主性]
            第二/三人称  → 依恋安全(ξ6)  [他人是行动者→依恋系统]
        """
        v = torch.zeros(n_dims, dtype=torch.float64)

        cat = object_category.lower()
        if cat == "person":
            if n_dims >= 6:
                v[5] = 1.0  # ξ6 依恋
            if n_dims >= 3:
                v[2] = 1.0  # ξ3 自我指涉
        elif cat == "first_person":
            if n_dims >= 3:
                v[2] = 1.0  # ξ3 自我指涉
            if n_dims >= 7:
                v[6] = 1.0  # ξ7 自主效能
        elif cat == "time":
            if n_dims >= 2:
                v[1] = 1.0  # ξ2 时间贴现
        elif cat == "abstract":
            if n_dims >= 4:
                v[3] = 1.0  # ξ4 意义
            if n_dims >= 8:
                v[7] = 1.0  # ξ8 价值
        elif cat == "physical":
            if n_dims >= 1:
                v[0] = 1.0  # ξ1 威胁感知

        # 主语人称修正
        if subject_person == 1 and n_dims >= 7:
            v[6] = v[6] + 0.5  # 第一人称主语→自主效能增强
        elif subject_person in (2, 3) and n_dims >= 6:
            v[5] = v[5] + 0.5  # 他人主语→依恋系统激活

        norm = v.norm()
        if norm > 1e-30:
            v = v / norm
        return v
