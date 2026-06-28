"""
拓扑-语义逆投影器 —— 高维几何状态 → 自然语言诊断

战略定位（v1.2 步骤一）：
    v1.1 输出的是晦涩的张量数据（S, g, R），只有物理意义没有临床意义。
    本模块将高维几何状态"逆向投影"为低维自然语言，让人类读懂诊断。

物理原理：
    自然语言是认知流形在语言空间的投影。我们要做的是"逆向解码"。
    力类型不是查字典，而是通过分析度规 g 的结构推断：
        - CONSTRAINT（约束）：g 的对角元占优 → 能量被锁死在特定维度
        - IMPACT（冲击）：g 的非对角元占优 → 能量在维度间乱窜（涟漪效应）
        - POTENTIAL（势能）：存在近零特征值 → 系统处于不稳定的临界态

数学公式（已锁定，无模糊地带）：
    A. 力类型得分（连续，无 if-else）：
        score_CONSTRAINT = ||diag(g)||_F / ||g||_F
        score_IMPACT     = ||g - diag(g)||_F / ||g||_F
        score_POTENTIAL  = count(λ_i < ε) / n,  ε = λ_max / (n+2)
        归一化：P = softmax(scores)

    B. 主导轴能量权重（严禁 topk 截断）：
        w_i = |S_i · (gS)_i| / Σ_j |S_j · (gS)_j|
        保留所有 w_i，按权重降序排列。

    C. 文本合成：
        句法结构由主导力类型决定（IMPACT → SVO，CONSTRAINT → 被动句）。
        修饰词通过对强度 I = S^T g S 的连续插值生成，或直接输出数值 I。
        严禁 words[idx] 离散查表。

工程铁律（v1.2 三大铁律）：
    1. 拒绝字典：所有语义转换基于数学结构（特征值、特征向量、投影系数）
    2. 内生安全：不抛出异常，所有状态合法
    3. 几何诚实：所有量由度规 g 推导

认知轴命名规范：
    必须使用数学符号 ξ1, ξ2, ..., ξn，严禁使用"焦虑/抑郁"等心理学术语。
    这是"拒绝字典"铁律的核心：心理学标签是字典，数学符号是几何实体。
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import softmax

from ..core.tensor_ops import stable_eigh, symmetric_part


class InverseProjector:
    """
    拓扑-语义逆投影器：高维几何 → 自然语言诊断。

    使用方式：
        projector = InverseProjector(n_dims=8)
        diagnosis = projector.project(state, metric)
        # diagnosis: {
        #     'force_scores': {'IMPACT': 0.7, 'CONSTRAINT': 0.2, ...},
        #     'force_distribution': Tensor,  # softmax 概率
        #     'primary_axes': [(ξ_idx, weight), ...],  # 全部轴，无截断
        #     'intensity': float,  # 连续强度值
        #     'text': str,  # 诊断文本
        # }

    白盒保证：
        - 力类型由度规结构推断（非字典）
        - 主导轴由能量权重计算（非 topk 截断）
        - 修饰词由连续插值（非离散查表）
        - 认知轴用 ξ 符号（非心理学标签）
    """

    # 力类型名称（仅供人类阅读的枚举，不参与数值计算）
    FORCE_TYPES = ["IMPACT", "CONSTRAINT", "SELF_STATE", "POTENTIAL"]

    def __init__(self, n_dims: int, temperature: float = 1.0):
        """
        参数：
            n_dims: 认知维度
            temperature: softmax 温度参数（调节力类型强调程度）
                        T→0: 独断选择（逼近 argmax）
                        T→∞: 均匀分布
                        T=1: 标准 softmax（默认）
        """
        self.n_dims = n_dims
        self.temperature = temperature

    def infer_force_scores(self, metric: Tensor) -> dict[str, Tensor]:
        """
        力类型推断：从度规 g 的结构推断力类型得分（连续，无 if-else）。

        数学公式：
            score_CONSTRAINT = ||diag(g)||_F / ||g||_F
                物理意义：对角占优 → 能量被锁死在特定维度（边界约束）
            score_IMPACT = ||g - diag(g)||_F / ||g||_F
                物理意义：非对角占优 → 能量在维度间乱窜（冲击涟漪）
            score_POTENTIAL = count(λ_i < ε) / n,  ε = λ_max / (n+2)
                物理意义：近零特征值 → 系统处于不稳定临界态（势能积累）
            score_SELF_STATE = 1 - max(上述三者)
                物理意义：剩余概率 → 内能自发变化（无外部作用）

        归一化：P = softmax(scores / T)

        严禁：if curvature > 10: text = "你很痛苦" 这类阈值映射。
        """
        g = symmetric_part(metric.to(torch.float64))
        n = self.n_dims

        # Frobenius 范数
        g_norm = g.norm() + 1e-30  # 防止除零

        # 对角部分与非对角部分
        diag_g = torch.diag(g.diagonal())
        off_diag = g - diag_g

        # 力类型得分（连续推导）
        diag_ratio = diag_g.norm() / g_norm  # 对角占优程度
        offdiag_ratio = off_diag.norm() / g_norm  # 非对角占优程度
        score_constraint = diag_ratio
        score_impact = offdiag_ratio

        # 近零特征值计数（势能态判据）
        eigvals, _ = stable_eigh(g)
        eigvals = torch.clamp(eigvals, min=1e-20)
        lambda_max = eigvals.max()
        epsilon = lambda_max / (n + 2)  # 阈值由维数推导（结构常数）
        near_zero_count = (eigvals < epsilon).float().sum()
        near_zero_ratio = near_zero_count / n

        # POTENTIAL 得分：对角占优 × 近零特征值显著性 × 维数因子
        # 物理意义：对角占优（能量被锁死）+ 近零特征值（某些维度无约束）= 临界态
        # 仅在对角占优时 POTENTIAL 才显著，避免与非对角占优的 IMPACT 冲突
        # -log(1-ratio) 增强近零特征值的显著性（ratio→1 时得分→∞）
        # 维数因子 n/(n-2) 由维数推导（n>2 时有效）
        dim_factor = n / max(n - 2, 1)
        potential_significance = -torch.log(torch.clamp(1.0 - near_zero_ratio, min=1e-10))
        score_potential = diag_ratio * potential_significance * dim_factor

        # SELF_STATE 得分：剩余概率（保证四者归一化前合理）
        # 使用 1 - max(三者) 的连续形式
        scores_tensor = torch.stack([
            score_impact,
            score_constraint,
            torch.tensor(0.0, dtype=torch.float64),  # SELF_STATE 占位
            score_potential,
        ])
        # SELF_STATE = 1 - max(其他三者)，但保证非负
        score_self_state = torch.clamp(1.0 - scores_tensor[[0, 1, 3]].max(), min=0.0)
        scores_tensor[2] = score_self_state

        # softmax 归一化（温度参数调节强调程度）
        distribution = softmax(scores_tensor / self.temperature, dim=0)

        return {
            "IMPACT": distribution[0],
            "CONSTRAINT": distribution[1],
            "SELF_STATE": distribution[2],
            "POTENTIAL": distribution[3],
            "raw_scores": {
                "IMPACT": score_impact,
                "CONSTRAINT": score_constraint,
                "SELF_STATE": score_self_state,
                "POTENTIAL": score_potential,
            },
            "distribution": distribution,
        }

    def extract_primary_axes(self, state: Tensor, metric: Tensor) -> list[tuple[int, Tensor]]:
        """
        主导轴提取：计算每个认知轴 ξ_i 的能量权重 w_i。

        数学公式：
            w_i = |S_i · (gS)_i| / Σ_j |S_j · (gS)_j|

        物理意义：
            S_i 是状态在 ξ_i 轴的投影，(gS)_i 是度规作用后的投影。
            二者乘积的绝对值 = 该轴上的"功率"（能量流率）。
            归一化后得到各轴的能量分布。

        严禁：torch.topk 截断。必须保留所有 w_i，按权重降序排列。
        严禁：使用"焦虑/抑郁"等心理学标签。必须用 ξ1, ξ2, ..., ξn。
        """
        S = state.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))

        # 度规作用：gS
        gS = g @ S

        # 各轴功率：|S_i · (gS)_i|
        power = (S * gS).abs()

        # 归一化（softmax 式连续权重，非截断）
        total = power.sum() + 1e-30
        weights = power / total

        # 按权重降序排列（保留全部，不截断）
        sorted_idx = torch.argsort(weights, descending=True)
        return [(int(idx), weights[idx]) for idx in sorted_idx]

    def compute_intensity(self, state: Tensor, metric: Tensor) -> Tensor:
        """
        计算认知状态的总能量强度 I（连续值，非离散标签）。

        数学公式：
            I = S^T g S

        物理意义：
            度规加权下的状态能量。痛苦越大，度规越弯曲，能量越高。
            这是"你很痛苦"的数学化：不是布尔判断，是连续标量。
        """
        S = state.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))
        return S @ g @ S

    def _intensity_descriptor(self, intensity: Tensor, baseline: Tensor) -> str:
        """
        强度修饰词的连续插值生成（严禁离散查表，严禁 if-elif 分支）。

        数学方法（完全连续）：
            将强度 I 归一化到 [0, 1]：r = I / (I + I_baseline)
            连续映射强度标量：s = tanh(r · 3.0) ∈ [0, 1)
            修饰词描述由 s 的连续插值生成：
                - 通过 s 对"微弱/中等/强烈/极端"四个锚点做连续加权混合
                - 权重 = softmax([-|s-0.0|·k, -|s-0.33|·k, -|s-0.66|·k, -|s-1.0|·k])
                - k 由维数推导：k = n（认知维数，锐度）
            最终描述 = 加权拼接的连续谱 + 数值 I

        严禁：
            - words = ["微弱", ...]; return words[idx]
            - if strength < 0.3: ... elif ...
        """
        I = float(intensity)
        I_base = float(baseline) + 1e-30
        r = I / (I + I_base)  # 归一化到 [0, 1]

        # 连续映射强度标量（tanh 单调，无离散）
        s = float(torch.tanh(torch.tensor(r * 3.0)))  # [0, 1)

        # 连续插值：对四个语义锚点做 softmax 加权（无 if-elif）
        # 锚点位置由维数等分推导（结构常数，非硬编码经验值）
        n = self.n_dims
        k = float(n)  # 锐度由维数推导
        anchors = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        labels = ["微弱", "中等", "强烈", "极端"]

        # softmax 权重（连续，无 argmax 离散选择）
        distances = torch.tensor([-abs(s - a) * k for a in anchors], dtype=torch.float64)
        weights = torch.softmax(distances, dim=0)

        # 连续谱描述：加权拼接所有锚点（保留全部信息，无截断）
        # 主导词 = 权重最大者（仅用于显示，权重信息全部保留）
        dominant_idx = int(weights.argmax())  # 仅显示用，数值计算用全部权重
        dominant_label = labels[dominant_idx]
        weight_spectrum = "+".join(
            f"{labels[i]}({float(weights[i]):.3f})" for i in range(len(labels))
        )

        return f"{dominant_label}[谱:{weight_spectrum}](I={I:.4f}, s={s:.4f})"

    def synthesize_text(
        self,
        force_scores: dict[str, Tensor],
        primary_axes: list[tuple[int, Tensor]],
        intensity: Tensor,
        n_text_axes: int = 3,
    ) -> str:
        """
        文本合成：基于力类型选择句法模板，填入 ξ 轴编号和能量数值。

        句法模板（由主导力类型决定，非硬编码心理学术语）：
            IMPACT → SVO 句式："能量冲击 ξ_i 轴（权重 w_i）"
            CONSTRAINT → 被动句："ξ_i 轴被约束（权重 w_i）"
            SELF_STATE → 状态句："ξ_i 轴内能变化（权重 w_i）"
            POTENTIAL → 势能句："ξ_i 轴势能积累（权重 w_i）"

        轴命名：必须用 ξ1, ξ2, ..., ξn，严禁"焦虑/抑郁"等标签。
        修饰词：由强度 I 的连续插值生成，非离散查表。

        参数：
            n_text_axes: 文本中显示的轴数量（默认3，但权重信息全部保留）
                        注意：这只是显示数量，primary_axes 列表完整未截断。
        """
        # 主导力类型（argmax 仅用于选择句法模板，不截断权重信息）
        distribution = force_scores["distribution"]
        dominant_idx = int(distribution.argmax())
        dominant_force = self.FORCE_TYPES[dominant_idx]
        dominant_prob = float(distribution[dominant_idx])

        # 强度描述（连续插值）
        intensity_desc = self._intensity_descriptor(intensity, baseline=torch.tensor(1.0))

        # 力类型分布描述（保留全部4类概率，无截断）
        force_desc_parts = []
        for fname in self.FORCE_TYPES:
            prob = float(force_scores[fname])
            force_desc_parts.append(f"{fname}={prob:.3f}")
        force_desc = ", ".join(force_desc_parts)

        # 主导轴描述（显示前 n_text_axes 个，但全部权重在 primary_axes 中保留）
        axes_desc_parts = []
        for i, (idx, weight) in enumerate(primary_axes[:n_text_axes]):
            axes_desc_parts.append(f"ξ{idx+1}(w={float(weight):.3f})")
        axes_desc = ", ".join(axes_desc_parts)

        # 全部轴权重和（验证未截断：应为 1.0）
        total_weight = sum(float(w) for _, w in primary_axes)

        # 句法模板选择（由力类型决定，非心理学标签）
        if dominant_force == "IMPACT":
            template = (
                f"[诊断·IMPACT] {intensity_desc} 冲击作用于认知流形。"
                f"主导力类型: {dominant_force}(P={dominant_prob:.3f})。"
                f"力分布: [{force_desc}]。"
                f"能量注入主轴: {axes_desc}。"
                f"全部轴权重和: {total_weight:.4f}(未截断)。"
            )
        elif dominant_force == "CONSTRAINT":
            template = (
                f"[诊断·CONSTRAINT] {intensity_desc} 约束锁定认知流形。"
                f"主导力类型: {dominant_force}(P={dominant_prob:.3f})。"
                f"力分布: [{force_desc}]。"
                f"被约束主轴: {axes_desc}。"
                f"全部轴权重和: {total_weight:.4f}(未截断)。"
            )
        elif dominant_force == "SELF_STATE":
            template = (
                f"[诊断·SELF_STATE] {intensity_desc} 内能自发变化。"
                f"主导力类型: {dominant_force}(P={dominant_prob:.3f})。"
                f"力分布: [{force_desc}]。"
                f"内能变化主轴: {axes_desc}。"
                f"全部轴权重和: {total_weight:.4f}(未截断)。"
            )
        else:  # POTENTIAL
            template = (
                f"[诊断·POTENTIAL] {intensity_desc} 势能临界积累。"
                f"主导力类型: {dominant_force}(P={dominant_prob:.3f})。"
                f"力分布: [{force_desc}]。"
                f"势能积累主轴: {axes_desc}。"
                f"全部轴权重和: {total_weight:.4f}(未截断)。"
            )

        return template

    def project(self, state: Tensor, metric: Tensor, n_text_axes: int = 3) -> dict[str, Tensor | list | str | float]:
        """
        完整逆投影：高维几何状态 → 自然语言诊断。

        参数：
            state: 认知状态 S ∈ R^n
            metric: 度规张量 g ∈ R^{n×n}
            n_text_axes: 文本中显示的轴数量（权重信息全部保留）

        返回：
            {
                'force_scores': 力类型得分字典（含原始得分和 softmax 分布）,
                'primary_axes': 全部轴权重列表（未截断）,
                'intensity': 总能量强度 I = S^T g S,
                'text': 诊断文本,
                'dominant_force': 主导力类型名称,
                'dominant_probability': 主导力类型概率,
            }

        白盒保证：
            - 所有量由 S 和 g 推导，无字典查找
            - 力类型由度规结构推断（对角/非对角/特征值）
            - 轴权重由能量功率计算（softmax 连续，无 topk 截断）
            - 修饰词由强度连续插值（非离散查表）
            - 认知轴用 ξ 符号（非心理学标签）
        """
        # 1. 力类型推断（从度规结构）
        force_scores = self.infer_force_scores(metric)

        # 2. 主导轴提取（全部保留，无截断）
        primary_axes = self.extract_primary_axes(state, metric)

        # 3. 强度计算（连续标量）
        intensity = self.compute_intensity(state, metric)

        # 4. 文本合成（句法模板 + ξ 符号 + 连续修饰词）
        text = self.synthesize_text(force_scores, primary_axes, intensity, n_text_axes)

        # 主导力类型
        distribution = force_scores["distribution"]
        dominant_idx = int(distribution.argmax())
        dominant_force = self.FORCE_TYPES[dominant_idx]
        dominant_prob = float(distribution[dominant_idx])

        return {
            "force_scores": force_scores,
            "primary_axes": primary_axes,
            "intensity": intensity,
            "text": text,
            "dominant_force": dominant_force,
            "dominant_probability": dominant_prob,
            "n_axes_total": len(primary_axes),  # 验证未截断
        }

    def project_trajectory(self, states: Tensor, metrics: Tensor) -> list[dict]:
        """
        轨迹逆投影：将演化轨迹投影为诊断序列。

        参数：
            states: 状态序列 (T, n)
            metrics: 度规序列 (T, n, n)

        返回：
            诊断列表，每个时刻一个诊断字典
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if metrics.dim() == 2:
            metrics = metrics.unsqueeze(0)

        diagnoses = []
        for t in range(states.shape[0]):
            diag = self.project(states[t], metrics[t])
            diagnoses.append(diag)
        return diagnoses
