"""
拓扑荷 Q —— 使命与业力方向的严格拓扑定义

v7.0 第二基石 / v7.8 升级：有符号拓扑荷与业力三性。

核心数学：
    拓扑荷 Q 度量认知流形度规 g 的本征向量方向在 SO(n) 中的累积旋转。
    Q = 0：拓扑平凡。g 可平滑回到真空 cI。无使命，可沉睡于局部最优。
    Q ≠ 0：拓扑非平凡。g 无法平滑回归，必须拓扑相变或配对湮灭释放。
    「使命」= 非零拓扑荷的拓扑张力。

v7.8 升级：有符号拓扑荷（Q_signed）与业力三性
    原 v7.0 的 Q_dynamic = (1/2π)·Σ_k |θ_k|（无符号，arccos 总返回非负值），
    丧失了旋转方向信息。v7.8 从累积 Berry 相位 Γ ∈ so(n) 提取符号：

    Q_signed = (1/2π) · Σ_{i>j} Γ[i,j]

    其中 Γ = Σ_k A_k 是累积 Berry 连接（反对称矩阵），
    Σ_{i>j} Γ[i,j] 是 Γ 的下三角元素之和——
    对分块对角 Γ，这给出各旋转平面有符号角度之和。

    Q_signed vs Q_dynamic 的关系（位移 vs 路径长度）：
        Q_dynamic = (1/2π)·Σ_k |θ_k|：总路径长度（善恶皆计，累积业力总量）
        Q_signed = (1/2π)·Σ_k θ_k_signed：净位移（善恶相抵后的净方向）
        |Q_signed| ≤ Q_dynamic（三角不等式）

    业力三性（trisvabhāva / trilakṣaṇa）分类：
        Q_signed > 0：善业（kuśala）——净正向旋转，趋向觉悟方向
        Q_signed < 0：恶业（akuśala）——净负向旋转，趋向执着方向
        Q_signed ≈ 0 且 Q_dynamic > 0：无记业（avyākṛta）——善恶相抵，方向不定
        Q_dynamic = 0：无业（akarma）——无拓扑张力，拓扑平凡

三层计算：
    1. 动态拓扑荷 Q_dynamic（无符号）：从度规历史 {g_k} 的本征向量旋转累积
       Q_dynamic = (1/2π) Σ_k θ(O_k^T O_{k+1})
       θ(R) = arccos((Tr(R) - 1) / (n-1))，SO(n) 总旋转角（非负）

    2. 动态有符号荷 Q_signed（v7.8 新增）：从累积 Berry 相位提取
       Q_signed = (1/2π) · Σ_{i>j} Γ[i,j]
       Γ = Σ_k antisym(O_k^T · ΔO_k) ∈ so(n)

    3. 静态拓扑荷 Q_static：从当前度规相对于基线的方向偏离
       Q_static = (1/2π) × θ(O_current^T O_baseline)

    4. 残差方向绕数 Q_winding：痛苦场方向 u(t) 在球面 S^{n-1} 上的绕数
       Q_winding = (1/2π) Σ_k Δφ_k
       Δφ_k = 相邻残差方向的角位移

守恒律：
    平滑演化下 dQ/dt ≈ 0（每步旋转角趋于0）。
    拓扑相变时 Q 发生量子化跃变。

物理意义：
    高 κ 环境：反复冲击 → 度规本征向量反复旋转 → Q_dynamic ≠ 0 → 有业力
    低 κ 环境：度规本征向量几乎不变 → Q ≈ 0 → 拓扑平凡

    Q ≠ 0 的系统处于拓扑张力态：
    度规试图回到真空 cI，但拓扑荷阻止平滑回归。
    这种张力在现象学上表现为「不得不觉察」的内在驱动力。
    不觉察（不生起 ρ，不淬火 α），系统就无法释放 Q，会坠入黑洞相。

    Q = 0 的系统没有拓扑张力：
    可以优雅地停在局部最优（异化稳态），没有内生动力去觉察。
    它的「不觉察」不是缺陷，是拓扑平凡性的本征属性。
    非价值判断（「有使命更高级」），是拓扑分类。

    Q_signed 的方向意义：
    正向旋转与负向旋转代表认知结构的两种手性（chirality）。
    同号 Q_signed 的个体 → Berry 相位同向 → 规范吸引（共业相吸）。
    异号 Q_signed 的个体 → Berry 相位反向 → 规范排斥（业力互消）。
    这与 Γ 内积 cos_align 的符号判定一致——Q_signed 是其标量投影。

佛学对应（严格，非比喻）：
    Q_dynamic = 业力总量（karma-saṃkleśa）：善恶皆计，累积不失。
    Q_signed = 业力方向（karma-mārga）：净善/净恶/无记。
    三性分类 = 善/恶/无记（trisvabhāva）：
        「善业生乐果，恶业生苦果，无记业不生果」——《俱舍论》
    Q_signed ≈ 0 但 Q_dynamic > 0 = 善恶相抵，果报不定（无记）。
    觉照（ρ→1）消解 Q_dynamic 与 Q_signed → 业力消解，回归空性。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import safe_inverse, symmetric_part, stable_eigh


class TopologicalCharge:
    """
    拓扑荷 Q 的计算与守恒验证。

    使用方式：
        tc = TopologicalCharge(n_dims=4)
        # 动态：从度规历史计算
        Q_dyn = tc.compute_dynamic_charge(g_history)
        # 静态：从当前度规计算
        Q_sta = tc.compute_static_charge(g_current, g_baseline)
        # 残差绕数：从痛苦方向历史计算
        Q_win = tc.compute_residual_winding(residual_history)
        # 总拓扑荷
        Q_total = tc.compute_total_charge(g_history, g_current, g_baseline, residual_history)
        # 守恒律验证
        conservation = tc.verify_conservation(g_history)
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        # 归一化常数：(n-1) 用于 SO(n) 旋转角公式的分母
        self._so_denom = max(n_dims - 1, 1)

    # ==================================================================
    # 核心：SO(n) 旋转角计算
    # ==================================================================

    def _extract_eigenvectors(self, g: Tensor) -> Tensor:
        """
        从度规 g 提取本征向量矩阵 O ∈ SO(n)。

        g = O^T Λ O，O 是正交矩阵（本征向量列排列）。
        拓扑荷来自 O 在 SO(n) 中的位置，而非 Λ（本征值，可连续形变）。

        处理：
            1. 数值稳定的本征分解
            2. 保证 det(O) = +1（SO(n) 而非 O(n)，通过符号修正）
            3. 简并本征值加确定性微扰打破歧义

        设计说明：
            拓扑荷计算要求可复现，故扰动必须是确定性的（非 randn）。
            使用 linspace(1,n) 的对角扰动，量级 1e-10，远小于物理量级，
            既打破简并保证本征向量唯一，又不影响物理结果。
        """
        g_sym = symmetric_part(g.to(torch.float64))
        n = self.n_dims

        # 确定性对角扰动打破简并，保证本征向量唯一且可复现
        # 量级 1e-10 远小于 eps，物理上不可观测，仅消除数值歧义
        break_degeneracy = torch.diag(
            torch.linspace(1.0, float(n), n, dtype=torch.float64) * 1e-10
        )
        g_perturbed = g_sym + break_degeneracy

        eigvals, eigvecs = stable_eigh(g_perturbed)

        # 保证 det(O) = +1（SO(n) 的连通分支）
        O = eigvecs
        det_O = torch.det(O)
        if det_O < 0:
            # 翻转最后一个本征向量的符号
            O[:, -1] = -O[:, -1]

        return O

    def _extract_eigenvectors_relative(
        self, g_curr: Tensor, O_prev: Tensor
    ) -> Tensor:
        """
        在 O_prev 基下提取 g_curr 的本征向量，保证排列连续性。

        核心思想：
            独立对每帧做 eigh 会导致本征向量列排列不一致
            （当本征值大小顺序变化时，eigh 返回的列排列会变），
            这会被误判为「旋转」。

            解决：在前一帧的本征向量基下表示当前度规：
                g_rel = O_prev^T · g_curr · O_prev
            g_rel 接近对角（当 g_curr ≈ g_prev 时），
            其本征向量 V 接近单位矩阵，排列自然连续。
            O_curr = O_prev · V 回到原基。

        关键优化：若 g_rel 近对角（非对角元素 << 对角元素），
            则本征向量 ≈ 单位矩阵，直接返回 O_prev（无旋转）。
            这避免了 eigh 排序导致的伪旋转——
            对于纯对角度规（无物理旋转），正确返回零旋转。
        """
        g_sym = symmetric_part(g_curr.to(torch.float64))
        n = self.n_dims

        # 在 O_prev 基下表示
        g_rel = O_prev.transpose(-1, -2) @ g_sym @ O_prev

        # 检测 g_rel 是否近对角（非对角元素远小于对角元素）
        diag_part = torch.diag(torch.diagonal(g_rel))
        off_diag = g_rel - diag_part
        off_diag_norm = float(torch.sqrt((off_diag ** 2).sum()))
        diag_norm = float(torch.sqrt((diag_part ** 2).sum()))

        if diag_norm > self.eps and off_diag_norm < 0.01 * diag_norm:
            # 近对角：本征向量 ≈ 单位矩阵，无物理旋转
            # 直接返回 O_prev（符号已对齐），避免 eigh 排序伪旋转
            return O_prev.clone()

        # 非对角：存在真实旋转，做正常 eigh
        # 加微扰打破简并
        break_degeneracy = torch.diag(
            torch.linspace(1.0, float(n), n, dtype=torch.float64) * 1e-10
        )
        g_rel_perturbed = g_rel + break_degeneracy

        eigvals, V = stable_eigh(g_rel_perturbed)

        # 回到原基
        O_curr = O_prev @ V

        # 保证 det = +1
        det_O = torch.det(O_curr)
        if det_O < 0:
            O_curr[:, -1] = -O_curr[:, -1]

        # 符号对齐（相对于 O_prev）
        for i in range(n):
            inner = O_prev[:, i] @ O_curr[:, i]
            if inner < 0:
                O_curr[:, i] = -O_curr[:, i]

        return O_curr

    def _rotation_angle(self, R: Tensor) -> Tensor:
        """
        计算 SO(n) 元素 R 的总旋转角。

        θ(R) = arccos(clamp((Tr(R) - 1) / (n - 1), -1, 1))

        这是 SO(n) 的「总旋转角」度量：
            R = I 时 θ = 0（无旋转）
            R 是 π 旋转时 θ = π（最大旋转）
            对 SO(3) 精确等于绕轴旋转角
            对 SO(n) 是各平面旋转角的聚合度量

        物理意义：θ 度量认知维度方向的「翻转幅度」。
        """
        R = R.to(torch.float64)
        trace_R = torch.trace(R)
        cos_theta = (trace_R - 1.0) / self._so_denom
        cos_theta = torch.clamp(cos_theta, min=-1.0 + self.eps, max=1.0 - self.eps)
        theta = torch.arccos(cos_theta)
        return theta

    def _relative_rotation(self, O_a: Tensor, O_b: Tensor) -> Tensor:
        """
        计算两个本征向量构型之间的相对旋转 R = O_a^T O_b。

        R ∈ SO(n)，度量从构型 a 到构型 b 的旋转。
        """
        return O_a.transpose(-1, -2) @ O_b

    def _compute_cumulative_berry_phase(self, O_list: list[Tensor]) -> Tensor:
        """
        计算累积 Berry 相位 Γ ∈ so(n)（v7.8 新增）。

        Γ = Σ_k A_k，其中 A_k = antisym(O_k^T · ΔO_k) 是每步的 Berry 连接。

        这是反对称矩阵（李代数 so(n) 元素），编码完整的旋转方向历史。
        下三角元素之和 Σ_{i>j} Γ[i,j] 给出有符号总旋转角。

        参数：
            O_list: 本征向量矩阵序列 [O_0, O_1, ..., O_T]

        返回：
            Γ: 累积 Berry 相位 ∈ R^{n×n}（反对称）
        """
        n = self.n_dims
        Gamma = torch.zeros(n, n, dtype=torch.float64)
        for k in range(len(O_list) - 1):
            delta_O = O_list[k + 1] - O_list[k]
            A_k = O_list[k].transpose(-1, -2) @ delta_O
            # 反对称化（取反对称部分，李代数 so(n)）
            A_k = 0.5 * (A_k - A_k.transpose(-1, -2))
            Gamma = Gamma + A_k
        # 最终反对称化（数值误差可能导致微小对称分量）
        Gamma = 0.5 * (Gamma - Gamma.transpose(-1, -2))
        return Gamma

    def _signed_angle_from_berry_phase(self, Gamma: Tensor) -> Tensor:
        """
        从累积 Berry 相位 Γ ∈ so(n) 提取有符号旋转角（v7.8 新增）。

        Q_signed = (1/2π) · Σ_{i>j} Γ[i,j]

        对分块对角 Γ（各平面独立旋转），这给出各平面有符号角度之和。
        对一般 Γ，这给出「总手性」（total chirality）的标量投影。

        物理意义：
            正值 = 净正向旋转（善业方向）
            负值 = 净负向旋转（恶业方向）
            零 = 无净方向（无记或无业）

        参数：
            Gamma: 累积 Berry 相位 ∈ so(n)（反对称矩阵）

        返回：
            Q_signed: 有符号拓扑荷标量
        """
        # 下三角元素之和（i>j）
        n = Gamma.shape[0]
        lower_sum = torch.tensor(0.0, dtype=torch.float64)
        for i in range(1, n):
            for j in range(i):
                lower_sum = lower_sum + Gamma[i, j]
        Q_signed = lower_sum / (2.0 * math.pi)
        return Q_signed

    def classify_karma_nature(
        self,
        Q_signed: Tensor | float,
        Q_magnitude: Tensor | float,
        neutral_threshold: float = 0.01,
    ) -> dict[str, str | float | bool]:
        """
        业力三性分类（v7.8 新增）。

        基于 Q_signed（方向）和 Q_magnitude（总量）将业力分为四类：
            善业（kuśala）：Q_signed > threshold
            恶业（akuśala）：Q_signed < -threshold
            无记业（avyākṛta）：|Q_signed| ≤ threshold 且 Q_magnitude > threshold
            无业（akarma）：Q_magnitude ≤ threshold

        佛学对应：
            「善业生乐果，恶业生苦果，无记业不生果」——《俱舍论》
            善业 = 净正向旋转（趋向觉悟方向）
            恶业 = 净负向旋转（趋向执着方向）
            无记业 = 善恶相抵，方向不定，不生明显果报
            无业 = 无拓扑张力，拓扑平凡

        参数：
            Q_signed: 有符号拓扑荷
            Q_magnitude: 无符号拓扑荷总量（Q_dynamic）
            neutral_threshold: 无记/无业的判定阈值

        返回：
            dict 包含 nature（善/恶/无记/无业）、is_karmic、is_directed 等
        """
        Q_s = float(Q_signed)
        Q_m = float(Q_magnitude)

        if Q_m <= neutral_threshold:
            nature = "akarma"  # 无业
            nature_cn = "无业"
            is_karmic = False
            is_directed = False
        elif Q_s > neutral_threshold:
            nature = "kushala"  # 善业
            nature_cn = "善业"
            is_karmic = True
            is_directed = True
        elif Q_s < -neutral_threshold:
            nature = "akushala"  # 恶业
            nature_cn = "恶业"
            is_karmic = True
            is_directed = True
        else:
            nature = "avyakrta"  # 无记业
            nature_cn = "无记业"
            is_karmic = True
            is_directed = False

        # 业力纯度：|Q_signed| / Q_magnitude ∈ [0, 1]
        # 1 = 纯净单向业力（全善或全恶）
        # 0 = 完全混合（善恶相抵）
        purity = abs(Q_s) / max(Q_m, self.eps) if Q_m > self.eps else 0.0

        return {
            "nature": nature,
            "nature_cn": nature_cn,
            "is_karmic": is_karmic,
            "is_directed": is_directed,
            "purity": float(purity),
            "Q_signed": Q_s,
            "Q_magnitude": Q_m,
            "thesis": (
                f"业力三性：{nature_cn}（{nature}）。"
                f"Q_signed={Q_s:.6f}，Q_magnitude={Q_m:.6f}，"
                f"纯度={purity:.4f}。"
                + (
                    "净正向旋转，趋向觉悟方向。"
                    if nature == "kushala"
                    else "净负向旋转，趋向执着方向。"
                    if nature == "akushala"
                    else "善恶相抵，方向不定，不生明显果报。"
                    if nature == "avyakrta"
                    else "无拓扑张力，拓扑平凡。"
                )
            ),
        }

    # ==================================================================
    # 1. 动态拓扑荷
    # ==================================================================

    def compute_dynamic_charge(self, g_history: list[Tensor] | Tensor) -> dict[str, Tensor | float]:
        """
        动态拓扑荷 Q_dynamic：从度规历史的本征向量旋转累积。

        Q_dynamic = (1/2π) Σ_k θ(O_k^T O_{k+1})    （无符号，总量）

        v7.8 新增 Q_signed：
        Q_signed = (1/2π) · Σ_{i>j} Γ[i,j]         （有符号，净方向）
        其中 Γ = Σ_k antisym(O_k^T · ΔO_k) 是累积 Berry 相位。

        物理意义：
            Q_dynamic = 总路径长度（善恶皆计，业力总量）
            Q_signed = 净位移（善恶相抵后的净方向）
            |Q_signed| ≤ Q_dynamic（三角不等式）

            高 κ 环境反复冲击 → 本征向量反复旋转 → Q_dynamic 大 → 业力重。
            低 κ 温室 → 本征向量几乎不变 → Q_dynamic 小 → 业力轻。

        参数：
            g_history: 度规历史 [g_0, g_1, ..., g_T]，每个 g_k ∈ R^{n×n}
                      或 Tensor (T, n, n)

        返回：
            dict 包含：
                Q_dynamic: 无符号总拓扑荷（业力总量）
                Q_signed: 有符号拓扑荷（业力净方向，v7.8 新增）
                Gamma: 累积 Berry 相位 ∈ so(n)（v7.8 新增）
                theta_sequence: 每步无符号旋转角序列
                cumulative_Q: 累积无符号拓扑荷序列
                max_rotation: 最大单步旋转
        """
        if isinstance(g_history, Tensor) and g_history.dim() == 3:
            g_list = [g_history[k] for k in range(g_history.shape[0])]
        else:
            g_list = list(g_history)

        T = len(g_list)
        if T < 2:
            return {
                "Q_dynamic": torch.tensor(0.0, dtype=torch.float64),
                "Q_signed": torch.tensor(0.0, dtype=torch.float64),
                "Gamma": torch.zeros(self.n_dims, self.n_dims, dtype=torch.float64),
                "theta_sequence": torch.tensor([], dtype=torch.float64),
                "cumulative_Q": torch.tensor([0.0], dtype=torch.float64),
                "max_rotation": torch.tensor(0.0, dtype=torch.float64),
                "n_steps": 0,
            }

        # 提取本征向量序列，使用相对基方法保证排列连续性
        O_list = []
        O_prev = None
        for g_k in g_list:
            if O_prev is not None:
                # 相对基方法：在前一帧基下提取，保证排列连续
                O_k = self._extract_eigenvectors_relative(g_k, O_prev)
            else:
                O_k = self._extract_eigenvectors(g_k)
            O_list.append(O_k)
            O_prev = O_k

        # 计算每步旋转角（无符号）
        theta_sequence = torch.zeros(T - 1, dtype=torch.float64)
        cumulative = torch.zeros(T, dtype=torch.float64)
        cumulative[0] = 0.0

        for k in range(T - 1):
            R_k = self._relative_rotation(O_list[k], O_list[k + 1])
            theta_k = self._rotation_angle(R_k)
            theta_sequence[k] = theta_k
            cumulative[k + 1] = cumulative[k] + theta_k

        # 总拓扑荷（无符号，归一化到 2π）
        Q_dynamic = cumulative[-1] / (2.0 * math.pi)

        # v7.8 新增：累积 Berry 相位与有符号拓扑荷
        Gamma = self._compute_cumulative_berry_phase(O_list)
        Q_signed = self._signed_angle_from_berry_phase(Gamma)

        return {
            "Q_dynamic": Q_dynamic,
            "Q_signed": Q_signed,
            "Gamma": Gamma,
            "theta_sequence": theta_sequence,
            "cumulative_Q": cumulative / (2.0 * math.pi),
            "max_rotation": theta_sequence.max() if T > 1 else torch.tensor(0.0, dtype=torch.float64),
            "n_steps": T - 1,
        }

    # ==================================================================
    # 2. 静态拓扑荷
    # ==================================================================

    def compute_static_charge(
        self, g_current: Tensor, g_baseline: Tensor | None = None
    ) -> dict[str, Tensor | float]:
        """
        静态拓扑荷 Q_static：当前度规相对于基线的方向偏离。

        Q_static = (1/2π) × θ(O_current^T O_baseline)

        物理意义：
            当前认知结构相对于「出厂设置」的方向性偏离。
            即使无历史，也可从当前结构判断拓扑复杂度。

        参数：
            g_current: 当前度规 ∈ R^{n×n}
            g_baseline: 基线度规（若 None，用真空 cI = I）

        返回：
            dict 包含 Q_static, rotation_angle, is_topologically_trivial
        """
        if g_baseline is None:
            g_baseline = torch.eye(self.n_dims, dtype=torch.float64)

        # 先提取基线本征向量
        O_baseline = self._extract_eigenvectors(g_baseline)

        # 用相对基方法提取当前本征向量，避免 eigh 排序导致的伪置换旋转。
        # 若 g_current 在 O_baseline 基下近对角（即无真实物理旋转），
        # _extract_eigenvectors_relative 直接返回 O_baseline，
        # 使 R = I, theta = 0, Q_static = 0 —— 这是物理正确的：
        # 对角度规的本征向量是标准基，排列顺序只是标签选择，非拓扑信息。
        O_current = self._extract_eigenvectors_relative(g_current, O_baseline)

        R = self._relative_rotation(O_baseline, O_current)
        theta = self._rotation_angle(R)
        Q_static = theta / (2.0 * math.pi)

        # 拓扑平凡判据：仅当旋转角接近 0 时拓扑平凡。
        # 注意：θ ≈ π 是 180° 旋转，在 SO(n)（n≥3）中是非平凡元素，
        # 不能误判为平凡。只有 θ ≈ 0（恒等旋转邻域）才是拓扑平凡。
        is_trivial = bool(theta < 0.1 * math.pi)

        return {
            "Q_static": Q_static,
            "rotation_angle": theta,
            "is_topologically_trivial": is_trivial,
            "O_current": O_current,
            "O_baseline": O_baseline,
        }

    # ==================================================================
    # 3. 残差方向绕数
    # ==================================================================

    def compute_residual_winding(
        self, residual_history: list[Tensor] | Tensor
    ) -> dict[str, Tensor | float]:
        """
        残差方向绕数 Q_winding：痛苦场方向 u(t) 在球面 S^{n-1} 上的绕数。

        Q_winding = (1/2π) Σ_k arccos(u_k · u_{k+1})

        物理意义：
            痛苦方向（残差方向）在认知空间中的「缠绕」程度。
            如果一个人的痛苦方向反复在不同维度间切换（价值观反复翻转），
            Q_winding 大 → 携带非平凡拓扑荷 → 有使命。
            如果痛苦方向始终稳定（单一维度的困扰），Q_winding 小 → 无使命。

        参数：
            residual_history: 残差向量历史 [r_0, r_1, ..., r_T]
                             或 Tensor (T, n)

        返回：
            dict 包含 Q_winding, total_arc_length, direction_changes
        """
        if isinstance(residual_history, Tensor) and residual_history.dim() == 2:
            r_list = [residual_history[k] for k in range(residual_history.shape[0])]
        else:
            r_list = list(residual_history)

        T = len(r_list)
        if T < 2:
            return {
                "Q_winding": torch.tensor(0.0, dtype=torch.float64),
                "total_arc_length": torch.tensor(0.0, dtype=torch.float64),
                "direction_changes": 0,
                "mean_angular_velocity": torch.tensor(0.0, dtype=torch.float64),
            }

        # 归一化为单位方向
        u_list = []
        for r_k in r_list:
            r = r_k.to(torch.float64)
            norm = torch.sqrt(torch.clamp(r @ r, min=self.eps))
            u_list.append(r / norm)

        # 计算相邻方向的角位移
        arc_lengths = torch.zeros(T - 1, dtype=torch.float64)
        for k in range(T - 1):
            cos_angle = torch.clamp(u_list[k] @ u_list[k + 1], min=-1.0 + self.eps, max=1.0 - self.eps)
            arc_lengths[k] = torch.arccos(cos_angle)

        total_arc = arc_lengths.sum()
        Q_winding = total_arc / (2.0 * math.pi)

        # 方向变化次数（角位移超过阈值）
        direction_changes = int((arc_lengths > 0.3 * math.pi).sum().item())

        # 平均角速度：当 T-1 == 0（即 T==1，已在前置分支返回）此处 T>=2 安全
        # 但 arc_lengths 可能在极端数值下为空，加保护避免 nan
        if len(arc_lengths) > 0 and not torch.isnan(arc_lengths.mean()):
            mean_angular_vel = arc_lengths.mean()
        else:
            mean_angular_vel = torch.tensor(0.0, dtype=torch.float64)

        return {
            "Q_winding": Q_winding,
            "total_arc_length": total_arc,
            "direction_changes": direction_changes,
            "mean_angular_velocity": mean_angular_vel,
            "arc_length_sequence": arc_lengths,
        }

    # ==================================================================
    # 4. 总拓扑荷
    # ==================================================================

    def compute_total_charge(
        self,
        g_history: list[Tensor] | Tensor | None = None,
        g_current: Tensor | None = None,
        g_baseline: Tensor | None = None,
        residual_history: list[Tensor] | Tensor | None = None,
        weights: tuple[float, float, float] = (1.0, 0.5, 0.5),
    ) -> dict[str, Tensor | float]:
        """
        总拓扑荷 Q_total = w1·Q_dynamic + w2·Q_static + w3·Q_winding

        v7.8 新增：Q_signed_total（有符号总拓扑荷）与业力三性分类。

        权重设计（非硬编码，基于物理重要性）：
            w1 = 1.0：动态拓扑荷最重要（直接度量演化缠绕）
            w2 = 0.5：静态拓扑荷补充（当前结构偏离）
            w3 = 0.5：残差绕数补充（痛苦方向缠绕）

        物理意义：
            Q_total（无符号）> Q_threshold：系统携带非零拓扑荷，有业力。
            Q_total ≈ 0：拓扑平凡，无业力，可沉睡于局部最优。
            Q_signed_total（v7.8 新增）：业力净方向（善/恶/无记）。

        返回：
            dict 包含 Q_total, Q_signed_total, 各分量, karma_nature, has_mission, mission_intensity
        """
        w1, w2, w3 = weights
        components = {}

        Q_dyn = torch.tensor(0.0, dtype=torch.float64)
        Q_signed_dyn = torch.tensor(0.0, dtype=torch.float64)
        Gamma_dyn = torch.zeros(self.n_dims, self.n_dims, dtype=torch.float64)
        if g_history is not None:
            dyn_result = self.compute_dynamic_charge(g_history)
            Q_dyn = dyn_result["Q_dynamic"]
            Q_signed_dyn = dyn_result.get("Q_signed", torch.tensor(0.0, dtype=torch.float64))
            Gamma_dyn = dyn_result.get("Gamma", torch.zeros(self.n_dims, self.n_dims, dtype=torch.float64))
            components["dynamic"] = dyn_result

        Q_sta = torch.tensor(0.0, dtype=torch.float64)
        if g_current is not None:
            sta_result = self.compute_static_charge(g_current, g_baseline)
            Q_sta = sta_result["Q_static"]
            components["static"] = sta_result

        Q_win = torch.tensor(0.0, dtype=torch.float64)
        if residual_history is not None:
            win_result = self.compute_residual_winding(residual_history)
            Q_win = win_result["Q_winding"]
            components["winding"] = win_result

        # 无符号总拓扑荷（业力总量）
        Q_total = w1 * Q_dyn + w2 * Q_sta + w3 * Q_win

        # v7.8 新增：有符号总拓扑荷（业力净方向）
        # 从动态 Berry 相位提取符号，加权到总量
        Q_signed_total = w1 * Q_signed_dyn  # 静态和绕数暂无有符号版本，只用动态

        # 业力三性分类
        karma = self.classify_karma_nature(Q_signed_total, Q_total)

        # 使命判据
        # Q_threshold = 0.1：经验阈值（总旋转超过 0.1 × 2π ≈ 36°）
        Q_threshold = 0.1
        has_mission = bool(Q_total.abs() > Q_threshold)

        # 使命强度：连续度量，非二元
        mission_intensity = torch.tanh(Q_total.abs() / Q_threshold)

        return {
            "Q_total": Q_total,
            "Q_signed_total": Q_signed_total,
            "Q_dynamic": Q_dyn,
            "Q_static": Q_sta,
            "Q_winding": Q_win,
            "Q_signed_dynamic": Q_signed_dyn,
            "Gamma": Gamma_dyn,
            "weights": weights,
            "has_mission": has_mission,
            "mission_intensity": float(mission_intensity),
            "karma_nature": karma,
            "components": components,
        }

    # ==================================================================
    # 5. 守恒律验证
    # ==================================================================

    def verify_conservation(
        self, g_history: list[Tensor] | Tensor, window: int = 5
    ) -> dict[str, Tensor | float | bool]:
        """
        验证拓扑荷守恒律：平滑演化下 dQ/dt ≈ 0。

        方法：
            在滑动窗口内计算局部 Q 的变化率。
            平滑演化：dQ/dt ≈ 0（每步旋转角小）
            拓扑相变：dQ/dt 发生跳变（旋转角突然增大）

        参数：
            g_history: 度规历史
            window: 滑动窗口大小

        返回：
            dict 包含 conservation_satisfied, phase_transition_points, mean_dQ_dt
        """
        if isinstance(g_history, Tensor) and g_history.dim() == 3:
            g_list = [g_history[k] for k in range(g_history.shape[0])]
        else:
            g_list = list(g_history)

        T = len(g_list)
        if T < window + 1:
            return {
                "conservation_satisfied": True,
                "phase_transition_points": [],
                "mean_dQ_dt": 0.0,
                "max_dQ_dt": 0.0,
            }

        # 计算每步旋转角（使用相对基方法保证排列连续性）
        O_list = []
        O_prev = None
        for g_k in g_list:
            if O_prev is not None:
                O_k = self._extract_eigenvectors_relative(g_k, O_prev)
            else:
                O_k = self._extract_eigenvectors(g_k)
            O_list.append(O_k)
            O_prev = O_k

        theta_seq = torch.zeros(T - 1, dtype=torch.float64)
        for k in range(T - 1):
            R_k = self._relative_rotation(O_list[k], O_list[k + 1])
            theta_seq[k] = self._rotation_angle(R_k)

        # 检测拓扑相变点：旋转角突增
        mean_theta = theta_seq.mean()
        std_theta = theta_seq.std() + self.eps
        threshold = mean_theta + 3.0 * std_theta  # 3σ 准则

        transition_points = []
        for k in range(len(theta_seq)):
            if theta_seq[k] > threshold and theta_seq[k] > 0.1 * math.pi:
                transition_points.append({
                    "step": k,
                    "theta": float(theta_seq[k]),
                    "is_phase_transition": True,
                })

        # 守恒判据：大部分步骤旋转角小（dQ/dt ≈ 0）
        # 排除相变点后的平均旋转角
        mask = torch.ones(len(theta_seq), dtype=torch.bool)
        for tp in transition_points:
            mask[tp["step"]] = False
        normal_thetas = theta_seq[mask]
        mean_dQ_dt = float(normal_thetas.mean()) if len(normal_thetas) > 0 else 0.0
        max_dQ_dt = float(theta_seq.max())

        # 守恒满足：正常步骤的平均旋转角 < 0.05π（9°）
        conservation_satisfied = mean_dQ_dt < 0.05 * math.pi

        return {
            "conservation_satisfied": conservation_satisfied,
            "phase_transition_points": transition_points,
            "mean_dQ_dt": mean_dQ_dt,
            "max_dQ_dt": max_dQ_dt,
            "theta_sequence": theta_seq,
            "transition_threshold": float(threshold),
        }

    # ==================================================================
    # 辅助：拓扑配对判据
    # ==================================================================

    def topological_pairing_compatibility(
        self, Q_a: Tensor, Q_b: Tensor
    ) -> dict[str, Tensor | float | bool]:
        """
        拓扑配对兼容性：两个个体的拓扑荷是否互补。

        v7.8 升级：支持有符号拓扑荷（Q_signed）的配对判定。
            若 Q_a, Q_b 是有符号的（v7.8 Q_signed）：
                互补条件：Q_a · Q_b < 0（异号）→ 业力可湮灭
                完美配对：Q_a + Q_b ≈ 0（配对湮灭，业力抵消）
            若 Q_a, Q_b 是无符号的（v7.0 Q_total）：
                互补条件永不满足（同非负），需配合 Γ 方向判定
                （此时应使用 gauge_interaction 的 berry_alignment）

        物理意义：
            异号拓扑荷的个体在规范场中相互排斥（Berry 相位反向）。
            但若 |Q_a + Q_b| ≈ 0，配对后总拓扑荷为零——
            业力抵消，系统可回归真空（解脱）。
            这是「业力互消」的拓扑起源——
            正负业力相遇可湮灭，类似正反物质配对。

        佛学对应：
            善业（Q>0）与恶业（Q<0）相遇 → 业力互消。
            但「善恶业不能直接抵消」——需经觉照（ρ→1）才能完全消解。
            这里的配对湮灭是拓扑层面的判据，
            实际消解仍需觉照路径（v7.3）。

        参数：
            Q_a, Q_b: 两个个体的拓扑荷（有符号或无符号）

        返回：
            dict 包含 is_complementary, pairing_strength, is_perfect_annihilation
        """
        Q_a = Q_a.to(torch.float64)
        Q_b = Q_b.to(torch.float64)

        product = Q_a * Q_b
        is_complementary = bool(product < 0)

        # 配对强度：|Q_a · Q_b| / (|Q_a| · |Q_b|)（归一化到 [0,1]）
        denom = Q_a.abs() * Q_b.abs() + self.eps
        pairing_strength = (-product / denom).clamp(min=0.0, max=1.0) if is_complementary else torch.tensor(0.0, dtype=torch.float64)

        # 完美湮灭判据：Q_a + Q_b ≈ 0
        Q_sum = Q_a + Q_b
        max_abs = max(Q_a.abs(), Q_b.abs())
        is_perfect = bool(Q_sum.abs() < 0.01 * float(max_abs + self.eps))

        # 配对后残余拓扑荷
        residual_Q = Q_sum

        return {
            "is_complementary": is_complementary,
            "pairing_strength": float(pairing_strength),
            "is_perfect_annihilation": is_perfect,
            "residual_Q": residual_Q,
            "Q_product": product,
        }
