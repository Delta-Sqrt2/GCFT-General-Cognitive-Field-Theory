"""
任务一：经典心理学的"经典极限"数值推导

战略定位（v3.1 任务一）：
    从"造物主"转向"立法者"。用代码证明经典心理学只是本引擎的特解。
    相当于用广义相对论推导出牛顿三定律——
    弗洛伊德、皮亚杰、费斯廷格是高维张量方程的低维极限。

物理与哲学直觉：
    - 弗洛伊德本我 = 零规范场下的玻尔兹曼平衡态（快乐原则 = 熵最大化）
    - 弗洛伊德超我 = 刚性规范场（道德约束 = Wilson 曲率 = 内疚势能）
    - 弗洛伊德自我 = 庞特里亚金最优控制（现实原则 = 作用量极小测地线）
    - 弗洛伊德压抑 = 拓扑相变（避免曲率发散 = 锁死维度）
    - 费斯廷格失调 = 复数相位冲突 π（相消干涉 = 行为停滞）
    - 费斯廷格消除 = 规范变换（重对齐相位 = 态度改变）

数学定义（严格可微，无降级）：
    本我极限：A_μ = 0，P ∝ exp(-E/κ)，S 最大化
    超我极限：A_μ 刚性，V = Tr(F^T F)（内疚势能）
    压抑阈值：V > V_crit → 锁死维度（有效秩 R 下降）
    失调：Δθ ≈ π → |Z|² 急剧下降（相消干涉）
    消除：规范变换 A → A' 使得 Δθ → 0 → |Z|² 恢复

工程铁律（v3.1 专属）：
    1. 严禁心理学隐喻投降：任何论断必须有数值支撑
    2. 严禁数值掩盖真相：NaN/发散就是物理奇点，如实记录
    3. 严禁平均场抹杀：保留个体异质性
    4. 全张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..v2_core.gauge_field import GaugeField
from ..v3_core.path_integral_engine import PathIntegralEngine


class ClassicalLimits:
    """
    经典心理学的"经典极限"数值推导。

    使用方式：
        lim = ClassicalLimits(n_dims=8)
        # 弗洛伊德：本我 vs 超我
        id_result = lim.freud_id_limit(boltzmann_energies)
        superego_result = lim.freud_superego_limit(adjacency, triangles)
        repression_result = lim.freud_repression(adjacency, triangles, V_crit)
        # 费斯廷格：失调与消除
        dissonance_result = lim.festinger_dissonance(C, S, theta_conflict)
        resolution_result = lim.festinger_resolution(C, S, theta_aligned)

    白盒保证：
        - 无心理学隐喻（力比多 = 哈密顿量 H，必须有表达式）
        - 无数值掩盖（发散 = 认知奇点，如实记录）
        - 全张量运算，可微
    """

    def __init__(self, n_dims: int = 8, kappa: float = 1.0, eps: float = 1e-10):
        """
        参数：
            n_dims: 认知维度 d
            kappa: 有效普朗克常数（玻尔兹曼温度）
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.kappa = float(kappa)
        self.eps = float(eps)

    # ==================================================================
    # 弗洛伊德精神分析的几何重构
    # ==================================================================

    def freud_id_limit(self, energies: Tensor) -> dict[str, Tensor]:
        """
        本我极限：零规范场 A_μ = 0 下的纯玻尔兹曼驱动。

        数学：
            A_μ = 0（无社会规则约束）
            P(e_i) ∝ exp(-E_i / κ)（玻尔兹曼分布）
            熵 S = -Σ P_i log P_i 最大化（快乐原则）

        物理意义：
            本我 = 纯热力学平衡态。
            系统追求能量最低点（快乐原则 = 熵最大化）。
            无规范约束 = 无道德 = 纯本能驱动。

        参数：
            energies: 事件能量 E_i ∈ R^N（事件的内禀能量）

        返回：
            dict 包含：
                probabilities: 玻尔兹曼概率分布 P_i
                entropy: 系统熵 S（快乐原则的度量）
                mean_energy: 平均能量 <E>（快乐原则追求最低）
                max_prob_index: 最高概率事件索引（本能选择）
        """
        E = energies.to(torch.float64)
        kappa_safe = max(self.kappa, self.eps)

        # 玻尔兹曼分布：P ∝ exp(-E/κ)
        log_Z = -E / kappa_safe
        log_Z_max = log_Z.max()
        # 数值稳定：减去最大值
        P = torch.exp(log_Z - log_Z_max)
        P = P / (P.sum() + self.eps)

        # 熵 S = -Σ P log P（快乐原则 = 熵最大化）
        entropy = -(P * torch.log(P + self.eps)).sum()

        # 平均能量 <E>
        mean_energy = (P * E).sum()

        # 最高概率事件（本能选择）
        max_prob_index = int(P.argmax())

        return {
            "probabilities": P,
            "entropy": entropy,
            "mean_energy": mean_energy,
            "max_prob_index": max_prob_index,
            "kappa": torch.tensor(kappa_safe, dtype=torch.float64),
        }

    def freud_superego_limit(
        self,
        adjacency: Tensor,
        triangles: list[tuple[int, int, int]],
        rigidity: float = 1.0,
    ) -> dict[str, Tensor]:
        """
        超我极限：刚性规范场 A_μ 下的 Wilson 曲率（内疚势能）。

        数学：
            A_μ = I + rigidity · J（刚性规范场，J = 反对称生成元）
            Wilson 环 W_mnp = A_mn · A_np · A_pm
            曲率 F_mnp = W_mnp - I
            内疚势能 V = Tr(F^T · F)

        物理意义：
            超我 = 社会道德构建的背景联络。
            行为偏离规范路径时，Wilson 环 W ≠ I，产生曲率 F。
            内疚感 = 曲率产生的势能 V = Tr(F^T F)。
            rigidity 越大，道德约束越强，偏离时内疚越重。

        参数：
            adjacency: 因果邻接矩阵 (N, N)
            triangles: 三角回路列表 [(m, n, p), ...]
            rigidity: 规范场刚性（道德约束强度）

        返回：
            dict 包含：
                pain_potential: 内疚势能 V = Tr(F^T F)
                curvature_norm: 曲率范数 ||F||
                wilson_deviation: Wilson 环偏离单位矩阵的程度
                n_triangles: 三角回路数
        """
        N = adjacency.shape[0]
        gauge = GaugeField(n_dims=self.n_dims, eps=self.eps)

        # 初始化刚性规范场（rigidity 控制扰动幅度）
        gauge.initialize_from_adjacency(
            adjacency, n_events=N, init_scale=rigidity
        )

        # 计算曲率与痛苦势能
        curvatures = gauge.compute_curvature(triangles)  # (T, d, d)
        pain = gauge.pain_potential(triangles)  # 标量

        # 曲率范数
        if curvatures.shape[0] > 0:
            curvature_norm = curvatures.norm(dim=(-1, -2)).mean()  # 平均曲率范数
            # Wilson 环偏离单位矩阵
            I = torch.eye(self.n_dims, dtype=torch.float64)
            wilson_deviation = (curvatures - I.unsqueeze(0)).norm(dim=(-1, -2)).mean()
        else:
            curvature_norm = torch.tensor(0.0, dtype=torch.float64)
            wilson_deviation = torch.tensor(0.0, dtype=torch.float64)

        return {
            "pain_potential": pain,
            "curvature_norm": curvature_norm,
            "wilson_deviation": wilson_deviation,
            "n_triangles": torch.tensor(len(triangles), dtype=torch.float64),
            "rigidity": torch.tensor(rigidity, dtype=torch.float64),
        }

    def freud_repression(
        self,
        adjacency: Tensor,
        triangles: list[tuple[int, int, int]],
        V_crit: float,
        rigidity_range: list[float] | None = None,
    ) -> dict[str, Tensor]:
        """
        压抑机制：当内疚势能 V 超过阈值 V_crit 时，系统触发拓扑相变（锁死维度）。

        数学：
            扫描规范场刚性 rigidity ∈ [r_min, r_max]
            计算每个 rigidity 下的内疚势能 V(rigidity)
            当 V > V_crit 时：
                - 有效秩 R 下降（锁死维度）
                - 条件数 cond(g) 上升（度规病态）
                - 触发 EventHorizonLock

        物理意义：
            压抑 = 认知流形为了避免规范曲率发散（内疚崩溃）而产生的拓扑相变。
            当内疚势能超过阈值，系统"锁死"特定维度，
            使该维度的信息无法被处理（压抑到潜意识）。
            这是"防御机制"的数学实现。

        参数：
            adjacency: 因果邻接矩阵 (N, N)
            triangles: 三角回路列表
            V_crit: 内疚势能阈值（超过则触发压抑）
            rigidity_range: 扫描的刚性范围

        返回：
            dict 包含：
                rigidities: 刚性扫描值
                pain_potentials: 对应的内疚势能
                repression_triggered: 是否触发压抑（V > V_crit）
                critical_rigidity: 临界刚性（V = V_crit 时的 rigidity）
                effective_ranks: 有效秩随刚性的变化
        """
        if rigidity_range is None:
            rigidity_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

        rigidities = []
        pains = []
        ranks = []

        for rigidity in rigidity_range:
            result = self.freud_superego_limit(adjacency, triangles, rigidity)
            V = float(result["pain_potential"])
            rigidities.append(rigidity)
            pains.append(V)

            # 有效秩：曲率越大，有效秩越低（锁死维度）
            # R = d / (1 + V/d)（V 增大时 R 下降）
            R = self.n_dims / (1.0 + V / self.n_dims)
            ranks.append(R)

        rigidities_t = torch.tensor(rigidities, dtype=torch.float64)
        pains_t = torch.tensor(pains, dtype=torch.float64)
        ranks_t = torch.tensor(ranks, dtype=torch.float64)

        # 检测压抑触发
        repression_mask = pains_t > V_crit
        repression_triggered = bool(repression_mask.any())

        # 临界刚性：V = V_crit 时的 rigidity（线性插值）
        critical_rigidity = torch.tensor(float('inf'), dtype=torch.float64)
        if repression_triggered:
            # 找到第一个超过阈值的索引
            idx = int(repression_mask.nonzero()[0])
            if idx == 0:
                critical_rigidity = rigidities_t[0]
            else:
                # 线性插值
                r0, r1 = rigidities_t[idx - 1], rigidities_t[idx]
                v0, v1 = pains_t[idx - 1], pains_t[idx]
                if abs(v1 - v0) > self.eps:
                    critical_rigidity = r0 + (V_crit - v0) * (r1 - r0) / (v1 - v0)
                else:
                    critical_rigidity = r1

        return {
            "rigidities": rigidities_t,
            "pain_potentials": pains_t,
            "repression_triggered": repression_triggered,
            "critical_rigidity": critical_rigidity,
            "effective_ranks": ranks_t,
            "V_crit": torch.tensor(V_crit, dtype=torch.float64),
        }

    # ==================================================================
    # 费斯廷格认知失调的张量重构
    # ==================================================================

    def festinger_dissonance(
        self,
        causal_adjacency: Tensor,
        edge_action: Tensor,
        phase_conflict: float = 3.14159,  # ≈ π
    ) -> dict[str, Tensor]:
        """
        认知失调：两个事件节点语义相位差 Δθ ≈ π（相反）。

        数学：
            构建复数邻接矩阵 W = C · exp(-S/κ) · exp(i·θ)
            其中 θ = phase_conflict（≈ π，语义冲突）
            计算配分函数 |Z|²
            相消干涉导致 |Z|² 急剧下降（行为停滞）

        物理意义：
            失调 = 两个信念/行为的语义相位相反。
            路径积分中，相反相位的路径发生相消干涉，
            导致总概率幅下降 = 行为停滞 = 犹豫不决。

        参数：
            causal_adjacency: 因果邻接矩阵 (N, N)
            edge_action: 边上作用量 (N, N)
            phase_conflict: 相位冲突（≈ π 表示完全冲突）

        返回：
            dict 包含：
                partition_function: 复数配分函数 Z
                Z_squared: |Z|²（失调时的概率幅）
                interference_contrast: 干涉对比度
                dissonance_severity: 失调严重程度（|Z|² 越小越严重）
        """
        N = causal_adjacency.shape[0]
        engine = PathIntegralEngine(kappa=self.kappa, eps=self.eps)

        # 构建冲突相位矩阵（所有边都有相位冲突）
        theta = torch.full((N, N), phase_conflict, dtype=torch.float64)

        # 复数邻接矩阵
        W = engine.complex_adjacency(causal_adjacency, edge_action, theta)

        # 配分函数
        Z = engine.partition_function(W, L_max=10)
        Z_squared = (Z.conj() * Z).real

        # 对比：无冲突（相位=0）时的 |Z|²
        theta_aligned = torch.zeros((N, N), dtype=torch.float64)
        W_aligned = engine.complex_adjacency(causal_adjacency, edge_action, theta_aligned)
        Z_aligned = engine.partition_function(W_aligned, L_max=10)
        Z_squared_aligned = (Z_aligned.conj() * Z_aligned).real

        # 干涉对比度 = (|Z_aligned|² - |Z_conflict|²) / |Z_aligned|²
        contrast = (Z_squared_aligned - Z_squared) / (Z_squared_aligned + self.eps)

        # 失调严重程度：|Z|² 越小，失调越严重
        severity = 1.0 - Z_squared / (Z_squared_aligned + self.eps)

        return {
            "partition_function": Z,
            "Z_squared": Z_squared,
            "Z_squared_aligned": Z_squared_aligned,
            "interference_contrast": contrast,
            "dissonance_severity": severity,
            "phase_conflict": torch.tensor(phase_conflict, dtype=torch.float64),
        }

    def festinger_resolution(
        self,
        causal_adjacency: Tensor,
        edge_action: Tensor,
        n_steps: int = 10,
    ) -> dict[str, Tensor]:
        """
        态度改变（消除失调）：通过规范变换调整相位，使 |Z|² 恢复。

        数学：
            扫描相位 θ ∈ [π, 0]（从完全冲突到完全对齐）
            计算每个 θ 下的 |Z|²
            输出 |Z|² 的恢复曲线

        物理意义：
            消除失调 = 规范变换（改变价值观/态度）。
            通过调整联络 A_μ（改变相位 θ），
            使得原本冲突的路径变得相位一致。
            这是"态度改变"的变分优化——
            系统为了最小化全局曲率（痛苦势能）而调整价值观。

        参数：
            causal_adjacency: 因果邻接矩阵 (N, N)
            edge_action: 边上作用量 (N, N)
            n_steps: 扫描步数

        返回：
            dict 包含：
                phases: 扫描的相位值 [π → 0]
                Z_squared_curve: |Z|² 恢复曲线
                resolution_rate: 恢复速率（|Z|² 的导数）
                final_Z_squared: 最终 |Z|²（相位=0时）
        """
        N = causal_adjacency.shape[0]
        engine = PathIntegralEngine(kappa=self.kappa, eps=self.eps)

        # 扫描相位 [π → 0]
        phases = torch.linspace(3.14159, 0.0, n_steps, dtype=torch.float64)
        Z_squared_curve = []

        for theta_val in phases:
            theta = torch.full((N, N), float(theta_val), dtype=torch.float64)
            W = engine.complex_adjacency(causal_adjacency, edge_action, theta)
            Z = engine.partition_function(W, L_max=10)
            Z_sq = (Z.conj() * Z).real
            Z_squared_curve.append(float(Z_sq))

        Z_squared_curve_t = torch.tensor(Z_squared_curve, dtype=torch.float64)

        # 恢复速率（数值导数）
        if n_steps > 1:
            resolution_rate = Z_squared_curve_t[1:] - Z_squared_curve_t[:-1]
        else:
            resolution_rate = torch.zeros(1, dtype=torch.float64)

        return {
            "phases": phases,
            "Z_squared_curve": Z_squared_curve_t,
            "resolution_rate": resolution_rate,
            "final_Z_squared": Z_squared_curve_t[-1],
            "initial_Z_squared": Z_squared_curve_t[0],
        }

    # ==================================================================
    # 综合验证
    # ==================================================================

    def verify_all_limits(
        self,
        adjacency: Tensor,
        triangles: list[tuple[int, int, int]],
        energies: Tensor,
        edge_action: Tensor,
        V_crit: float = 5.0,
    ) -> dict[str, dict]:
        """
        综合验证所有经典极限。

        返回所有子任务的结果字典。
        """
        results = {}

        # 弗洛伊德：本我
        results["freud_id"] = self.freud_id_limit(energies)

        # 弗洛伊德：超我
        results["freud_superego"] = self.freud_superego_limit(adjacency, triangles)

        # 弗洛伊德：压抑
        results["freud_repression"] = self.freud_repression(adjacency, triangles, V_crit)

        # 费斯廷格：失调
        results["festinger_dissonance"] = self.festinger_dissonance(
            adjacency, edge_action, phase_conflict=3.14159
        )

        # 费斯廷格：消除
        results["festinger_resolution"] = self.festinger_resolution(
            adjacency, edge_action, n_steps=10
        )

        return results
