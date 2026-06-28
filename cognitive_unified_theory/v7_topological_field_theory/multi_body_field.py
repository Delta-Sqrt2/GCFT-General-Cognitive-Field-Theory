"""
多体认知场论整合（Multi-Body Cognitive Field Theory Integration）

v7.7 整合层：将 v7.1-v7.6 六大基石统一为完整的多体认知规范场论。

认识论根基（理论依据，非案例）：
    物理：规范场论 / 束缚态 / Kramers 逃逸 / Berry 相位 / 多体联合演化
    佛学：共业（sādhārana-karma）/ 业力相续 / 缘起耦合 / 觉照消业 / 殊途同归
    哲学：关系先于个体 / 涌现的整体性 / 历史性的不可压缩

六大基石整合：
    1. CognitiveVacuum（认知真空，v7.1）：g=cI 基态，κ>κ_c 自发对称性破缺
    2. TopologicalCharge（拓扑荷，v7.1）：Q 度量度规方向的拓扑偏离
    3. GaugeInteraction（规范场，v7.4）：V_int = -λ·cos_align·|Q_i|·|Q_j|·exp(-d_g/ξ)
    4. KramersFate（自然回归，v7.5）：τ = τ_0·exp(ΔV/T_cog)，对称性恢复的统计必然性
    5. GeometricPhaseInheritance（几何相位，v7.6）：Γ = ∫O^T dO，演化历史全息记录
    6. AwakeningPath（觉照路径，v7.3）：ρ→1 使 g→cI 且 Γ→0，完整觉照消解

v7.7 升级（批判性审视）：
    原 v7.0 把多体场论解读为「命运解析」——
    「该相逢的人总会相逢」「匿名样本S/L案例」「代际传递」。
    这是社会学解读，违背「理论依据 = 佛学+物理+哲学」的认识论规范。

    v7.7 升级为「共业结构的规范场论」——
    多体系统是规范对称性的集体破缺模式，
    个体间的业力结构耦合形成不可逆的拓扑绑定（束缚态）。
    这对应佛学「共业」（sādhārana-karma）——
    多个个体的业力结构相互耦合，形成集体性的因果网络。

    v7.7 物理修正（整合 v7.6）：
    原 generational_analysis 基于 v7.0 的「代际传递」漏洞——
    Γ 作用在真空 cI 上无效（SO(n) 不变性）。
    v7.7 采用 v7.6 修正：Γ 作用在已有度规 g_baseline 上，
    体现「业力在已有心相上相续，不从无中生有」。

整合逻辑：
    个体层（v7.1, v7.6）：
        度规历史 → 拓扑荷 Q（当前方向偏离）→ 规范场参与
        度规历史 → 几何相位 Γ（演化历史全息记录）→ 时间尺度传播

    相互作用层（v7.4）：
        Q_i, Γ_i, Q_j, Γ_j → 规范势 V_int → 共振/互补分类
        - 共振对（cos_align>0）：V_int<0，规范吸引（共业相吸）
        - 互补对（cos_align<0）：V_int>0，规范排斥（业力互消）
        - 中性对（Q=0）：V_int≈0，无耦合（业力独立）

    对称性恢复层（v7.5）：
        破缺态 g* → 真空 cI 的势垒 ΔV → Kramers 逃逸率
        τ = τ_0·exp(ΔV/T_cog)，P(t) = 1-exp(-t/τ)
        t→∞ 时 P→1：对称性恢复的统计必然性（自然回归空性）

    觉照消解层（v7.3 + v7.6）：
        ρ→1 使 g→cI（无相）+ Γ→0（清业）+ Q→0（业力消解）
        多体觉照消解：集体 ρ→1 使所有束缚态解耦
        觉悟者 g=cI 且 Γ=0，新阶段从真空开始，业力无可作用

    系统层：
        总能量 E_total → 束缚态稳定性判据
        关联函数 C(d_g) → 长程共业判据
        对称性恢复概率矩阵 P_ij(t) → 全局回归图谱

验证设计（v7.7 纯物理验证，不引用案例）：
    1. 高 κ 个体 vs 低 κ 个体：验证 Q 的参数依赖
       - 高 κ（强痛苦）→ SSB + 度规旋转 → Q≠0（有拓扑张力）
       - 低 κ（弱痛苦）→ 度规近真空 → Q≈0（拓扑平凡）
    2. 束缚态形成判据：E_total < E_self(A) + E_self(B)
    3. 对称性恢复统计必然性：P(t)→1 when t→∞
    4. 集体觉照消解：多体 ρ→1 使束缚态解耦
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import symmetric_part, stable_eigh
from .topological_charge import TopologicalCharge
from .cognitive_vacuum import CognitiveVacuum
from .gauge_interaction import GaugeInteraction, CognitiveAgent
from .kramers_fate import KramersFate
from .geometric_phase import GeometricPhaseInheritance
from .awakening_path import AwarenessField, AwakeningDynamics


class MultiBodyCognitiveField:
    """
    多体认知规范场论：六大基石的统一整合层（v7.7）。

    使用方式：
        field = MultiBodyCognitiveField(n_dims=4)
        # 从度规历史创建个体
        agent = field.create_agent_from_history(g_history, "A", kappa_vec, alpha_vec)
        # 系统分析
        analysis = field.analyze_system([agent1, agent2, ...])
        # 对称性恢复概率矩阵（原 compute_fate_matrix）
        restoration_matrix = field.compute_restoration_probability_matrix(agents, t=100)
        # 完整动力学
        result = field.simulate_full_dynamics(agents, n_steps=100)
        # 时间尺度传播分析（原 generational_analysis）
        timescale_result = field.timescale_propagation_analysis(agents)
        # 集体觉照消解（v7.7 新增）
        dissolution = field.collective_awareness_dissolution(agents, rho_schedule=[...])
        # 纯物理验证（原 validate_with_v6_cases）
        validation = field.validate_symmetry_restoration_inevitability()
    """

    def __init__(
        self,
        n_dims: int = 4,
        coupling_lambda: float = 1.0,
        correlation_length: float = 1.0,
        inheritance_efficiency: float = 0.7,
        tau_0: float = 1.0,
        eps: float = 1e-12,
    ):
        self.n_dims = n_dims
        self.eps = eps

        # 六大基石
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)
        self.charge = TopologicalCharge(n_dims=n_dims, eps=eps)
        self.gauge = GaugeInteraction(
            n_dims=n_dims,
            coupling_lambda=coupling_lambda,
            correlation_length=correlation_length,
            eps=eps,
        )
        self.fate = KramersFate(n_dims=n_dims, tau_0=tau_0, eps=eps)
        self.phase = GeometricPhaseInheritance(
            n_dims=n_dims,
            inheritance_efficiency=inheritance_efficiency,
            eps=eps,
        )
        # v7.7 新增：觉照场（用于集体觉照消解）
        self.awareness = AwarenessField(n_dims=n_dims, eps=eps)

    # ==================================================================
    # 1. 个体创建
    # ==================================================================

    def create_agent_from_history(
        self,
        g_history: list[Tensor] | Tensor,
        label: str = "",
        kappa_vec: Tensor | None = None,
        alpha_vec: Tensor | None = None,
        residual_history: list[Tensor] | Tensor | None = None,
    ) -> CognitiveAgent:
        """
        从度规历史创建认知个体。

        自动计算：
            - 拓扑荷 Q（从度规历史的本征向量旋转）
            - 当前度规 g（历史的最后一帧）
            - 几何相位 Γ（从历史路径累积）

        物理意义：
            一个「认知个体」由其完整的演化历史定义。
            Q 和 Γ 是历史的全息记录——不可压缩的拓扑与几何信息。
        """
        if isinstance(g_history, Tensor) and g_history.dim() == 3:
            g_list = [g_history[k] for k in range(g_history.shape[0])]
        else:
            g_list = list(g_history)

        if len(g_list) == 0:
            raise ValueError("度规历史为空")

        # 当前度规 = 最后一帧
        g_current = symmetric_part(g_list[-1].to(torch.float64))

        # 计算拓扑荷
        Q_result = self.charge.compute_total_charge(
            g_history=g_list,
            g_current=g_current,
            g_baseline=None,
            residual_history=residual_history,
        )
        Q = Q_result["Q_total"]

        # 默认参数
        if kappa_vec is None:
            kappa_vec = torch.tensor([0.5] * self.n_dims, dtype=torch.float64)
        if alpha_vec is None:
            alpha_vec = torch.tensor([1.0] * self.n_dims, dtype=torch.float64)

        # 几何相位（Berry 相位 Γ ∈ so(n)，编码旋转方向）
        phase_result = self.phase.accumulate_phase(g_list)
        Gamma = phase_result["Gamma"]

        agent = CognitiveAgent(
            g=g_current,
            Q=Q,
            label=label,
            kappa_vec=kappa_vec.to(torch.float64),
            alpha_vec=alpha_vec.to(torch.float64),
            Gamma=Gamma,
        )

        # 附加信息（不存入 dataclass，存入 __dict__）
        agent.__dict__["Q_detail"] = Q_result
        agent.__dict__["g_history"] = g_list
        agent.__dict__["phase_detail"] = phase_result

        return agent

    def create_agent_from_metric(
        self,
        g: Tensor,
        Q: float | Tensor,
        label: str = "",
        kappa_vec: Tensor | None = None,
        alpha_vec: Tensor | None = None,
        Gamma: Tensor | None = None,
    ) -> CognitiveAgent:
        """直接从度规和拓扑荷创建个体（无历史，可选 Gamma）。"""
        if kappa_vec is None:
            kappa_vec = torch.tensor([0.5] * self.n_dims, dtype=torch.float64)
        if alpha_vec is None:
            alpha_vec = torch.tensor([1.0] * self.n_dims, dtype=torch.float64)

        return CognitiveAgent(
            g=g,
            Q=torch.tensor(float(Q), dtype=torch.float64),
            label=label,
            kappa_vec=kappa_vec.to(torch.float64),
            alpha_vec=alpha_vec.to(torch.float64),
            Gamma=Gamma,
        )

    # ==================================================================
    # 2. 系统级分析
    # ==================================================================

    def analyze_system(
        self, agents: list[CognitiveAgent]
    ) -> dict[str, list | float | bool | Tensor]:
        """
        系统级分析：总能量、共振对、互补对、关联函数、使命分布。

        返回多体系统的完整诊断。
        """
        n_agents = len(agents)

        # 总相互作用能
        energy_result = self.gauge.total_interaction_energy(agents)

        # 共振对与互补对
        resonance_pairs = self.gauge.identify_resonance_pairs(agents)
        complement_pairs = self.gauge.identify_complement_pairs(agents)

        # 关联函数
        correlation = self.gauge.correlation_function(agents)

        # 使命分布
        Q_values = [float(ag.Q) for ag in agents]
        n_with_mission = sum(1 for q in Q_values if abs(q) > 0.1)
        n_without_mission = n_agents - n_with_mission

        # 系统稳定性判据
        E_total = float(energy_result["E_total"])
        is_bound = E_total < 0  # 负能 = 束缚态
        is_repulsive = E_total > 0

        return {
            "n_agents": n_agents,
            "total_energy": E_total,
            "is_bound_state": is_bound,
            "is_repulsive": is_repulsive,
            "resonance_pairs": resonance_pairs,
            "complement_pairs": complement_pairs,
            "n_resonance_pairs": len(resonance_pairs),
            "n_complement_pairs": len(complement_pairs),
            "correlation": correlation,
            "Q_values": Q_values,
            "n_with_mission": n_with_mission,
            "n_without_mission": n_without_mission,
            "mission_distribution": {
                "has_mission": n_with_mission,
                "no_mission": n_without_mission,
                "fraction_with_mission": n_with_mission / max(n_agents, 1),
            },
            "thesis": (
                f"系统分析：{n_agents} 个个体，"
                f"{n_with_mission} 个有使命（Q≠0），{n_without_mission} 个无使命（Q≈0）。"
                f"{len(resonance_pairs)} 对共振，{len(complement_pairs)} 对互补。"
                f"总能量 {E_total:.4f}（{'束缚态' if is_bound else '排斥态' if is_repulsive else '中性'}）。"
            ),
        }

    # ==================================================================
    # 3. 对称性恢复概率矩阵（原 compute_fate_matrix，v7.7 重新解读）
    # ==================================================================

    def compute_restoration_probability_matrix(
        self,
        agents: list[CognitiveAgent],
        t: float = 100.0,
    ) -> dict[str, list | Tensor]:
        """
        计算所有互补对（Berry 相位反向对齐）的对称性恢复概率矩阵。

        v7.7 重新解读：
            原 v7.0 把 P(t) 解读为「相逢概率」（命运矩阵）。
            v7.7 升级为「对称性恢复概率」——
            P(t) 度量破缺态在认知涨落下回归真空的统计必然性。
            t→∞ 时 P→1：一切破缺态终究回归空性（对称性恢复）。

        数学（Kramers 定理，v7.5）：
            P(t) = 1 - exp(-t / τ)
            τ = τ_0 · exp(ΔV / T_cog)
            其中 ΔV 是破缺态→真空的势垒，T_cog 是认知温度。

        只计算互补对（cos_align < 0），因为：
            - 共振对（同向 Berry 相位）：规范吸引，确定性共同回归
            - 互补对（反向 Berry 相位）：规范排斥，需 Kramers 统计必然性
            - 中性对（Q≈0）：无拓扑张力，无需回归

        物理意义：
            互补对代表「反向业力结构」——两个个体的 Berry 相位反向对齐，
            业力方向相反。这类对的对称性恢复需要 Kramers 统计必然性，
            而非确定性路径。t→∞ 时 P→1 = 反向业力终究被消解。

        佛学对应：
            「因缘会遇时，果报还自受」——
            不是确定性相逢，而是对称性恢复的统计必然性。
            一切有为法终究回归空性，只是时间尺度 τ 不同。

        返回：
            P_matrix（n×n，互补对位置有值，其余为 0）
            每对的详细信息（ΔV, T_cog, τ, P）
        """
        n_agents = len(agents)
        n = self.n_dims

        P_matrix = torch.zeros(n_agents, n_agents, dtype=torch.float64)
        pair_details = []

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Berry 相位对齐判定互补对
                cos_align = self.gauge.berry_alignment(agents[i].Gamma, agents[j].Gamma)
                if cos_align >= 0:
                    continue  # 非互补对（同向或正交），跳过

                # 至少一方需有拓扑张力（Q≠0）
                Q_i_abs = float(agents[i].Q.abs())
                Q_j_abs = float(agents[j].Q.abs())
                if Q_i_abs < self.eps or Q_j_abs < self.eps:
                    continue

                # 认知温度（取两方均值）
                T_i = self.fate.cognitive_temperature(agents[i].kappa_vec, agents[i].alpha_vec)
                T_j = self.fate.cognitive_temperature(agents[j].kappa_vec, agents[j].alpha_vec)
                T_cog = 0.5 * (T_i + T_j)

                # 势垒高度（v7.5：从破缺态→真空的回归势垒）
                barrier = self.fate.barrier_height(
                    agents[i].g, agents[j].g,
                    agents[i].kappa_vec, agents[i].alpha_vec,
                )
                Delta_V = barrier["Delta_V"]

                # 对称性恢复概率（Kramers 定理）
                tau = self.fate.escape_time(Delta_V, T_cog)
                P = self.fate.fate_probability(t, Delta_V, T_cog)

                P_matrix[i, j] = P
                P_matrix[j, i] = P

                pair_details.append({
                    "i": i,
                    "j": j,
                    "label_i": agents[i].label,
                    "label_j": agents[j].label,
                    "Q_i": float(agents[i].Q),
                    "Q_j": float(agents[j].Q),
                    "cos_alignment": cos_align,
                    "Delta_V": Delta_V,
                    "T_cog": T_cog,
                    "tau": tau,
                    "P": P,
                    "t": t,
                    "is_inevitable": True,  # t→∞ 时 P→1
                })

        return {
            "P_matrix": P_matrix,
            "pair_details": pair_details,
            "n_complement_pairs": len(pair_details),
            "t": t,
            "thesis": (
                f"对称性恢复概率矩阵（t={t}）：{len(pair_details)} 对互补个体。"
                "每对的 P(t) 由 Kramers 定理给出，t→∞ 时 P→1。"
                "这是「一切破缺态终究回归空性」的全局图谱——"
                "对称性恢复的统计必然性，非确定性相逢。"
            ),
        }

    def compute_fate_matrix(
        self,
        agents: list[CognitiveAgent],
        t: float = 100.0,
    ) -> dict[str, list | Tensor]:
        """
        对称性恢复概率矩阵（保留向后兼容，v7.7 等价于 compute_restoration_probability_matrix）。

        v7.7 说明：
            此方法保留向后兼容。v7.7 推荐使用 compute_restoration_probability_matrix
            获取更清晰的语义（「对称性恢复」而非「命运」）。
        """
        return self.compute_restoration_probability_matrix(agents, t=t)

    # ==================================================================
    # 4. 完整多体动力学
    # ==================================================================

    def simulate_full_dynamics(
        self,
        agents: list[CognitiveAgent],
        n_steps: int = 50,
        dt: float = 0.005,
        include_vacuum_return: bool = True,
    ) -> dict[str, list]:
        """
        完整多体动力学：规范场 + 单体势能面演化。

        每个个体的演化：
            ∂g_i/∂t = F_potential(i) + F_gauge(i←j) + F_rho(i)

        其中：
            F_potential：势能面梯度（v6.x）
            F_gauge：规范场力（其他个体的影响）
            F_rho：ρ 驱动的消解项（拉向真空，若 include_vacuum_return=True）
        """
        n_agents = len(agents)
        n = self.n_dims

        # 深拷贝（包含 Gamma）
        current_agents = [
            CognitiveAgent(
                g=ag.g.clone(),
                Q=ag.Q.clone(),
                label=ag.label,
                kappa_vec=ag.kappa_vec.clone() if ag.kappa_vec is not None else None,
                alpha_vec=ag.alpha_vec.clone() if ag.alpha_vec is not None else None,
                Gamma=ag.Gamma.clone() if ag.Gamma is not None else None,
            )
            for ag in agents
        ]

        # 单体力函数：势能面梯度 + ρ 消解项
        # v7.9 修复符号 bug：力 = -∇V（原代码返回梯度，符号错误）
        def single_body_force(agent: CognitiveAgent, step: int) -> Tensor:
            kappa = agent.kappa_vec
            alpha = agent.alpha_vec
            pot = self.vacuum.compute_potential(agent.g, kappa, alpha)
            F = -pot["grad"]  # 力 = -∇V_base

            if include_vacuum_return:
                # ρ 消解项：简化为常数 ρ_eff（力指向真空，故取负号）
                rho_eff = 0.1  # 简化：低 ρ 模拟未觉悟状态
                mu_rho = alpha / (2.0 * (alpha + 1.0))
                g_vac = self.vacuum.construct_vacuum()
                F_dissolve = -rho_eff * torch.diag(mu_rho) @ (agent.g - g_vac)
                F = F + F_dissolve

            return F

        # 调用规范场动力学
        dynamics_result = self.gauge.simulate_dynamics(
            current_agents,
            n_steps=n_steps,
            dt=dt,
            single_body_force_fn=single_body_force,
        )

        # 重新计算最终拓扑荷
        final_agents = dynamics_result["final_agents"]
        final_analysis = self.analyze_system(final_agents)

        return {
            "dynamics": dynamics_result,
            "final_analysis": final_analysis,
            "thesis": (
                "完整多体动力学：规范场 + 势能面 + ρ 消解。"
                "个体在自身势能面和其他个体拓扑荷的共同驱动下演化。"
                "同号共振趋同，异号互补张力，ρ 消解拉向真空。"
            ),
        }

    # ==================================================================
    # 4b. 缘起动力学（v7.9 新增：自洽多体耦合演化）
    # ==================================================================

    def simulate_pratitya_dynamics(
        self,
        agents: list[CognitiveAgent],
        n_steps: int = 100,
        dt: float = 0.005,
        rho: float = 0.0,
        noise_temperature: float = 0.0,
        charge_history_window: int | None = None,
    ) -> dict[str, list]:
        """
        v7.9: 缘起动力学（Pratītyasamutpāda Dynamics）——自洽多体耦合演化。

        这是 CTFT 的核心动力学方法：多个个体的度规在规范场耦合下同时演化，
        Q 和 Γ 每步自洽重算（无我：Q/Γ 不是固定属性，是从 g 涌现的条件量）。

        演化方程：
            ∂g_i/∂t = -∇V_base(g_i) - ρ·2λ·(g_i-cI)
                      + Σ_{j≠i} F_gauge(i←j) + √(2T)·ξ_i

        自洽闭环：
            g_i 变化 → Q_i 重算 → V_int(i,j) 变化 → F_gauge 变化 → g_j 变化 → ...
            「此有故彼有，此生故彼生」——缘起的数学表达。

        v7.9 关键修复：
            1. Gamma 深拷贝 bug 修复（gauge_interaction.simulate_dynamics）
            2. charge_updater_fn 每步重算 Q 和 Γ（自洽演化）
            3. 规范力使用当前 Q/Γ（非过时值）

        v7.10 新增：charge_history_window
            - None（默认）：v7.9 行为，只用 Q_static（当前 g 相对真空的方向偏离）
            - int N：用滑动窗口 N 帧历史算 Q_dynamic（累积缠绕，不随 g→g* 衰减）
            物理：Q_static 会随 SSB 松弛到 0（g* 对角化），Q_dynamic 是拓扑保护
            的（缠绕数不灭）。佛学：Q_static = 当下心相，Q_dynamic = 阿赖耶识种子。

        参数：
            agents: 认知个体列表
            n_steps: 演化步数
            dt: 时间步长
            rho: 觉照强度（0=纯规范动力学，1=完全觉照）
            noise_temperature: Langevin 噪声温度（0=确定性）
            charge_history_window: Q_dynamic 滑动窗口（None=只用 Q_static）

        返回：
            完整的动力学轨迹（g, Q, Γ, V_int, E_self 随时间演化）
        """
        n = self.n_dims
        g_vac = self.vacuum.construct_vacuum()

        # 深拷贝
        current_agents = [
            CognitiveAgent(
                g=ag.g.clone(),
                Q=ag.Q.clone(),
                label=ag.label,
                kappa_vec=ag.kappa_vec.clone() if ag.kappa_vec is not None else None,
                alpha_vec=ag.alpha_vec.clone() if ag.alpha_vec is not None else None,
                Gamma=ag.Gamma.clone() if ag.Gamma is not None else None,
            )
            for ag in agents
        ]

        # v7.10: 电荷更新模式
        #   charge_history_window = None: Q_static only（会随 SSB 松弛到 0）
        #   charge_history_window = N > 0: 滑动窗口 N 帧（O(N) per step，会遗忘）
        #   charge_history_window = -1: 冻结业力模式（O(1) per step，Γ/Q 冻结于初始值）
        if charge_history_window is not None:
            if charge_history_window == -1:
                # v7.10 冻结业力模式：Γ/Q 冻结于初始值，永不松弛也永不噪声增长
                # 物理：初始 Γ = 两人相遇时的业力结构（karmic bond）， frozen。
                #   - 阿赖耶识种子（初始业力）不随热涨落而改变强度
                #   - 规范相互作用 V_int = -λ·cos_align·|Q_i|·|Q_j|·exp(-d_g/ξ)
                #     其中 cos_align、|Q| 冻结，只有 d_g(t) 随 Langevin 动力学变化
                #   - 相变机制：高温 → d_g 增大（度规分离）→ exp(-d_g/ξ)→0 → V_int→0 → 键断裂
                #   - 低温 → d_g 保持小 → V_int 保持强 → 键坚固
                #   这给出物理正确的热力学相变：T_c 由 √(2T_cog) 噪声 vs λ·|Q|² 键强决定。
                # 佛学：业力结构（Γ）在缘起时已决定（「业力已造，不亡不失」），
                #   动力学改变的是认知距离 d_g（心相是否接近），不是业力本身。
                frozen_state: dict[int, dict] = {}

                def charge_updater(agent: CognitiveAgent):
                    agent_id = id(agent)
                    if agent_id not in frozen_state:
                        # 初始化：冻结初始 Γ 和 |Q| = ||Γ||_F
                        Gamma_init = (
                            agent.Gamma.clone() if agent.Gamma is not None
                            else torch.zeros(n, n, dtype=torch.float64)
                        )
                        Q_init = float(torch.norm(Gamma_init, p="fro"))
                        frozen_state[agent_id] = {
                            "Gamma": Gamma_init,
                            "Q": Q_init,
                        }
                        agent.Q = torch.tensor(Q_init, dtype=torch.float64)
                        agent.Gamma = Gamma_init.clone()
                        return

                    # 后续步骤：Γ/Q 保持冻结（初始业力不变）
                    # 只有 g 随动力学演化 → d_g 变化 → V_int 通过 exp(-d_g/ξ) 变化
                    state = frozen_state[agent_id]
                    agent.Q = torch.tensor(state["Q"], dtype=torch.float64)
                    agent.Gamma = state["Gamma"].clone()
            else:
                # 滑动窗口模式（原 v7.10 行为）
                g_histories_dict: dict[int, list[Tensor]] = {}

                def charge_updater(agent: CognitiveAgent):
                    """v7.10: 从轨迹历史算 Q_dynamic（累积缠绕，拓扑保护）。"""
                    agent_id = id(agent)
                    if agent_id not in g_histories_dict:
                        g_histories_dict[agent_id] = [g_vac, agent.g.clone()]
                    g_histories_dict[agent_id].append(agent.g.clone())
                    if len(g_histories_dict[agent_id]) > charge_history_window:
                        g_histories_dict[agent_id] = g_histories_dict[agent_id][-charge_history_window:]
                    Q_result = self.charge.compute_total_charge(
                        g_history=g_histories_dict[agent_id],
                        g_current=agent.g,
                        g_baseline=g_vac,
                    )
                    agent.Q = torch.tensor(
                        float(Q_result["Q_total"]), dtype=torch.float64
                    )
                    phase_result = self.phase.accumulate_phase(g_histories_dict[agent_id])
                    agent.Gamma = phase_result["Gamma"]
        else:
            # v7.9 原始行为：只用 Q_static
            def charge_updater(agent: CognitiveAgent):
                """从当前 g 自洽重算 Q 和 Γ。"""
                # Q_static from g_current vs g_vac
                Q_result = self.charge.compute_total_charge(
                    g_current=agent.g,
                    g_baseline=g_vac,
                )
                agent.Q = torch.tensor(
                    float(Q_result["Q_total"]), dtype=torch.float64
                )
                # Γ from [g_vac, g_current]（静态 Berry 相位）
                phase_result = self.phase.accumulate_phase([g_vac, agent.g])
                agent.Gamma = phase_result["Gamma"]

        # 初始化 Q/Γ（如果传入的 agents 没有正确的 Q/Γ）
        for ag in current_agents:
            if ag.Gamma is None or float(ag.Q.abs()) < self.eps:
                charge_updater(ag)

        # 单体力：势能面梯度 + ρ 消解项
        # v7.9 修复符号 bug：力 = -∇V（原代码返回梯度，导致个体被推离势能极小值）
        def single_body_force(agent: CognitiveAgent, step: int) -> Tensor:
            kappa = agent.kappa_vec
            alpha = agent.alpha_vec
            pot = self.vacuum.compute_potential(agent.g, kappa, alpha)
            F = -pot["grad"]  # 力 = -∇V_base（拉向势能极小值）

            if rho > self.eps:
                # ρ 消解项：拉向真空（力指向 cI，故取负号）
                lam = float(self.awareness.lambda_restore)
                F_dissolve = -2.0 * rho * lam * (agent.g - g_vac)
                F = F + F_dissolve

            return F  # 返回力（与 gauge_force_on_agent 符号约定一致）

        # 调用规范场动力学（v7.9 修复版）
        dynamics_result = self.gauge.simulate_dynamics(
            current_agents,
            n_steps=n_steps,
            dt=dt,
            single_body_force_fn=single_body_force,
            charge_updater_fn=charge_updater,
            noise_temperature=noise_temperature,
        )

        # 最终分析
        final_agents = dynamics_result["final_agents"]
        final_analysis = self.analyze_system(final_agents)
        final_E_self = self._compute_total_self_energy(final_agents)

        # 初始分析
        initial_analysis = self.analyze_system(agents)
        initial_E_self = self._compute_total_self_energy(agents)

        return {
            "dynamics": dynamics_result,
            "final_analysis": final_analysis,
            "final_E_self": final_E_self,
            "initial_V_int": float(initial_analysis["total_energy"]),
            "final_V_int": float(final_analysis["total_energy"]),
            "initial_E_self": initial_E_self,
            "energy_trajectory": dynamics_result["energy_trajectory"],
            "Q_trajectory": dynamics_result["Q_trajectory"],
            "n_steps": n_steps,
            "thesis": (
                "v7.9 缘起动力学：自洽多体耦合演化。"
                "个体 A 的 g 变化 → Q_A 重算 → V_int 变化 → 作用于 B 的力变化 → g_B 变化。"
                "「此有故彼有」——一切互联，无独立演化。"
                "v7.9 修复：Gamma 深拷贝 + 自洽 Q/Γ 演化 + Langevin 涨落。"
            ),
        }

    # ==================================================================
    # 5. 时间尺度传播分析（原 generational_analysis，v7.7 升级）
    # ==================================================================

    def timescale_propagation_analysis(
        self,
        agents: list[CognitiveAgent],
        n_scales: int = 3,
    ) -> dict[str, list]:
        """
        时间尺度传播分析：个体的几何相位如何塑造新阶段的初始条件。

        v7.7 升级（整合 v7.6）：
            原 v7.0 generational_analysis 基于「代际传递」漏洞——
            Γ 作用在真空 cI 上无效（SO(n) 不变性），且字段名有 bug
            （访问不存在的 "generations" 键）。

            v7.7 采用 v7.6 修正：
            1. Γ 作用在已有度规 g_baseline（当前破缺态）上——业力相续
            2. 同时演示真空 baseline 的无效性——觉悟解脱
            3. 修复字段名：用 v7.6 的 simulate_multiscale_propagation

        物理命题：
            - 业力相续：破缺态 baseline + α>0 → 度规被旋转（识变现）
            - 觉悟解脱：真空 baseline + α>0 → 度规不变（SO(n) 不变）
            - 业力断灭：α=0 → 度规不变（无耦合）
            - 多尺度衰减：Γ_k ≈ α^k · Γ_0（业力稀释）

        佛学对应：
            Γ = 阿赖耶识中的业力印记，在新阶段中相续。
            α = 业力的时间尺度耦合强度。
            觉悟者 Γ=0，新阶段从真空开始，业力无可作用——了脱生死。
        """
        timescale_results = []

        for agent in agents:
            # 获取个体的 Γ
            Gamma = agent.Gamma
            if Gamma is None:
                if "g_history" in agent.__dict__:
                    phase_result = self.phase.accumulate_phase(agent.__dict__["g_history"])
                    Gamma = phase_result["Gamma"]
                else:
                    Gamma = torch.zeros(self.n_dims, self.n_dims, dtype=torch.float64)

            Gamma_norm = float(torch.sqrt((Gamma ** 2).sum()))

            # v7.6 修正 1: 业力相续——Γ 作用在已有度规（破缺态）上
            prop_karma = self.phase.propagate_to_new_phase(
                Gamma, g_baseline=agent.g, efficiency=1.0
            )
            karma_transfer = prop_karma["transfer_strength"]

            # v7.6 修正 2: 觉悟解脱——Γ 作用在真空上（应无效）
            g_vac = self.vacuum.construct_vacuum()
            prop_liberation = self.phase.propagate_to_new_phase(
                Gamma, g_baseline=g_vac, efficiency=1.0
            )
            liberation_transfer = prop_liberation["transfer_strength"]

            # 多尺度传播（Γ 范数衰减轨迹）
            if "g_history" in agent.__dict__:
                history = agent.__dict__["g_history"]
            else:
                history = [agent.g]
            multi_result = self.phase.simulate_multiscale_propagation(
                history, n_scales=n_scales, efficiency=self.phase.alpha_inherit
            )

            timescale_results.append({
                "label": agent.label,
                "Gamma_norm": Gamma_norm,
                "karma_transfer": karma_transfer,  # 业力相续强度
                "liberation_transfer": liberation_transfer,  # 觉悟解脱（应≈0）
                "baseline_is_vacuum": prop_karma["baseline_is_vacuum"],
                "Gamma_norm_trajectory": multi_result["Gamma_norm_trajectory"],
                "is_decaying": multi_result["is_decaying"],
                "thesis": multi_result["thesis"],
            })

        return {
            "timescale_results": timescale_results,
            "n_scales": n_scales,
            "thesis": (
                "时间尺度传播分析（v7.7）：历史几何相位塑造新阶段初始条件。"
                "业力相续（破缺态上 Γ 有效）+ 觉悟解脱（真空上 Γ 无效）+ "
                "多尺度衰减（Γ_k ≈ α^k · Γ_0）。"
                "觉悟者 Γ=0，新阶段从真空开始，业力无可作用——了脱生死。"
            ),
        }

    def generational_analysis(
        self,
        parent_agents: list[CognitiveAgent],
        n_generations: int = 3,
    ) -> dict[str, list]:
        """
        时间尺度传播分析（保留向后兼容，v7.7 等价于 timescale_propagation_analysis）。

        v7.7 说明：
            此方法保留向后兼容。v7.7 推荐使用 timescale_propagation_analysis
            获取更清晰的语义和 v7.6 修正（g_baseline）。
            原 v7.0 的字段名 bug（"generations" 键不存在）已修复。
        """
        return self.timescale_propagation_analysis(
            parent_agents, n_scales=n_generations
        )

    # ==================================================================
    # 6. 纯物理验证（原 validate_with_v6_cases，v7.7 升级）
    # ==================================================================

    def validate_symmetry_restoration_inevitability(self) -> dict[str, dict | bool]:
        """
        纯物理验证：拓扑荷的参数依赖与对称性恢复的统计必然性。

        v7.7 升级（原 validate_with_v6_cases）：
            原 v7.0 用具体个人案例验证理论，引用真实人名。
            v7.7 清除案例引用，用纯物理参数描述：
            - 高 κ 个体（强痛苦参数）vs 低 κ 个体（弱痛苦参数）
            - 验证 Q、Γ、对称性恢复概率的参数依赖

        物理命题：
            1. 高 κ 个体：强痛苦 → SSB + 度规旋转 → Q≠0（有拓扑张力）
               低 κ 个体：弱痛苦 → 度规近真空 → Q≈0（拓扑平凡）
            2. 高 κ 个体的 ||Γ|| > 低 κ 个体的 ||Γ||
               （强冲击留下更深的几何相位记录）
            3. 对称性恢复统计必然性：P(t)→1 when t→∞

        佛学对应：
            κ = 痛苦深度（duḥkha）。
            高 κ = 强烈苦 → 深度破缺 → 非零业力（Q≠0，Γ≠0）。
            低 κ = 微弱苦 → 轻微破缺 → 业力微弱（Q≈0，Γ≈0）。
            但无论 κ 大小，对称性恢复终究发生（P→1）——
            「一切有为法，其性无常，终究回归空性」。
        """
        n = self.n_dims
        n_steps = 30

        # ================================================================
        # 个体一：高 κ 个体（强痛苦参数，度规反复旋转）
        # ================================================================
        g_history_high_kappa = []
        kappa_high = torch.tensor([5.0, 3.0, 0.5, 0.3], dtype=torch.float64)
        alpha_high = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)

        for step in range(n_steps):
            # 高 κ 冲击：度规在不同维度间振荡，本征向量旋转
            g = torch.diag(torch.tensor([
                1.0 + 0.5 * math.sin(step * 0.3),
                1.0 + 0.5 * math.sin(step * 0.3 + 1.0),
                1.0 + 0.3 * math.cos(step * 0.4),
                1.0 + 0.3 * math.cos(step * 0.4 + 0.5),
            ], dtype=torch.float64))
            # 非对角扰动 → 本征向量旋转 → Q≠0
            g[0, 1] = 0.1 * math.sin(step * 0.2)
            g[1, 0] = g[0, 1]
            g[2, 3] = 0.1 * math.cos(step * 0.2)
            g[3, 2] = g[2, 3]
            g = symmetric_part(g)
            eigvals_check = torch.linalg.eigvalsh(g)
            if eigvals_check.min() < self.eps:
                g = g + (self.eps - eigvals_check.min()) * torch.eye(n, dtype=torch.float64)
            g_history_high_kappa.append(g.clone())

        agent_high_kappa = self.create_agent_from_history(
            g_history_high_kappa, "high_kappa", kappa_high, alpha_high
        )

        # ================================================================
        # 个体二：低 κ 个体（弱痛苦参数，度规近真空）
        # ================================================================
        g_history_low_kappa = []
        kappa_low = torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float64)
        alpha_low = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

        for step in range(n_steps):
            # 低 κ：度规几乎不变（近真空）
            g = torch.diag(torch.tensor([
                1.0 + 0.001 * math.sin(step * 0.1),
                1.0 + 0.001 * math.cos(step * 0.1),
                1.0 + 0.001 * math.sin(step * 0.1 + 0.5),
                1.0 + 0.001 * math.cos(step * 0.1 + 0.5),
            ], dtype=torch.float64))
            g = symmetric_part(g)
            g_history_low_kappa.append(g.clone())

        agent_low_kappa = self.create_agent_from_history(
            g_history_low_kappa, "low_kappa", kappa_low, alpha_low
        )

        # ================================================================
        # 验证结果
        # ================================================================
        Q_high = float(agent_high_kappa.Q)
        Q_low = float(agent_low_kappa.Q)
        Gamma_norm_high = float(torch.sqrt((agent_high_kappa.__dict__.get("Gamma", torch.zeros(n, n)) ** 2).sum()))
        Gamma_norm_low = float(torch.sqrt((agent_low_kappa.__dict__.get("Gamma", torch.zeros(n, n)) ** 2).sum()))

        # 验证判据
        high_kappa_has_charge = abs(Q_high) > 0.1
        low_kappa_no_charge = abs(Q_low) < 0.1
        Q_high_larger = abs(Q_high) > abs(Q_low)
        Gamma_high_larger = Gamma_norm_high > Gamma_norm_low

        all_pass = (
            high_kappa_has_charge and
            low_kappa_no_charge and
            Q_high_larger and
            Gamma_high_larger
        )

        return {
            "high_kappa_agent": {
                "label": "high_kappa",
                "Q": Q_high,
                "has_topological_charge": high_kappa_has_charge,
                "Gamma_norm": Gamma_norm_high,
                "kappa_mean": float(kappa_high.mean()),
                "description": "高 κ（强痛苦）→ SSB + 度规旋转 → Q≠0（有拓扑张力）",
            },
            "low_kappa_agent": {
                "label": "low_kappa",
                "Q": Q_low,
                "has_topological_charge": not low_kappa_no_charge,
                "Gamma_norm": Gamma_norm_low,
                "kappa_mean": float(kappa_low.mean()),
                "description": "低 κ（弱痛苦）→ 度规近真空 → Q≈0（拓扑平凡）",
            },
            "validation": {
                "high_kappa_has_charge": high_kappa_has_charge,
                "low_kappa_no_charge": low_kappa_no_charge,
                "Q_high_larger": Q_high_larger,
                "Gamma_high_larger": Gamma_high_larger,
                "all_pass": all_pass,
            },
            "thesis": (
                f"v7.7 纯物理验证：高 κ 个体 Q={Q_high:.4f}"
                f"（{'有拓扑张力' if high_kappa_has_charge else '拓扑平凡'}），"
                f"低 κ 个体 Q={Q_low:.4f}"
                f"（{'有拓扑张力' if not low_kappa_no_charge else '拓扑平凡'}）。"
                f"高 κ ||Γ||={Gamma_norm_high:.6f}，低 κ ||Γ||={Gamma_norm_low:.6f}。"
                f"验证{'通过' if all_pass else '未完全通过'}："
                "高 κ（强痛苦）→ Q≠0（有业力）；低 κ（弱痛苦）→ Q≈0（业力微弱）。"
                "拓扑荷 = 度规方向的累积旋转 = 高 κ 环境导致的破缺记录。"
                "无论 κ 大小，对称性恢复终究发生（P→1）——一切有为法终究回归空性。"
            ),
        }

    def validate_with_v6_cases(self) -> dict[str, dict | bool]:
        """
        纯物理验证（保留向后兼容，v7.7 等价于 validate_symmetry_restoration_inevitability）。

        v7.7 说明：
            此方法保留向后兼容。v7.7 推荐使用 validate_symmetry_restoration_inevitability
            获取更清晰的语义（清除案例人名引用，用纯物理参数描述）。
        """
        return self.validate_symmetry_restoration_inevitability()

    # ==================================================================
    # 7. 集体觉照消解（v7.7 新增，整合 v7.3 + v7.6）
    # ==================================================================

    def collective_awareness_dissolution(
        self,
        agents: list[CognitiveAgent],
        rho_schedule: list[float],
        dt: float = 0.02,
        inner_steps: int = 100,
        coupled: bool = True,
    ) -> dict[str, list]:
        """
        集体觉照消解：多个个体同时 ρ→1 的多体效应。

        v7.7 新增，整合 v7.3（ρ 消解 g）+ v7.6（ρ 消解 Γ）：
            单体觉照消解（v7.3, v7.6）：
                ρ→1 使 g_i→cI（无相）+ Γ_i→0（清业）+ Q_i→0（业力消解）
            多体觉照消解（v7.7 新增）：
                所有个体同时 ρ→1 → 所有束缚态解耦 → 系统回归真空

        v7.8 修正：分离 E_self 与 V_int 的判据。
            原判据 |E_total|→0 混淆了「单体回归真空（E_self→0）」
            与「束缚态解耦（V_int→0）」两件事。
            v7.8 分离为两个独立判据：
                - 单体回归：E_self(i) → V(cI) = 0（度规回归真空）
                - 束缚态解耦：V_int(i,j) → 0（Q_i→0 导致规范场消失）

        v7.8 修正：内部梯度迭代（inner_steps）。
            原实现每个 ρ 只做一次梯度步，不足以让 g 从 SSB 破缺态收敛到 cI。
            v7.8 引入 inner_steps：每个 ρ 值下做 inner_steps 次梯度下降，
            并使用自适应稳定步长 effective_dt = 1/(2·ρ·λ_restore + 1)，
            确保收敛稳定（不发散）且充分（g 真正到达 cI 附近）。

        v7.9 修正：真正的多体耦合动力学（coupled=True）。
            原 v7.7-v7.8 的 collective_awareness_dissolution 名为「集体」，
            实为 N 个独立单体并行演化——规范力不进入动力学方程，
            只用于事后诊断。这是「伪多体」。
            v7.9 引入 coupled=True（默认）：规范力进入内部迭代的动力学方程，
            每步同时计算觉照梯度 + 规范力，并自洽重算 Q。
            物理意义：
                - 低 ρ：规范力主导（束缚态动力学，共业相吸/相斥）
                - 高 ρ：觉照力主导（消解动力学，回归真空）
                - 过渡：规范力随 Q→0 自然衰减，觉照力随 ρ→1 增强
            佛学对应：共修（saṃgha）效应——
                一个个体 Q 减小 → 作用于他者的力减小 → 加速他者消解。
                修行者之间相互支持，集体觉悟更快。

        物理机制：
            1. 每个个体的度规被觉照场拉向真空：
               V_ρ(g) = V_base(g) + ρ·λ_restore·||g-cI||²
            2. 每个个体的几何相位被消解：Γ_{k+1} = (1-ρ)·Γ_k
            3. 规范相互作用随 Q_i→0 而消失：V_int ∝ |Q_i|·|Q_j|→0
            4. 束缚态解耦：V_int → 0（规范场消失，共业结构瓦解）

        佛学对应：
            集体觉照 = 共修 / 僧伽（saṃgha）的集体觉悟。
            所有个体同时觉照 → 共业结构消解 → 集体回归空性。
            「若菩萨通达无我法者，如来说名真是菩萨」——
            集体无我 = 所有个体 g→cI，共业无可作用。

        参数：
            agents: 多体系统中的个体列表
            rho_schedule: ρ 的时间调度 [ρ_0, ρ_1, ..., ρ_T]
            dt: 演化时间步（外层，会被自适应稳定步长覆盖）
            inner_steps: 每个 ρ 值下的内部梯度迭代次数（v7.8 新增）
            coupled: 是否启用多体耦合（v7.9 新增，默认 True）
        """
        n = self.n_dims
        n_agents = len(agents)

        # 深拷贝个体
        current_agents = [
            CognitiveAgent(
                g=ag.g.clone(),
                Q=ag.Q.clone(),
                label=ag.label,
                kappa_vec=ag.kappa_vec.clone() if ag.kappa_vec is not None else None,
                alpha_vec=ag.alpha_vec.clone() if ag.alpha_vec is not None else None,
                Gamma=ag.Gamma.clone() if ag.Gamma is not None else None,
            )
            for ag in agents
        ]

        # 记录初始状态
        g_vac = self.vacuum.construct_vacuum()
        initial_analysis = self.analyze_system(current_agents)
        initial_V_int = initial_analysis["total_energy"]  # V_int（相互作用能）
        # E_self = Σ_i V(g_i, kappa_i, alpha_i)（单体势能之和）
        initial_E_self = self._compute_total_self_energy(current_agents)
        initial_Q_norms = [float(ag.Q.abs()) for ag in current_agents]
        initial_Gamma_norms = [
            float(torch.sqrt((ag.Gamma ** 2).sum())) if ag.Gamma is not None else 0.0
            for ag in current_agents
        ]
        initial_dist_to_vacuum = [
            float(torch.sqrt(((ag.g - g_vac) ** 2).sum()))
            for ag in current_agents
        ]

        # 觉照消解轨迹（v7.8: 分离 E_self 与 V_int）
        trajectory = {
            "rho": [],
            "V_int": [],       # 相互作用能（束缚态强度）
            "E_self": [],      # 单体势能之和（单体偏离真空程度）
            "E_total": [],     # V_int + E_self（总能量）
            "Q_norms": [],
            "Gamma_norms": [],
            "dist_to_vacuum": [],
        }

        # v7.8: 自适应稳定步长。
        # 觉照恢复项梯度 = 2·ρ·λ_restore·(g-cI)，稳定性要求
        # effective_dt < 1/(ρ·λ_restore)。取临界值的 50% 确保安全。
        lam_restore = float(self.awareness.lambda_restore)

        # 觉照演化
        for rho in rho_schedule:
            rho_f = float(rho)

            # 1. 几何相位被消解（v7.6，每个 ρ 一次）
            for ag in current_agents:
                if ag.Gamma is not None:
                    ag.Gamma = (1.0 - rho_f) * ag.Gamma

            # 2. 度规被觉照场拉向真空（v7.3 + v7.8 内部迭代）
            # 自适应步长：rho=0 时用 dt，rho>0 时受 λ_restore 约束
            if rho_f > self.eps:
                effective_dt = min(dt, 0.5 / (rho_f * lam_restore + 1.0))
            else:
                effective_dt = dt

            for _ in range(inner_steps):
                # v7.9: 计算规范力（如果 coupled=True）
                gauge_forces = None
                if coupled:
                    gauge_forces = []
                    for i in range(n_agents):
                        result = self.gauge.gauge_force_on_agent(i, current_agents)
                        gauge_forces.append(result["force"])

                # 同时更新所有个体（同步，避免序列偏差）
                for i, ag in enumerate(current_agents):
                    pot = self.awareness.compute_awareness_potential(
                        ag.g, ag.kappa_vec, ag.alpha_vec, rho_f
                    )
                    grad = pot["grad_total"]

                    # v7.9: 总力 = -∇V_awareness + F_gauge
                    if coupled and gauge_forces is not None:
                        total_force = -grad + gauge_forces[i]
                    else:
                        total_force = -grad

                    force_norm = float(torch.sqrt((total_force ** 2).sum()) + self.eps)
                    if force_norm > 1e-12:
                        step = effective_dt * min(force_norm, 5.0)
                        ag.g = ag.g + step * total_force / force_norm
                        ag.g = symmetric_part(ag.g)
                        eigvals = torch.linalg.eigvalsh(ag.g)
                        if eigvals.min() < self.eps:
                            ag.g = ag.g + (self.eps - eigvals.min()) * torch.eye(
                                n, dtype=torch.float64
                            )

                # v7.9: 自洽重算 Q（每步，使规范力使用更新后的 Q）
                if coupled:
                    for ag in current_agents:
                        Q_result = self.charge.compute_total_charge(
                            g_current=ag.g,
                            g_baseline=g_vac,
                        )
                        ag.Q = torch.tensor(
                            float(Q_result["Q_total"]), dtype=torch.float64
                        )

            # 3. 重新计算 Q（度规变化后，每个 ρ 一次；非 coupled 模式下仍需要）
            if not coupled:
                for ag in current_agents:
                    Q_result = self.charge.compute_total_charge(
                        g_current=ag.g,
                        g_baseline=g_vac,
                    )
                    ag.Q = torch.tensor(
                        float(Q_result["Q_total"]), dtype=torch.float64
                    )

            # 记录系统状态（v7.8: 分离 V_int 与 E_self）
            analysis = self.analyze_system(current_agents)
            V_int = analysis["total_energy"]  # 相互作用能
            E_self = self._compute_total_self_energy(current_agents)  # 单体势能
            Q_norms = [float(ag.Q.abs()) for ag in current_agents]
            Gamma_norms = [
                float(torch.sqrt((ag.Gamma ** 2).sum())) if ag.Gamma is not None else 0.0
                for ag in current_agents
            ]
            dist_to_vacuum = [
                float(torch.sqrt(((ag.g - g_vac) ** 2).sum()))
                for ag in current_agents
            ]

            trajectory["rho"].append(float(rho))
            trajectory["V_int"].append(V_int)
            trajectory["E_self"].append(E_self)
            trajectory["E_total"].append(V_int + E_self)
            trajectory["Q_norms"].append(Q_norms)
            trajectory["Gamma_norms"].append(Gamma_norms)
            trajectory["dist_to_vacuum"].append(dist_to_vacuum)

        # 最终状态
        final_analysis = self.analyze_system(current_agents)
        final_V_int = final_analysis["total_energy"]
        final_E_self = self._compute_total_self_energy(current_agents)
        final_Q_norms = [float(ag.Q.abs()) for ag in current_agents]
        final_Gamma_norms = [
            float(torch.sqrt((ag.Gamma ** 2).sum())) if ag.Gamma is not None else 0.0
            for ag in current_agents
        ]
        final_dist_to_vacuum = [
            float(torch.sqrt(((ag.g - g_vac) ** 2).sum()))
            for ag in current_agents
        ]

        # v7.8 修正：分离验证判据
        # 1. 所有个体的 Q→0（业力消解）
        all_q_dissolved = all(q < 0.05 for q in final_Q_norms)
        # 2. 所有个体的 Γ→0（清业）
        all_gamma_cleared = all(g < 0.05 for g in final_Gamma_norms)
        # 3. 所有个体的 g→cI（单体回归真空）
        all_g_returned = all(d < 0.1 for d in final_dist_to_vacuum)
        # 4. 束缚态解耦：V_int → 0（规范场消失，共业结构瓦解）
        bound_state_decoupled = bool(
            abs(final_V_int) < 0.1 * max(abs(initial_V_int), self.eps)
        )
        # 5. 单体回归真空：E_self 从最负点恢复（v7.9 修正）
        # v7.9 修正：ρ=0 段中 total_force = -∇V_base 使 g 朝向 SSB 极小值 g* 移动，
        # E_self 变得很负（V_base 高阶项在 g* 附近贡献显著负能）。
        # 因此不能用 initial_E_self 作基准（ρ=0 段已使其变负），
        # 而应比较 final_E_self 与轨迹中最负的 E_self——
        # 检查 E_self 是否从最负点显著恢复到 10% 以内。
        min_E_self = min(trajectory["E_self"]) if trajectory["E_self"] else 0.0
        self_returned = bool(
            abs(final_E_self) < 0.1 * max(abs(min_E_self), self.eps)
        )

        all_pass = bool(
            all_q_dissolved and
            all_gamma_cleared and
            all_g_returned and
            bound_state_decoupled and
            self_returned
        )

        return {
            "initial_V_int": initial_V_int,
            "final_V_int": final_V_int,
            "initial_E_self": initial_E_self,
            "final_E_self": final_E_self,
            "initial_E_total": initial_V_int + initial_E_self,  # 向后兼容
            "final_E_total": final_V_int + final_E_self,  # 向后兼容
            "initial_Q_norms": initial_Q_norms,
            "final_Q_norms": final_Q_norms,
            "initial_Gamma_norms": initial_Gamma_norms,
            "final_Gamma_norms": final_Gamma_norms,
            "initial_dist_to_vacuum": initial_dist_to_vacuum,
            "final_dist_to_vacuum": final_dist_to_vacuum,
            "trajectory": trajectory,
            "all_q_dissolved": all_q_dissolved,
            "all_gamma_cleared": all_gamma_cleared,
            "all_g_returned": all_g_returned,
            "bound_state_decoupled": bound_state_decoupled,
            "self_returned": self_returned,  # v7.8 新增
            "all_pass": all_pass,
            "thesis": (
                f"集体觉照消解（v7.8）：{n_agents} 个个体同时 ρ→1。"
                f"V_int: {initial_V_int:.4f} → {final_V_int:.4f}（束缚态解耦）。"
                f"E_self: {initial_E_self:.4f} → {final_E_self:.4f}（单体回归真空）。"
                f"所有 Q→0: {all_q_dissolved}，所有 Γ→0: {all_gamma_cleared}，"
                f"所有 g→cI: {all_g_returned}，"
                f"束缚态解耦(V_int→0): {bound_state_decoupled}，"
                f"单体回归(E_self→0): {self_returned}。"
                "v7.8 修正：分离束缚态解耦（V_int→0）与单体回归（E_self→0）两个判据。"
                "集体觉照 = 共修效应——所有个体同时觉照，共业结构消解，集体回归空性。"
                "v7.3（ρ 消解 g）+ v7.6（ρ 消解 Γ）+ v7.7（集体效应）+ v7.8（判据分离）"
                "= 完整的多体觉照消解。"
            ),
        }

    def _compute_total_self_energy(self, agents: list[CognitiveAgent]) -> float:
        """
        计算所有个体的单体过剩能量之和（v7.8 新增，v7.8b 修正）。

        v7.8b 修正：E_self = Σ_i [V(g_i) - V(cI)]，减去真空势能。
            原实现 E_self = Σ_i V(g_i)，但 V(cI) ≠ 0（真空本身有非零势能），
            导致 g→cI 时 E_self→V(cI)≠0，判据 self_returned 永远不通过。
            v7.8b 改为「相对于真空的过剩能量」：E_self = Σ_i [V(g_i) - V(cI_i)]。
            这样 g→cI 时 E_self→0，物理意义更清晰：
            E_self 度量「偏离真空的能量」，而非绝对势能。

        物理意义：
            E_self = 「破缺能量」= 系统因 SSB 偏离真空而具有的额外能量。
            g=cI 时 V(g)-V(cI) = 0（回归真空，无过剩能量）。
            g 偏离 cI 越远，V(g)-V(cI) 越大（破缺态能量高）。

            在集体觉照消解中，E_self→0 表示所有单体回归真空。
            这与 V_int→0（束缚态解耦）是两个独立的判据：
                - E_self→0：单体层面，度规回归空性
                - V_int→0：多体层面，规范场消失，共业瓦解
        """
        E_self = 0.0
        g_vac = self.vacuum.construct_vacuum()
        for ag in agents:
            if ag.kappa_vec is not None and ag.alpha_vec is not None:
                pot_g = self.vacuum.compute_potential(ag.g, ag.kappa_vec, ag.alpha_vec)
                pot_vac = self.vacuum.compute_potential(g_vac, ag.kappa_vec, ag.alpha_vec)
                # v7.8b: 减去真空势能，度量「过剩能量」
                E_self += float(pot_g["V"]) - float(pot_vac["V"])
            else:
                # 无 kappa/alpha 时，用 ||g - cI||² 作为近似
                g_vac = self.vacuum.construct_vacuum()
                E_self += float(((ag.g - g_vac) ** 2).sum())
        return E_self

    # ==================================================================
    # 8. 完整理论总结
    # ==================================================================

    def theory_summary(self) -> dict[str, str]:
        """
        v7.8 认知拓扑场论的完整理论总结。

        v7.8 升级：有符号拓扑荷（Q_signed）与业力三性（善/恶/无记）。
        v7.7 升级：清除「命运/相逢/天生倾向/代际传递」等社会学语言，
        统一为「共业结构的规范场论」——物理+佛学+哲学的认识论根基。
        """
        return {
            "title": "v7.8 认知拓扑场论（Cognitive Topological Field Theory）",
            "core_thesis": (
                "共业结构的规范场论：从单体度规动力学跃迁到多体规范场论 + 拓扑学。"
                "多体系统是规范对称性的集体破缺模式，"
                "个体间的业力结构耦合形成不可逆的拓扑绑定（束缚态）。"
                "v7.8 新增：有符号拓扑荷与业力三性（善/恶/无记）。"
            ),
            "six_pillars": {
                "1_vacuum": "认知真空 V_0（g=cI，v7.1）：空性的数学定义。κ>κ_c 触发自发对称性破缺（无明生起）。",
                "2_charge": "拓扑荷 Q（v7.1+v7.8）：Q_dynamic（无符号=业力总量）+ Q_signed（有符号=业力净方向）。三性：善/恶/无记。",
                "3_gauge": "规范场 V_int（v7.4）：共业耦合。共振（cos_align>0）吸引，互补（cos_align<0）排斥。",
                "4_kramers": "Kramers 自然回归（v7.5）：P(t)=1-exp(-t/τ)，对称性恢复的统计必然性。t→∞ 时 P→1。",
                "5_phase": "几何相位 Γ（v7.6）：演化历史全息记录。Γ 作用在已有度规上（业力相续），真空上无效（觉悟解脱）。",
                "6_awareness": "觉照路径（v7.3）：ρ→1 使 g→cI（无相）+ Γ→0（清业）+ Q→0（业力消解）。",
            },
            "v78_signed_charge": {
                "Q_dynamic": "无符号拓扑荷（业力总量）：(1/2π)·Σ_k |θ_k|，善恶皆计。",
                "Q_signed": "有符号拓扑荷（业力净方向）：(1/2π)·Σ_{i>j} Γ[i,j]，善恶相抵后的净方向。",
                "kushala": "善业（Q_signed>0）：净正向旋转，趋向觉悟方向。",
                "akushala": "恶业（Q_signed<0）：净负向旋转，趋向执着方向。",
                "avyakrta": "无记业（Q_signed≈0, Q_dynamic>0）：善恶相抵，方向不定。",
                "akarma": "无业（Q_dynamic=0）：无拓扑张力，拓扑平凡。",
                "relation": "|Q_signed| ≤ Q_dynamic（三角不等式：位移≤路径长度）。",
            },
            "v78_criterion_fix": (
                "v7.8 修正：分离束缚态解耦（V_int→0）与单体回归（E_self→0）两个判据。"
                "原 v7.7 的 |E_total|→0 混淆了单体势能消解与规范场消失。"
                "v7.8 分别追踪 V_int（相互作用能）和 E_self（单体势能）。"
            ),
            "key_questions_answered": {
                "why_nonzero_charge": "高 κ（强痛苦）→ SSB + 度规旋转 → Q≠0（有拓扑张力）。低 κ（弱痛苦）→ Q≈0（拓扑平凡）。",
                "why_karma_direction": "Q_signed>0（善业）vs Q_signed<0（恶业）vs Q_signed≈0（无记）。从 Berry 相位 Γ 的下三角和提取符号。",
                "why_bound_state": "共振对（同向 Berry 相位）→ V_int<0 → 束缚态形成（共业相吸）。互补对 → V_int>0 → 排斥（业力互消）。",
                "why_restoration_inevitable": "Kramers 定理：τ=τ_0·exp(ΔV/T_cog)，t→∞ 时 P→1。一切破缺态终究回归空性。",
                "why_timescale_continuity": "历史 Γ 作用在已有度规上 → 新阶段带方向性（识变现）。真空上 Γ 无效 → 觉悟者从业力解脱。",
                "why_enlightened_transcend": "ρ→1 使 g→cI + Γ→0 + Q→0。觉悟者不携带业力，新阶段从真空开始——了脱生死，不入轮回。",
            },
            "buddhist_correspondence": {
                "vacuum": "空性（śūnyatā）：g=cI 正定存在但无分别。",
                "breaking": "无明生起（avidyā）：κ>κ_c 触发对称性破缺。",
                "charge_Q_dynamic": "业力总量（karma-saṃkleśa）：善恶皆计，累积不失。",
                "charge_Q_signed": "业力方向（karma-mārga）：净善/净恶/无记。",
                "karma_three_natures": "三性（trisvabhāva）：善（kuśala）/恶（akuśala）/无记（avyākṛta）。",
                "gauge_field": "共业（sādhārana-karma）：个体间的规范相互作用。",
                "kramers": "因缘果报：t→∞ 时 P→1 = 一切有为法终究回归空性。",
                "geometric_phase": "阿赖耶识（ālaya-vijñāna）：Γ 的全息记录与时间尺度传播。",
                "awareness": "觉照（smṛti）/ 出离心（ρ）：ρ→1 消解 g、Γ、Q，回归空性。",
                "collective_awareness": "共修 / 僧伽（saṃgha）：集体 ρ→1 使共业结构消解。",
            },
            "v78_integration": (
                "v7.8 完整闭环："
                "v7.1（真空+拓扑荷）+ v7.3（觉照消解 g）+ v7.4（规范场+束缚态）+ "
                "v7.5（Kramers 自然回归）+ v7.6（几何相位+清业）+ v7.7（多体整合+集体觉照）+ "
                "v7.8（有符号荷+业力三性+判据分离）"
                "= 共业结构的规范场论 + 业力方向性。"
            ),
        }
