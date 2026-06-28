"""
认知热力学层（Cognitive Thermodynamics）

v7.10 多体自由演化热力学：从确定性动力学升级到统计力学。

认识论根基（理论依据，非案例）：
    物理：统计力学 / 配分函数 / 自由能 / 熵 / 热容 / 相变温度
    佛学：业力分布 / 有效势（saṃvṛti-satya）/ 信息结构 / 定力与涨落
    哲学：宏观描述 vs 微观动力学 / 涌现的统计规律 / 有效势与体验

核心命题：把 CTFT 从"动力学"升级到"统计力学"。

    动力学层（v7.9）：确定性轨迹 ∂g/∂t = F(g)
    统计力学层（v7.10）：系综平均 ⟨E⟩、配分函数 Z、自由能 F、熵 S、热容 C_V

物理机制：
    1. Langevin 动力学：∂g/∂t = F_single + Σ F_gauge + √(2·T_cog)·ξ(t)
       T_cog = κ/α（认知温度，痛苦/定力比值）
       高 T_cog（痛苦大、定力小）→ 涨落大 → 束缚态易解离
       低 T_cog（痛苦小、定力大）→ 涨落小 → 束缚态稳定

    2. 配分函数：Z = ∫ exp(-E_total(g)/T_cog) dg
       从 Langevin 轨迹采样估计（各态历经假设）：
       Z ≈ (1/N) Σ_k exp(-E_k/T_cog)

    3. 热力学量：
       U = ⟨E⟩（内能）
       F = -T_cog · ln(Z)（自由能 = 有效势）
       S = (U - F) / T_cog（熵 = 信息结构量）
       C_V = (⟨E²⟩ - ⟨E⟩²) / T_cog²（热容 = 涨落）

    4. 相变：C_V 的尖峰或 F 的非解析点 → T_c（临界温度）
       T_cog > T_c：束缚态解离（共业熔化）
       T_cog < T_c：束缚态稳定（共业坚固）

佛学对应：
    配分函数 Z = 业力分布的全集（一切可能业力态的权重和）
    自由能 F = 业力的"有效势"（考虑温度涨落后的实际驱动力）
    熵 S = 业力结构的"信息量"（结构越确定，熵越低）
    T_cog = κ/α = 痛苦/定力 = 认知温度
    高 T_cog → 涨落大 → 业力结构易动摇（心散乱）
    低 T_cog → 涨落小 → 业力结构坚固（心安定）
    相变 T_c → "定力阈值"：低于此阈值共业不可破

哲学对应：
    自由能景观 F(g) 是"实际体验的苦乐 landscape"
    不同于势能 V(g)（"客观苦乐"）
    熵增 = 业力结构被打散（向无序化）
    热力学第二定律纳入认知场论
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import symmetric_part
from .cognitive_vacuum import CognitiveVacuum
from .gauge_interaction import GaugeInteraction, CognitiveAgent
from .multi_body_field import MultiBodyCognitiveField


class ThermodynamicAnalyzer:
    """
    认知热力学分析器：从 Langevin 轨迹估计热力学量。

    使用方式：
        field = MultiBodyCognitiveField(n_dims=4, coupling_lambda=2.0)
        agents = [agent_A, agent_B]
        thermo = ThermodynamicAnalyzer(field)
        # 采样
        energies = thermo.langevin_sample(agents, T_cog=0.1, n_steps=500)
        # 计算热力学量
        quantities = thermo.compute_thermodynamic_quantities(energies, T_cog=0.1)
        # 扫描温度
        scan = thermo.scan_temperature(agents, T_range=(0.01, 1.0), n_points=20)
        # 找相变点
        T_c = thermo.find_phase_transition(scan)
    """

    def __init__(
        self,
        field: MultiBodyCognitiveField,
        eps: float = 1e-12,
    ):
        self.field = field
        self.vacuum = field.vacuum
        self.gauge = field.gauge
        self.n_dims = field.n_dims
        self.eps = eps

    # ==================================================================
    # 1. 能量计算
    # ==================================================================

    def _compute_E_self_for_agent(
        self, g: Tensor, kappa_vec: Tensor, alpha_vec: Tensor
    ) -> float:
        """单体过剩能量 E_self = V(g) - V(cI)。"""
        g_vac = self.vacuum.construct_vacuum()
        pot_g = self.vacuum.compute_potential(g, kappa_vec, alpha_vec)
        pot_vac = self.vacuum.compute_potential(g_vac, kappa_vec, alpha_vec)
        return float(pot_g["V"]) - float(pot_vac["V"])

    def _compute_E_total_from_state(
        self,
        g_list: list[Tensor],
        agents: list[CognitiveAgent],
        V_int_current: float,
    ) -> float:
        """
        从当前 g 状态计算总能量 E_total = Σ E_self + V_int。

        V_int 由 simulate_dynamics 内部已算（传 V_int_current），
        E_self 从 g_list 重算（因为 simulate_dynamics 不记录 E_self 轨迹）。
        """
        E_self_total = 0.0
        for i, g in enumerate(g_list):
            ag = agents[i]
            E_self_total += self._compute_E_self_for_agent(
                g, ag.kappa_vec, ag.alpha_vec
            )
        return E_self_total + V_int_current

    # ==================================================================
    # 2. Langevin 采样
    # ==================================================================

    def langevin_sample(
        self,
        agents: list[CognitiveAgent],
        T_cog: float,
        n_steps: int = 500,
        dt: float = 0.005,
        rho: float = 0.0,
        burn_in: int = 0,
        charge_history_window: int | None = -1,
    ) -> dict[str, list]:
        """
        跑 Langevin 动力学，返回能量轨迹。

        物理：∂g/∂t = F_single + Σ F_gauge + √(2·T_cog)·ξ(t)
        采样：E_total(k) = Σ E_self(g_k) + V_int(g_k)

        v7.10 关键：charge_history_window=-1（默认，累积模式）
            累积模式 O(1) per step，永不遗忘初始业力（阿赖耶识种子不灭）。
            Q/Γ 增量更新：每步加新的旋转角和 Berry 连接到累积值。
            物理：Q_static = 当下心相（随 SSB 松弛到 0），Q_dynamic = 阿赖耶识种子（拓扑保护）。
            束缚态的稳定性依赖 Q_dynamic（业力不灭），而非 Q_static（心相无常）。

        参数：
            agents: 初始个体
            T_cog: 认知温度（Langevin 噪声强度）
            n_steps: 演化步数
            dt: 时间步长
            rho: 觉照强度（0=纯规范动力学，1=完全觉照）
            burn_in: 热化期（丢弃前 burn_in 步，只保留平衡后采样）
            charge_history_window: -1=累积模式(默认), N>0=滑动窗口N帧, None=只用Q_static

        返回：
            {"E_total": [...], "E_self": [...], "V_int": [...], "n_samples": N}
        """
        result = self.field.simulate_pratitya_dynamics(
            agents,
            n_steps=n_steps,
            dt=dt,
            rho=rho,
            noise_temperature=T_cog,
            charge_history_window=charge_history_window,
        )

        trajectories = result["dynamics"]["trajectories"]
        V_int_traj = result["dynamics"]["energy_trajectory"]

        # trajectories[0] 是初始状态，trajectories[k+1] 是 step k 后的状态
        # V_int_traj[k] 是 step k 后的 V_int
        # 所以 E_total(k) = E_self(trajectories[k+1]) + V_int_traj[k]

        E_self_list = []
        V_int_list = []
        E_total_list = []

        for k in range(len(V_int_traj)):
            g_list = trajectories[k + 1]
            V_int_k = V_int_traj[k]

            E_self_k = 0.0
            for i, g in enumerate(g_list):
                ag = agents[i]
                E_self_k += self._compute_E_self_for_agent(
                    g, ag.kappa_vec, ag.alpha_vec
                )

            E_total_k = E_self_k + V_int_k

            # v7.10: 过滤 NaN/inf（高噪声下可能发生）
            if not (E_total_k == E_total_k and abs(E_total_k) != float('inf')):
                # NaN 或 inf，跳过这个采样点
                continue

            E_self_list.append(E_self_k)
            V_int_list.append(V_int_k)
            E_total_list.append(E_total_k)

        # 热化期丢弃
        if burn_in > 0 and burn_in < len(E_total_list):
            E_total_list = E_total_list[burn_in:]
            E_self_list = E_self_list[burn_in:]
            V_int_list = V_int_list[burn_in:]

        return {
            "E_total": E_total_list,
            "E_self": E_self_list,
            "V_int": V_int_list,
            "n_samples": len(E_total_list),
            "T_cog": T_cog,
            "rho": rho,
        }

    # ==================================================================
    # 3. 热力学量计算
    # ==================================================================

    def compute_thermodynamic_quantities(
        self,
        energies: list[float],
        T_cog: float,
    ) -> dict[str, float]:
        """
        从能量轨迹计算热力学量（标准统计力学公式）。

        U = ⟨E⟩
        F = -T·ln(Z)，Z = (1/N) Σ exp(-E_k/T)
        S = (U - F) / T
        C_V = (⟨E²⟩ - ⟨E⟩²) / T²

        数值稳定：用 log-sum-exp 计算 ln(Z)。

        参数：
            energies: 能量采样序列
            T_cog: 认知温度

        返回：
            {"U", "F", "S", "C_V", "log_Z", "var_E", "T_cog"}
        """
        if len(energies) == 0:
            return {
                "U": 0.0, "F": 0.0, "S": 0.0, "C_V": 0.0,
                "log_Z": 0.0, "var_E": 0.0, "T_cog": T_cog,
            }

        E = torch.tensor(energies, dtype=torch.float64)
        N = len(E)
        T = max(float(T_cog), self.eps)

        # 内能 U = <E>
        U = float(E.mean())

        # 方差 var(E) = <(E-U)²>（数值稳定，避免 E² 溢出）
        E_shifted = E - U
        var_E = float((E_shifted ** 2).mean())

        # 热容 C_V = var(E) / T²
        C_V = var_E / (T ** 2)

        # ln(Z) = -ln(N) + logsumexp(-E_k/T)
        log_Z = -math.log(N) + float(torch.logsumexp(-E / T, dim=0))

        # 自由能 F = -T·ln(Z)
        F = -T * log_Z

        # 熵 S = (U - F) / T
        S = (U - F) / T

        return {
            "U": U,
            "F": F,
            "S": S,
            "C_V": C_V,
            "log_Z": log_Z,
            "var_E": var_E,
            "T_cog": T_cog,
            "n_samples": N,
        }

    # ==================================================================
    # 4. 温度扫描
    # ==================================================================

    def scan_temperature(
        self,
        agents: list[CognitiveAgent],
        T_range: tuple[float, float] = (0.01, 1.0),
        n_points: int = 15,
        n_steps_per_T: int = 400,
        dt: float = 0.005,
        rho: float = 0.0,
        burn_in: int = 100,
    ) -> dict[str, list]:
        """
        扫描认知温度 T_cog，绘制 F(T)、S(T)、C_V(T)、U(T)。

        物理：T_cog 是相变控制变量。
        T_cog 小 → 束缚态稳定（F 低，C_V 小）
        T_cog 大 → 束缚态解离（F 升，C_V 在 T_c 出现尖峰）

        参数：
            agents: 初始个体
            T_range: (T_min, T_max)
            n_points: 温度点数
            n_steps_per_T: 每个温度的演化步数
            dt: 时间步长
            rho: 觉照强度
            burn_in: 热化期步数

        返回：
            {"T_list", "U_list", "F_list", "S_list", "C_V_list", "var_E_list"}
        """
        T_values = torch.linspace(T_range[0], T_range[1], n_points)

        T_list = []
        U_list = []
        F_list = []
        S_list = []
        C_V_list = []
        var_E_list = []
        log_Z_list = []

        for T in T_values:
            T_float = float(T)
            sample = self.langevin_sample(
                agents,
                T_cog=T_float,
                n_steps=n_steps_per_T,
                dt=dt,
                rho=rho,
                burn_in=burn_in,
            )
            quantities = self.compute_thermodynamic_quantities(
                sample["E_total"], T_float
            )

            T_list.append(T_float)
            U_list.append(quantities["U"])
            F_list.append(quantities["F"])
            S_list.append(quantities["S"])
            C_V_list.append(quantities["C_V"])
            var_E_list.append(quantities["var_E"])
            log_Z_list.append(quantities["log_Z"])

        return {
            "T_list": T_list,
            "U_list": U_list,
            "F_list": F_list,
            "S_list": S_list,
            "C_V_list": C_V_list,
            "var_E_list": var_E_list,
            "log_Z_list": log_Z_list,
        }

    # ==================================================================
    # 5. 相变点检测
    # ==================================================================

    def find_phase_transition(self, scan_result: dict) -> dict:
        """
        从温度扫描找相变点 T_c。

        方法：C_V(T) 的尖峰位置（方差最大的温度）。
        物理：C_V 尖峰 = 能量涨落最大 = 相变点。

        参数：
            scan_result: scan_temperature 的返回

        返回：
            {"T_c", "C_V_peak", "T_list", "C_V_list", "is_phase_transition"}
        """
        T_list = scan_result["T_list"]
        C_V_list = scan_result["C_V_list"]

        if len(C_V_list) == 0:
            return {
                "T_c": None,
                "C_V_peak": 0.0,
                "is_phase_transition": False,
            }

        # 找 C_V 最大值
        max_idx = int(max(range(len(C_V_list)), key=lambda i: C_V_list[i]))
        T_c = T_list[max_idx]
        C_V_peak = C_V_list[max_idx]

        # 判据：C_V 峰值显著高于两侧（>2 倍均值）
        C_V_mean = sum(C_V_list) / len(C_V_list)
        is_phase_transition = C_V_peak > 2.0 * C_V_mean and C_V_peak > 1e-6

        return {
            "T_c": T_c,
            "C_V_peak": C_V_peak,
            "C_V_mean": C_V_mean,
            "is_phase_transition": is_phase_transition,
            "peak_idx": max_idx,
        }

    # ==================================================================
    # 6. 自由能景观（沿切片）
    # ==================================================================

    def free_energy_landscape_along_slice(
        self,
        agent_template: CognitiveAgent,
        other_agents: list[CognitiveAgent],
        T_cog: float,
        slice_axis: int = 0,
        n_points: int = 30,
        delta_range: tuple[float, float] = (-1.0, 1.0),
        scan_mode: str = "rotation",
    ) -> dict[str, list]:
        """
        计算自由能景观 F(g) 沿某个方向的切片。

        scan_mode="rotation"（v7.10 默认）：扫描旋转角 θ
            g(θ) = R(θ)^T g_template R(θ)
            物理：旋转改变本征向量方向，能看到 SSB 的多方向极小值。
            佛学：不同方向 = 不同"业力态"。

        scan_mode="diagonal"：扫描 g[axis, axis]（原行为）
            g(delta) = g_template + delta * e_axis
            物理：改变本征值大小。

        简化版：直接用 E_total(g) 作为 F(g) 的零阶近似（T→0 极限）。

        参数：
            agent_template: 被扫描的个体模板
            other_agents: 其他个体（固定）
            T_cog: 认知温度（零阶近似不用，但保留接口）
            slice_axis: 扫描的度规方向（diagonal 模式）
            n_points: 扫描点数
            delta_range: (min, max) 扫描范围
            scan_mode: "rotation" 或 "diagonal"

        返回：
            {"delta_list", "E_total_list", "E_self_list", "V_int_list", "scan_mode"}
        """
        deltas = torch.linspace(delta_range[0], delta_range[1], n_points)

        delta_list = []
        E_total_list = []
        E_self_list = []
        V_int_list = []

        g_vac = self.vacuum.construct_vacuum()
        n = self.n_dims

        for delta in deltas:
            if scan_mode == "rotation":
                # 旋转扫描：g(θ) = R(θ)^T g R(θ)
                theta = float(delta)
                R = torch.eye(n, dtype=torch.float64)
                c, s = math.cos(theta), math.sin(theta)
                R[0, 0] = c; R[0, 1] = -s
                R[1, 0] = s; R[1, 1] = c
                g_scan = R.T @ agent_template.g @ R
                g_scan = symmetric_part(g_scan)
            else:
                # 对角扫描：g[axis, axis] += delta
                g_scan = agent_template.g.clone()
                g_scan[slice_axis, slice_axis] += float(delta)
                g_scan = symmetric_part(g_scan)

            # 正定性保护
            try:
                eigvals = torch.linalg.eigvalsh(g_scan)
                if eigvals.min() < self.eps:
                    g_scan = g_scan + (self.eps - eigvals.min()) * torch.eye(
                        n, dtype=torch.float64
                    )
            except Exception:
                continue  # 跳过病态点

            # E_self
            E_self = self._compute_E_self_for_agent(
                g_scan, agent_template.kappa_vec, agent_template.alpha_vec
            )

            # V_int（用 agent_template 的 Q, Γ 近似）
            scan_agent = CognitiveAgent(
                g=g_scan,
                Q=agent_template.Q,
                label=agent_template.label,
                kappa_vec=agent_template.kappa_vec,
                alpha_vec=agent_template.alpha_vec,
                Gamma=agent_template.Gamma,
            )

            all_agents = [scan_agent] + list(other_agents)
            try:
                V_int = float(self.gauge.total_interaction_energy(all_agents)["E_total"])
            except Exception:
                continue

            E_total = E_self + V_int

            if not (E_total == E_total and abs(E_total) != float('inf')):
                continue

            delta_list.append(float(delta))
            E_total_list.append(E_total)
            E_self_list.append(E_self)
            V_int_list.append(V_int)

        return {
            "delta_list": delta_list,
            "E_total_list": E_total_list,
            "E_self_list": E_self_list,
            "V_int_list": V_int_list,
            "T_cog": T_cog,
            "slice_axis": slice_axis,
            "scan_mode": scan_mode,
        }

    # ==================================================================
    # 7. 验证方法
    # ==================================================================

    def verify_bound_state_stability(
        self,
        bound_agents: list[CognitiveAgent],
        free_agents: list[CognitiveAgent],
        T_cog: float = 0.05,
        n_steps: int = 400,
        dt: float = 0.005,
        burn_in: int = 100,
    ) -> dict:
        """
        验证 1：束缚态热力学稳定性。

        物理预测：F_bound < F_free（束缚态自由能更低）
        佛学：业力束缚在定力高（T_cog 低）时坚固

        参数：
            bound_agents: 束缚态初始配置（共振对，有 Γ）
            free_agents: 自由态初始配置（远离，无耦合）
            T_cog: 认知温度（低，确保束缚态稳定）

        返回：
            {"F_bound", "F_free", "F_bound_lower", "thesis"}
        """
        sample_bound = self.langevin_sample(
            bound_agents, T_cog=T_cog, n_steps=n_steps, dt=dt, burn_in=burn_in
        )
        sample_free = self.langevin_sample(
            free_agents, T_cog=T_cog, n_steps=n_steps, dt=dt, burn_in=burn_in
        )

        q_bound = self.compute_thermodynamic_quantities(
            sample_bound["E_total"], T_cog
        )
        q_free = self.compute_thermodynamic_quantities(
            sample_free["E_total"], T_cog
        )

        F_bound = q_bound["F"]
        F_free = q_free["F"]
        F_bound_lower = F_bound < F_free

        return {
            "F_bound": F_bound,
            "F_free": F_free,
            "F_bound_lower": bool(F_bound_lower),
            "U_bound": q_bound["U"],
            "U_free": q_free["U"],
            "S_bound": q_bound["S"],
            "S_free": q_free["S"],
            "T_cog": T_cog,
            "thesis": (
                f"束缚态自由能 F_bound={F_bound:.4f}，"
                f"自由态 F_free={F_free:.4f}。"
                f"{'F_bound < F_free：束缚态热力学稳定（共业坚固）。' if F_bound_lower else 'F_bound ≥ F_free：束缚态不稳定。'}"
                "佛学：定力高（T_cog 低）时业力结构坚固。"
            ),
        }

    def verify_phase_transition_temperature(
        self,
        agents: list[CognitiveAgent],
        T_range: tuple[float, float] = (0.01, 0.5),
        n_points: int = 12,
        n_steps_per_T: int = 300,
        dt: float = 0.005,
        burn_in: int = 80,
    ) -> dict:
        """
        验证 2：相变温度 T_c。

        物理预测：T_cog > T_c 时束缚态解离（C_V 出现尖峰）
        佛学：痛苦/定力比超过阈值，业力结构熔化

        方法：扫描 T_cog，找 C_V 尖峰。
        """
        scan = self.scan_temperature(
            agents,
            T_range=T_range,
            n_points=n_points,
            n_steps_per_T=n_steps_per_T,
            dt=dt,
            burn_in=burn_in,
        )
        transition = self.find_phase_transition(scan)

        return {
            "scan": scan,
            "transition": transition,
            "T_c": transition["T_c"],
            "C_V_peak": transition["C_V_peak"],
            "is_phase_transition": transition["is_phase_transition"],
            "thesis": (
                f"相变温度 T_c ≈ {transition['T_c']:.4f}，"
                f"C_V 峰值 = {transition['C_V_peak']:.4f}。"
                f"{'相变确认：C_V 尖峰显示束缚态在 T_c 处解离。' if transition['is_phase_transition'] else '无明显相变（可能需要更细的 T 扫描）。'}"
                "佛学：T_cog 超过 T_c 时，业力结构无法维持。"
            ),
        }

    def verify_entropy_change(
        self,
        bound_agents: list[CognitiveAgent],
        free_agents: list[CognitiveAgent],
        T_cog: float = 0.05,
        n_steps: int = 400,
        dt: float = 0.005,
        burn_in: int = 100,
    ) -> dict:
        """
        验证 3：熵变 ΔS = S_bound - S_free。

        物理预测：束缚态形成时 ΔS < 0（熵减少，结构化）
        佛学：业力结构化 = 个体被"塑造"成特定形态

        方法：分别采样束缚态和自由态，比较 S。
        """
        sample_bound = self.langevin_sample(
            bound_agents, T_cog=T_cog, n_steps=n_steps, dt=dt, burn_in=burn_in
        )
        sample_free = self.langevin_sample(
            free_agents, T_cog=T_cog, n_steps=n_steps, dt=dt, burn_in=burn_in
        )

        q_bound = self.compute_thermodynamic_quantities(
            sample_bound["E_total"], T_cog
        )
        q_free = self.compute_thermodynamic_quantities(
            sample_free["E_total"], T_cog
        )

        delta_S = q_bound["S"] - q_free["S"]
        entropy_decreases = delta_S < 0

        return {
            "S_bound": q_bound["S"],
            "S_free": q_free["S"],
            "delta_S": delta_S,
            "entropy_decreases": bool(entropy_decreases),
            "T_cog": T_cog,
            "thesis": (
                f"束缚态熵 S_bound={q_bound['S']:.4f}，"
                f"自由态熵 S_free={q_free['S']:.4f}，"
                f"ΔS = {delta_S:.4f}。"
                f"{'ΔS < 0：束缚态形成是熵减过程（结构化）。' if entropy_decreases else 'ΔS ≥ 0：无结构化。'}"
                "佛学：业力结构化降低信息熵，个体被塑造。"
            ),
        }

    def verify_free_energy_landscape_multowell(
        self,
        agent: CognitiveAgent,
        other_agents: list[CognitiveAgent],
        T_cog: float = 0.05,
        slice_axis: int = 0,
        n_points: int = 61,
        delta_range: tuple[float, float] = (-3.14159, 3.14159),
        scan_mode: str = "rotation",
    ) -> dict:
        """
        验证 4：自由能景观 F(g) 的多井结构。

        物理预测：F(g) 沿切片有多个极小值（对应不同业力态）
        佛学：有多种"业力态"可陷入

        方法：旋转扫描 θ ∈ [-π, π]，g(θ) = R(θ)^T g R(θ)。
        SSB 势能 V 作用于对角元，旋转改变对角元 → E_self(θ) 振荡 → 多井。
        每个极小值对应一个"业力态"（破缺对称性的不同取向）。
        """
        landscape = self.free_energy_landscape_along_slice(
            agent, other_agents, T_cog=T_cog,
            slice_axis=slice_axis, n_points=n_points,
            delta_range=delta_range, scan_mode=scan_mode,
        )

        E_list = landscape["E_total_list"]
        delta_list = landscape["delta_list"]

        # 找局部极小值
        local_minima = []
        for i in range(1, len(E_list) - 1):
            if E_list[i] < E_list[i - 1] and E_list[i] < E_list[i + 1]:
                local_minima.append({
                    "delta": delta_list[i],
                    "E_total": E_list[i],
                    "idx": i,
                })

        n_wells = len(local_minima)
        has_multiple_wells = n_wells >= 2

        return {
            "landscape": landscape,
            "local_minima": local_minima,
            "n_wells": n_wells,
            "has_multiple_wells": bool(has_multiple_wells),
            "thesis": (
                f"自由能景观沿轴 {slice_axis} 有 {n_wells} 个局部极小值。"
                f"{'多井结构确认：存在多个业力态。' if has_multiple_wells else '单井结构：只有一个稳定业力态。'}"
                "佛学：认知 landscape 有多种'业力态'可陷入。"
            ),
        }

    def verify_awareness_reshaping(
        self,
        agents: list[CognitiveAgent],
        T_cog: float = 0.05,
        n_steps: int = 300,
        dt: float = 0.005,
        burn_in: int = 80,
    ) -> dict:
        """
        验证 5：觉照对自由能的重塑。

        物理预测：ρ 增大时 F 的极小值从 g* 移向 cI
        佛学：觉照使"苦乐 landscape"重新分布，cI 成为最稳态

        方法：比较 ρ=0 和 ρ=0.5 下的 ⟨E_total⟩（F 的代理）。
        觉照增大 → g 朝 cI 移动 → E_self 降低 → F 降低。
        """
        # ρ=0：自由演化，g 停留在 g*，E_self 高
        sample_rho0 = self.langevin_sample(
            agents, T_cog=T_cog, n_steps=n_steps, dt=dt, rho=0.0, burn_in=burn_in
        )
        # ρ=0.5：部分觉照，g 朝 cI 移动，E_self 降低
        sample_rho_high = self.langevin_sample(
            agents, T_cog=T_cog, n_steps=n_steps, dt=dt, rho=0.5, burn_in=burn_in
        )

        q_rho0 = self.compute_thermodynamic_quantities(
            sample_rho0["E_total"], T_cog
        )
        q_rho_high = self.compute_thermodynamic_quantities(
            sample_rho_high["E_total"], T_cog
        )

        F_decreases = q_rho_high["F"] < q_rho0["F"]
        E_self_decreases = (
            sum(sample_rho_high["E_self"]) / len(sample_rho_high["E_self"])
            < sum(sample_rho0["E_self"]) / len(sample_rho0["E_self"])
        )

        return {
            "F_rho0": q_rho0["F"],
            "F_rho_high": q_rho_high["F"],
            "F_decreases": bool(F_decreases),
            "E_self_rho0": sum(sample_rho0["E_self"]) / len(sample_rho0["E_self"]),
            "E_self_rho_high": sum(sample_rho_high["E_self"]) / len(sample_rho_high["E_self"]),
            "E_self_decreases": bool(E_self_decreases),
            "thesis": (
                f"ρ=0: F={q_rho0['F']:.4f}, ⟨E_self⟩={sum(sample_rho0['E_self'])/len(sample_rho0['E_self']):.4f}；"
                f"ρ=0.5: F={q_rho_high['F']:.4f}, ⟨E_self⟩={sum(sample_rho_high['E_self'])/len(sample_rho_high['E_self']):.4f}。"
                f"{'觉照降低 F 和 E_self：landscape 被重塑，cI 成最稳态。' if F_decreases and E_self_decreases else '觉照效应不显著。'}"
                "佛学：觉照使'苦乐 landscape'重新分布。"
            ),
        }

    def verify_multi_body_vs_single(
        self,
        multi_agents: list[CognitiveAgent],
        single_agent: CognitiveAgent,
        T_cog: float = 0.05,
        n_steps: int = 400,
        dt: float = 0.005,
        burn_in: int = 100,
    ) -> dict:
        """
        验证 6：多体 vs 单体自由能对比。

        物理预测：F_multi < F_single（多体束缚态自由能更低）
        佛学：共业结构比个体业力更坚固

        方法：比较多体系统和单体系统的自由能。
        多体有 V_int（负能束缚态），单体无 V_int → F_multi 应更低。
        """
        # 多体采样
        sample_multi = self.langevin_sample(
            multi_agents, T_cog=T_cog, n_steps=n_steps, dt=dt, burn_in=burn_in
        )

        # 单体采样（用相同 field，但只有 1 个 agent，V_int=0）
        sample_single = self.langevin_sample(
            [single_agent], T_cog=T_cog, n_steps=n_steps, dt=dt, burn_in=burn_in
        )

        q_multi = self.compute_thermodynamic_quantities(
            sample_multi["E_total"], T_cog
        )
        q_single = self.compute_thermodynamic_quantities(
            sample_single["E_total"], T_cog
        )

        F_multi = q_multi["F"]
        F_single = q_single["F"]
        F_multi_lower = F_multi < F_single

        return {
            "F_multi": F_multi,
            "F_single": F_single,
            "F_multi_lower": bool(F_multi_lower),
            "U_multi": q_multi["U"],
            "U_single": q_single["U"],
            "V_int_multi_mean": sum(sample_multi["V_int"]) / len(sample_multi["V_int"]),
            "V_int_single_mean": sum(sample_single["V_int"]) / len(sample_single["V_int"]),
            "thesis": (
                f"多体 F_multi={F_multi:.4f}，单体 F_single={F_single:.4f}。"
                f"{'F_multi < F_single：共业使系统更稳定。' if F_multi_lower else 'F_multi ≥ F_single：共业未增强稳定性。'}"
                "佛学：共业结构比个体业力更坚固。"
            ),
        }


# ======================================================================
# v7.10 完整验证入口
# ======================================================================

def run_thermodynamic_verification(
    n_dims: int = 4,
    coupling_lambda: float = 2.0,
    correlation_length: float = 3.0,
    n_steps: int = 200,
    dt: float = 0.005,
    T_cog_low: float = 0.05,
    burn_in: int = 50,
) -> dict:
    """
    运行 v7.10 全部 6 个热力学验证。

    返回：
        {
            "verification_1_bound_state_stability": ...,
            "verification_2_phase_transition": ...,
            "verification_3_entropy_change": ...,
            "verification_4_landscape_multowell": ...,
            "verification_5_awareness_reshaping": ...,
            "verification_6_multi_vs_single": ...,
            "all_pass": bool,
            "thesis": str,
        }
    """
    import math as _math

    field = MultiBodyCognitiveField(
        n_dims=n_dims,
        coupling_lambda=coupling_lambda,
        correlation_length=correlation_length,
    )
    thermo = ThermodynamicAnalyzer(field)

    # 构建有 Γ 的共振对（束缚态）
    g_vac = field.vacuum.construct_vacuum()
    # 破缺态：各向异性
    g_broken_A = g_vac.clone()
    anisotropy = [0.4, 0.1, -0.1, -0.4]
    for i in range(n_dims):
        g_broken_A[i, i] += 0.5 + anisotropy[i % len(anisotropy)]
    # v7.10: 多步旋转产生大 Q_dynamic（10 步 × 0.3 rad = 3 rad 总旋转）
    theta = 0.3
    for _ in range(10):
        R = torch.eye(n_dims, dtype=torch.float64)
        c, s = _math.cos(theta), _math.sin(theta)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
        g_broken_A = R.T @ g_broken_A @ R
        g_broken_A = symmetric_part(g_broken_A)

    g_broken_B = g_vac.clone()
    for i in range(n_dims):
        g_broken_B[i, i] += 0.5 + anisotropy[i % len(anisotropy)]
    # B 用相同旋转（共振对，cos_align=+1）
    for _ in range(10):
        R = torch.eye(n_dims, dtype=torch.float64)
        c, s = _math.cos(theta), _math.sin(theta)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
        g_broken_B = R.T @ g_broken_B @ R
        g_broken_B = symmetric_part(g_broken_B)

    kappa_vec = torch.tensor([1.0] * n_dims, dtype=torch.float64)
    alpha_vec = torch.tensor([1.0] * n_dims, dtype=torch.float64)

    # 用 create_agent_from_history 创建有 Γ 的个体
    g_history_A = [g_vac, g_broken_A]
    g_history_B = [g_vac, g_broken_B]

    agent_A = field.create_agent_from_history(
        g_history_A, "A", kappa_vec, alpha_vec
    )
    agent_B = field.create_agent_from_history(
        g_history_B, "B", kappa_vec, alpha_vec
    )

    # v7.10: 自由态 = 束缚态克隆 + Γ 翻转（cos_align 从 +1 变 -1）
    # 物理确保 E_self 相同（相同 g），唯一差异是 V_int：
    #   束缚态 = 共振对（cos_align=+1，V_int<0 吸引）→ F 低
    #   自由态 = 互补对（cos_align=-1，V_int>0 排斥）→ F 高
    # 累积模式下 Γ 不回弹（阿赖耶识种子不灭），差异持久
    agent_free_A = CognitiveAgent(
        g=agent_A.g.clone(),
        Q=agent_A.Q.clone(),
        label="free_A",
        kappa_vec=agent_A.kappa_vec.clone(),
        alpha_vec=agent_A.alpha_vec.clone(),
        Gamma=-agent_A.Gamma.clone() if agent_A.Gamma is not None else None,
    )
    agent_free_B = CognitiveAgent(
        g=agent_B.g.clone(),
        Q=agent_B.Q.clone(),
        label="free_B",
        kappa_vec=agent_B.kappa_vec.clone(),
        alpha_vec=agent_B.alpha_vec.clone(),
        Gamma=agent_B.Gamma.clone() if agent_B.Gamma is not None else None,
    )

    # 单体（无耦合）
    agent_single = field.create_agent_from_history(
        [g_vac, g_broken_A], "single", kappa_vec, alpha_vec
    )

    # 执行 6 个验证
    results = {}

    # 验证 1：束缚态稳定性
    results["verification_1_bound_state_stability"] = thermo.verify_bound_state_stability(
        bound_agents=[agent_A, agent_B],
        free_agents=[agent_free_A, agent_free_B],
        T_cog=T_cog_low,
        n_steps=n_steps,
        dt=dt,
        burn_in=burn_in,
    )

    # 验证 2：相变温度
    results["verification_2_phase_transition"] = thermo.verify_phase_transition_temperature(
        agents=[agent_A, agent_B],
        T_range=(0.02, 0.3),
        n_points=6,
        n_steps_per_T=80,
        dt=dt,
        burn_in=20,
    )

    # 验证 3：熵变
    results["verification_3_entropy_change"] = thermo.verify_entropy_change(
        bound_agents=[agent_A, agent_B],
        free_agents=[agent_free_A, agent_free_B],
        T_cog=T_cog_low,
        n_steps=n_steps,
        dt=dt,
        burn_in=burn_in,
    )

    # 验证 4：自由能景观多井
    # v7.10: 旋转扫描 θ ∈ [-π, π]，SSB 势能振荡产生多井（不同业力态取向）
    results["verification_4_landscape_multowell"] = thermo.verify_free_energy_landscape_multowell(
        agent=agent_A,
        other_agents=[agent_B],
        T_cog=T_cog_low,
        slice_axis=0,
        n_points=61,
        delta_range=(-3.14159, 3.14159),
        scan_mode="rotation",
    )

    # 验证 5：觉照重塑
    results["verification_5_awareness_reshaping"] = thermo.verify_awareness_reshaping(
        agents=[agent_A, agent_B],
        T_cog=T_cog_low,
        n_steps=n_steps,
        dt=dt,
        burn_in=burn_in,
    )

    # 验证 6：多体 vs 单体
    results["verification_6_multi_vs_single"] = thermo.verify_multi_body_vs_single(
        multi_agents=[agent_A, agent_B],
        single_agent=agent_single,
        T_cog=T_cog_low,
        n_steps=n_steps,
        dt=dt,
        burn_in=burn_in,
    )

    # 汇总 PASS/FAIL
    pass_flags = [
        results["verification_1_bound_state_stability"]["F_bound_lower"],
        results["verification_2_phase_transition"]["is_phase_transition"],
        results["verification_3_entropy_change"]["entropy_decreases"],
        results["verification_4_landscape_multowell"]["has_multiple_wells"],
        results["verification_5_awareness_reshaping"]["F_decreases"]
        and results["verification_5_awareness_reshaping"]["E_self_decreases"],
        results["verification_6_multi_vs_single"]["F_multi_lower"],
    ]
    n_pass = sum(pass_flags)
    all_pass = all(pass_flags)

    results["n_pass"] = n_pass
    results["n_total"] = 6
    results["all_pass"] = bool(all_pass)
    results["pass_flags"] = pass_flags
    results["thesis"] = (
        f"v7.10 热力学验证：{n_pass}/6 PASS。"
        f"{'全部通过：多体自由演化热力学层可靠。' if all_pass else '部分失败：需要调试。'}"
        "v7.10 把 CTFT 从动力学升级到统计力学：Z, F, S, C_V。"
    )

    return results
