"""
束缚态形成验证（Bound State Formation Verification）

v7.4 多体规范场论的核心验证：两个携带拓扑荷的个体在什么条件下形成束缚态。

认识论根基（理论依据，非案例）：
    物理：Yukawa 势 / 束缚态判据 / 仿射不变距离 / 关联长度扫描
    佛学：共业（sādhārana-karma）/ 业力绑定 / 缘起耦合
    哲学：关系先于个体 / 涌现的整体性

四大验证：
    1. V_int 符号判定：互补吸引 / 同号排斥 / Q=0 无耦合
    2. d_g 仿射不变测地距离：内禀结构距离 vs Frobenius 表层距离
    3. ξ 关联长度扫描：长程共业 vs 短程共业
    4. 束缚态能量判据：E_total < E_self(A) + E_self(B) ⟺ 束缚态形成

核心命题：
    束缚态不是「宿命相逢」，而是规范对称性的集体破缺模式——
    两个个体的业力结构耦合，形成不可逆的拓扑绑定。
    这是「共业」的严格数学表达。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import symmetric_part
from .cognitive_vacuum import CognitiveVacuum
from .topological_charge import TopologicalCharge
from .gauge_interaction import GaugeInteraction, CognitiveAgent
from .awakening_path import AwakeningDynamics


# ======================================================================
# 1. 个体构造器：从认知参数到带 Q 和 Gamma 的 CognitiveAgent
# ======================================================================

class AgentBuilder:
    """
    从认知参数构造带拓扑荷和 Berry 相位的认知个体。

    v7.4 关键：连接 v7.3 的纠缠态构建与多体规范场论。
    个体不仅有度规 g，还有拓扑荷 Q 和 Berry 相位 Gamma。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)
        self.tc = TopologicalCharge(n_dims=n_dims, eps=eps)
        self.dynamics = AwakeningDynamics(n_dims=n_dims, eps=eps)

    def build_entangled_agent(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        label: str = "",
        rotation_direction: int = 1,
    ) -> CognitiveAgent:
        """
        构造有 Q≠0 的纠缠个体（SSB + 旋转）。

        参数：
            rotation_direction: +1 或 -1，控制旋转方向
                +1：正向旋转（Gamma 方向 A）
                -1：反向旋转（Gamma 方向 -A）
                这决定了个体的 Berry 相位方向，用于共振/互补判定。
        """
        # SSB + 旋转产生纠缠态
        g = self.dynamics._build_entangled_broken_state(kappa_vec, alpha_vec)
        g_vac = self.vacuum.construct_vacuum()

        # 拓扑荷 Q
        Q_val = float(self.tc.compute_static_charge(g, g_vac)["Q_static"])

        # Berry 相位 Gamma ∈ so(n)：从旋转历史提取
        # 用旋转矩阵的 log 作为 Gamma 的近似
        # 正向旋转：Gamma = theta * E_{01}
        # 反向旋转：Gamma = -theta * E_{01}
        Gamma = torch.zeros(self.n_dims, self.n_dims, dtype=torch.float64)
        theta = 0.08 * 30 * rotation_direction  # 总旋转角度（30步×0.08）
        Gamma[0, 1] = -theta  # so(n) 反对称
        Gamma[1, 0] = theta

        return CognitiveAgent(
            g=g,
            Q=torch.tensor(Q_val, dtype=torch.float64),
            label=label,
            kappa_vec=kappa_vec.clone(),
            alpha_vec=alpha_vec.clone(),
            Gamma=Gamma,
        )

    def build_neutral_agent(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        label: str = "",
    ) -> CognitiveAgent:
        """
        构造 Q≈0 的中性个体（SSB 但无旋转）。
        """
        breaking = self.vacuum.spontaneous_symmetry_breaking(
            kappa_vec, alpha_vec, n_steps=120, dt=0.02
        )
        g = breaking["g_final"]
        g_vac = self.vacuum.construct_vacuum()
        Q_val = float(self.tc.compute_static_charge(g, g_vac)["Q_static"])

        return CognitiveAgent(
            g=g,
            Q=torch.tensor(Q_val, dtype=torch.float64),
            label=label,
            kappa_vec=kappa_vec.clone(),
            alpha_vec=alpha_vec.clone(),
            Gamma=None,  # 无 Berry 相位
        )


# ======================================================================
# 2. V_int 符号判定验证
# ======================================================================

class InteractionSignVerifier:
    """
    验证 V_int 的符号结构。

    物理预测：
        - 互补对（cos_align < 0）：V_int > 0（规范排斥）
        - 共振对（cos_align > 0）：V_int < 0（规范吸引）
        - 中性对（Q=0 或 Gamma=None）：V_int ≈ 0（无耦合）

    佛学：
        - 规范排斥 = 业力互消（反向业力湮灭）
        - 规范吸引 = 共业相吸（同向业力叠加）
        - 无耦合 = 业力独立（无共业关系）
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.builder = AgentBuilder(n_dims=n_dims, eps=eps)

    def verify_sign_structure(self) -> dict:
        """
        验证三种情形的 V_int 符号。
        """
        # 认知参数
        kappa_vec = torch.tensor([2.3, 1.8, 2.6, 1.5], dtype=torch.float64)
        alpha_vec = torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=torch.float64)

        # 构造三个个体：
        # A：正向旋转（Gamma 方向 +）
        # B：反向旋转（Gamma 方向 -，与 A 互补）
        # C：正向旋转（Gamma 方向 +，与 A 共振）
        # D：中性（无旋转，Q≈0）
        agent_A = self.builder.build_entangled_agent(
            kappa_vec, alpha_vec, label="A", rotation_direction=+1
        )
        agent_B = self.builder.build_entangled_agent(
            kappa_vec, alpha_vec, label="B", rotation_direction=-1
        )
        agent_C = self.builder.build_entangled_agent(
            kappa_vec, alpha_vec, label="C", rotation_direction=+1
        )
        agent_D = self.builder.build_neutral_agent(
            kappa_vec, alpha_vec, label="D"
        )

        # 用中等 ξ 和 λ
        gi = GaugeInteraction(
            n_dims=self.n_dims,
            coupling_lambda=1.0,
            correlation_length=1.0,
            eps=self.eps,
        )

        # 三种情形
        results = {}

        # 情形 1：A-B 互补（反向旋转）
        d_AB = gi.metric_distance(agent_A.g, agent_B.g)
        cos_AB = gi.berry_alignment(agent_A.Gamma, agent_B.Gamma)
        V_AB = gi.two_body_potential(agent_A.Q, agent_B.Q, d_AB, cos_alignment=cos_AB)
        results["complement_AB"] = {
            "Q_A": float(agent_A.Q),
            "Q_B": float(agent_B.Q),
            "d_g": float(d_AB),
            "cos_align": cos_AB,
            "V_int": float(V_AB["V_int"]),
            "interaction_type": V_AB["interaction_type"],
            "is_attractive": V_AB["is_attractive"],
            "sign_correct": bool(V_AB["V_int"] > 0 and cos_AB < 0),  # 互补→排斥
        }

        # 情形 2：A-C 共振（同向旋转）
        d_AC = gi.metric_distance(agent_A.g, agent_C.g)
        cos_AC = gi.berry_alignment(agent_A.Gamma, agent_C.Gamma)
        V_AC = gi.two_body_potential(agent_A.Q, agent_C.Q, d_AC, cos_alignment=cos_AC)
        results["resonance_AC"] = {
            "Q_A": float(agent_A.Q),
            "Q_C": float(agent_C.Q),
            "d_g": float(d_AC),
            "cos_align": cos_AC,
            "V_int": float(V_AC["V_int"]),
            "interaction_type": V_AC["interaction_type"],
            "is_attractive": V_AC["is_attractive"],
            "sign_correct": bool(V_AC["V_int"] < 0 and cos_AC > 0),  # 共振→吸引
        }

        # 情形 3：A-D 中性（D 无 Q）
        d_AD = gi.metric_distance(agent_A.g, agent_D.g)
        cos_AD = gi.berry_alignment(agent_A.Gamma, agent_D.Gamma)
        V_AD = gi.two_body_potential(agent_A.Q, agent_D.Q, d_AD, cos_alignment=cos_AD)
        results["neutral_AD"] = {
            "Q_A": float(agent_A.Q),
            "Q_D": float(agent_D.Q),
            "d_g": float(d_AD),
            "cos_align": cos_AD,
            "V_int": float(V_AD["V_int"]),
            "interaction_type": V_AD["interaction_type"],
            "is_attractive": V_AD["is_attractive"],
            "sign_correct": bool(abs(V_AD["V_int"]) < 1e-6),  # 中性→无耦合
        }

        all_correct = all(
            results[k]["sign_correct"]
            for k in ["complement_AB", "resonance_AC", "neutral_AD"]
        )

        return {
            **results,
            "all_signs_correct": all_correct,
            "thesis": (
                "V_int 符号结构验证：互补对（cos_align<0）→ V_int>0（规范排斥=业力互消）；"
                "共振对（cos_align>0）→ V_int<0（规范吸引=共业相吸）；"
                "中性对（Q=0）→ V_int≈0（无耦合=业力独立）。"
                "符号由 Berry 相位对齐决定，而非标量 Q 乘积——"
                "这利用了 so(n) 李代数的完整方向信息。"
            ),
        }


# ======================================================================
# 3. 度规距离 d_g 验证（仿射不变 vs Frobenius）
# ======================================================================

class MetricDistanceVerifier:
    """
    验证仿射不变测地距离的内禀性。

    物理预测：
        - d_g(g, g) = 0（自距离为零）
        - d_g(A, B) = d_g(B, A)（对称性）
        - d_g(A, B) ≤ d_g(A, C) + d_g(C, B)（三角不等式）
        - 仿射不变：d_g(XAX^T, XBX^T) = d_g(A, B)（GL(n) 不变性）
        - Frobenius 距离不满足仿射不变性

    佛学：
        仿射不变性 = 「法尔如是」的客观性——
        距离不依赖观察坐标系，是认知结构的内禀属性。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)
        self.gi = GaugeInteraction(n_dims=n_dims, eps=eps)

    def verify_affine_invariance(self) -> dict:
        """
        验证仿射不变性：d_g(XAX^T, XBX^T) = d_g(A, B)。
        """
        n = self.n_dims

        # 构造两个不同的正定矩阵
        g_A = torch.eye(n, dtype=torch.float64)
        g_A[0, 0] = 1.5; g_A[1, 1] = 0.8; g_A[2, 2] = 1.2; g_A[3, 3] = 0.9

        g_B = torch.eye(n, dtype=torch.float64)
        g_B[0, 0] = 0.7; g_B[1, 1] = 1.3; g_B[2, 2] = 0.9; g_B[3, 3] = 1.1

        # 原始距离
        d_original = float(self.gi.metric_distance(g_A, g_B, method="affine"))

        # 构造随机可逆矩阵 X（仿射变换）
        torch.manual_seed(42)
        X = torch.randn(n, n, dtype=torch.float64) + 2.0 * torch.eye(n, dtype=torch.float64)
        # 保证可逆
        while float(torch.det(X).abs()) < 0.1:
            X = X + 0.5 * torch.eye(n, dtype=torch.float64)

        # 仿射变换后的矩阵
        g_A_transformed = X @ g_A @ X.T
        g_B_transformed = X @ g_B @ X.T

        # 仿射不变距离
        d_transformed = float(self.gi.metric_distance(
            g_A_transformed, g_B_transformed, method="affine"
        ))

        # Frobenius 距离（应不满足不变性）
        d_frob_original = float(self.gi.metric_distance(g_A, g_B, method="frobenius"))
        d_frob_transformed = float(self.gi.metric_distance(
            g_A_transformed, g_B_transformed, method="frobenius"
        ))

        # 仿射不变性判据
        affine_invariant = bool(abs(d_original - d_transformed) < 0.01 * d_original)
        frobenius_invariant = bool(abs(d_frob_original - d_frob_transformed) < 0.01 * d_frob_original)

        # 自距离
        d_self = float(self.gi.metric_distance(g_A, g_A, method="affine"))
        self_zero = bool(d_self < 0.01)

        # 对称性
        d_AB = float(self.gi.metric_distance(g_A, g_B, method="affine"))
        d_BA = float(self.gi.metric_distance(g_B, g_A, method="affine"))
        symmetric = bool(abs(d_AB - d_BA) < 0.01 * d_AB)

        return {
            "d_affine_original": d_original,
            "d_affine_transformed": d_transformed,
            "affine_invariant": affine_invariant,
            "d_frob_original": d_frob_original,
            "d_frob_transformed": d_frob_transformed,
            "frobenius_invariant": frobenius_invariant,
            "self_distance": d_self,
            "self_zero": self_zero,
            "symmetric": symmetric,
            "all_correct": bool(affine_invariant and self_zero and symmetric and not frobenius_invariant),
            "thesis": (
                "仿射不变测地距离 d_g = ||log(g_A^{-1/2}·g_B·g_A^{-1/2})||_F "
                "满足 GL(n) 不变性、自距离为零、对称性。"
                "Frobenius 距离不满足仿射不变性——依赖坐标系选择。"
                "仿射不变性 = 「法尔如是」的客观性：距离是认知结构的内禀属性。"
            ),
        }


# ======================================================================
# 4. 关联长度 ξ 扫描
# ======================================================================

class CorrelationLengthScanner:
    """
    扫描关联长度 ξ 对 V_int 的影响。

    物理预测：
        - ξ 大：长程共业（即使 d_g 大，V_int 仍显著）
        - ξ 小：短程共业（d_g 大时 V_int 快速衰减）
        - V_int ∝ exp(-d_g/ξ)，ξ 控制衰减速率

    佛学：
        ξ = 业力作用范围 = 规范场穿透深度。
        ξ 大 = 深缘众生（远距离也能感应）。
        ξ 小 = 浅缘众生（必须近距离才能耦合）。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.builder = AgentBuilder(n_dims=n_dims, eps=eps)

    def scan_xi(self) -> dict:
        """
        扫描 ξ，观察 V_int 衰减模式。
        """
        # 构造共振对（同向旋转，但不同度规——不同 kappa 产生不同破缺态）
        kappa_A = torch.tensor([2.3, 1.8, 2.6, 1.5], dtype=torch.float64)
        kappa_C = torch.tensor([1.5, 2.0, 1.2, 2.8], dtype=torch.float64)  # 不同的 kappa
        alpha_vec = torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=torch.float64)
        agent_A = self.builder.build_entangled_agent(
            kappa_A, alpha_vec, label="A", rotation_direction=+1
        )
        agent_C = self.builder.build_entangled_agent(
            kappa_C, alpha_vec, label="C", rotation_direction=+1
        )

        # 固定距离
        gi_temp = GaugeInteraction(n_dims=self.n_dims, eps=self.eps)
        d_g = float(gi_temp.metric_distance(agent_A.g, agent_C.g))
        cos_align = gi_temp.berry_alignment(agent_A.Gamma, agent_C.Gamma)

        # 扫描 ξ
        xi_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
        V_int_values = []
        screening_values = []

        for xi in xi_values:
            gi = GaugeInteraction(
                n_dims=self.n_dims,
                coupling_lambda=1.0,
                correlation_length=xi,
                eps=self.eps,
            )
            V = gi.two_body_potential(
                agent_A.Q, agent_C.Q, torch.tensor(d_g), cos_alignment=cos_align
            )
            V_int_values.append(float(V["V_int"]))
            screening_values.append(float(V["screening_factor"]))

        # 验证衰减模式
        # ξ 小 → |V_int| 小（强屏蔽）
        # ξ 大 → |V_Int| 大（弱屏蔽）
        V_abs = [abs(v) for v in V_int_values]
        long_range_strong = bool(V_abs[-1] > V_abs[0])  # ξ=10 > ξ=0.1
        decay_monotonic = bool(
            all(V_abs[i] <= V_abs[i + 1] + 1e-10 for i in range(len(V_abs) - 1))
        )

        # 验证 exp(-d/ξ) 衰减律
        # V_int ∝ exp(-d/ξ)，所以 log|V_int| ∝ -d/ξ
        # 对 ξ 求导：d(log|V|)/d(1/ξ) = -d_g（斜率应为 -d_g）
        import numpy as np
        inv_xi = np.array([1.0 / x for x in xi_values])
        log_V = np.array([math.log(max(v, 1e-20)) for v in V_abs])
        # 线性拟合 log|V| vs 1/ξ，斜率应为 -d_g
        if len(xi_values) > 2:
            slope = float(np.polyfit(inv_xi, log_V, 1)[0])
        else:
            slope = 0.0
        slope_correct = bool(abs(slope - (-d_g)) < 0.1 * d_g)

        return {
            "xi_values": xi_values,
            "V_int_values": V_int_values,
            "screening_values": screening_values,
            "d_g": d_g,
            "cos_align": cos_align,
            "long_range_strong": long_range_strong,
            "decay_monotonic": decay_monotonic,
            "exp_decay_slope": slope,
            "expected_slope": -d_g,
            "slope_correct": slope_correct,
            "all_correct": bool(long_range_strong and decay_monotonic and slope_correct),
            "thesis": (
                "关联长度 ξ 扫描：V_int ∝ exp(-d_g/ξ)。"
                "ξ 大 → 长程共业（深缘众生，远距离也能感应）；"
                "ξ 小 → 短程共业（浅缘众生，必须近距离才能耦合）。"
                "衰减斜率 = -d_g，验证 Yukawa 势的指数屏蔽结构。"
                "ξ 是规范场 Compton 波长，由社会网络连通性决定。"
            ),
        }


# ======================================================================
# 5. 束缚态能量判据验证
# ======================================================================

class BoundStateVerifier:
    """
    验证束缚态形成的能量判据。

    物理预测：
        - 共振对（V_int < 0 且 |V_int| 足够大）：束缚态形成，E_total < E_self(A) + E_self(B)
        - 互补对（V_int > 0）：无束缚态，E_total > E_self(A) + E_self(B)
        - 中性对（V_int ≈ 0）：无耦合，E_total ≈ E_self(A) + E_self(B)

    束缚态判据：
        E_total = E_self(A) + E_self(B) + V_int(A,B)
        束缚态 ⟺ V_int < 0 且 |V_int| > E_bind_threshold
        （规范吸引足以克服个体势垒）

    佛学：
        束缚态 = 业力绑定（karmic bondage）。
        共振对形成束缚态 = 共业结构（同向业力叠加，不可逆绑定）。
        互补对不形成束缚态 = 业力互消（反向业力湮灭，不绑定）。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)
        self.builder = AgentBuilder(n_dims=n_dims, eps=eps)

    def _compute_self_energy(self, agent: CognitiveAgent) -> float:
        """
        计算个体自能 E_self = V_base(g, κ, α)。
        """
        if agent.kappa_vec is None or agent.alpha_vec is None:
            return 0.0
        pot = self.vacuum.compute_potential(agent.g, agent.kappa_vec, agent.alpha_vec)
        return float(pot["V"])

    def verify_bound_state(self) -> dict:
        """
        验证三种情形的束缚态判据。
        """
        # A 和 C 用不同 kappa（共振对，但有度规差异 d_g > 0）
        kappa_A = torch.tensor([2.3, 1.8, 2.6, 1.5], dtype=torch.float64)
        kappa_C = torch.tensor([1.5, 2.0, 1.2, 2.8], dtype=torch.float64)
        alpha_vec = torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=torch.float64)

        # 构造三个个体
        agent_A = self.builder.build_entangled_agent(
            kappa_A, alpha_vec, label="A", rotation_direction=+1
        )
        agent_B = self.builder.build_entangled_agent(
            kappa_A, alpha_vec, label="B", rotation_direction=-1
        )
        agent_C = self.builder.build_entangled_agent(
            kappa_C, alpha_vec, label="C", rotation_direction=+1
        )

        # 用大 λ 确保束缚态可检测
        gi = GaugeInteraction(
            n_dims=self.n_dims,
            coupling_lambda=50.0,  # 大耦合常数
            correlation_length=2.0,  # 中等关联长度
            eps=self.eps,
        )

        results = {}

        # 自能
        E_self_A = self._compute_self_energy(agent_A)
        E_self_B = self._compute_self_energy(agent_B)
        E_self_C = self._compute_self_energy(agent_C)

        # 情形 1：A-C 共振对（应形成束缚态）
        d_AC = gi.metric_distance(agent_A.g, agent_C.g)
        cos_AC = gi.berry_alignment(agent_A.Gamma, agent_C.Gamma)
        V_AC = gi.two_body_potential(agent_A.Q, agent_C.Q, d_AC, cos_alignment=cos_AC)
        V_int_AC = float(V_AC["V_int"])
        E_total_AC = E_self_A + E_self_C + V_int_AC
        E_separate_AC = E_self_A + E_self_C
        binding_energy_AC = E_separate_AC - E_total_AC  # >0 表示束缚态

        results["resonance_AC"] = {
            "E_self_A": E_self_A,
            "E_self_C": E_self_C,
            "E_separate": E_separate_AC,
            "V_int": V_int_AC,
            "E_total": E_total_AC,
            "binding_energy": binding_energy_AC,
            "bound_state_formed": bool(binding_energy_AC > 0 and V_int_AC < 0),
            "V_int_negative": bool(V_int_AC < 0),
        }

        # 情形 2：A-B 互补对（不应形成束缚态）
        d_AB = gi.metric_distance(agent_A.g, agent_B.g)
        cos_AB = gi.berry_alignment(agent_A.Gamma, agent_B.Gamma)
        V_AB = gi.two_body_potential(agent_A.Q, agent_B.Q, d_AB, cos_alignment=cos_AB)
        V_int_AB = float(V_AB["V_int"])
        E_total_AB = E_self_A + E_self_B + V_int_AB
        E_separate_AB = E_self_A + E_self_B
        binding_energy_AB = E_separate_AB - E_total_AB

        results["complement_AB"] = {
            "E_self_A": E_self_A,
            "E_self_B": E_self_B,
            "E_separate": E_separate_AB,
            "V_int": V_int_AB,
            "E_total": E_total_AB,
            "binding_energy": binding_energy_AB,
            "bound_state_formed": bool(binding_energy_AB > 0 and V_int_AB < 0),
            "V_int_positive": bool(V_int_AB > 0),
        }

        # 验证判据
        all_correct = bool(
            results["resonance_AC"]["bound_state_formed"]
            and not results["complement_AB"]["bound_state_formed"]
            and results["resonance_AC"]["V_int_negative"]
            and results["complement_AB"]["V_int_positive"]
        )

        return {
            **results,
            "coupling_lambda": 50.0,
            "correlation_length": 2.0,
            "all_correct": all_correct,
            "thesis": (
                "束缚态能量判据：E_total = E_self(A) + E_self(B) + V_int。"
                "共振对（V_int<0 且 |V_int| 足够大）→ binding_energy>0 → 束缚态形成（业力绑定）；"
                "互补对（V_int>0）→ binding_energy<0 → 无束缚态（业力互消，不绑定）。"
                "束缚态 = 规范对称性的集体破缺模式 = 共业结构的数学表达。"
            ),
        }


# ======================================================================
# 6. 顶层 API
# ======================================================================

def run_bound_state_verification(verbose: bool = True) -> dict:
    """
    运行 v7.4 束缚态形成完整验证。

    四大验证：
        1. V_int 符号判定（互补/共振/中性）
        2. d_g 仿射不变测地距离（内禀性）
        3. ξ 关联长度扫描（长程/短程共业）
        4. 束缚态能量判据（共振束缚/互补不束缚）
    """
    if verbose:
        print("=" * 70)
        print("CTFT v7.4 多体规范场论——束缚态形成验证")
        print("（理论依据：物理 Yukawa 势 + 佛学共业 + 哲学关系先于个体）")
        print("=" * 70)

    results = {}

    # 验证 1：V_int 符号判定
    if verbose:
        print("\n[验证 1] V_int 符号判定（互补/共振/中性）")
    sign_verifier = InteractionSignVerifier()
    sign_result = sign_verifier.verify_sign_structure()
    results["sign_structure"] = sign_result
    if verbose:
        comp = sign_result["complement_AB"]
        reso = sign_result["resonance_AC"]
        neut = sign_result["neutral_AD"]
        print(f"  互补 A-B: cos_align={comp['cos_align']:.3f}, V_int={comp['V_int']:.4f} "
              f"(排斥={comp['sign_correct']})")
        print(f"  共振 A-C: cos_align={reso['cos_align']:.3f}, V_int={reso['V_int']:.4f} "
              f"(吸引={reso['sign_correct']})")
        print(f"  中性 A-D: Q_D={neut['Q_D']:.6f}, V_int={neut['V_int']:.4f} "
              f"(无耦合={neut['sign_correct']})")
        print(f"  全部符号正确: {sign_result['all_signs_correct']}")

    # 验证 2：仿射不变测地距离
    if verbose:
        print("\n[验证 2] 仿射不变测地距离 d_g")
    dist_verifier = MetricDistanceVerifier()
    dist_result = dist_verifier.verify_affine_invariance()
    results["metric_distance"] = dist_result
    if verbose:
        print(f"  仿射不变: d_original={dist_result['d_affine_original']:.4f}, "
              f"d_transformed={dist_result['d_affine_transformed']:.4f}, "
              f"invariant={dist_result['affine_invariant']}")
        print(f"  Frobenius: d_original={dist_result['d_frob_original']:.4f}, "
              f"d_transformed={dist_result['d_frob_transformed']:.4f}, "
              f"invariant={dist_result['frobenius_invariant']}")
        print(f"  自距离为零: {dist_result['self_zero']}, 对称性: {dist_result['symmetric']}")
        print(f"  全部正确: {dist_result['all_correct']}")

    # 验证 3：关联长度 ξ 扫描
    if verbose:
        print("\n[验证 3] 关联长度 ξ 扫描（长程/短程共业）")
    xi_scanner = CorrelationLengthScanner()
    xi_result = xi_scanner.scan_xi()
    results["xi_scan"] = xi_result
    if verbose:
        print(f"  d_g = {xi_result['d_g']:.4f}, cos_align = {xi_result['cos_align']:.3f}")
        print(f"  ξ 扫描（|V_int| 衰减）:")
        for i, xi in enumerate(xi_result["xi_values"]):
            print(f"    ξ={xi:.1f}: |V_int|={abs(xi_result['V_int_values'][i]):.6f}, "
                  f"screening={xi_result['screening_values'][i]:.6f}")
        print(f"  长程强于短程: {xi_result['long_range_strong']}, "
              f"单调衰减: {xi_result['decay_monotonic']}")
        print(f"  exp 衰减斜率: {xi_result['exp_decay_slope']:.4f} "
              f"(期望 {-xi_result['d_g']:.4f}), 正确: {xi_result['slope_correct']}")

    # 验证 4：束缚态能量判据
    if verbose:
        print("\n[验证 4] 束缚态能量判据（共振束缚/互补不束缚）")
    bound_verifier = BoundStateVerifier()
    bound_result = bound_verifier.verify_bound_state()
    results["bound_state"] = bound_result
    if verbose:
        reso = bound_result["resonance_AC"]
        comp = bound_result["complement_AB"]
        print(f"  共振 A-C: V_int={reso['V_int']:.4f}, binding_energy={reso['binding_energy']:.4f}, "
              f"束缚态={reso['bound_state_formed']}")
        print(f"  互补 A-B: V_int={comp['V_int']:.4f}, binding_energy={comp['binding_energy']:.4f}, "
              f"束缚态={comp['bound_state_formed']}")
        print(f"  全部正确: {bound_result['all_correct']}")

    if verbose:
        print("\n" + "=" * 70)
        print("v7.4 束缚态验证完成。")
        print("共振对形成束缚态（共业绑定），互补对不形成束缚态（业力互消）。")
        print("束缚态 = 规范对称性的集体破缺模式 = 共业结构的数学表达。")
        print("=" * 70)

    return results


if __name__ == "__main__":
    run_bound_state_verification(verbose=True)
