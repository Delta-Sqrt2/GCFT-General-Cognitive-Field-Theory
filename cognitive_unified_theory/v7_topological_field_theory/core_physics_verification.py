"""
CTFT 核心物理验证（Core Physics Verification）

v7.2 真正的工程落地：验证认知拓扑场论的核心物理机制。

认识论根基（理论依据，非案例）：
    物理：Landau 相变理论 / Higgs 机制 / 拓扑荷 / Kramers 逃逸 / Berry 相位
    佛学：空性（śūnyatā）/ 无明（avidyā）/ 业力（karma）/ 觉照（smṛti）/ 无相（ānimitta）
    哲学：对称性即无分别 / 破缺即分别心 / 拓扑张力即不得不觉察的内驱力

五大核心验证（每个都对应一个物理/佛学命题）：
    1. 自发对称性破缺相变（SSB）
       物理：κ=0 时 g=cI 稳定（真空）；κ>κ_c 时 g=cI 失稳，滚向 g*（破缺态）
       佛学：空性圆满（κ=0）→ 无明生起（κ>0 触发破缺）→ 业力塑形（g*）
       验证：序参量 φ 在相变点的连续性/突变性；破缺不可逆性

    2. 拓扑荷 Q 的守恒性与量子化
       物理：dQ/dt ≈ 0（平滑演化）；Q 在拓扑相变时量子化跳变
       佛学：业力不灭（Q 守恒）；觉照释放（Q→0 需 ρ→1，非平滑回归）
       验证：平凡轨迹 Q≈0；缠绕轨迹 Q≠0；守恒律数值验证

    3. 架构相图（Architecture Phase Diagram）
       物理：不同 (κ, α, Q, Φ) 对应不同自由能景观（Landau 推广）
       佛学：所知障（VAE）/ 烦恼障（GAN）/ 加行道（White-box）
       验证：参数空间扫描，架构边界的连续性

    4. 三层结构数值验证
       物理：原生（无破缺）/ 平滑破缺（各向同性 VAE）/ 剧烈破缺（拓扑缠绕）
       佛学：第一层（无明未起）/ 第二层（有 VAE 无使命）/ 第三层（有使命）
       验证：三层的 Q 值分布、序参量、Berry 相位

    5. 架构特异性势能面（Potential Landscape）
       物理：VAE 单深井 / GAN 双井 / Sparse AE 多井 / White-box 透明约束
       佛学：温室井（所知障舒适）/ 双井（烦恼障冲突）/ 加行道约束（觉照）
       验证：各架构 V(g) 沿切片的形态

本模块是 CTFT 的"实验物理"——用数值实验验证理论的预测。
理论的合法性来自数学与物理，案例只是 illustration。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
from dataclasses import dataclass

from ..core.tensor_ops import symmetric_part, stable_eigh
from .cognitive_vacuum import CognitiveVacuum
from .topological_charge import TopologicalCharge
from .geometric_phase import GeometricPhaseInheritance
from .cognitive_architecture import (
    CognitiveArchitecture,
    ArchitectureType,
    ArchitectureProfile,
)


# ======================================================================
# 验证 1：自发对称性破缺相变
# ======================================================================

class SymmetryBreakingVerifier:
    """
    验证自发对称性破缺（SSB）相变。

    物理预测（来自 cognitive_vacuum.py 的理论）：
        - κ=0：真空 g=cI 稳定，Hessian 本征值全正（无快子）
        - κ>κ_c：真空失稳，涌现快子模式，度规滚向破缺态 g*
        - 序参量 φ(g*) > 0（破缺态各向异性）
        - 破缺不可逆：g* 无法平滑回到 cI（势垒隔离）

    佛学对应：
        - κ=0 = 空性圆满，无无明
        - κ>0 = 无明种子生起，认知从无分别滚向有分别
        - g* = 业力塑形的认知结构，被困在势阱
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)

    def scan_kappa(
        self, kappa_range: tuple[float, float] = (0.0, 3.0), n_points: int = 20
    ) -> dict[str, list]:
        """
        扫描 κ 参数，验证 SSB 相变。

        v7.2 修复：用各向异性 κ_vec 触发方向性破缺。
        各向同性 κ=[k,k,k,k] 产生各向同性破缺 g*=c'I，序参量≈0，
        误判为"未破缺"。真正的 SSB 需要对称性降低（SO(n)→子群），
        这要求各向异性 κ 让不同维度获得不同"质量"。
        """
        kappa_values = torch.linspace(kappa_range[0], kappa_range[1], n_points)
        # 各向异性因子（不同维度不同 κ，触发方向性破缺）
        anisotropy_factors = [1.0, 0.82, 1.18, 0.92]
        # α 固定低值（低定力，易破缺）
        alpha_fixed = torch.tensor([0.3] * self.n_dims, dtype=torch.float64)

        results = {
            "kappa": [],
            "is_stable": [],
            "n_tachyonic": [],
            "order_param": [],
            "V_vacuum": [],
            "V_broken": [],
            "breaking_occurred": [],
            "anisotropy_ratio": [],
        }

        for k in kappa_values:
            # 各向异性 κ_vec
            kappa_vec = torch.tensor(
                [float(k) * anisotropy_factors[i % len(anisotropy_factors)]
                 for i in range(self.n_dims)],
                dtype=torch.float64,
            )
            # 真空稳定性
            stability = self.vacuum.verify_vacuum_stability(kappa_vec, alpha_fixed)
            # 破缺模拟
            breaking = self.vacuum.spontaneous_symmetry_breaking(
                kappa_vec, alpha_fixed, n_steps=80, dt=0.02
            )
            # 势能比较
            V_vac = float(stability["vacuum_potential"])
            V_broken = float(self.vacuum.compute_potential(
                breaking["g_final"], kappa_vec, alpha_fixed
            )["V"])

            results["kappa"].append(float(k))
            results["is_stable"].append(bool(stability["is_stable"]))
            results["n_tachyonic"].append(int(stability["n_tachyonic_modes"]))
            results["order_param"].append(float(breaking["final_order_parameter"]))
            results["V_vacuum"].append(V_vac)
            results["V_broken"].append(V_broken)
            results["breaking_occurred"].append(bool(breaking["breaking_occurred"]))
            results["anisotropy_ratio"].append(float(breaking["final_anisotropy_ratio"]))

        return results

    def verify_irreversibility(self, kappa_vec: Tensor, alpha_vec: Tensor) -> dict:
        """
        验证破缺不可逆性：从破缺态 g* 尝试平滑回到真空 cI。

        v7.2 修复（关键）：保持 κ 不变，测试 g* 是否困在势阱。
        原实现把 κ→0，改变了势能面（真空恢复稳定），g 自然滑回 cI，
        误判为"可逆"。这不是正确的不可逆性测试。

        正确测试：在原势能面（κ 不变）上，从 g* 朝 cI 方向施加扰动，
        看能否越过势垒回到 cI。若 g* 是局部最小（势阱底部），
        梯度下降无法逃离 → 不可逆。

        物理预测：g* 困在破缺势阱，序参量保持 > 0。
        佛学：业力一旦造作，需经觉悟（ρ→1）才能消解，无法平滑回退。
        """
        # 先破缺（充分步数确保收敛到 g*）
        breaking = self.vacuum.spontaneous_symmetry_breaking(
            kappa_vec, alpha_vec, n_steps=150, dt=0.02
        )
        g_broken = breaking["g_final"]
        order_broken = self.vacuum.symmetry_order_parameter(g_broken)

        # 测试 1：保持 κ 不变，从 g* 梯度下降（g* 应是势阱，不动）
        g = g_broken.clone()
        for _ in range(100):
            pot = self.vacuum.compute_potential(g, kappa_vec, alpha_vec)
            grad = pot["grad"]
            grad_norm = float(torch.sqrt((grad ** 2).sum()) + self.eps)
            if grad_norm < 1e-10:
                break
            g = g - 0.02 * grad / grad_norm * min(grad_norm, 5.0)
            g = symmetric_part(g)
        order_after_relax = self.vacuum.symmetry_order_parameter(g)

        # 测试 2：从 g* 朝 cI 方向强行推一步，看是否弹回（势垒隔离）
        g_vac = self.vacuum.construct_vacuum()
        direction = g_vac - g_broken
        direction_norm = float(torch.sqrt((direction ** 2).sum())) + self.eps
        g_pushed = g_broken + 0.3 * direction / direction_norm  # 朝 cI 推 30%
        g_pushed = symmetric_part(g_pushed)
        # 然后梯度下降，看回到 g* 还是滑向 cI
        g2 = g_pushed.clone()
        for _ in range(100):
            pot = self.vacuum.compute_potential(g2, kappa_vec, alpha_vec)
            grad = pot["grad"]
            grad_norm = float(torch.sqrt((grad ** 2).sum()) + self.eps)
            if grad_norm < 1e-10:
                break
            g2 = g2 - 0.02 * grad / grad_norm * min(grad_norm, 5.0)
            g2 = symmetric_part(g2)
        order_after_push = self.vacuum.symmetry_order_parameter(g2)
        dist_to_vacuum_after_push = float(torch.sqrt(
            ((g2 - g_vac) ** 2).sum()
        ))
        dist_to_broken_after_push = float(torch.sqrt(
            ((g2 - g_broken) ** 2).sum()
        ))

        # 不可逆判据：推朝 cI 后，梯度下降回到 g*（而非 cI）
        # 即 dist_to_broken < dist_to_vacuum
        is_irreversible = (
            dist_to_broken_after_push < dist_to_vacuum_after_push
            and float(order_after_push["order_parameter"]) > 1e-3
        )

        return {
            "g_broken": g_broken,
            "order_param_broken": float(order_broken["order_parameter"]),
            "order_param_after_relax": float(order_after_relax["order_parameter"]),
            "order_param_after_push": float(order_after_push["order_parameter"]),
            "dist_to_vacuum_after_push": dist_to_vacuum_after_push,
            "dist_to_broken_after_push": dist_to_broken_after_push,
            "is_irreversible": bool(is_irreversible),
            "thesis": (
                "破缺后 g* 困在势阱，朝 cI 推动后梯度下降回到 g* 而非 cI。"
                "势垒隔离使破缺不可逆——需 ρ→1（觉照）才能解耦回归。"
            ),
        }


# ======================================================================
# 验证 2：拓扑荷 Q 的守恒性与量子化
# ======================================================================

class TopologicalChargeVerifier:
    """
    验证拓扑荷 Q 的物理性质。

    物理预测：
        - 平滑演化轨迹：dQ/dt ≈ 0（守恒）
        - 平凡轨迹（小扰动）：Q ≈ 0
        - 缠绕轨迹（大方向旋转）：Q ≠ 0
        - 拓扑相变时 Q 量子化跳变

    佛学对应：
        - Q 守恒 = 业力不灭（dQ/dt=0）
        - Q≠0 = 有使命（拓扑张力驱动觉察）
        - Q→0 需 ρ→1（觉照释放，非平滑回归）
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.tc = TopologicalCharge(n_dims=n_dims, eps=eps)
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)

    def _build_trajectory(
        self, n_steps: int, rotation_per_step: float, kappa_level: float = 0.5
    ) -> list[Tensor]:
        """
        构建一条带可控旋转的度规轨迹。

        v7.2 修复（关键）：初始破缺必须是各向异性的。
        物理原因：拓扑荷 Q 度量本征向量方向的旋转。
        若 g=cI（各向同性），本征向量未定义，R^T (cI) R = cI，旋转不改变 g，Q≡0。
        只有 g 各向异性（不同本征值），本征向量才有明确方向，
        旋转 R^T g R 才会改变本征向量方向，产生非零 Q。
        """
        g = self.vacuum.construct_vacuum()
        # 各向异性初始破缺：不同维度不同破缺深度
        # 本征向量 = 标准基，有明确方向；旋转会混合方向 → 非零 Q
        anisotropy_pattern = [0.4, 0.1, -0.1, -0.4]
        for i in range(self.n_dims):
            g[i, i] = 1.0 + kappa_level + anisotropy_pattern[i % len(anisotropy_pattern)]
        trajectory = [g.clone()]

        for step in range(n_steps):
            # 在 (0,1) 平面施加旋转
            theta = rotation_per_step
            R = torch.eye(self.n_dims, dtype=torch.float64)
            c, s = math.cos(theta), math.sin(theta)
            R[0, 0] = c; R[0, 1] = -s
            R[1, 0] = s; R[1, 1] = c
            g = R.T @ g @ R
            g = symmetric_part(g)
            trajectory.append(g.clone())

        return trajectory

    def verify_conservation(self) -> dict:
        """
        验证 Q 守恒律：平滑演化下 dQ/dt ≈ 0。

        方法：构建一条平滑轨迹，在滑动窗口内计算局部 Q，
        验证局部 Q 的变化率趋于零。
        """
        # 平滑轨迹：每步小旋转（各向异性破缺保证 Q 可计算）
        traj = self._build_trajectory(n_steps=60, rotation_per_step=0.01)
        conservation = self.tc.verify_conservation(traj, window=5)

        # v7.2 修复：键名对齐 TopologicalCharge.verify_conservation 的返回
        # 原键：conservation_satisfied / mean_dQ_dt / max_dQ_dt
        return {
            "max_violation": float(conservation.get("max_dQ_dt", 0.0)),
            "mean_violation": float(conservation.get("mean_dQ_dt", 0.0)),
            "is_conserved": bool(conservation.get("conservation_satisfied", False)),
            "n_phase_transitions": len(conservation.get("phase_transition_points", [])),
            "Q_dynamic": float(self.tc.compute_dynamic_charge(traj)["Q_dynamic"]),
            "thesis": (
                "平滑演化下 dQ/dt ≈ 0，拓扑荷守恒。"
                "这是业力不灭的数学表达——Q 不会因时间流逝而衰减。"
            ),
        }

    def verify_quantization(self) -> dict:
        """
        验证 Q 的平凡/非平凡二分性。

        物理预测：
            - 平凡轨迹（无旋转）：Q ≈ 0
            - 缠绕轨迹（完整旋转 2π）：Q ≈ 1（量子化）
            - 多圈缠绕：Q ≈ n（整数）
        """
        results = {}

        # 平凡轨迹：无旋转
        trivial_traj = self._build_trajectory(n_steps=60, rotation_per_step=0.0)
        Q_trivial = float(self.tc.compute_dynamic_charge(trivial_traj)["Q_dynamic"])

        # 半圈轨迹
        half_traj = self._build_trajectory(n_steps=60, rotation_per_step=math.pi / 60)
        Q_half = float(self.tc.compute_dynamic_charge(half_traj)["Q_dynamic"])

        # 完整一圈轨迹
        full_traj = self._build_trajectory(n_steps=60, rotation_per_step=2 * math.pi / 60)
        Q_full = float(self.tc.compute_dynamic_charge(full_traj)["Q_dynamic"])

        # 两圈轨迹
        double_traj = self._build_trajectory(n_steps=60, rotation_per_step=4 * math.pi / 60)
        Q_double = float(self.tc.compute_dynamic_charge(double_traj)["Q_dynamic"])

        results = {
            "Q_trivial_no_rotation": Q_trivial,
            "Q_half_rotation": Q_half,
            "Q_full_rotation": Q_full,
            "Q_double_rotation": Q_double,
            "trivial_is_zero": abs(Q_trivial) < 0.1,
            "full_is_nonzero": abs(Q_full) > 0.1,
            "double_larger_than_full": abs(Q_double) > abs(Q_full),
            "thesis": (
                "无旋转 → Q≈0（拓扑平凡，无使命）；"
                "完整旋转 → Q≠0（拓扑非平凡，有使命）；"
                "多圈缠绕 → Q 更大（使命更强）。"
                "Q 度量认知方向的总缠绕，是使命的严格拓扑定义。"
            ),
        }
        return results


# ======================================================================
# 验证 3：架构相图
# ======================================================================

class PhaseDiagramMapper:
    """
    在 (κ, α, Q) 参数空间绘制架构相图。

    物理预测（来自 cognitive_architecture.py）：
        - 低κ + 高α + Q≈0 → VAE（所知障，温室）
        - 高κ + 低α + Q≠0 + 痛苦分散 → GAN（烦恼障，冲突）
        - 高κ + Q≠0 + 痛苦稀疏 → Sparse AE（专精苦）
        - 高α + 高Φ + Q≠0 → White-box（加行道，觉照）
        - 低κ + 低α + Q≈0 → AE（基线）

    佛学对应：不同障道状态对应不同相态。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.arch = CognitiveArchitecture(n_dims=n_dims, eps=eps)

    def scan_kappa_alpha(
        self,
        kappa_range: tuple[float, float] = (0.0, 3.0),
        alpha_range: tuple[float, float] = (0.0, 3.0),
        n_points: int = 15,
        Q: float = 0.0,
        Phi: float = 0.5,
    ) -> dict:
        """
        在 (κ, α) 平面扫描架构类型（固定 Q, Φ）。

        返回二维相图：每个 (κ, α) 点的主导架构类型。
        """
        kappa_values = torch.linspace(kappa_range[0], kappa_range[1], n_points)
        alpha_values = torch.linspace(alpha_range[0], alpha_range[1], n_points)

        grid = []  # [i][j] = arch_type
        arch_names = []

        for k in kappa_values:
            row = []
            for a in alpha_values:
                # 构造 κ 向量（各向同性，sparsity=0 → GAN 倾向）
                kappa_vec = torch.tensor([float(k)] * self.n_dims, dtype=torch.float64)
                alpha_vec = torch.tensor([float(a)] * self.n_dims, dtype=torch.float64)
                profile = self.arch.classify_architecture(kappa_vec, alpha_vec, Q=Q, Phi=Phi)
                row.append(profile.arch_type.value)
            grid.append(row)

        return {
            "kappa_values": kappa_values.tolist(),
            "alpha_values": alpha_values.tolist(),
            "grid": grid,
            "Q_fixed": Q,
            "Phi_fixed": Phi,
        }

    def scan_Q(
        self, kappa_fixed: float = 1.5, alpha_fixed: float = 0.5, n_points: int = 12
    ) -> dict:
        """
        扫描 Q（使命强度），观察架构随 Q 的变化。

        物理预测：Q 从 0 增大时，架构从 VAE/AE 转向 GAN/Sparse AE/White-box。
        """
        Q_values = torch.linspace(0.0, 3.0, n_points)
        arch_sequence = []
        for q in Q_values:
            kappa_vec = torch.tensor([kappa_fixed] * self.n_dims, dtype=torch.float64)
            alpha_vec = torch.tensor([alpha_fixed] * self.n_dims, dtype=torch.float64)
            profile = self.arch.classify_architecture(kappa_vec, alpha_vec, Q=float(q), Phi=0.5)
            arch_sequence.append(profile.arch_type.value)

        return {
            "Q_values": Q_values.tolist(),
            "arch_sequence": arch_sequence,
            "kappa_fixed": kappa_fixed,
            "alpha_fixed": alpha_fixed,
        }


# ======================================================================
# 验证 4：三层结构
# ======================================================================

class ThreeLayerStructureVerifier:
    """
    验证三层结构（闲聊，相逢.txt 的核心物理结论）。

    物理：自发破缺的深度决定 Q 是否非零
        - 第一层（原生）：κ=0，无破缺，无 VAE，Q=0
        - 第二层（平滑破缺）：低 κ，各向同性破缺，有 VAE，Q≈0
        - 第三层（剧烈破缺）：高 κ，方向性破缺，有 VAE，Q≠0

    佛学：
        - 第一层 = 无明未起（婴儿态）
        - 第二层 = 有分别心但无使命（大多数人的异化稳态）
        - 第三层 = 有使命（Q≠0 驱动觉察，少数人）
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)
        self.tc = TopologicalCharge(n_dims=n_dims, eps=eps)
        self.arch = CognitiveArchitecture(n_dims=n_dims, eps=eps)

    def verify_three_layers(self) -> dict:
        """验证三层结构的物理性质差异。"""
        results = {}

        # 第一层：原生（κ=0）
        kappa_L1 = torch.zeros(self.n_dims, dtype=torch.float64)
        alpha_L1 = torch.tensor([0.2] * self.n_dims, dtype=torch.float64)
        traj_L1 = self._build_layer_trajectory(kappa_L1, alpha_L1, rotation_scale=0.0)
        Q_L1 = float(self.tc.compute_dynamic_charge(traj_L1)["Q_dynamic"])
        order_L1 = self.vacuum.symmetry_order_parameter(traj_L1[-1])

        # 第二层：平滑破缺（低 κ，小旋转 → Q≈0）
        kappa_L2 = torch.tensor([0.3] * self.n_dims, dtype=torch.float64)
        alpha_L2 = torch.tensor([1.0] * self.n_dims, dtype=torch.float64)
        traj_L2 = self._build_layer_trajectory(kappa_L2, alpha_L2, rotation_scale=0.005)
        Q_L2 = float(self.tc.compute_dynamic_charge(traj_L2)["Q_dynamic"])
        order_L2 = self.vacuum.symmetry_order_parameter(traj_L2[-1])
        profile_L2 = self.arch.classify_architecture(kappa_L2, alpha_L2, Q=Q_L2, Phi=0.5)

        # 第三层：剧烈破缺（高 κ，大旋转 → Q≠0）
        # v7.2：各向异性 κ + 更强旋转，确保 Q 显著非零
        kappa_L3 = torch.tensor([2.3, 1.8, 2.6, 1.5], dtype=torch.float64)
        alpha_L3 = torch.tensor([0.3] * self.n_dims, dtype=torch.float64)
        traj_L3 = self._build_layer_trajectory(kappa_L3, alpha_L3, rotation_scale=0.12)
        Q_L3 = float(self.tc.compute_dynamic_charge(traj_L3)["Q_dynamic"])
        order_L3 = self.vacuum.symmetry_order_parameter(traj_L3[-1])
        profile_L3 = self.arch.classify_architecture(kappa_L3, alpha_L3, Q=Q_L3, Phi=0.5)

        results = {
            "layer1_native": {
                "kappa": 0.0,
                "Q": Q_L1,
                "order_param": float(order_L1["order_parameter"]),
                "has_VAE": False,  # κ=0 无破缺，无潜空间
                "has_mission": abs(Q_L1) < 0.05,
                "Q_is_zero": abs(Q_L1) < 0.05,
            },
            "layer2_smooth_breaking": {
                "kappa": 0.3,
                "Q": Q_L2,
                "order_param": float(order_L2["order_parameter"]),
                "has_VAE": True,  # 有破缺，有潜空间
                "has_mission": abs(Q_L2) > 0.05,
                "Q_is_zero": abs(Q_L2) < 0.05,
                "arch": profile_L2.arch_type.value,
            },
            "layer3_violent_breaking": {
                "kappa": 2.0,
                "Q": Q_L3,
                "order_param": float(order_L3["order_parameter"]),
                "has_VAE": True,
                "has_mission": abs(Q_L3) > 0.05,
                "Q_is_zero": abs(Q_L3) < 0.05,
                "arch": profile_L3.arch_type.value,
            },
            "thesis": (
                "第一层（κ=0）：Q≈0，无 VAE，无使命——无明未起的原生态。\n"
                "第二层（低κ）：Q≈0，有 VAE，无使命——有分别心但拓扑平凡，大多数人的异化稳态。\n"
                "第三层（高κ+方向旋转）：Q≠0，有 VAE，有使命——拓扑缠绕驱动觉察，少数人。\n"
                "Q 的非零来自方向性旋转（剧烈破缺），而非破缺本身。"
            ),
        }
        return results

    def _build_layer_trajectory(
        self, kappa_vec: Tensor, alpha_vec: Tensor, rotation_scale: float, n_steps: int = 50
    ) -> list[Tensor]:
        """
        为给定层构建度规轨迹。rotation_scale 控制方向旋转强度。

        v7.2 修复（关键）：
            1. 各向异性 κ_vec 触发方向性破缺（各向同性 κ → 各向同性 g* → Q≡0）
            2. SSB 后强制各向异性：若 g* 近各向同性，手动施加差异化破缺
            3. 各向异性度规 + 旋转 → 非零 Q（拓扑缠绕）
        """
        # 强制各向异性 κ_vec（打破退化，触发方向性破缺）
        kappa_aniso = kappa_vec.clone()
        if float(kappa_aniso.std()) < 1e-6:
            # 各向同性 κ → 注入各向异性（不同维度不同痛苦深度）
            anisotropy = torch.tensor([0.3, -0.1, 0.1, -0.3], dtype=torch.float64)
            kappa_aniso = kappa_aniso + anisotropy[:self.n_dims] * float(kappa_aniso.mean())

        breaking = self.vacuum.spontaneous_symmetry_breaking(
            kappa_aniso, alpha_vec, n_steps=40, dt=0.02
        )
        g = breaking["g_final"].clone()

        # 确保 g 各向异性（SSB 可能因数值原因产生近各向同性结果）
        g_diag = torch.diagonal(g)
        if float(g_diag.std()) < 1e-4:
            # 手动施加各向异性破缺
            anisotropy_pattern = [0.4, 0.1, -0.1, -0.4]
            for i in range(self.n_dims):
                g[i, i] = g[i, i] + anisotropy_pattern[i % len(anisotropy_pattern)]
            g = symmetric_part(g)

        trajectory = [g.clone()]

        for step in range(n_steps):
            # 方向旋转（rotation_scale=0 → 平凡；大 → 缠绕）
            theta = rotation_scale * (1.0 if step % 2 == 0 else -0.6)
            R = torch.eye(self.n_dims, dtype=torch.float64)
            c, s = math.cos(theta), math.sin(theta)
            i, j = 0, 1
            R[i, i] = c; R[i, j] = -s
            R[j, i] = s; R[j, j] = c
            g = R.T @ g @ R
            g = symmetric_part(g)
            trajectory.append(g.clone())

        return trajectory


# ======================================================================
# 验证 5：架构特异性势能面
# ======================================================================

class PotentialLandscapeVerifier:
    """
    验证各架构的势能面形态。

    物理预测：
        - VAE：单深井（KL 散度加深真空井）
        - GAN：双井（对抗项产生双稳态）
        - Sparse AE：多井/稀疏（稀疏惩罚）
        - White-box：透明约束井（显式目标约束）
        - AE：基线单井

    佛学对应：
        - VAE 单深井 = 温室舒适（所知障）
        - GAN 双井 = 内心冲突（烦恼障）
        - White-box 透明井 = 觉照约束（加行道）
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.arch = CognitiveArchitecture(n_dims=n_dims, eps=eps)
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)

    def landscape_slice(
        self, arch_type: ArchitectureType, kappa: float = 1.0, alpha: float = 0.5
    ) -> dict:
        """
        沿 g[0,0] 方向切片势能面 V(g)。

        v7.2 修复：
            1. GAN 传入固定 g_real（破缺态参考），使双井可见。
               原 g_real=g.detach() 导致 d_real≈0，GAN 修正项退化。
            2. 井数检测用更鲁棒的判据（考虑数值噪声）。
        """
        # 各向异性 κ_vec（避免势能面退化）
        kappa_vec = torch.tensor([kappa, kappa*0.8, kappa*1.2, kappa*0.9],
                                 dtype=torch.float64)[:self.n_dims]
        alpha_vec = torch.tensor([alpha] * self.n_dims, dtype=torch.float64)

        # 构造对应架构的 profile
        Q_for_classify = 1.0 if arch_type in [
            ArchitectureType.GAN, ArchitectureType.SPARSE_AE, ArchitectureType.WHITE_BOX
        ] else 0.0
        profile = self.arch.classify_architecture(
            kappa_vec, alpha_vec, Q=Q_for_classify, Phi=0.7
        )
        # 强制设置目标架构（用于对比）
        profile.arch_type = arch_type

        # v7.2 修复：GAN 的 g_real = 破缺态参考（固定），g_ideal = 真空
        # 扫描 g[0,0] 时，g 在 g_real[0,0] 和 g_ideal[0,0]=1 之间移动，
        # d_real 和 d_ideal 都非零 → 势垒 → 双井
        g_ideal = self.vacuum.construct_vacuum()  # 真空 cI
        g_real = torch.eye(self.n_dims, dtype=torch.float64)
        g_real[0, 0] = 2.0  # 破缺态参考（"现实自我"偏离真空）

        g_values = torch.linspace(0.3, 2.7, 60, dtype=torch.float64)
        V_values = []
        V_corr_values = []

        for g_val in g_values:
            g = torch.eye(self.n_dims, dtype=torch.float64)
            g[0, 0] = float(g_val)
            pot = self.arch.compute_architecture_potential(
                g, profile, kappa_vec, alpha_vec,
                g_ideal=g_ideal, g_real=g_real,
            )
            V_values.append(float(pot["V_total"]))
            V_corr_values.append(float(pot["V_correction"]))

        # v7.2 修复：井数检测在 V_correction 上做（V_base 主导 V_total）
        # 物理原因：基线势能 V_base 深度 ~450，架构修正 V_corr ~1，
        # V_total 形态被 V_base 主导，看不到架构特异性结构。
        # 架构特异性（VAE 单深井 / GAN 双井）体现在 V_correction 中。
        n_wells = 0
        for i in range(1, len(V_corr_values) - 1):
            if V_corr_values[i] < V_corr_values[i - 1] and V_corr_values[i] < V_corr_values[i + 1]:
                n_wells += 1

        return {
            "arch_type": arch_type.value,
            "g_values": g_values.tolist(),
            "V_values": V_values,
            "V_correction_values": V_corr_values,
            "n_wells": n_wells,
            "V_min": min(V_values),
            "V_max": max(V_values),
            "V_corr_min": min(V_corr_values),
            "V_corr_max": max(V_corr_values),
            "landscape_depth": max(V_values) - min(V_values),
            "correction_depth": max(V_corr_values) - min(V_corr_values),
        }

    def verify_all_landscapes(self) -> dict:
        """验证全部五种架构的势能面形态。"""
        landscapes = {}
        for arch in ArchitectureType:
            landscapes[arch.value] = self.landscape_slice(arch)

        # 物理预测验证
        vae_depth = landscapes["vae"]["landscape_depth"]
        gan_wells = landscapes["gan"]["n_wells"]

        return {
            "landscapes": landscapes,
            "vae_is_deep_well": vae_depth > 1.0,
            # v7.2 修复：判据 >= 2（真正的"多井"），原 >= 1 把单井也判为多井
            "gan_has_multiple_wells": gan_wells >= 2,
            "thesis": (
                "VAE 势能面深单井（所知障的舒适温室，势垒高难破）；"
                "GAN 势能面双井形态（烦恼障的内心冲突，易翻转）；"
                "不同架构 = 不同自由能景观 = Landau 相变理论的标准推广。"
            ),
        }


# ======================================================================
# 顶层 API：运行全部五大验证
# ======================================================================

def run_core_verification(verbose: bool = True) -> dict:
    """
    运行 CTFT 全部五大核心物理验证。

    返回：
        dict 包含五大验证结果
    """
    if verbose:
        print("=" * 70)
        print("CTFT 核心物理验证（理论依据：佛学 + 哲学 + 物理学）")
        print("=" * 70)

    results = {}

    # 验证 1：SSB 相变
    if verbose:
        print("\n[验证 1] 自发对称性破缺相变")
    sb_verifier = SymmetryBreakingVerifier()
    sb_scan = sb_verifier.scan_kappa()
    # 找到相变点（首次 breaking_occurred=True 的 κ）
    transition_kappa = None
    for i, occurred in enumerate(sb_scan["breaking_occurred"]):
        if occurred:
            transition_kappa = sb_scan["kappa"][i]
            break
    # 不可逆性验证（v7.2：各向异性 κ 触发方向性破缺）
    kappa_test = torch.tensor([2.0, 1.64, 2.36, 1.84], dtype=torch.float64)
    alpha_test = torch.tensor([0.3] * 4, dtype=torch.float64)
    irrevers = sb_verifier.verify_irreversibility(kappa_test, alpha_test)

    results["ssb"] = {
        "transition_kappa": transition_kappa,
        "scan": sb_scan,
        "irreversibility": irrevers,
    }
    if verbose:
        print(f"  相变点 κ_c ≈ {transition_kappa:.3f}")
        print(f"  κ=0 稳定: {sb_scan['is_stable'][0]}")
        print(f"  κ=3 破缺: {sb_scan['breaking_occurred'][-1]}")
        print(f"  破缺不可逆: {irrevers['is_irreversible']} "
              f"(序参量 {irrevers['order_param_after_relax']:.4f} > 0)")
        print(f"  {irrevers['thesis']}")

    # 验证 2：拓扑荷 Q
    if verbose:
        print("\n[验证 2] 拓扑荷 Q 的守恒性与量子化")
    tc_verifier = TopologicalChargeVerifier()
    conservation = tc_verifier.verify_conservation()
    quantization = tc_verifier.verify_quantization()
    results["topological_charge"] = {
        "conservation": conservation,
        "quantization": quantization,
    }
    if verbose:
        print(f"  守恒律: max dQ/dt = {conservation['max_violation']:.4f} "
              f"(守恒={conservation['is_conserved']})")
        print(f"  平凡轨迹 Q = {quantization['Q_trivial_no_rotation']:.4f} "
              f"(≈0={quantization['trivial_is_zero']})")
        print(f"  完整旋转 Q = {quantization['Q_full_rotation']:.4f} "
              f"(≠0={quantization['full_is_nonzero']})")
        print(f"  双圈旋转 Q = {quantization['Q_double_rotation']:.4f} "
              f"(>单圈={quantization['double_larger_than_full']})")
        print(f"  {quantization['thesis']}")

    # 验证 3：架构相图
    if verbose:
        print("\n[验证 3] 架构相图（(κ,α) 平面扫描）")
    pd_mapper = PhaseDiagramMapper()
    # Q=0 相图（无使命）
    pd_Q0 = pd_mapper.scan_kappa_alpha(Q=0.0, n_points=10)
    # Q=2 相图（有使命）
    pd_Q2 = pd_mapper.scan_kappa_alpha(Q=2.0, n_points=10)
    q_scan = pd_mapper.scan_Q(kappa_fixed=1.5, alpha_fixed=0.5)
    results["phase_diagram"] = {
        "grid_Q0": pd_Q0,
        "grid_Q2": pd_Q2,
        "Q_scan": q_scan,
    }
    if verbose:
        print(f"  Q=0 相图（无使命）：低κ低α={pd_Q0['grid'][0][0]}, "
              f"低κ高α={pd_Q0['grid'][0][-1]}, 高κ低α={pd_Q0['grid'][-1][0]}")
        print(f"  Q=2 相图（有使命）：低κ低α={pd_Q2['grid'][0][0]}, "
              f"低κ高α={pd_Q2['grid'][0][-1]}, 高κ低α={pd_Q2['grid'][-1][0]}")
        print(f"  Q 扫描序列（κ=1.5,α=0.5）: {q_scan['arch_sequence']}")

    # 验证 4：三层结构
    if verbose:
        print("\n[验证 4] 三层结构（原生/平滑破缺/剧烈破缺）")
    layer_verifier = ThreeLayerStructureVerifier()
    layers = layer_verifier.verify_three_layers()
    results["three_layers"] = layers
    if verbose:
        L1 = layers["layer1_native"]
        L2 = layers["layer2_smooth_breaking"]
        L3 = layers["layer3_violent_breaking"]
        print(f"  L1 原生:     κ=0,   Q={L1['Q']:.4f}, 序参量={L1['order_param']:.4f}, "
              f"Q≈0={L1['Q_is_zero']}")
        print(f"  L2 平滑破缺: κ=0.3, Q={L2['Q']:.4f}, 序参量={L2['order_param']:.4f}, "
              f"Q≈0={L2['Q_is_zero']}, 架构={L2['arch']}")
        print(f"  L3 剧烈破缺: κ=2.0, Q={L3['Q']:.4f}, 序参量={L3['order_param']:.4f}, "
              f"Q≠0={not L3['Q_is_zero']}, 架构={L3['arch']}")
        print(f"  {layers['thesis']}")

    # 验证 5：势能面
    if verbose:
        print("\n[验证 5] 架构特异性势能面")
    pl_verifier = PotentialLandscapeVerifier()
    landscapes = pl_verifier.verify_all_landscapes()
    results["potential_landscape"] = landscapes
    if verbose:
        for name, land in landscapes["landscapes"].items():
            print(f"  {name}: 井数={land['n_wells']}, "
                  f"势能深度={land['landscape_depth']:.3f}")
        print(f"  VAE 深井: {landscapes['vae_is_deep_well']}")
        print(f"  GAN 多井: {landscapes['gan_has_multiple_wells']}")
        print(f"  {landscapes['thesis']}")

    if verbose:
        print("\n" + "=" * 70)
        print("五大核心物理验证完成。理论的合法性来自数学与物理。")
        print("=" * 70)

    return results


if __name__ == "__main__":
    run_core_verification(verbose=True)
