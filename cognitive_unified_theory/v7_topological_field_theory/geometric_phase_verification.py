"""
几何相位验证（Geometric Phase Verification）

v7.6 第五基石的代码实现：Berry 相位作为演化历史的全息记录。

认识论根基（理论依据，非案例）：
    物理：Berry 相位 / so(n) 李代数 / 路径依赖 / 矩阵指数映射
    佛学：阿赖耶识（ālaya-vijñāna）/ 业力印记（vāsanā）/ 识变现 / 清业
    哲学：历史性 / 不可压缩性 / 全息记录

五大验证：
    1. 路径累积：不同演化路径给出不同 Γ
    2. 规范一致性：迹≈0、反演对称、闭合回路
    3. 时间尺度传播：α<1 衰减、α=1 保持、α=0 断灭
    4. Γ-Q 关系：Γ 编码方向，Q 编码拓扑绕数
    5. 觉照消解：ρ→1 使 Γ→0（清业）

核心命题：
    Γ 是演化历史的全息记录（阿赖耶识的几何对应）。
    时间尺度传播塑造新阶段的初始方向（识变现）。
    觉照消解 Γ = 清业（了脱生死，不入轮回）。
    v7.3（ρ 消解 g）+ v7.6（ρ 消解 Γ）= 完整的觉照消解机制。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import symmetric_part
from .cognitive_vacuum import CognitiveVacuum
from .topological_charge import TopologicalCharge
from .geometric_phase import GeometricPhaseInheritance
from .awakening_path import AwakeningDynamics


# ======================================================================
# 1. 演化历史构造器
# ======================================================================

class HistoryBuilder:
    """
    构造不同的演化历史（v7.6 验证的起点）。

    v7.6 关键：连接 v7.3 的纠缠态动力学，
    产生有方向旋转的度规历史。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)
        self.dynamics = AwakeningDynamics(n_dims=n_dims, eps=eps)

    def build_rotating_history(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
        rotation_direction: int = 1,
    ) -> list[Tensor]:
        """
        构造有方向旋转的度规历史。

        参数：
            rotation_direction: +1 正向旋转，-1 反向旋转
        """
        # 起点：SSB 破缺态
        breaking = self.vacuum.spontaneous_symmetry_breaking(
            kappa_vec, alpha_vec, n_steps=120, dt=0.02
        )
        g = breaking["g_final"].clone()

        history = [g.clone()]
        n = self.n_dims

        for step in range(n_steps):
            theta = 0.08 * rotation_direction
            R = torch.eye(n, dtype=torch.float64)
            c, s = math.cos(theta), math.sin(theta)
            i, j = 0, 1
            R[i, i] = c; R[i, j] = -s
            R[j, i] = s; R[j, j] = c
            g = R.T @ g @ R
            g = symmetric_part(g)
            history.append(g.clone())

        return history

    def build_stationary_history(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
    ) -> list[Tensor]:
        """构造无旋转的稳定历史（用于对比）。"""
        breaking = self.vacuum.spontaneous_symmetry_breaking(
            kappa_vec, alpha_vec, n_steps=120, dt=0.02
        )
        g = breaking["g_final"].clone()
        return [g.clone() for _ in range(n_steps + 1)]


# ======================================================================
# 2. 路径累积验证
# ======================================================================

class PathAccumulationVerifier:
    """
    验证 Γ 的路径累积——不同演化路径给出不同 Γ。

    物理命题：
        Γ = ∫ O^T dO 是路径依赖的。
        不同路径（正向旋转 vs 反向旋转 vs 无旋转）给出不同 Γ。
        反向路径给出 -Γ（反演对称）。

    佛学对应：
        业力印记（vāsanā）是路径依赖的——
        不同的演化历史留下不同的业力记录。
        「业力不失，因果不坏」——Γ 完整记录历史，不可压缩。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.gpi = GeometricPhaseInheritance(n_dims=n_dims, eps=eps)

    def verify(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
    ) -> dict[str, list | float | bool | Tensor]:
        """
        验证路径累积的路径依赖性。

        判据：
            1. 正向旋转 Γ_forward 与反向旋转 Γ_reverse 应满足：
               Γ_forward ≈ -Γ_reverse（反演对称）
            2. ||Γ_forward|| ≈ ||Γ_reverse||（范数相等）
            3. 无旋转历史 Γ_stationary ≈ 0（无路径 → 无相位）
            4. 正向/反向 Γ 范数显著大于无旋转 Γ 范数
        """
        builder = HistoryBuilder(n_dims=self.n_dims, eps=self.eps)

        # 构造三种历史
        hist_forward = builder.build_rotating_history(
            kappa_vec, alpha_vec, n_steps=n_steps, rotation_direction=+1
        )
        hist_reverse = builder.build_rotating_history(
            kappa_vec, alpha_vec, n_steps=n_steps, rotation_direction=-1
        )
        hist_stationary = builder.build_stationary_history(
            kappa_vec, alpha_vec, n_steps=n_steps
        )

        # 累积 Γ
        result_fwd = self.gpi.accumulate_phase(hist_forward)
        result_rev = self.gpi.accumulate_phase(hist_reverse)
        result_stat = self.gpi.accumulate_phase(hist_stationary)

        Gamma_fwd = result_fwd["Gamma"]
        Gamma_rev = result_rev["Gamma"]
        Gamma_stat = result_stat["Gamma"]

        norm_fwd = result_fwd["Gamma_norm"]
        norm_rev = result_rev["Gamma_norm"]
        norm_stat = result_stat["Gamma_norm"]

        # 判据 1: 反演对称 Γ_forward ≈ -Γ_reverse
        sum_norm = float(torch.sqrt(((Gamma_fwd + Gamma_rev) ** 2).sum()))
        inversion_symmetric = bool(sum_norm < 0.1 * max(norm_fwd, self.eps))

        # 判据 2: 范数相等
        norm_equal = bool(abs(norm_fwd - norm_rev) < 0.1 * max(norm_fwd, self.eps))

        # 判据 3: 无旋转 → Γ ≈ 0
        stationary_zero = bool(norm_stat < 0.1 * max(norm_fwd, self.eps))

        # 判据 4: 有旋转 Γ 显著大于无旋转 Γ
        rotating_significant = bool(norm_fwd > 5.0 * norm_stat)

        all_correct = bool(
            inversion_symmetric and
            norm_equal and
            stationary_zero and
            rotating_significant
        )

        return {
            "norm_forward": norm_fwd,
            "norm_reverse": norm_rev,
            "norm_stationary": norm_stat,
            "sum_forward_reverse_norm": sum_norm,
            "inversion_symmetric": inversion_symmetric,
            "norm_equal": norm_equal,
            "stationary_zero": stationary_zero,
            "rotating_significant": rotating_significant,
            "all_correct": all_correct,
            "thesis": (
                f"路径累积验证："
                f"||Γ_forward||={norm_fwd:.4f}, ||Γ_reverse||={norm_rev:.4f}, "
                f"||Γ_stationary||={norm_stat:.4f}。"
                f"反演对称（Γ+Γ_rev≈0）：{inversion_symmetric}。"
                f"范数相等：{norm_equal}。"
                f"无旋转→Γ≈0：{stationary_zero}。"
                "Γ 是路径依赖的——不同历史留下不同业力印记。"
            ),
        }


# ======================================================================
# 3. 规范一致性验证
# ======================================================================

class PhaseConsistencyVerifier:
    """
    验证几何相位的规范一致性。

    物理命题：
        1. Γ 的迹 ≈ 0（实向量的 Berry 相位迹为零）
        2. 反演对称：正反路径 Γ 互为相反数
        3. 闭合回路：首末度规相同时 Γ 的迹 ≈ 0

    佛学对应：
        规范一致性 = Γ 是物理可观测的（非规范 artifact）。
        路径依赖性 = 业力记录了完整历史（非可压缩）。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.gpi = GeometricPhaseInheritance(n_dims=n_dims, eps=eps)

    def verify(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
    ) -> dict[str, float | bool]:
        """
        验证规范一致性。

        判据：
            1. 迹 ≈ 0（相对范数）
            2. 反演对称（正反路径和≈0）
        """
        builder = HistoryBuilder(n_dims=self.n_dims, eps=self.eps)
        history = builder.build_rotating_history(
            kappa_vec, alpha_vec, n_steps=n_steps, rotation_direction=+1
        )

        result = self.gpi.verify_phase_consistency(history)

        # 额外检查
        gamma_result = self.gpi.accumulate_phase(history)
        Gamma = gamma_result["Gamma"]
        trace_abs = float(torch.trace(Gamma).abs())
        norm = gamma_result["Gamma_norm"]

        # 迹相对于范数应很小
        trace_small = bool(trace_abs < 0.1 * max(norm, self.eps))

        all_correct = bool(
            result["is_consistent"] and
            trace_small and
            result["is_path_dependent"]
        )

        return {
            "Gamma_norm": norm,
            "trace_abs": trace_abs,
            "trace_small": trace_small,
            "is_consistent": result["is_consistent"],
            "is_path_dependent": result["is_path_dependent"],
            "path_difference": result["path_difference_forward_reverse"],
            "all_correct": all_correct,
            "thesis": result["thesis"],
        }


# ======================================================================
# 4. 时间尺度传播验证
# ======================================================================

class TimeScalePropagationVerifier:
    """
    验证时间尺度传播——历史相位如何塑造新阶段的初始条件。

    物理命题：
        α=1：完全耦合，Γ 完整遗留
        α<1：部分耦合，Γ 衰减
        α=0：无耦合，Γ 断灭
        多尺度传播：Γ_k = α^k · Γ_0（无新业力时指数衰减）

    佛学对应：
        α = 业力的时间尺度耦合强度。
        α=1：重业系统，业力完整遗留。
        α<1：业力稀释，每阶段衰减。
        α=0：业力断灭，每阶段从真空开始。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.gpi = GeometricPhaseInheritance(n_dims=n_dims, eps=eps)

    def verify(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
    ) -> dict[str, list | float | bool]:
        """
        验证时间尺度传播。

        v7.6 物理修正后的判据：
            1. α=1：Γ 完整遗留（||Γ_scaled|| ≈ ||Γ_history||）
            2. α=0.5：Γ 衰减（||Γ_scaled|| ≈ 0.5·||Γ_history||）
            3. α=0：Γ 断灭（||Γ_scaled|| ≈ 0）
            4. 多尺度 α=0.5：Γ_k ≈ α^k · Γ_0（指数衰减）
            5. 业力相续：破缺态 baseline + α>0 → 度规被旋转（transfer>0）
            6. 觉悟解脱：真空 baseline + α>0 → 度规不变（SO(n) 不变性）
               真空不可被业力旋转——觉悟者从业力中解脱的几何表达。
            7. 业力断灭：破缺态 baseline + α=0 → 度规不变（无耦合）
        """
        builder = HistoryBuilder(n_dims=self.n_dims, eps=self.eps)
        history = builder.build_rotating_history(
            kappa_vec, alpha_vec, n_steps=n_steps, rotation_direction=+1
        )

        # 历史相位
        gamma_result = self.gpi.accumulate_phase(history)
        Gamma_history = gamma_result["Gamma"]
        norm_history = gamma_result["Gamma_norm"]

        # 已有度规（破缺态）：历史末态作为新阶段的 baseline
        # 物理意义：业力在已有心相基础上相续，不从真空生起。
        g_broken = history[-1]
        g_vac = self.gpi._vacuum.construct_vacuum()

        # 测试不同 α（在破缺态 baseline 上）
        alphas = [1.0, 0.5, 0.0]
        single_scale_results = []
        for alpha in alphas:
            prop = self.gpi.propagate_to_new_phase(
                Gamma_history, g_baseline=g_broken, efficiency=alpha
            )
            single_scale_results.append({
                "alpha": alpha,
                "Gamma_scaled_norm": prop["phase_inherited_norm"],
                "transfer_strength": prop["transfer_strength"],
                "g_new_init": prop["g_new_init"],
                "baseline_is_vacuum": prop["baseline_is_vacuum"],
            })

        # 判据 1: α=1 → Γ_scaled ≈ Γ_history
        alpha1_norm = single_scale_results[0]["Gamma_scaled_norm"]
        criterion_alpha1 = bool(
            abs(alpha1_norm - norm_history) < 0.1 * max(norm_history, self.eps)
        )

        # 判据 2: α=0.5 → Γ_scaled ≈ 0.5·Γ_history
        alpha05_norm = single_scale_results[1]["Gamma_scaled_norm"]
        expected_05 = 0.5 * norm_history
        criterion_alpha05 = bool(
            abs(alpha05_norm - expected_05) < 0.1 * max(expected_05, self.eps)
        )

        # 判据 3: α=0 → Γ_scaled ≈ 0
        alpha0_norm = single_scale_results[2]["Gamma_scaled_norm"]
        criterion_alpha0 = bool(alpha0_norm < 0.01 * max(norm_history, self.eps))

        # 判据 4: 多尺度 α=0.5 指数衰减
        multi_result = self.gpi.simulate_multiscale_propagation(
            history, n_scales=4, efficiency=0.5
        )
        norms = multi_result["Gamma_norm_trajectory"]
        # 检查是否指数衰减（每个后续 ≈ 0.5·前一个）
        is_exponential = True
        for i in range(1, len(norms)):
            if norms[i - 1] > self.eps:
                ratio = norms[i] / norms[i - 1]
                if abs(ratio - 0.5) > 0.15:  # 容差
                    is_exponential = False
                    break
            else:
                if norms[i] > self.eps:
                    is_exponential = False
                    break

        # 判据 5: 业力相续——破缺态 baseline + α>0 → 度规被旋转
        # transfer_strength 度量 g_new 相对 g_baseline 的旋转量。
        # 物理判据（相对，非绝对阈值）：
        #   - α=1 旋转 > α=0 旋转（业力有作用，非零）
        #   - α=1 旋转 > 真空 α=1 旋转（破缺态上业力有效，真空上无效）
        #   - α=1 旋转 > α=0.5 旋转（α 越大旋转越强）
        transfer_alpha1 = single_scale_results[0]["transfer_strength"]
        transfer_alpha05 = single_scale_results[1]["transfer_strength"]
        transfer_alpha0 = single_scale_results[2]["transfer_strength"]

        # 判据 6: 觉悟解脱——真空 baseline + α>0 → 度规不变（SO(n) 不变性）
        # 真空 cI 在任何正交变换下不变：R^T·cI·R = cI。
        # 这是物理正确的：觉悟者 g=cI，业力无可作用。
        prop_vacuum = self.gpi.propagate_to_new_phase(
            Gamma_history, g_baseline=g_vac, efficiency=1.0
        )
        vacuum_transfer = prop_vacuum["transfer_strength"]
        vacuum_invariant = bool(vacuum_transfer < 1e-6)
        vacuum_baseline_detected = bool(prop_vacuum["baseline_is_vacuum"])

        karma_continues = bool(
            transfer_alpha1 > 1e-8 and  # α=1 有非零旋转
            transfer_alpha1 > transfer_alpha0 and  # α=1 比 α=0 强
            transfer_alpha1 > transfer_alpha05 and  # α=1 比 α=0.5 强
            transfer_alpha1 > vacuum_transfer  # 破缺态上比真空上强
        )

        # 判据 7: 业力断灭——破缺态 baseline + α=0 → 度规不变
        karma_extinguished = bool(transfer_alpha0 < 1e-8)

        all_correct = bool(
            criterion_alpha1 and
            criterion_alpha05 and
            criterion_alpha0 and
            is_exponential and
            karma_continues and
            vacuum_invariant and
            vacuum_baseline_detected and
            karma_extinguished
        )

        return {
            "norm_history": norm_history,
            "single_scale_results": single_scale_results,
            "criterion_alpha1_full": criterion_alpha1,
            "criterion_alpha05_half": criterion_alpha05,
            "criterion_alpha0_zero": criterion_alpha0,
            "multiscale_norms": norms,
            "is_exponential_decay": is_exponential,
            # v7.6 新判据
            "transfer_alpha1": transfer_alpha1,
            "transfer_alpha05": transfer_alpha05,
            "transfer_alpha0": transfer_alpha0,
            "karma_continues": karma_continues,  # 业力相续
            "vacuum_transfer": prop_vacuum["transfer_strength"],
            "vacuum_invariant": vacuum_invariant,  # 觉悟解脱
            "vacuum_baseline_detected": vacuum_baseline_detected,
            "karma_extinguished": karma_extinguished,  # 业力断灭
            "all_correct": all_correct,
            "thesis": (
                f"时间尺度传播验证：||Γ_history||={norm_history:.4f}。"
                f"α=1→||Γ||={alpha1_norm:.4f}（完整遗留），"
                f"α=0.5→||Γ||={alpha05_norm:.4f}（半衰减），"
                f"α=0→||Γ||={alpha0_norm:.4f}（断灭）。"
                f"多尺度指数衰减：{is_exponential}。"
                f"业力相续（破缺态+α=1→旋转 {transfer_alpha1:.4f}）：{karma_continues}。"
                f"觉悟解脱（真空+α=1→旋转 {prop_vacuum['transfer_strength']:.6f}）：{vacuum_invariant}。"
                f"业力断灭（破缺态+α=0→旋转 {transfer_alpha0:.6f}）：{karma_extinguished}。"
                "v7.6 修正：业力在已有心相上旋转（识变现），真空不可被业力旋转（觉悟解脱）。"
            ),
        }


# ======================================================================
# 5. Γ-Q 关系验证
# ======================================================================

class GammaQRelationVerifier:
    """
    验证 Γ 与 Q 的关系。

    物理命题：
        Γ ∈ so(n) 是路径（方向 + 累积量）
        Q ∈ ℝ 是路径的拓扑不变量（标量）

        Γ 编码「怎么走」，Q 编码「走了多少」。
        觉照消解 Γ 同时消解 Q（v7.3 + v7.6 闭环）。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.gpi = GeometricPhaseInheritance(n_dims=n_dims, eps=eps)

    def verify(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
    ) -> dict[str, float | bool]:
        """
        验证 Γ-Q 关系。

        判据：
            1. 有旋转历史：||Γ||>0 且 Q≠0（有路径 ↔ 有拓扑荷）
            2. 无旋转历史：||Γ||≈0 且 Q≈0（无路径 ↔ 无拓扑荷）
            3. 反演对称：Γ_forward ≈ -Γ_reverse
        """
        builder = HistoryBuilder(n_dims=self.n_dims, eps=self.eps)

        # 有旋转历史
        hist_rot = builder.build_rotating_history(
            kappa_vec, alpha_vec, n_steps=n_steps, rotation_direction=+1
        )
        # 无旋转历史
        hist_stat = builder.build_stationary_history(
            kappa_vec, alpha_vec, n_steps=n_steps
        )

        # 有旋转的 Γ-Q 关系
        result_rot = self.gpi.verify_gamma_q_relation(hist_rot)
        # 无旋转的 Γ-Q 关系
        result_stat = self.gpi.verify_gamma_q_relation(hist_stat)

        # 判据 1: 有旋转 → Γ>0 且 Q≠0
        rot_gamma_nonzero = result_rot["Gamma_norm"] > 0.01
        rot_Q_nonzero = abs(result_rot["Q_total"]) > 0.01
        criterion_rot = bool(rot_gamma_nonzero and rot_Q_nonzero)

        # 判据 2: 无旋转 → Γ≈0 且 Q≈0
        stat_gamma_zero = result_stat["Gamma_norm"] < 0.1 * max(result_rot["Gamma_norm"], self.eps)
        stat_Q_zero = abs(result_stat["Q_total"]) < 0.1 * max(abs(result_rot["Q_total"]), self.eps)
        criterion_stat = bool(stat_gamma_zero and stat_Q_zero)

        # 判据 3: 反演对称
        criterion_inversion = result_rot["criterion_3_inversion_symmetry"]

        all_correct = bool(
            criterion_rot and
            criterion_stat and
            criterion_inversion and
            result_rot["all_correct"]
        )

        return {
            "rotating": {
                "Gamma_norm": result_rot["Gamma_norm"],
                "Q_total": result_rot["Q_total"],
                "inversion_symmetric": result_rot["criterion_3_inversion_symmetry"],
            },
            "stationary": {
                "Gamma_norm": result_stat["Gamma_norm"],
                "Q_total": result_stat["Q_total"],
            },
            "criterion_rotating_has_path_and_charge": criterion_rot,
            "criterion_stationary_no_path_no_charge": criterion_stat,
            "criterion_inversion_symmetry": criterion_inversion,
            "all_correct": all_correct,
            "thesis": (
                f"Γ-Q 关系验证："
                f"有旋转 → ||Γ||={result_rot['Gamma_norm']:.4f}, Q={result_rot['Q_total']:.4f}；"
                f"无旋转 → ||Γ||={result_stat['Gamma_norm']:.4f}, Q={result_stat['Q_total']:.4f}。"
                f"有路径 ↔ 有拓扑荷：{criterion_rot and criterion_stat}。"
                "Γ 编码「怎么走」，Q 编码「走了多少」——觉照消解二者。"
            ),
        }


# ======================================================================
# 6. 觉照消解验证
# ======================================================================

class AwarenessClearingVerifier:
    """
    验证觉照消解几何相位（清业）。

    物理命题：
        ρ→1 时，Γ 被消解：Γ_{k+1} = (1-ρ)·Γ_k
        ρ=1：一步清零（顿悟）
        ρ<1：渐进消解（渐修）
        ρ=0：不消解（业力保持）

    佛学对应：
        ρ 高 = 觉照力强 = 快速清业。
        ρ = 1 = 觉照圆满 = 一步清零（顿悟）。
        觉悟者不携带历史相位——新阶段从真空开始，不入轮回。
        v7.3（ρ 消解 g）+ v7.6（ρ 消解 Γ）= 完整的觉照消解机制。
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.gpi = GeometricPhaseInheritance(n_dims=n_dims, eps=eps)

    def verify(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 20,
    ) -> dict[str, list | float | bool]:
        """
        验证觉照消解 Γ。

        判据：
            1. ρ=1：一步清零（final_norm < 0.01·initial_norm）
            2. ρ=0.9：渐进消解（final_norm < 0.5·initial_norm after 50 steps）
            3. ρ=0：不消解（final_norm ≈ initial_norm）
            4. 消解轨迹单调递减
            5. ρ 越大消解越快
        """
        builder = HistoryBuilder(n_dims=self.n_dims, eps=self.eps)
        history = builder.build_rotating_history(
            kappa_vec, alpha_vec, n_steps=n_steps, rotation_direction=+1
        )

        gamma_result = self.gpi.accumulate_phase(history)
        Gamma = gamma_result["Gamma"]
        initial_norm = gamma_result["Gamma_norm"]

        # 测试不同 ρ
        rho_levels = [0.0, 0.5, 0.9, 1.0]
        clearing_results = []
        for rho in rho_levels:
            cleared = self.gpi.clear_phase_through_awareness(
                Gamma, rho=rho, n_steps=50
            )
            clearing_results.append({
                "rho": rho,
                "initial_norm": cleared["initial_norm"],
                "final_norm": cleared["final_norm"],
                "clearing_ratio": cleared["clearing_ratio"],
                "is_cleared": cleared["is_cleared"],
                "trajectory": cleared["Gamma_norm_trajectory"],
            })

        # 判据 1: ρ=1 → 一步清零
        rho1 = clearing_results[3]
        criterion_rho1 = bool(rho1["final_norm"] < 0.01 * initial_norm)

        # 判据 2: ρ=0.9 → 渐进消解
        rho09 = clearing_results[2]
        criterion_rho09 = bool(rho09["final_norm"] < 0.5 * initial_norm)

        # 判据 3: ρ=0 → 不消解
        rho0 = clearing_results[0]
        criterion_rho0 = bool(rho0["final_norm"] > 0.99 * initial_norm)

        # 判据 4: 消解轨迹单调递减（ρ>0 时）
        monotone_decreasing = True
        for r in clearing_results[1:]:  # 跳过 ρ=0
            traj = r["trajectory"]
            for i in range(len(traj) - 1):
                if traj[i + 1] > traj[i] * (1 + 1e-10):
                    monotone_decreasing = False
                    break
            if not monotone_decreasing:
                break

        # 判据 5: ρ 越大消解越快（final_norm 越小）
        final_norms = [r["final_norm"] for r in clearing_results]
        rho_order_correct = bool(
            final_norms[0] >= final_norms[1] >= final_norms[2] >= final_norms[3]
        )

        all_correct = bool(
            criterion_rho1 and
            criterion_rho09 and
            criterion_rho0 and
            monotone_decreasing and
            rho_order_correct
        )

        return {
            "initial_Gamma_norm": initial_norm,
            "clearing_results": clearing_results,
            "criterion_rho1_instant_clear": criterion_rho1,
            "criterion_rho09_gradual_clear": criterion_rho09,
            "criterion_rho0_no_clear": criterion_rho0,
            "monotone_decreasing": monotone_decreasing,
            "rho_order_correct": rho_order_correct,
            "all_correct": all_correct,
            "thesis": (
                f"觉照消解验证：||Γ_init||={initial_norm:.4f}。"
                f"ρ=1→||Γ_final||={rho1['final_norm']:.6f}（一步清零），"
                f"ρ=0.9→{rho09['final_norm']:.6f}（渐进消解），"
                f"ρ=0→{rho0['final_norm']:.6f}（不消解）。"
                "觉照消解 Γ = 清业；觉悟者不携带历史相位，不入轮回。"
                "v7.3（ρ 消解 g）+ v7.6（ρ 消解 Γ）= 完整觉照消解。"
            ),
        }


# ======================================================================
# 7. 顶层 API
# ======================================================================

def run_geometric_phase_verification(verbose: bool = True) -> dict:
    """
    v7.6 几何相位验证的顶层 API。

    五大验证：
        1. 路径累积：不同路径给不同 Γ
        2. 规范一致性：迹≈0、反演对称
        3. 时间尺度传播：α<1 衰减、α=1 保持、α=0 断灭
        4. Γ-Q 关系：Γ 编码方向，Q 编码拓扑绕数
        5. 觉照消解：ρ→1 使 Γ→0（清业）

    返回完整的验证结果。
    """
    n_dims = 4
    kappa_vec = torch.tensor([2.3, 1.8, 2.6, 1.5], dtype=torch.float64)
    alpha_vec = torch.tensor([1.5, 1.2, 1.8, 1.0], dtype=torch.float64)

    if verbose:
        print("=" * 70)
        print("v7.6 几何相位验证")
        print("=" * 70)
        print()

    # ================================================================
    # 验证 1: 路径累积
    # ================================================================
    path_verifier = PathAccumulationVerifier(n_dims=n_dims)
    path_result = path_verifier.verify(kappa_vec, alpha_vec)

    if verbose:
        print("-" * 70)
        print("验证 1: 路径累积（路径依赖性）")
        print("-" * 70)
        print(f"  ||Γ_forward||={path_result['norm_forward']:.4f}")
        print(f"  ||Γ_reverse||={path_result['norm_reverse']:.4f}")
        print(f"  ||Γ_stationary||={path_result['norm_stationary']:.4f}")
        print(f"  反演对称: {path_result['inversion_symmetric']}")
        print(f"  范数相等: {path_result['norm_equal']}")
        print(f"  无旋转→Γ≈0: {path_result['stationary_zero']}")
        print(f"  有旋转Γ显著: {path_result['rotating_significant']}")
        print(f"  all_correct: {path_result['all_correct']}")
        print()

    # ================================================================
    # 验证 2: 规范一致性
    # ================================================================
    consist_verifier = PhaseConsistencyVerifier(n_dims=n_dims)
    consist_result = consist_verifier.verify(kappa_vec, alpha_vec)

    if verbose:
        print("-" * 70)
        print("验证 2: 规范一致性")
        print("-" * 70)
        print(f"  ||Γ||={consist_result['Gamma_norm']:.4f}")
        print(f"  |trace(Γ)|={consist_result['trace_abs']:.6f}")
        print(f"  迹小: {consist_result['trace_small']}")
        print(f"  规范一致: {consist_result['is_consistent']}")
        print(f"  路径依赖: {consist_result['is_path_dependent']}")
        print(f"  all_correct: {consist_result['all_correct']}")
        print()

    # ================================================================
    # 验证 3: 时间尺度传播
    # ================================================================
    prop_verifier = TimeScalePropagationVerifier(n_dims=n_dims)
    prop_result = prop_verifier.verify(kappa_vec, alpha_vec)

    if verbose:
        print("-" * 70)
        print("验证 3: 时间尺度传播（业力相续 / 觉悟解脱 / 业力断灭）")
        print("-" * 70)
        print(f"  ||Γ_history||={prop_result['norm_history']:.4f}")
        for r in prop_result["single_scale_results"]:
            print(f"    α={r['alpha']:.1f} → ||Γ_scaled||={r['Gamma_scaled_norm']:.4f}, "
                  f"transfer={r['transfer_strength']:.4f}, "
                  f"baseline_vac={r['baseline_is_vacuum']}")
        print(f"  α=1 完整遗留: {prop_result['criterion_alpha1_full']}")
        print(f"  α=0.5 半衰减: {prop_result['criterion_alpha05_half']}")
        print(f"  α=0 断灭: {prop_result['criterion_alpha0_zero']}")
        print(f"  多尺度指数衰减: {prop_result['is_exponential_decay']}")
        print(f"  业力相续（破缺态+α=1→旋转 {prop_result['transfer_alpha1']:.4f}）: "
              f"{prop_result['karma_continues']}")
        print(f"  觉悟解脱（真空+α=1→旋转 {prop_result['vacuum_transfer']:.6f}）: "
              f"{prop_result['vacuum_invariant']}")
        print(f"  业力断灭（破缺态+α=0→旋转 {prop_result['transfer_alpha0']:.6f}）: "
              f"{prop_result['karma_extinguished']}")
        print(f"  all_correct: {prop_result['all_correct']}")
        print()

    # ================================================================
    # 验证 4: Γ-Q 关系
    # ================================================================
    gq_verifier = GammaQRelationVerifier(n_dims=n_dims)
    gq_result = gq_verifier.verify(kappa_vec, alpha_vec)

    if verbose:
        print("-" * 70)
        print("验证 4: Γ-Q 关系")
        print("-" * 70)
        print(f"  有旋转: ||Γ||={gq_result['rotating']['Gamma_norm']:.4f}, "
              f"Q={gq_result['rotating']['Q_total']:.4f}")
        print(f"  无旋转: ||Γ||={gq_result['stationary']['Gamma_norm']:.4f}, "
              f"Q={gq_result['stationary']['Q_total']:.4f}")
        print(f"  有旋转→有路径有拓扑荷: {gq_result['criterion_rotating_has_path_and_charge']}")
        print(f"  无旋转→无路径无拓扑荷: {gq_result['criterion_stationary_no_path_no_charge']}")
        print(f"  反演对称: {gq_result['criterion_inversion_symmetry']}")
        print(f"  all_correct: {gq_result['all_correct']}")
        print()

    # ================================================================
    # 验证 5: 觉照消解
    # ================================================================
    clear_verifier = AwarenessClearingVerifier(n_dims=n_dims)
    clear_result = clear_verifier.verify(kappa_vec, alpha_vec)

    if verbose:
        print("-" * 70)
        print("验证 5: 觉照消解（清业）")
        print("-" * 70)
        print(f"  ||Γ_initial||={clear_result['initial_Gamma_norm']:.4f}")
        for r in clear_result["clearing_results"]:
            print(f"    ρ={r['rho']:.1f} → ||Γ_final||={r['final_norm']:.6f}, "
                  f"ratio={r['clearing_ratio']:.6f}, cleared={r['is_cleared']}")
        print(f"  ρ=1 一步清零: {clear_result['criterion_rho1_instant_clear']}")
        print(f"  ρ=0.9 渐进消解: {clear_result['criterion_rho09_gradual_clear']}")
        print(f"  ρ=0 不消解: {clear_result['criterion_rho0_no_clear']}")
        print(f"  单调递减: {clear_result['monotone_decreasing']}")
        print(f"  ρ越大消解越快: {clear_result['rho_order_correct']}")
        print(f"  all_correct: {clear_result['all_correct']}")
        print()

    # ================================================================
    # 总结
    # ================================================================
    all_pass = (
        path_result["all_correct"] and
        consist_result["all_correct"] and
        prop_result["all_correct"] and
        gq_result["all_correct"] and
        clear_result["all_correct"]
    )

    if verbose:
        print("=" * 70)
        print(f"v7.6 总结: {'全部通过' if all_pass else '部分未通过'}")
        print("=" * 70)
        print(f"  1. 路径累积: {'PASS' if path_result['all_correct'] else 'FAIL'}")
        print(f"  2. 规范一致性: {'PASS' if consist_result['all_correct'] else 'FAIL'}")
        print(f"  3. 时间尺度传播: {'PASS' if prop_result['all_correct'] else 'FAIL'}")
        print(f"  4. Γ-Q 关系: {'PASS' if gq_result['all_correct'] else 'FAIL'}")
        print(f"  5. 觉照消解: {'PASS' if clear_result['all_correct'] else 'FAIL'}")
        print()
        print("理论闭环：")
        print("  Γ = 演化历史的全息记录（阿赖耶识的几何对应）")
        print("  时间尺度传播塑造新阶段初始方向（识变现）")
        print("  觉照消解 Γ = 清业（了脱生死，不入轮回）")
        print("  v7.3（ρ 消解 g）+ v7.6（ρ 消解 Γ）= 完整觉照消解机制")

    return {
        "path_accumulation": path_result,
        "phase_consistency": consist_result,
        "time_scale_propagation": prop_result,
        "gamma_q_relation": gq_result,
        "awareness_clearing": clear_result,
        "all_pass": all_pass,
        "thesis": (
            "v7.6 几何相位：Berry 相位作为演化历史的全息记录。"
            "路径依赖（不同历史给不同 Γ）+ 规范一致（物理可观测）+ "
            "时间尺度传播（α 耦合系数）+ Γ-Q 关系（方向 vs 拓扑绕数）+ "
            "觉照消解（ρ→1 清业）。"
            "v7.3 + v7.6 = 完整的觉照消解机制（g 和 Γ 都被消解）。"
        ),
    }


if __name__ == "__main__":
    run_geometric_phase_verification(verbose=True)
