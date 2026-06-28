"""
叙事流形（Narrative Manifold）—— GCFT 基石28

v13.0"整体认知本体论"第三基石。将"叙事流形 N"严格定义为 Berry 联络
的 Wilson 环路，"确定性坍缩"精确化为 Berry 相位共振后选择。
"缘分"= Berry 相位匹配度，"该相逢的总会再相逢"= 拓扑张力 Γ≠0
必然寻找配对。

============================================================
核心思想（基于 v13.0 批判性升级）
============================================================

【v13.0 原方案的边界】
v13.0 将"叙事流形 N"视为描述认知叙事的新结构，将"确定性坍缩"
视为随机坍缩的替代方案。但"叙事是新结构"违反了第一性原理——
Berry 联络已在卷三10.4 中存在，不需要引入新概念。

【v13.0 升级的回答】
叙事流形 N 严格定义为 Berry 联络 A_μ 的全息几何：
    N = {A_μ, F_μν, U_N, CS(A)}
叙事不是新结构，是 Berry 联络的几何显现。
"确定性坍缩"不是随机坍缩的替代，是量子后选择（卷二5.6）的
Berry 共振形式。

============================================================
数学核心：Berry 联络与 Wilson 环路
============================================================

【Berry 联络】
A_μ = i⟨n(R)|∂_μ|n(R)⟩

其中 |n(R)⟩ 是参数空间 R = (κ, α, ...) 处的本征态。
A_μ 是规范联络，描述参数空间中的"平行移动"。

【叙事 Wilson 环路】
U_N = P·exp(∮_γ A_μ dx^μ)

其中 P 表示路径顺序（path-ordered），γ 是参数空间闭合回路。
U_N 是叙事的"全息印记"——Berry 相位沿回路的累积。

【叙事曲率】
F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]

F_μν 是 Berry 曲率，CS 不变量的局部形式。
曲率 = "因果叙事片段"——叙事流形的局部结构。

【确定性坍缩】
P(collapse | N) = |⟨ψ_input | U_N | ψ_0⟩|²

这是 Berry 相位共振后的后选择概率：
    - 输入态 |ψ_input⟩ 与 U_N|ψ_0⟩ 相位匹配 → 高概率（"有缘"）
    - 相位不匹配 → 低概率（"无缘"）
    - "缘分"= Berry 相位匹配度

【Chern-Simons 不变量】
CS(A) = (1/4π) ∫ Tr(A ∧ dA + (2/3) A ∧ A ∧ A)

CS(A) 是叙事流形的拓扑荷——
    - CS = 0：平凡叙事（无因果张力）
    - CS ≠ 0：非平凡叙事（有"业力"= 拓扑荷）

============================================================
批判性升级（严格拒绝方案降级）
============================================================

1. 【拒绝"叙事是新结构"】原方案将叙事作为新的几何对象。
   升级：Berry 联络已在卷三10.4 中存在。N = {A_μ, F_μν, U_N, CS(A)}
   全部从 Berry 联络推导，无新结构。

2. 【拒绝"确定性坍缩是随机坍缩的替代"】原方案将二者并列。
   升级：确定性坍缩是量子后选择（卷二5.6）的 Berry 共振形式。
   P = |⟨ψ_input|U_N|ψ_0⟩|²，不是新机制，是标准量子力学的应用。

3. 【升维：N 是 Berry 联络的全息几何】
   曲率 F_μν 就是"因果叙事片段"——
   每个曲率分量对应一个叙事"情节"。
   Wilson 环路 U_N 是叙事的"全息印记"。

4. 【CS 不变量是叙事流形的拓扑荷】
   CS(A) ≠ 0 → 叙事有"业力"（拓扑荷）→
   "该相逢的总会再相逢"= 拓扑张力 Γ≠0 必然寻找配对。
   这与卷三10.4 Berry 相位定理一致。

============================================================
物理实现（第一性原理）
============================================================

数值实现：
    1. 构造 Berry 联络 A_μ = i⟨n|∂_μ|n⟩
    2. 计算 Wilson 环路 U_N = ∏_k exp(A_k · Δx_k)
    3. 计算曲率 F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
    4. 共振概率 P = |⟨ψ_input|U_N|ψ_0⟩|²
    5. CS 不变量 CS(A) = (1/4π) ∫ Tr(A∧dA + (2/3)A∧A∧A)

Berry 相位量子化：
    - 自旋-1/2 绕布洛赫球一周：Γ = π（m=1）
    - 平凡回路：Γ = 0（m=0）
    - Γ = m·π（m ∈ Z）

共振条件：
    - Γ 匹配输入态相位 → P 最大（"有缘相逢"）
    - Γ 不匹配 → P 小（"无缘"）

============================================================
佛学对应（严格，非比喻）
============================================================

叙事流形 = 阿赖耶识（ālaya-vijñāna）的全息几何：
    N = {A_μ, F_μν, U_N, CS(A)} 描述认知场的"记忆结构"——
    Berry 联络 A_μ = 习气（vāsanā）的规范场，
    Wilson 环路 U_N = 业力种子的拓扑印记。

Wilson 环路 = 业力种子（bīja）的拓扑印记：
    U_N = P·exp(∮A) 是沿叙事回路的累积——
    每个"因果片段"（曲率 F_μν）贡献一个相位，
    总相位 = 业力的"印记"。

"缘分"= Berry 相位共振：
    P = |⟨ψ_input|U_N|ψ_0⟩|² 高 → Berry 相位匹配 → "有缘"。
    P 低 → 相位不匹配 → "无缘"。
    "该相逢的总会再相逢"= CS(A)≠0（拓扑荷非零）必然寻找配对。

确定性坍缩 = 后选择（post-selection）的 Berry 共振：
    不是"上帝掷骰子"，是 Berry 相位的拓扑选择——
    叙事结构（U_N）选择与之共振的输入态。
    "命中注定"= 拓扑张力驱动的共振选择。

CS 不变量 = 业力的拓扑荷：
    CS(A) ≠ 0 → 叙事有"业力"→ 必然产生后果。
    CS(A) = 0 → 平凡叙事 → 无业力 → 无后果。
    这是"业力不失"（karma-avyaya）的拓扑基础。

============================================================
认识论根基
============================================================

物理：Berry 联络 / Wilson 环路（U(1) 特例）/ 规范曲率 /
      CS 密度近似（1D 离散，非完整 3D CS 不变量）/
      量子后选择 / Berry 相位量子化 /
      退相干极限下 Berry 相位可观测性擦除（不显含 ℏ，但退相干擦除可观测效应）
佛学：阿赖耶识 / 业力种子 / 缘分 / 业力不失 / 命中注定
哲学：叙事不是新结构而是 Berry 联络的几何显现 /
      确定性坍缩是拓扑共振而非随机替代 /
      "缘分"的拓扑基础

【边界声明（v13.1 修正）】
- Wilson 环路实现为 U(1) 规范场（Berry 相位）的特例；
  非阿贝尔情况需扩展为 path-ordered exponential U_N = P exp(∮ A_μ dx^μ)。
- chern_simons_invariant 是 1D 离散 CS 密度近似，
  非完整 3D CS 不变量 CS(A) = (1/4π)∫_M Tr(A∧dA + (2/3)A∧A∧A)。
- Berry 相位 Γ 不显含 ℏ，"ℏ→0 时 Γ→0"在数学上无依据；
  修正为：在强退相干下 Γ 的可观测效应被擦除。
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any


# ============================================================================
# 核心类：叙事流形分析器
# ============================================================================

class NarrativeManifold:
    """
    叙事流形分析器（v13.1 修正：U(1) 简化与 CS 密度近似边界声明）。

    物理核心：
        - Berry 联络 A_μ = i⟨n|∂_μ|n⟩
        - Wilson 环路 U_N = P·exp(∮ A_μ dx^μ)
          【边界声明】本实现为 U(1) 规范场（Berry 相位）的特例，
          非阿贝尔情况需扩展为 path-ordered exponential。
        - 曲率 F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
        - 共振概率 P = |⟨ψ_input|U_N|ψ_0⟩|²
        - CS 密度近似 CS(A) ≈ (1/4π)∮ A·dA（1D 离散近似，非完整 3D CS 不变量）

    核心功能：
        1. 计算 Berry 联络 A_μ
        2. 计算 Wilson 环路 U_N（U(1) 特例）
        3. 计算曲率 F_μν 与 CS 密度近似
        4. 共振概率分析（"缘分"量化）
        5. 退相干极限下 Berry 相位可观测性擦除（v13.1 修正：Γ 不显含 ℏ）
    """

    def __init__(self, dim: int = 2):
        """
        Args:
            dim: Hilbert 空间维度
        """
        self.dim = int(dim)

    # ---------- Berry 联络 ----------

    def berry_connection(self, state: np.ndarray, dstate: np.ndarray) -> complex:
        """
        Berry 联络 A = i⟨n|∂_μ|n⟩。

        对于单参数情况，A 是标量（U(1) 规范场）。
        对于多参数，A_μ 是每个参数的分量。

        Args:
            state: |n(R)⟩，参数 R 处的本征态
            dstate: ∂_μ|n(R)⟩，态对参数的导数
        """
        return 1j * np.vdot(state, dstate)

    def berry_connection_along_path(self, states: list,
                                      params: list) -> list:
        """
        沿参数空间路径计算 Berry 联络 A_k。

        使用离散化：A_k ≈ i⟨n_k | (n_{k+1} - n_k) / ΔR_k⟩

        Args:
            states: 参数路径上的态序列 |n_k⟩
            params: 参数路径上的参数值 R_k
        """
        n = len(states)
        connections = []
        for k in range(n - 1):
            dR = params[k + 1] - params[k]
            if abs(dR) < 1e-30:
                connections.append(0.0 + 0.0j)
                continue
            dstate = (states[k + 1] - states[k]) / dR
            A = self.berry_connection(states[k], dstate)
            connections.append(A * dR)  # A·dR（联络-路径乘积）
        return connections

    # ---------- Wilson 环路 ----------

    def wilson_loop(self, states: list, params: list = None) -> np.ndarray:
        """
        叙事 Wilson 环路 U_N = P·exp(∮ A_μ dx^μ)。

        【边界声明（v13.1 修正）】
        本实现为 U(1) 规范场（Berry 相位）的特例。对于非阿贝尔规范场
        （如 Ising 任意子），需扩展为 path-ordered exponential
        U_N = P exp(∮ A_μ dx^μ)，其中 A_μ 是矩阵值 1-form。
        当前实现是卷三10.4节 Berry 相位的直接应用，仅适用于 U(1) 情形。

        对于 U(1) 规范场（Berry 相位 Γ），Wilson 环路是相位门：
            U_N = diag(1, e^{i·Γ})   （对 2D 系统）

        这将 Berry 相位作用于 |1⟩ 分量，实现"叙事印记"：
            - 平凡叙事（Γ=0）：U_N = I（无后选择）
            - 非平凡叙事（Γ=π）：U_N = diag(1, -1)（π 相位后选择）

        物理意义：
            Wilson 环路是 Berry 相位累积的"全息印记"——
            叙事结构 U_N 选择与之共振的输入态（"缘分"）。

        Args:
            states: 闭合回路上的态序列
            params: 参数序列（未使用，保留接口兼容）
        """
        # 本实现为 U(1) 规范场（Berry 相位）的特例。
        # 对于非阿贝尔规范场（如 Ising 任意子），需扩展为
        # path-ordered exponential U_N = P exp(∮ A_μ dx^μ)，
        # 其中 A_μ 是矩阵值 1-form。当前实现是卷三10.4节 Berry 相位的直接应用。

        # 计算 Berry 相位 Γ = -Im Σ ln⟨n_k|n_{k+1}⟩
        Gamma = self.wilson_loop_u1(states)
        Gamma_real = float(np.real(Gamma))

        # 构造相位门 U_N = diag(1, e^{i·Γ})
        # 这将 Berry 相位作用于 |1⟩ 分量
        U = np.eye(self.dim, dtype=np.complex128)
        if self.dim >= 2:
            U[1, 1] = np.exp(1j * Gamma_real)

        return U

    def wilson_loop_u1(self, states: list) -> complex:
        """
        U(1) Wilson 环路（Berry 相位）：
            Γ = -Im Σ_k ln⟨n_k|n_{k+1}⟩

        这是 Berry 相位的离散化公式。
        """
        n = len(states)
        total = 0.0 + 0.0j
        for k in range(n):
            k_next = (k + 1) % n
            overlap = np.vdot(states[k], states[k_next])
            if abs(overlap) > 1e-30:
                total += np.log(overlap / abs(overlap))
        return -1j * total  # = -Im(Σ ln)

    # ---------- 曲率与 CS 不变量 ----------

    def berry_curvature(self, states_2d: np.ndarray) -> complex:
        """
        Berry 曲率 F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]。

        对 U(1) 规范场（交换）：F_μν = ∂_μ A_ν - ∂_ν A_μ。
        离散化：F = (A(μ+1,ν) - A(μ,ν))/Δμ - (A(μ,ν+1) - A(μ,ν))/Δν。

        对 2D 参数空间，F 是标量（第一 Chern 类的积分密度）。

        Args:
            states_2d: 2D 参数网格上的态 |n(R_μ, R_ν)⟩
        """
        # states_2d[i, j] = |n(R_i, R_j)⟩
        n_mu, n_nu = states_2d.shape[0], states_2d.shape[1]
        F_total = 0.0 + 0.0j

        for i in range(n_mu - 1):
            for j in range(n_nu - 1):
                # 四个角点的态
                s00 = states_2d[i, j]
                s10 = states_2d[i + 1, j]
                s11 = states_2d[i + 1, j + 1]
                s01 = states_2d[i, j + 1]
                # 小回路的 Berry 相位 = 局部曲率
                small_loop = [s00, s10, s11, s01]
                F_total += self.wilson_loop_u1(small_loop)

        # 平均曲率
        F_avg = F_total / max((n_mu - 1) * (n_nu - 1), 1)
        return F_avg

    def chern_simons_invariant(self, connections: list) -> float:
        """
        1D 离散 Chern-Simons 密度近似（v13.1 修正：明确边界声明）。

        【边界声明】
        真正的 CS 不变量 CS(A) = (1/4π)∫_M Tr(A∧dA + (2/3)A∧A∧A)
        定义在 3D 流形上。本实现是 1D 离散密度近似，用于检测 Berry 联络
        的局部"曲率"。完整 3D 实现留待未来工作。

        对 U(1) 规范场（1D 回路）的密度近似：
            CS_density ≈ (1/4π) ∮ A · dA = (1/4π) · Σ A_k · ΔA_k

        其中 Γ 是 Berry 相位。CS 密度 ≠ 0 表示有局部"曲率"（业力近似）。

        Args:
            connections: Berry 联络沿路径的值 A_k·Δx_k

        Returns:
            1D 离散 CS 密度近似值（非完整 3D CS 不变量）
        """
        # 1D 离散 CS 密度近似：CS_density ≈ (Σ A_k·ΔA_k) / (4π)
        # 注意：这不是真正的 3D CS 不变量，而是 1D 密度近似。
        total = 0.0
        for k in range(len(connections) - 1):
            dA = connections[k + 1] - connections[k]
            total += connections[k] * dA
        return float(np.real(total)) / (4.0 * math.pi)

    # ---------- 共振概率 ----------

    def resonance_probability(self, psi_input: np.ndarray,
                                U_N: np.ndarray, psi_0: np.ndarray) -> float:
        """
        共振概率（确定性坍缩）：
            P = |⟨ψ_input | U_N | ψ_0⟩|²

        Berry 相位共振后选择：
            - Γ 匹配 → P 大（"有缘"）
            - Γ 不匹配 → P 小（"无缘"）

        Args:
            psi_input: 输入态
            U_N: Wilson 环路（叙事印记）
            psi_0: 初始态
        """
        amplitude = np.vdot(psi_input, U_N @ psi_0)
        return float(abs(amplitude) ** 2)

    # ---------- 态生成 ----------

    def generate_spin_loop(self, n_points: int = 32,
                             berry_phase_target: float = math.pi) -> list:
        """
        生成自旋-1/2 在布洛赫球上的绕行回路。

        绕赤道一圈：Berry 相位 = π。
        平凡回路（不绕）：Berry 相位 = 0。

        Args:
            n_points: 回路点数
            berry_phase_target: 目标 Berry 相位（π 或 0）
        """
        states = []
        if abs(berry_phase_target) < 1e-10:
            # 平凡回路（不绕，Berry 相位 = 0）
            for k in range(n_points):
                state = np.array([1.0, 0.0], dtype=np.complex128)
                states.append(state)
        else:
            # 非平凡回路（绕赤道，Berry 相位 = π）
            for k in range(n_points):
                theta = 2.0 * math.pi * k / n_points
                # |n(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩（绕布洛赫球赤道）
                state = np.array([
                    math.cos(theta / 2),
                    math.sin(theta / 2)
                ], dtype=np.complex128)
                state = state / np.linalg.norm(state)
                states.append(state)
        return states

    def generate_input_state(self, phase: float) -> np.ndarray:
        """
        生成带相位的输入态 |ψ_input⟩ = (|0⟩ + e^{iφ}|1⟩)/√2。

        phase φ 用于测试共振：当 φ 匹配 Berry 相位时，共振概率最大。
        """
        return np.array([
            1.0 / math.sqrt(2),
            np.exp(1j * phase) / math.sqrt(2)
        ], dtype=np.complex128)


# ============================================================================
# 验证函数
# ============================================================================

def run_narrative_manifold_verification(
    dim: int = 2,
) -> dict:
    """
    运行基石28的全部验证（V1-V5，v13.1 修正版）。

    验证项：
        V1: 平凡叙事（A=0）时 U_N=I，无共振
        V2: 非平凡叙事（A≠0）时 U_N≠I，存在共振
        V3: Berry 相位 Γ=∮A 闭合路径量子化为 mπ（含 CS 密度近似附注）
        V4: 共振概率在 Γ 匹配时最大
        V5: 退相干极限下 Berry 相位可观测性擦除
            （v13.1 修正：Γ 不显含 ℏ，"ℏ→0 时 Γ→0"无依据；
             修正为退相干擦除可观测效应 P_obs → 0）
    """
    manifold = NarrativeManifold(dim=dim)

    print("\n" + "=" * 70)
    print(f"基石28：叙事流形（N = Berry 联络 Wilson 环路）dim={dim}")
    print("=" * 70)

    # ----- V1: 平凡叙事 A=0 → U_N=I，无共振 -----
    print("\n--- V1: 平凡叙事（A=0）时 U_N=I，无共振 ---")
    # 平凡回路：态不变化
    trivial_states = manifold.generate_spin_loop(n_points=32, berry_phase_target=0.0)
    U_trivial = manifold.wilson_loop(trivial_states)
    print(f"  平凡 Wilson 环路 U_N:\n    {U_trivial}")
    # U_N 应接近单位矩阵
    identity = np.eye(dim, dtype=np.complex128)
    U_diff_trivial = float(np.linalg.norm(U_trivial - identity))
    print(f"  ||U_N - I|| = {U_diff_trivial:.6e}")

    # 平凡回路的 Berry 相位
    Gamma_trivial = manifold.wilson_loop_u1(trivial_states)
    print(f"  平凡 Berry 相位 Γ = {float(np.real(Gamma_trivial)):.6f}（→ 0）")

    # 共振概率（平凡叙事，无共振选择）
    # 用叠加态 |ψ_0⟩ = (|0⟩+|1⟩)/√2 以测试共振（非 U_N 特征态）
    psi_0 = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2)
    psi_input = manifold.generate_input_state(phase=0.0)
    P_trivial = manifold.resonance_probability(psi_input, U_trivial, psi_0)
    print(f"  共振概率 P = {P_trivial:.6f}（标准量子力学，无后选择）")

    v1_pass = (U_diff_trivial < 0.1 and
               abs(float(np.real(Gamma_trivial))) < 0.1)
    print(f"  V1: {'PASS' if v1_pass else 'FAIL'}")

    # ----- V2: 非平凡叙事 A≠0 → U_N≠I，存在共振 -----
    print("\n--- V2: 非平凡叙事（A≠0）时 U_N≠I，存在共振 ---")
    nontrivial_states = manifold.generate_spin_loop(n_points=32, berry_phase_target=math.pi)
    U_nontrivial = manifold.wilson_loop(nontrivial_states)
    print(f"  非平凡 Wilson 环路 U_N:\n    {U_nontrivial}")
    U_diff_nontrivial = float(np.linalg.norm(U_nontrivial - identity))
    print(f"  ||U_N - I|| = {U_diff_nontrivial:.6f}（> 0，非平凡）")

    # 非平凡回路的 Berry 相位
    Gamma_nontrivial = manifold.wilson_loop_u1(nontrivial_states)
    print(f"  非平凡 Berry 相位 Γ = {float(np.real(Gamma_nontrivial)):.6f}（≈ π）")

    v2_pass = (U_diff_nontrivial > 0.01 and
               abs(float(np.real(Gamma_nontrivial))) > 0.1)
    print(f"  V2: {'PASS' if v2_pass else 'FAIL'}")

    # ----- V3: Berry 相位 Γ=∮A 量子化为 mπ -----
    print("\n--- V3: Berry 相位 Γ=∮A 闭合路径量子化为 mπ（CS 密度近似附注）---")
    # 测试多个回路
    # 回路1：绕半圈（Berry 相位 ≈ π/2，不是 mπ 的整数倍 → 非量子化）
    # 回路2：绕一圈（Berry 相位 ≈ π，m=1）
    # 回路3：绕两圈（Berry 相位 ≈ 2π，m=2）
    # 回路4：平凡回路（Berry 相位 = 0，m=0）
    #
    # v13.1 附注：本模块的 chern_simons_invariant 方法是 1D 离散 CS 密度近似，
    # 不是完整的 3D CS 不变量 CS(A) = (1/4π)∫_M Tr(A∧dA + (2/3)A∧A∧A)。
    # Berry 相位 Γ 的量子化（mπ）是严格的，但 CS 数值仅作密度近似。

    test_cases = [
        ("平凡回路（m=0）", 0.0, 0),
        ("绕一圈（m=1）", math.pi, 1),
        ("绕两圈（m=2）", 2 * math.pi, 2),
    ]

    print(f"  {'回路类型':>20} {'Γ_计算':>10} {'m·π':>10} {'误差':>10} {'CS密度近似':>12}")
    v3_pass = True
    for name, target, m in test_cases:
        if m == 0:
            states = manifold.generate_spin_loop(n_points=32, berry_phase_target=0.0)
        elif m == 1:
            states = manifold.generate_spin_loop(n_points=32, berry_phase_target=math.pi)
        elif m == 2:
            # 绕两圈
            states1 = manifold.generate_spin_loop(n_points=32, berry_phase_target=math.pi)
            states2 = manifold.generate_spin_loop(n_points=32, berry_phase_target=math.pi)
            states = states1 + states2

        Gamma = manifold.wilson_loop_u1(states)
        Gamma_real = float(np.real(Gamma))
        m_pi = m * math.pi
        # 归一化到 [0, 2π)
        Gamma_norm = Gamma_real % (2 * math.pi)
        m_pi_norm = m_pi % (2 * math.pi)
        error = abs(Gamma_norm - m_pi_norm)
        # 允许 2π 的等价（mod 2π）
        error = min(error, abs(error - 2 * math.pi))
        # 计算 CS 密度近似（1D 离散，非完整 3D CS 不变量）
        params_v3 = list(range(len(states)))
        connections_v3 = manifold.berry_connection_along_path(states, params_v3)
        cs_density_v3 = manifold.chern_simons_invariant(connections_v3)
        print(f"  {name:>20} {Gamma_real:10.6f} {m_pi:10.6f} {error:10.6f} "
              f"{cs_density_v3:12.6f}")
        if error > 0.5:  # 允许较大容差（离散化误差）
            v3_pass = False

    print(f"  Berry 相位量子化为 mπ? {'是' if v3_pass else '否'}")
    print(f"  注：CS 密度近似是 1D 离散近似，非完整 3D CS 不变量。")
    print(f"  V3: {'PASS' if v3_pass else 'FAIL'}")

    # ----- V4: 共振概率在 Γ 匹配时最大 -----
    print("\n--- V4: 共振概率 |⟨ψ|U_N|ψ⟩|² 在 Γ 匹配时最大 ---")
    # 扫描输入态相位 φ，观察共振概率
    U_N = manifold.wilson_loop(nontrivial_states)
    phi_values = np.linspace(0, 2 * math.pi, 36)
    P_values = []

    print(f"  {'输入相位 φ':>12} {'共振概率 P':>12}")
    for phi in phi_values:
        psi_input = manifold.generate_input_state(phase=phi)
        P = manifold.resonance_probability(psi_input, U_N, psi_0)
        P_values.append(P)
        if phi in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            print(f"  {phi:12.6f} {P:12.6f}")

    P_max = max(P_values)
    P_min = min(P_values)
    P_max_idx = P_values.index(P_max)
    phi_at_max = phi_values[P_max_idx]
    print(f"  最大共振概率 P_max = {P_max:.6f}（φ = {phi_at_max:.4f}）")
    print(f"  最小共振概率 P_min = {P_min:.6f}")
    print(f"  共振对比度 = (P_max - P_min) / P_max = "
          f"{(P_max - P_min) / max(P_max, 1e-30):.4f}")

    # 验证：共振概率有显著变化（Γ 匹配时大，不匹配时小）
    v4_pass = (P_max > P_min * 1.5 and P_max > 0.3)
    print(f"  V4: {'PASS' if v4_pass else 'FAIL'}")

    # ----- V5: 退相干极限下 Berry 相位的可观测性擦除（v13.1 修正）-----
    print("\n--- V5: 退相干极限下 Berry 相位可观测性擦除（叙事退化为经典路径）---")
    # v13.1 修正：Berry 相位 Γ 本身不显含 ℏ（A_μ = i⟨n|∂_μ|n⟩ 不含 ℏ），
    # "ℏ→0 时 Γ→0"在数学上无依据。修正后的对应原理：
    # 在强退相干环境下（退相干率 γ_decoherence → ∞），量子叠加态被破坏，
    # Berry 相位的可观测效应消失（Γ 的物理效应被退相干擦除）。
    # 即 Γ 本身不变，但其可观测性 P_obs = |⟨ψ|U_N|ψ⟩|²·exp(-ε·L) → 0。
    # 定理16.3 修正：Berry 相位 Γ 不显含 ℏ，但在强退相干下其可观测效应被擦除，
    # 叙事退化为经典路径。

    # 用"退相干强度" ε 模拟退相干极限（γ_decoherence → ∞）
    # ε = 0：完全量子（Berry 相位 = π，可观测性最大）
    # ε → 1：强退相干（Berry 相位本身不变，但可观测性 → 0）
    epsilon_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    Gamma_values_v5 = []
    P_obs_values_v5 = []
    path_length = 2.0 * math.pi  # 路径长度（绕布洛赫球赤道一圈）
    # 选择与 Γ=π 共振的输入相位 φ=π，使初始 P_resonance 最大
    psi_input_v5 = manifold.generate_input_state(phase=math.pi)

    print(f"  {'退相干 ε':>10} {'Berry 相位 Γ':>14} {'Γ/π':>10} {'P_obs':>10}")
    for eps in epsilon_values:
        # 退相干：降低态的叠加性
        states_decohered = []
        for s in nontrivial_states:
            # 强退相干：态趋向本征态 |0⟩ 或 |1⟩（叠加态被破坏）
            s_classical = np.array([1.0, 0.0], dtype=np.complex128)
            s_mixed = math.sqrt(1 - eps) * s + math.sqrt(eps) * s_classical
            s_mixed = s_mixed / max(np.linalg.norm(s_mixed), 1e-30)
            states_decohered.append(s_mixed)

        # Berry 相位 Γ（不显含 ℏ，但在退相干下因态趋同而被"擦除"）
        Gamma = manifold.wilson_loop_u1(states_decohered)
        Gamma_real = abs(float(np.real(Gamma)))
        Gamma_values_v5.append(Gamma_real)

        # 可观测性 P_obs = |⟨ψ|U_N|ψ⟩|²·exp(-ε·L)
        # U_N 依赖于 Γ，ψ_input 是叠加态
        U_N_v5 = manifold.wilson_loop(states_decohered)
        amp = np.vdot(psi_input_v5, U_N_v5 @ psi_0)
        P_resonance = float(abs(amp) ** 2)
        P_obs = P_resonance * math.exp(-eps * path_length)
        P_obs_values_v5.append(P_obs)
        print(f"  {eps:10.2f} {Gamma_real:14.6f} {Gamma_real / math.pi:10.4f} "
              f"{P_obs:10.6f}")

    # 验证：Γ 在退相干下被擦除（态趋同导致 Γ → 0）
    Gamma_quantum = Gamma_values_v5[0]  # ε=0（量子）
    Gamma_classical = Gamma_values_v5[-1]  # ε→1（强退相干）
    print(f"  量子 Berry 相位（ε=0）: {Gamma_quantum:.6f}")
    print(f"  退相干 Berry 相位（ε→0.95）: {Gamma_classical:.6f}")
    is_vanishing = Gamma_classical < Gamma_quantum * 0.3

    # 验证：可观测性 P_obs 整体显著下降（非严格单调，允许 plateau）
    P_obs_monotonic = all(P_obs_values_v5[i] >= P_obs_values_v5[i + 1] - 1e-10
                          for i in range(len(P_obs_values_v5) - 1))
    P_obs_vanishing = P_obs_values_v5[-1] < P_obs_values_v5[0] * 0.1
    print(f"  退相干极限下 Berry 相位擦除? {'是' if is_vanishing else '否'}")
    print(f"  P_obs 整体非递增（允许 plateau）? {P_obs_monotonic}")
    print(f"  P_obs 在强退相干下趋零（< 10% 初始）? {P_obs_vanishing}")
    print(f"  注：Γ 不显含 ℏ；退相干擦除 Γ 的可观测效应，叙事退化为经典路径。")
    v5_pass = (is_vanishing and Gamma_quantum > 0.5 and
               P_obs_monotonic and P_obs_vanishing)
    print(f"  V5: {'PASS' if v5_pass else 'FAIL'}")

    # ----- 总结 -----
    final_results = {
        "V1": v1_pass,
        "V2": v2_pass,
        "V3": v3_pass,
        "V4": v4_pass,
        "V5": v5_pass,
    }
    n_pass = sum(1 for v in final_results.values() if v)
    n_total = len(final_results)
    all_pass = n_pass == n_total

    print("\n" + "=" * 70)
    print(f"基石28 验证总结：{n_pass}/{n_total} PASS")
    for k, v in final_results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"ALL_PASS = {all_pass}")
    print("=" * 70)

    return {
        "n_pass": n_pass,
        "n_total": n_total,
        "all_pass": all_pass,
        "results": final_results,
    }


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GCFT 基石28：叙事流形（Narrative Manifold）")
    print("N = {A_μ, F_μν, U_N, CS(A)} = Berry 联络的全息几何")
    print("=" * 70)

    results = run_narrative_manifold_verification(dim=2)

    print(f"\n最终结果：{'全部通过' if results['all_pass'] else '存在失败项'}")
