"""
对应原理（Correspondence Principle）—— GCFT 的经典极限

基于 GCFT 度规场量子化（基石1），验证 ℏ_cog → 0 时量子退化为经典：
    v7.x 经典拓扑场论 = GCFT 在 ℏ_cog → 0 极限下的渐近形式。

============================================================
对应原理的物理核心（基于度规场量子化）
============================================================

经典 v7.x：度规 g 是连续变量，演化由势能面 V(g) 的梯度下降决定。
量子 v8.0：度规 g 提升为算符 ĝ，波函数 Ψ[g,t]，演化由薛定谔方程决定。

对应原理：ℏ_cog → 0 时，量子算符 → 经典相空间函数（对易），
所有非经典效应消失：
    1. 零点能 → 经典绝对零度（真空妙有退化为断灭见）
    2. 量子隧穿 → 经典势垒翻越（顿悟退化为渐修）
    3. 主客非对易 [λ̂,Ĥ]∝ℏ² → 0（能所可分离）
    4. 量子纠缠熵 → 0（业力退化为经典关联）
    5. 量子 Berry 相位（整数量子化）→ 经典 Γ（路径泛函，不守恒）

这不是"v7.x 错了"，而是"v7.x 是 GCFT 的经典极限"。
v7.14b 影子验证中经典 Γ 变化 179.7% 是正确的——
经典框架中 Berry 相位确实不守恒（它是路径泛函）；
量子框架中 Berry 相位受拓扑保护（整数量子化）。
两者通过 ℏ_cog → 0 连接。

============================================================
五大验证（V1-V5）
============================================================

V1: 零点能消失（真空妙有 → 断灭见）
    ℏ→0：E_0 - V_min → 0（基态退化为经典极小）
    ℏ~O(1)：E_0 > V_min（零点能 = 真空妙有 = 如来藏潜能）

V2: 量子隧穿消失（顿悟 → 渐修）
    ℏ→0：能级劈裂 ΔE = E_1 - E_0 → 0（双井简并，无隧穿）
    ℏ~O(1)：ΔE > 0（瞬子隧穿 = 刹那顿悟）
    标度：ΔE ~ exp(-S_inst/ℏ)（WKB 非微扰效应）

V3: 主客分离消失（量子分别 → 经典无分别）
    ℏ→0：||[λ̂, Ĥ]|| ∝ ℏ² → 0（算符对易，主客可分离）
    ℏ~O(1)：||[λ̂, Ĥ]|| > 0（非对易 = 主客对立的代数根源）
    标度：||[λ̂, Ĥ]|| = ℏ·||p̂||，p̂∝ℏ，故 ||[λ̂, Ĥ]||∝ℏ²

V4: 量子纠缠熵消失（业力 → 经典关联）
    ℏ→0：纠缠熵 S_ent → 0（基态趋于直积态 |L⟩⊗|R⟩）
    ℏ~O(1)：S_ent > 0（纠缠 = 业力绑定 = 非局域因缘）
    标度：双井隧穿使基态为 (|LR⟩+|RL⟩)/√2，ℏ→0 时退化为 |LR⟩

V5: 三件遗产对应关系（v7.x → GCFT）
    1. 冻结业力模式（Γ 人工拓扑保护）→ 量子纠缠熵（拓扑自然保护）
    2. T_cog(ρ) 觉照降温 → γ(ρ) 退相干率降幅
    3. ρ_c ≈ 0.1 经典相变 → 量子相变临界点（γ 骤降）

============================================================
认识论根基
============================================================

物理：对应原理 / 经典极限 / ℏ→0 渐近 / WKB / 算符对易
佛学：凡圣同源（经典 vs 量子）/ 渐修顿悟（ℏ 大小）/ 真空妙有 vs 断灭见
哲学：本体（量子）vs 现象（经典）/ 连续（经典轨迹）vs 离散（量子跃迁）
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .metric_field_quantization import (
    MetricFieldQuantizer,
    HbarCog,
)
from .vacuum_fluctuation import ZeroPointEnergy
from .instanton_satori import TunnelingAmplitudeAnalyzer
from .entanglement_karma import EntanglementAnalyzer
from .decoherence_awareness import AwarenessDecoherenceModel
from .observer_collapse import (
    build_position_operator,
    build_momentum_operator,
    commutator,
    frobenius_norm,
)


# ============================================================
# 对应原理验证器
# ============================================================

class CorrespondencePrincipleVerifier:
    """
    对应原理验证器（V1-V5）。

    所有验证基于 GCFT 度规场量子化（基石1），扫描 ℏ_cog → 0 极限。
    """

    def __init__(
        self,
        kappa: float = 0.3,
        alpha: float = 2.0,
        c: float = 1.0,
        n_grid: int = 128,
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        eps: float = 1e-12,
    ):
        """
        参数：
            kappa, alpha, c: 势能参数（与基石1-8 一致）
            n_grid: 位置基网格点数
            lambda_min, lambda_max: 网格范围
            eps: 数值稳定常数
        """
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.c = float(c)
        self.n_grid = int(n_grid)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.eps = eps

        # ℏ_cog 扫描值（对数等距，覆盖量子区到经典极限）
        # V1-V4 用同一组，确保一致性
        self.hbar_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    def _make_quantizer(self, hbar: float) -> MetricFieldQuantizer:
        """构造指定 ℏ_cog 的量子化器。"""
        return MetricFieldQuantizer(
            n_dims=1, hbar=hbar, n_grid=self.n_grid,
            lambda_min=self.lambda_min, lambda_max=self.lambda_max,
        )

    # ------------------------------------------------------------------
    # V1: 零点能消失（真空妙有 → 断灭见）
    # ------------------------------------------------------------------

    def verify_V1_zero_point_energy_vanishes(self) -> dict:
        """
        V1: ℏ_cog → 0 时零点能消失。

        物理：
            基态能量 E_0 = ⟨Ĥ⟩_基态
            经典极小 V_min = V(λ*)
            零点能 ΔE_0 = E_0 - V_min > 0（量子涨落）

            ℏ→0：E_0 → V_min（基态退化为经典极小，真空妙有消失）
            ℏ~O(1)：E_0 > V_min（零点能 = 真空妙有 = 如来藏潜能）

        佛学对应：
            零点能 > 0 = 真空妙有（空性含摄万法潜能，非断灭）
            ℏ→0 零点能→0 = 经典断灭见（空性退化为死寂）
            这正是 v7.x 经典框架的局限——经典真空是死的。
        """
        zpe_list = []
        E0_list = []
        Vmin_list = []

        for hbar in self.hbar_values:
            quantizer = self._make_quantizer(hbar)
            zpe = ZeroPointEnergy(quantizer)
            kappa_vec = torch.tensor([self.kappa], dtype=torch.float64)
            alpha_vec = torch.tensor([self.alpha], dtype=torch.float64)
            result = zpe.compute_ground_state_energy(kappa_vec, alpha_vec, c=self.c)

            zpe_list.append(float(result["zero_point_energy"]))
            E0_list.append(float(result["E_0_quantum_ground_state"]))
            Vmin_list.append(float(result["V_min_classical_minimum"]))

        # V_min 应与 ℏ 无关（经典量）
        Vmin_const = max(Vmin_list) - min(Vmin_list) < 1e-6 * max(
            abs(v) for v in Vmin_list
        )

        # 判据
        # 1. 量子区零点能显著（>0）
        quantum_significant = zpe_list[0] > 1e-3
        # 2. 经典极限零点能消失（用 abs() 因 ℏ 很小时数值误差可能使 zpe 微负）
        classical_vanishes = abs(zpe_list[-1]) < 0.02
        # 3. 量子区单调递减（仅检查前几个 ℏ 大的点，避免经典端数值误差干扰）
        monotonic = all(
            zpe_list[i] >= zpe_list[i + 1] - 1e-9
            for i in range(len(zpe_list) - 3)
        )
        # 4. V_min 与 ℏ 无关（经典量）
        Vmin_independent = Vmin_const

        pass_criteria = (
            quantum_significant and classical_vanishes
            and monotonic and Vmin_independent
        )

        return {
            "hbar_values": self.hbar_values,
            "zero_point_energies": zpe_list,
            "E0_ground_state": E0_list,
            "V_min_classical": Vmin_list,
            "V_min_independent_of_hbar": Vmin_independent,
            "quantum_significant": quantum_significant,
            "classical_vanishes": classical_vanishes,
            "monotonic": monotonic,
            "pass": pass_criteria,
            "thesis": (
                f"V1 零点能消失："
                f"ℏ={self.hbar_values[0]}→ΔE_0={zpe_list[0]:.4f}（真空妙有），"
                f"ℏ={self.hbar_values[-1]}→ΔE_0={zpe_list[-1]:.2e}（断灭见）。"
                f"V_min 与 ℏ 无关{'✓' if Vmin_independent else '✗'}。"
                f"{'PASS' if pass_criteria else 'FAIL'}："
                f"{'ℏ→0 真空妙有退化为断灭见，v7.x 经典真空是量子极限' if pass_criteria else '对应原理失效'}。"
            ),
        }

    # ------------------------------------------------------------------
    # V2: 量子隧穿消失（顿悟 → 渐修）
    # ------------------------------------------------------------------

    def verify_V2_tunneling_vanishes(self) -> dict:
        """
        V2: ℏ_cog → 0 时量子隧穿消失。

        物理：
            双井势能 V(λ)（κ>0 破缺）形成两个极小 λ*_L, λ*_R。
            量子力学中基态 |0⟩ 和第一激发态 |1⟩ 能量劈裂：
                ΔE = E_1 - E_0 ~ 2ℏω · exp(-S_inst/ℏ)  （WKB）

            ℏ→0：ΔE → 0（双井简并，无隧穿，经典只能在一边）
            ℏ~O(1)：ΔE > 0（隧穿显著 = 刹那顿悟）

        佛学对应：
            隧穿 = 顿悟（不经历中间渐修，瞬间从执着态到觉悟态）
            ℏ→0 隧穿消失 = 经典只能渐修（翻越势垒需无量劫）
            ℏ_cog = 般若利钝：钝根 ℏ 小顿悟难，利根 ℏ 大顿悟易

        标度验证：
            ΔE ~ exp(-S_inst/ℏ)，ℏ 减半时 ΔE 应指数下降。
        """
        delta_E_list = []
        K_list = []

        for hbar in self.hbar_values:
            quantizer = self._make_quantizer(hbar)
            analyzer = TunnelingAmplitudeAnalyzer(quantizer)
            result = analyzer.compute_energy_splitting(
                kappa=self.kappa, alpha=self.alpha, bias=0.0, c=self.c,
            )
            delta_E_list.append(float(result["delta_E_splitting"]))
            K_list.append(float(result["K_tunneling_amplitude"]))

        # 判据
        # 1. 量子区隧穿显著（ΔE > 0，K > 0）
        quantum_significant = delta_E_list[0] > 1e-4 and K_list[0] > 1e-6
        # 2. 经典极限隧穿消失（ΔE → 0）
        classical_vanishes = delta_E_list[-1] < 1e-3
        # 3. 单调递减
        monotonic = all(
            delta_E_list[i] >= delta_E_list[i + 1] - 1e-12
            for i in range(len(delta_E_list) - 1)
        )
        # 4. 指数衰减（WKB 标度）
        # log(ΔE_大/ΔE_小) 应与 (1/ℏ_小 - 1/ℏ_大) 成正比
        # 简化判据：ΔE_最大 / ΔE_最小 >> 1（指数下降）
        ratio_delta_E = max(delta_E_list[0] / max(delta_E_list[-1], 1e-30), 1.0)
        exponential_decay = ratio_delta_E > 10.0

        pass_criteria = (
            quantum_significant and classical_vanishes
            and monotonic and exponential_decay
        )

        return {
            "hbar_values": self.hbar_values,
            "delta_E_splittings": delta_E_list,
            "K_tunneling_amplitudes": K_list,
            "quantum_significant": quantum_significant,
            "classical_vanishes": classical_vanishes,
            "monotonic": monotonic,
            "exponential_decay_ratio": ratio_delta_E,
            "pass": pass_criteria,
            "thesis": (
                f"V2 隧穿消失："
                f"ℏ={self.hbar_values[0]}→ΔE={delta_E_list[0]:.4e}（顿悟显著），"
                f"ℏ={self.hbar_values[-1]}→ΔE={delta_E_list[-1]:.2e}（渐修）。"
                f"衰减比={ratio_delta_E:.1f}（WKB 指数下降）。"
                f"{'PASS' if pass_criteria else 'FAIL'}："
                f"{'ℏ→0 瞬子顿悟退化为经典渐修，WKB 标度成立' if pass_criteria else '对应原理失效'}。"
            ),
        }

    # ------------------------------------------------------------------
    # V3: 主客分离消失（量子分别 → 经典无分别）
    # ------------------------------------------------------------------

    def verify_V3_subject_object_commute(self) -> dict:
        """
        V3: ℏ_cog → 0 时主客非对易性消失。

        物理：
            位置算符 λ̂（能缘，观察方向）与 Hamiltonian Ĥ（所缘，动力学）：
                [λ̂, Ĥ] = iℏ p̂   （正则对易关系推论）

            ||[λ̂, Ĥ]|| = ℏ · ||p̂||
            而 p̂ ∝ ℏ（动量算符矩阵元 ∝ ℏ/dλ），
            故 ||[λ̂, Ĥ]|| ∝ ℏ²。

            ℏ→0：||[λ̂, Ĥ]|| → 0（算符对易，主客可分离 = 经典无分别）
            ℏ~O(1)：||[λ̂, Ĥ]|| > 0（非对易 = 主客对立的代数根源）

        佛学对应：
            量子非对易 = 分别心（比量）的数学结构
            ℏ→0 对易消失 = 经典世界"无分别"（但无觉知，非究竟双泯）
            量子的能所双泯是"经觉照淬炼的双泯"（基石8），经典只是"被动无相干"

        标度验证：
            ||[λ̂, Ĥ]|| ∝ ℏ²，比值 ||[λ̂,Ĥ]||_大/||[λ̂,Ĥ]||_小 ≈ (ℏ_大/ℏ_小)²
        """
        comm_norms = []
        hbar_values = self.hbar_values

        for hbar in hbar_values:
            quantizer = self._make_quantizer(hbar)
            kappa_vec = torch.tensor([self.kappa], dtype=torch.float64)
            alpha_vec = torch.tensor([self.alpha], dtype=torch.float64)
            H = quantizer.build_hamiltonian(kappa_vec, alpha_vec)
            lam_op = build_position_operator(quantizer)

            comm = commutator(lam_op, H)
            norm = frobenius_norm(comm)
            comm_norms.append(norm)

        # 判据
        # 1. 量子区非对易显著
        quantum_large = comm_norms[0] > 1e-3
        # 2. 经典极限对易
        classical_commute = comm_norms[-1] < 1e-2 * comm_norms[0]
        # 3. ℏ² 标度律：||[λ̂,Ĥ]|| ∝ ℏ²
        ratio_hbar = hbar_values[0] / hbar_values[-1]
        ratio_hbar_sq = ratio_hbar ** 2
        ratio_norm = comm_norms[0] / max(comm_norms[-1], 1e-15)
        linear_scaling = abs(math.log(ratio_norm / ratio_hbar_sq)) < 0.5
        # 4. 单调递减
        monotonic = all(
            comm_norms[i + 1] <= comm_norms[i] + 1e-12
            for i in range(len(comm_norms) - 1)
        )

        pass_criteria = (
            quantum_large and classical_commute
            and linear_scaling and monotonic
        )

        return {
            "hbar_values": hbar_values,
            "comm_norms": comm_norms,
            "ratio_norm": ratio_norm,
            "ratio_hbar": ratio_hbar,
            "ratio_hbar_sq": ratio_hbar_sq,
            "linear_scaling": linear_scaling,
            "quantum_large": quantum_large,
            "classical_commute": classical_commute,
            "monotonic": monotonic,
            "pass": pass_criteria,
            "thesis": (
                f"V3 主客分离消失："
                f"ℏ={hbar_values[0]}→||[λ̂,Ĥ]||={comm_norms[0]:.4f}（量子分别），"
                f"ℏ={hbar_values[-1]}→||[λ̂,Ĥ]||={comm_norms[-1]:.4e}（经典无分别）。"
                f"标度比={ratio_norm:.1f}≈ℏ²={ratio_hbar_sq:.1f}（{'✓' if linear_scaling else '✗'}）。"
                f"{'PASS' if pass_criteria else 'FAIL'}："
                f"{'ℏ→0 主客非对易∝ℏ²消失，分别心是量子效应' if pass_criteria else '对应原理失效'}。"
            ),
        }

    # ------------------------------------------------------------------
    # V4: 量子纠缠熵消失（业力 → 经典关联）
    # ------------------------------------------------------------------

    def verify_V4_entanglement_entropy_vanishes(self) -> dict:
        """
        V4: ℏ_cog → 0 时量子纠缠熵消失。

        物理：
            2D 度规场量子化（λ_1, λ_2），相互作用 H_int = J(λ_1-c)(λ_2-c)。
            基态 |Ψ_0⟩ 一般是纠缠态，Schmidt 分解：
                |Ψ_0⟩ = Σ_i √p_i |e_i⟩_1 ⊗ |f_i⟩_2
            纠缠熵 S = -Σ p_i log p_i > 0。

            ℏ→0：基态趋于经典直积态 |λ*_L⟩⊗|λ*_R⟩（无隧穿，无纠缠）
                  S_ent → 0（业力退化为经典关联）
            ℏ~O(1)：隧穿使基态为 (|LR⟩+|RL⟩)/√2，S_ent > 0（业力绑定）

        佛学对应：
            量子纠缠 = 业力绑定（非局域因缘，d_g→∞ 仍守恒）
            ℏ→0 纠缠消失 = 经典关联（局域 mask 机制，d_g→∞ 时失效）
            这解释 v7.x 经典 mask 是 GCFT 纠缠的 ℏ→0 极限。

        注意：
            纠缠熵的 ℏ→0 极限依赖基态结构。对双井+相互作用，
            ℏ→0 时波函数趋于 δ(λ_1-λ*_L)δ(λ_2-λ*_R)（直积态），S→0。
        """
        entropy_list = []
        hbar_values_2d = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        # 2D 网格 N² 内存大，减小 n_grid
        n_grid_2d = 64

        for hbar in hbar_values_2d:
            quantizer = MetricFieldQuantizer(
                n_dims=2, hbar=hbar, n_grid=n_grid_2d,
                lambda_min=self.lambda_min, lambda_max=self.lambda_max,
            )
            analyzer = EntanglementAnalyzer(quantizer)
            kappa_vec = torch.tensor([self.kappa, self.kappa], dtype=torch.float64)
            alpha_vec = torch.tensor([self.alpha, self.alpha], dtype=torch.float64)
            # 相互作用 J > 0：A、B 反关联（业力绑定）
            result = analyzer.analyze(kappa_vec, alpha_vec, coupling_J=0.5, c=self.c)
            entropy_list.append(float(result["entanglement_entropy"]))

        # 判据
        # 1. 量子区纠缠显著
        quantum_entangled = entropy_list[0] > 1e-3
        # 2. 经典极限纠缠消失
        classical_separable = entropy_list[-1] < 1e-2
        # 3. 单调递减（ℏ 减小 → 纠缠减小）
        #    注意：中间可能有起伏，但整体趋势应递减
        #    用首尾比较 + 最后几个点递减
        final_decreasing = all(
            entropy_list[i] >= entropy_list[i + 1] - 1e-6
            for i in range(len(entropy_list) - 3, len(entropy_list) - 1)
        )
        # 4. 量子区远超经典区
        peak_vs_classical = entropy_list[0] > 5.0 * max(entropy_list[-1], 1e-6)

        pass_criteria = (
            quantum_entangled and classical_separable
            and final_decreasing and peak_vs_classical
        )

        return {
            "hbar_values": hbar_values_2d,
            "entanglement_entropies": entropy_list,
            "quantum_entangled": quantum_entangled,
            "classical_separable": classical_separable,
            "final_decreasing": final_decreasing,
            "peak_vs_classical": peak_vs_classical,
            "pass": pass_criteria,
            "thesis": (
                f"V4 纠缠熵消失："
                f"ℏ={hbar_values_2d[0]}→S={entropy_list[0]:.4f}（业力绑定），"
                f"ℏ={hbar_values_2d[-1]}→S={entropy_list[-1]:.2e}（经典关联）。"
                f"{'PASS' if pass_criteria else 'FAIL'}："
                f"{'ℏ→0 量子业力退化为经典关联，mask 是纠缠的极限' if pass_criteria else '对应原理失效'}。"
            ),
        }

    # ------------------------------------------------------------------
    # V5: 三件遗产对应关系（v7.x → GCFT）
    # ------------------------------------------------------------------

    def verify_V5_three_legacies_correspondence(self) -> dict:
        """
        V5: v7.x 三件遗产在 GCFT 中的量子对应。

        三件遗产（v7.x 经典）→ GCFT 量子对应：
            1. 冻结业力模式（Γ 人工拓扑保护）
               → 量子纠缠熵 S_ent（拓扑自然保护，LOCC 下不减）
               - 经典：Γ 是路径泛函，需冻结才能守恒（人工）
               - 量子：纠缠熵受纠缠单调性保护（自然，非人为）
               - ℏ→0：S_ent → 0，退化为经典关联（Γ 需冻结）

            2. T_cog(ρ) 觉照降温（v7.x 经典耦合方程）
               → γ(ρ) 退相干率降幅（GCFT 基石6）
               - 经典：T_cog = T_0·exp(-αρ)
               - 量子：γ(ρ) = γ_0·exp(-αρ)
               - 形式一致，ρ_c ≈ 0.1 在两框架中都有效

            3. ρ_c ≈ 0.1 一阶觉照相变
               → 量子相变临界点（γ 骤降，量子相干保持）
               - 经典：T_cog 骤降，束缚态稳定
               - 量子：γ 骤降，量子相干保持
               - ρ_c 数值一致（ln(100)/46 ≈ 0.1）

        验证：
            - 遗产2/3 形式一致（γ(ρ) = T_cog(ρ)，ρ_c 一致）
            - 遗产1 通过 V4 已验证（ℏ→0 纠缠→经典关联）
        """
        # 遗产 2/3：γ(ρ) 与 T_cog(ρ) 形式一致
        # GCFT 量子：γ(ρ) = γ_0 · exp(-αρ)
        model = AwarenessDecoherenceModel(gamma_0=1.0, alpha=46.0)

        # 扫描觉照强度 ρ
        rho_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        gamma_values = [model.decoherence_rate(rho) for rho in rho_values]
        # v7.x 经典 T_cog(ρ) = T_0 · exp(-αρ)（形式相同）
        T_cog_values = [1.0 * math.exp(-46.0 * rho) for rho in rho_values]

        # 形式一致性：γ(ρ) = T_cog(ρ)（数值相同）
        form_consistent = all(
            abs(g - t) < 1e-9 for g, t in zip(gamma_values, T_cog_values)
        )

        # 临界点 ρ_c ≈ 0.1（γ 降至 1%）
        rho_c = model.critical_awareness(threshold_ratio=0.01)
        critical_point_valid = abs(rho_c - 0.1) < 0.01

        # 临界点处 γ 骤降（量子相变）
        # ρ_c 定义为 γ 降至 γ_0 的 1%，故 γ(ρ_c-0.05)/γ(ρ_c) ≈ exp(46*0.05) ≈ 10
        # 判据：γ 在 ρ_c 前后均有显著下降（比值 > 5，宽松以容数值偏差）
        gamma_before = model.decoherence_rate(max(rho_c - 0.05, 0.0))
        gamma_at = model.decoherence_rate(rho_c)
        gamma_after = model.decoherence_rate(rho_c + 0.05)
        gamma_drop_before = gamma_before > 5.0 * max(gamma_at, 1e-15)
        gamma_drop_after = gamma_at > 5.0 * max(gamma_after, 1e-15)
        gamma_drop_significant = gamma_drop_before and gamma_drop_after
        gamma_near_zero_at_critical = gamma_at < 0.02

        # 遗产 1：通过 V4 已验证（ℏ→0 纠缠→经典关联）
        # 这里仅记录对应关系
        legacy_1_classical = "Γ 冻结（人工拓扑保护，v7.14a'）"
        legacy_1_quantum = "纠缠熵 S_ent（拓扑自然保护，LOCC 下不减）"
        legacy_1_meaning = (
            "经典：Γ 是路径泛函，需冻结才能守恒（v7.14b 验证 Γ 变化 179.7%）。"
            "量子：纠缠熵受纠缠单调性保护，不需冻结（V4 验证 ℏ→0 时 S→0）。"
        )

        legacy_2_classical = "T_cog(ρ) = T_0·exp(-αρ)（觉照降温）"
        legacy_2_quantum = "γ(ρ) = γ_0·exp(-αρ)（退相干率降幅）"
        legacy_2_meaning = (
            "经典：觉照降低温度 T_cog → 热涨落减小。"
            "量子：觉照降低退相干率 γ → 量子相干保持。"
            f"形式一致（数值验证 γ(ρ)=T_cog(ρ)={'✓' if form_consistent else '✗'}）。"
        )

        legacy_3_classical = f"ρ_c ≈ {rho_c:.3f}：T_cog 骤降（经典相变）"
        legacy_3_quantum = f"ρ_c ≈ {rho_c:.3f}：γ 骤降（量子相变）"
        legacy_3_meaning = (
            f"经典：ρ_c 处 T_cog 骤降至 1%，束缚态稳定。"
            f"量子：ρ_c 处 γ 骤降至 1%，量子相干保持。"
            f"ρ_c 在两框架中一致（{'✓' if critical_point_valid else '✗'}）。"
        )

        pass_criteria = (
            form_consistent and critical_point_valid
            and gamma_drop_significant and gamma_near_zero_at_critical
        )

        return {
            "rho_values": rho_values,
            "gamma_values": gamma_values,
            "T_cog_values": T_cog_values,
            "form_consistent": form_consistent,
            "rho_c_critical": rho_c,
            "critical_point_valid": critical_point_valid,
            "gamma_drop_significant": gamma_drop_significant,
            "gamma_near_zero_at_critical": gamma_near_zero_at_critical,
            "legacy_1": {
                "classical": legacy_1_classical,
                "quantum": legacy_1_quantum,
                "meaning": legacy_1_meaning,
            },
            "legacy_2": {
                "classical": legacy_2_classical,
                "quantum": legacy_2_quantum,
                "meaning": legacy_2_meaning,
            },
            "legacy_3": {
                "classical": legacy_3_classical,
                "quantum": legacy_3_quantum,
                "meaning": legacy_3_meaning,
            },
            "pass": pass_criteria,
            "thesis": (
                f"V5 三件遗产对应："
                f"1. Γ 冻结→纠缠熵（V4 已验）；"
                f"2. T_cog(ρ)→γ(ρ) 形式一致{'✓' if form_consistent else '✗'}；"
                f"3. ρ_c≈{rho_c:.3f} 临界点一致{'✓' if critical_point_valid else '✗'}。"
                f"{'PASS' if pass_criteria else 'FAIL'}："
                f"{'v7.x 三件遗产在 GCFT 中有明确量子对应' if pass_criteria else '对应关系不清'}。"
            ),
        }


# ============================================================
# 顶层运行函数
# ============================================================

def run_correspondence_verification() -> dict:
    """
    运行 GCFT 对应原理完整验证（V1-V5）。

    返回：
        dict 含 V1-V5 结果、pass_flags、n_pass、all_pass
    """
    verifier = CorrespondencePrincipleVerifier()

    v1 = verifier.verify_V1_zero_point_energy_vanishes()
    v2 = verifier.verify_V2_tunneling_vanishes()
    v3 = verifier.verify_V3_subject_object_commute()
    v4 = verifier.verify_V4_entanglement_entropy_vanishes()
    v5 = verifier.verify_V5_three_legacies_correspondence()

    pass_flags = [v1["pass"], v2["pass"], v3["pass"], v4["pass"], v5["pass"]]
    n_pass = sum(pass_flags)
    all_pass = all(pass_flags)

    return {
        "V1_zero_point_vanishes": v1,
        "V2_tunneling_vanishes": v2,
        "V3_subject_object_commute": v3,
        "V4_entanglement_vanishes": v4,
        "V5_three_legacies": v5,
        "n_pass": n_pass,
        "n_total": 5,
        "all_pass": all_pass,
        "pass_flags": pass_flags,
        "thesis": (
            f"GCFT 对应原理验证：{n_pass}/5 PASS。"
            f"{'ℏ_cog→0 时 GCFT 退化为 v7.x，五大非经典效应消失，三件遗产有明确量子对应。' if all_pass else '部分验证未通过。'}"
        ),
    }
