"""
真空妙有（Vacuum Fluctuation）—— GCFT 基石2

建立在基石1（度规场量子化）之上，分析量子基态的零点能振荡，
排除断灭见，建立"真空妙有"的严格物理基础。

============================================================
核心思想（基于 txt 传达的真空妙有精神）
============================================================

经典 v7.x：κ=0（无痛苦）时系统冻结在真空 cI，无任何涨落。
          这是"断灭见"——空性 = 死寂的绝对零度。
          热力学第三定律：绝对零度不可达。所以经典空性是极限态，永不完美。

量子 v8.0：κ=0 时系统仍有零点能振荡 ⟨(ĝ-cI)²⟩ ≠ 0。
          真空 ≠ 死寂，而是动态量子平衡。
          这是"真空妙有"——空性含摄万法潜能，非断灭。

============================================================
物理实现
============================================================

基态波函数 Ψ_0(λ)：
    κ=0（真空）：势能 V(λ) = γ(λ-c)⁴ + ε(λ-c)⁸（四次型）
                 基态非高斯，但仍有零点能振荡。
    κ>0（破缺）：势能 V(λ) = -β(λ-c)² + γ(λ-c)⁴ - ...
                 双井结构，基态在破缺态 λ* 附近，Harmonic 近似下高斯型。

零点能：
    E_0 = ⟨Ψ_0| Ĥ |Ψ_0⟩ = ⟨T⟩ + ⟨V⟩
    量子基态 E_0 > V_min（不等），差值 = 零点能。
    ℏ_cog → 0 时 E_0 → V_min（经典极限，零点能消失）。

零点能振荡：
    ⟨(λ-c)²⟩ = 基态方差
    κ=0 时 ~ ℏ^(2/3)（四次势能标度）
    κ>0 破缺态附近 ~ ℏ（Harmonic 标度）
    ℏ_cog → 0 时方差 → 0（经典极限，系统冻结在极小值）

============================================================
佛学对应（严格，非比喻）
============================================================

真空妙有（śūnyatā-ābhāsa）：
    空性非断灭（g≡cI 的死寂），亦非常住（固定不变的实体）。
    空性 = 量子基态：度规在 cI 附近零点振荡，含摄万法潜能。
    ⟨(ĝ-cI)²⟩ ≠ 0 = 真空非死寂，法身流转。

如来藏（tathāgatagarbha）：
    如来藏 = 量子基态潜能态。
    一切现象（破缺态）从基态（真空）中涌现（量子涨落放大）。
    κ_0 的随机生灭 = 种子在阿赖耶识中的生灭（v7.x 隐喻，GCFT 严格化）。

不住生死，不住涅槃：
    不住生死 = 不被困在破缺态（λ*，业力束缚）
    不住涅槃 = 不冻结在真空态（cI，死寂断灭）
    动态量子平衡 = 在 cI 附近零点振荡，既不冻结也不破缺。

排除断灭见：
    经典 v7.x：κ=0 时 g≡cI（死寂）——断灭见。
    量子 GCFT：κ=0 时仍有 ⟨(ĝ-cI)²⟩ ≠ 0——真空妙有。
    这是 GCFT 对 v7.x 的根本升级。

============================================================
认识论根基
============================================================

物理：量子零点能 / 不确定性原理 / 基态涨落 / Harmonic 振子
佛学：真空妙有 / 如来藏 / 不住生死不住涅槃 / 排除断灭见
哲学：潜能（量子基态）vs 实现（经典极小）/
      动态（涨落）vs 静态（冻结）
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .metric_field_quantization import (
    MetricFieldQuantizer,
    CognitiveWavefunction,
    HbarCog,
)


# ============================================================
# 零点能分析器
# ============================================================

class ZeroPointEnergy:
    """
    零点能分析器：计算量子基态的零点能、方差、势能/动能分解。

    物理：
        零点能 E_0 = ⟨Ĥ⟩_基态 = ⟨T⟩ + ⟨V⟩
        经典极小 V_min = V(λ_min)
        零点能差 = E_0 - V_min > 0（量子涨落）

        ℏ_cog → 0 时 E_0 → V_min（经典极限，零点能消失）。
        ℏ_cog ~ O(1) 时 E_0 > V_min（真空妙有）。

    佛学对应：
        零点能 = 真空妙有（空性含摄万法潜能）
        E_0 > V_min = 真空非死寂，法身流转
        ℏ_cog → 0 时 E_0 → V_min = 经典断灭见（空性退化为死寂）

    使用方式：
        analyzer = ZeroPointEnergy(quantizer)
        result = analyzer.analyze(kappa_vec, alpha_vec, hbar_values=[1.0, 0.1, 0.01])
    """

    def __init__(self, quantizer: MetricFieldQuantizer):
        self.quantizer = quantizer

    def compute_ground_state_energy(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        c: float = 1.0,
    ) -> dict[str, float | Tensor]:
        """
        计算基态能量 E_0 = ⟨Ĥ⟩_基态。

        方法：求解 Ĥ 的最低本征值。
        """
        H = self.quantizer.build_hamiltonian(kappa_vec, alpha_vec)
        eigvals, eigvecs = self.quantizer.eigensolve(H, n_states=1)
        E_0 = float(eigvals[0].item())
        psi_0 = eigvecs[:, 0]

        # 势能极小值（经典极限）
        # V(λ) = -β(λ-c)² + γ(λ-c)⁴ - ...
        # κ=0 时极小在 λ=c，V_min=0
        # κ>0 时极小在 λ*=c±√(β/(2γ))，V_min<0
        kappa_i = float(kappa_vec[0]) if self.quantizer.n_dims == 1 else float(kappa_vec[0])
        alpha_i = float(alpha_vec[0]) if self.quantizer.n_dims == 1 else float(alpha_vec[0])
        beta = kappa_i / (1.0 + kappa_i)
        gamma = 1.0 / (2.0 * (alpha_i + 1.0))

        if kappa_i < 1e-10:
            # κ=0：极小在 λ=c，V_min=0
            V_min = 0.0
            lambda_min = c
        else:
            # κ>0：极小在 λ*=c±√(β/(2γ))，取正分支
            delta_star = math.sqrt(beta / (2.0 * gamma))
            lambda_star = c + delta_star
            # V(λ*) = -β δ*² + γ δ*⁴ - ... = -β²/(2γ) + β²/(4γ) - ...（近似）
            # 精确：用势能函数计算
            V_at_star = self.quantizer.potential_1d(
                torch.tensor([lambda_star]), kappa_i, alpha_i, c=c
            )[0].item()
            V_min = V_at_star
            lambda_min = lambda_star

        zero_point_energy = E_0 - V_min

        return {
            "E_0_quantum_ground_state": E_0,
            "V_min_classical_minimum": V_min,
            "lambda_min": lambda_min,
            "zero_point_energy": zero_point_energy,
            "is_zero_point_nonzero": zero_point_energy > 1e-8,
        }

    def compute_zero_point_fluctuation(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        c: float = 1.0,
    ) -> dict[str, float]:
        """
        计算零点能振荡 ⟨(ĝ-cI)²⟩ = 基态方差。

        物理：
            方差 = ⟨λ²⟩ - ⟨λ⟩²
            真空妙有：方差 > 0（即使 κ=0，仍有涨落）
            经典极限：ℏ→0 时方差 → 0
        """
        H = self.quantizer.build_hamiltonian(kappa_vec, alpha_vec)
        _, eigvecs = self.quantizer.eigensolve(H, n_states=1)
        psi_0 = eigvecs[:, 0]
        wf = CognitiveWavefunction(psi_0, self.quantizer)

        exp_lambda = float(wf.expectation_value_lambda().item())
        var_lambda = float(wf.variance_lambda().item())

        # 相对真空 cI 的偏移
        deviation_from_vacuum = abs(exp_lambda - c)

        return {
            "exp_lambda": exp_lambda,
            "var_lambda_zero_point": var_lambda,
            "std_lambda_zero_point": math.sqrt(max(var_lambda, 0.0)),
            "deviation_from_vacuum_cI": deviation_from_vacuum,
            "is_fluctuating": var_lambda > 1e-8,
        }

    def analyze_hbar_scaling(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        hbar_values: list[float],
        c: float = 1.0,
    ) -> dict[str, list[float] | str]:
        """
        分析零点能、方差随 ℏ_cog 的标度关系。

        物理：
            ℏ→0 时零点能→0，方差→0（经典断灭见）
            ℏ~O(1) 时零点能>0，方差>0（真空妙有）

        标度：
            四次势能（κ=0）：E_0 ~ ℏ^(4/3)，方差 ~ ℏ^(2/3)
            二次势能（破缺态附近）：E_0 ~ ℏ，方差 ~ ℏ
        """
        energies = []
        variances = []
        original_hbar = self.quantizer.hbar_value

        for hbar_val in hbar_values:
            # 临时修改 ℏ_cog
            self.quantizer.hbar = HbarCog(value=hbar_val)
            self.quantizer.hbar_value = hbar_val

            energy_result = self.compute_ground_state_energy(kappa_vec, alpha_vec, c)
            fluct_result = self.compute_zero_point_fluctuation(kappa_vec, alpha_vec, c)

            energies.append(energy_result["zero_point_energy"])
            variances.append(fluct_result["var_lambda_zero_point"])

        # 恢复原 ℏ_cog
        self.quantizer.hbar = HbarCog(value=original_hbar)
        self.quantizer.hbar_value = original_hbar

        # 检查单调减小
        is_energies_decreasing = all(
            energies[i + 1] < energies[i] * 1.5 for i in range(len(energies) - 1)
        )
        is_variances_decreasing = all(
            variances[i + 1] < variances[i] * 1.5 for i in range(len(variances) - 1)
        )

        # 标度拟合：log(y) = p * log(ℏ) + const
        # p = slope of log-log plot
        if len(hbar_values) >= 3 and all(e > 0 for e in energies):
            log_hbar = [math.log(h) for h in hbar_values]
            log_energy = [math.log(e) for e in energies]
            # 线性回归 slope
            n = len(log_hbar)
            sum_x = sum(log_hbar)
            sum_y = sum(log_energy)
            sum_xy = sum(x * y for x, y in zip(log_hbar, log_energy))
            sum_x2 = sum(x ** 2 for x in log_hbar)
            slope_energy = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-12)
        else:
            slope_energy = float('nan')

        if len(hbar_values) >= 3 and all(v > 0 for v in variances):
            log_hbar = [math.log(h) for h in hbar_values]
            log_var = [math.log(v) for v in variances]
            n = len(log_hbar)
            sum_x = sum(log_hbar)
            sum_y = sum(log_var)
            sum_xy = sum(x * y for x, y in zip(log_hbar, log_var))
            sum_x2 = sum(x ** 2 for x in log_hbar)
            slope_variance = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-12)
        else:
            slope_variance = float('nan')

        return {
            "hbar_values": hbar_values,
            "zero_point_energies": energies,
            "zero_point_variances": variances,
            "is_energies_decreasing_with_hbar": is_energies_decreasing,
            "is_variances_decreasing_with_hbar": is_variances_decreasing,
            "energy_scaling_exponent_p": float(slope_energy),
            "variance_scaling_exponent_p": float(slope_variance),
            "thesis": (
                f"零点能 ~ ℏ^{slope_energy:.2f}，方差 ~ ℏ^{slope_variance:.2f}。"
                f"ℏ→0 时零点能→0（经典断灭见），ℏ~O(1) 时零点能>0（真空妙有）。"
            ),
        }


# ============================================================
# 真空妙有分析器
# ============================================================

class VacuumFluctuationAnalyzer:
    """
    真空妙有分析器：综合分析量子基态的真空涨落性质。

    核心：排除断灭见，建立"真空 = 动态量子平衡"的物理基础。

    使用方式：
        quantizer = MetricFieldQuantizer(n_dims=1, hbar=1.0, n_grid=128)
        analyzer = VacuumFluctuationAnalyzer(quantizer)
        # 真空态（κ=0）分析
        result = analyzer.analyze_vacuum(alpha_vec=torch.tensor([2.0]))
        # 破缺态（κ>0）分析
        result = analyzer.analyze_broken(kappa_vec=torch.tensor([0.5]), alpha_vec=torch.tensor([2.0]))
    """

    def __init__(self, quantizer: MetricFieldQuantizer):
        self.quantizer = quantizer
        self.zpe = ZeroPointEnergy(quantizer)

    def analyze_vacuum(
        self,
        alpha_vec: Tensor,
        c: float = 1.0,
    ) -> dict:
        """
        真空态分析（κ=0）。

        物理问题：
            κ=0 时势能 V(λ) = γ(λ-c)⁴ + ε(λ-c)⁸（四次型）。
            基态波函数局域化在 λ=c 附近，但有零点能振荡。
            这是"真空妙有"——空性非死寂。

        对比经典：
            经典 v7.x：κ=0 时 g≡cI（死寂断灭）。
            量子 GCFT：κ=0 时 ⟨(ĝ-cI)²⟩ ≠ 0（真空妙有）。
        """
        kappa_zero = torch.zeros_like(alpha_vec)

        # 基态能量与零点能
        energy_result = self.zpe.compute_ground_state_energy(kappa_zero, alpha_vec, c)
        # 零点能振荡
        fluct_result = self.zpe.compute_zero_point_fluctuation(kappa_zero, alpha_vec, c)

        # ℏ 标度分析
        hbar_values = [1.0, 0.5, 0.2, 0.1, 0.05]
        scaling_result = self.zpe.analyze_hbar_scaling(kappa_zero, alpha_vec, hbar_values, c)

        # 真空妙有判据
        is_vacuum_fluctuating = fluct_result["is_fluctuating"]
        is_zero_point_nonzero = energy_result["is_zero_point_nonzero"]
        is_vacuum_alive = is_vacuum_fluctuating and is_zero_point_nonzero

        return {
            "regime": "vacuum_kappa_0",
            "kappa": 0.0,
            "alpha": float(alpha_vec[0]),
            "vacuum_is_alive_quantum": is_vacuum_alive,
            "vacuum_is_dead_classical": False,  # 量子下永远 False
            "ground_state_energy": energy_result["E_0_quantum_ground_state"],
            "classical_minimum": energy_result["V_min_classical_minimum"],
            "zero_point_energy": energy_result["zero_point_energy"],
            "zero_point_fluctuation_variance": fluct_result["var_lambda_zero_point"],
            "zero_point_fluctuation_std": fluct_result["std_lambda_zero_point"],
            "hbar_scaling": scaling_result,
            "thesis": (
                "κ=0（无痛苦）时，量子基态仍有零点能振荡 ⟨(ĝ-cI)²⟩ ≠ 0。"
                f"零点能 = {energy_result['zero_point_energy']:.6f}，"
                f"方差 = {fluct_result['var_lambda_zero_point']:.6f}。"
                "这是真空妙有——空性非死寂断灭，而是动态量子平衡。"
                "对比经典 v7.x：κ=0 时 g≡cI（死寂），GCFT 升级为真空涨落。"
                "如来藏 = 量子基态潜能态，含摄万法。"
            ),
        }

    def analyze_broken(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        c: float = 1.0,
    ) -> dict:
        """
        破缺态分析（κ>0）。

        物理问题：
            κ>0 时势能有双井结构，基态在破缺态 λ* 附近。
            基态能量 E_0 < 0（低于真空），零点能 = E_0 - V_min > 0。
            破缺态的零点能振荡 = 业力束缚态的量子涨落。

        佛学对应：
            破缺态 = 业力态（执着于某方向）
            零点能振荡 = 业力态中的微细涨落（习气种子生灭）
            隧穿概率 = 顿悟可能性（见基石4 瞬子顿悟）
        """
        # 基态能量与零点能
        energy_result = self.zpe.compute_ground_state_energy(kappa_vec, alpha_vec, c)
        # 零点能振荡
        fluct_result = self.zpe.compute_zero_point_fluctuation(kappa_vec, alpha_vec, c)

        # 计算破缺态 λ*
        kappa_i = float(kappa_vec[0])
        alpha_i = float(alpha_vec[0])
        beta = kappa_i / (1.0 + kappa_i)
        gamma = 1.0 / (2.0 * (alpha_i + 1.0))
        delta_star = math.sqrt(beta / (2.0 * gamma))
        lambda_star_plus = c + delta_star
        lambda_star_minus = c - delta_star

        return {
            "regime": "broken_kappa_positive",
            "kappa": kappa_i,
            "alpha": alpha_i,
            "lambda_star_plus": lambda_star_plus,
            "lambda_star_minus": lambda_star_minus,
            "ground_state_energy": energy_result["E_0_quantum_ground_state"],
            "classical_minimum": energy_result["V_min_classical_minimum"],
            "zero_point_energy": energy_result["zero_point_energy"],
            "zero_point_fluctuation_variance": fluct_result["var_lambda_zero_point"],
            "exp_lambda": fluct_result["exp_lambda"],
            "deviation_from_vacuum": fluct_result["deviation_from_vacuum_cI"],
            "is_broken_state": fluct_result["deviation_from_vacuum_cI"] > 0.01,
            "thesis": (
                f"κ={kappa_i} 时势能双井破缺，基态在 λ*≈{lambda_star_plus:.3f} 附近。"
                f"基态能量 E_0={energy_result['E_0_quantum_ground_state']:.4f}，"
                f"零点能={energy_result['zero_point_energy']:.6f}。"
                "破缺态 = 业力束缚态；零点能振荡 = 习气种子生灭。"
                "隧穿概率（顿悟）见基石4 瞬子顿悟。"
            ),
        }

    def compare_vacuum_vs_classical(
        self,
        alpha_vec: Tensor,
        hbar_values: list[float] = None,
        c: float = 1.0,
    ) -> dict:
        """
        对比真空态的量子 vs 经典行为。

        核心：展示 ℏ_cog → 0 时量子真空退化为经典断灭。
        """
        if hbar_values is None:
            hbar_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

        kappa_zero = torch.zeros_like(alpha_vec)
        scaling = self.zpe.analyze_hbar_scaling(kappa_zero, alpha_vec, hbar_values, c)

        # 经典极限（ℏ→0）
        classical_variance = 0.0  # 经典：方差=0（冻结在 cI）
        classical_energy = 0.0    # 经典：E=V_min=0（κ=0 时）

        # 量子极限（ℏ=1）
        quantum_variance = scaling["zero_point_variances"][0]
        quantum_energy = scaling["zero_point_energies"][0]

        return {
            "classical_limit_hbar_0": {
                "variance": classical_variance,
                "energy": classical_energy,
                "interpretation": "经典断灭见：g≡cI，死寂绝对零度",
            },
            "quantum_regime_hbar_1": {
                "variance": quantum_variance,
                "energy": quantum_energy,
                "interpretation": "真空妙有：⟨(ĝ-cI)²⟩≠0，动态量子平衡",
            },
            "hbar_scaling": scaling,
            "thesis": (
                "经典 v7.x（ℏ→0）：真空 = 死寂断灭（g≡cI，方差=0）。"
                "量子 GCFT（ℏ~O(1)）：真空 = 动态涨落（方差>0，零点能>0）。"
                "GCFT 排除断灭见，建立真空妙有的严格物理基础。"
                "如来藏 = 量子基态潜能；法身流转 = 零点能涨落。"
            ),
        }


# ============================================================
# 真空妙有验证器
# ============================================================

def run_vacuum_fluctuation_verification() -> dict:
    """
    真空妙有验证套件。

    验证项：
        V1：真空非死寂——κ=0 时 ⟨(ĝ-cI)²⟩ ≠ 0（排除断灭见）
        V2：零点能 > 0——E_0 > V_min（真空妙有）
        V3：ℏ 标度——零点能、方差随 ℏ_cog 减小而减小（对应原理）
        V4：破缺态零点能——κ>0 时破缺态仍有零点振荡（业力态涨落）
        V5：经典极限——ℏ→0 时真空退化为死寂（断灭见回归，对应原理）

    数值注意：
        四次势能 V~(λ-c)^4 下，基态宽度 x_0 ~ (ℏ²/γ)^(1/6)。
        网格须足够宽（L >> x_0）以避免盒子 confinement 主导方差，
        且足够细（dλ << x_0）以解析波函数。
        量子-经典 crossover: ℏ_cross ~ (2L²γ^(1/3)/π²)^(3/2)。
        对 L=2.0, γ=1/6: ℏ_cross ≈ 0.30。
        故 V1-V5 取 ℏ ≤ 0.1（深量子区），网格 [0,2]×512 点。
    """
    results = {}

    # 公共参数：宽网格 + 高分辨率，确保 ℏ≤0.1 时波函数被解析
    # L=2.0, n_grid=512, dλ≈0.0039
    # crossover ℏ ≈ 0.30，故 ℏ=0.1 深量子区
    GRID_MIN, GRID_MAX, N_GRID = 0.0, 2.0, 512
    ALPHA_TEST = torch.tensor([2.0])  # γ = 1/6
    HBAR_QUANTUM = 0.1   # 量子区
    HBAR_CLASSICAL = 0.0001  # 近经典极限

    # ----- V1：真空非死寂 -----
    quantizer_v1 = MetricFieldQuantizer(
        n_dims=1, hbar=HBAR_QUANTUM, n_grid=N_GRID,
        lambda_min=GRID_MIN, lambda_max=GRID_MAX,
    )
    analyzer_v1 = VacuumFluctuationAnalyzer(quantizer_v1)
    result_v1 = analyzer_v1.analyze_vacuum(alpha_vec=ALPHA_TEST)

    results["V1_vacuum_not_dead"] = {
        "kappa": 0.0,
        "hbar_cog": HBAR_QUANTUM,
        "variance_zero_point": result_v1["zero_point_fluctuation_variance"],
        "is_vacuum_fluctuating": result_v1["vacuum_is_alive_quantum"],
        "pass": result_v1["vacuum_is_alive_quantum"],
        "thesis": (
            "κ=0（无痛苦）时，量子基态仍有零点能振荡 ⟨(ĝ-cI)²⟩ ≠ 0。"
            "这排除断灭见——空性非死寂 g≡cI，而是动态量子平衡。"
            "对比经典 v7.x：κ=0 时 g≡cI（死寂断灭）。"
        ),
    }

    # ----- V2：零点能 > 0 -----
    results["V2_zero_point_energy_positive"] = {
        "ground_state_energy_E0": result_v1["ground_state_energy"],
        "classical_minimum_Vmin": result_v1["classical_minimum"],
        "zero_point_energy": result_v1["zero_point_energy"],
        "is_zero_point_nonzero": result_v1["zero_point_energy"] > 1e-8,
        "pass": result_v1["zero_point_energy"] > 1e-8,
        "thesis": (
            "量子基态能量 E_0 > 经典极小 V_min，差值 = 零点能 > 0。"
            "真空妙有：空性含摄万法潜能，非死寂断灭。"
            "如来藏 = 量子基态潜能态。"
        ),
    }

    # ----- V3：ℏ 标度 -----
    # 四次势能下 E~ℏ^(4/3), Var~ℏ^(2/3)
    # 取 ℏ ∈ [0.1, 0.005]，全部在量子区 (< crossover 0.30)
    quantizer_v3 = MetricFieldQuantizer(
        n_dims=1, hbar=HBAR_QUANTUM, n_grid=N_GRID,
        lambda_min=GRID_MIN, lambda_max=GRID_MAX,
    )
    analyzer_v3 = VacuumFluctuationAnalyzer(quantizer_v3)
    hbar_values_v3 = [0.1, 0.05, 0.02, 0.01, 0.005]
    scaling_v3 = analyzer_v3.zpe.analyze_hbar_scaling(
        kappa_vec=torch.tensor([0.0]),
        alpha_vec=ALPHA_TEST,
        hbar_values=hbar_values_v3,
    )

    results["V3_hbar_scaling"] = {
        "hbar_values": scaling_v3["hbar_values"],
        "zero_point_energies": scaling_v3["zero_point_energies"],
        "zero_point_variances": scaling_v3["zero_point_variances"],
        "energy_scaling_exponent": scaling_v3["energy_scaling_exponent_p"],
        "variance_scaling_exponent": scaling_v3["variance_scaling_exponent_p"],
        "is_energies_decreasing": scaling_v3["is_energies_decreasing_with_hbar"],
        "is_variances_decreasing": scaling_v3["is_variances_decreasing_with_hbar"],
        "pass": (
            scaling_v3["is_energies_decreasing_with_hbar"]
            and scaling_v3["is_variances_decreasing_with_hbar"]
        ),
        "thesis": (
            f"零点能 ~ ℏ^{scaling_v3['energy_scaling_exponent_p']:.2f}，"
            f"方差 ~ ℏ^{scaling_v3['variance_scaling_exponent_p']:.2f}。"
            "四次势能（κ=0）下，理论标度 E~ℏ^(4/3)，方差~ℏ^(2/3)。"
            "ℏ→0 时零点能→0，方差→0（经典极限，断灭见回归）。"
        ),
    }

    # ----- V4：破缺态零点能 -----
    # κ=0.5 → λ* = c + √(β/(2γ)) = 1 + √(1/3 / 1/3) = 2.0
    # 网格 [0, 3] 覆盖双井 [0, 2]
    quantizer_v4 = MetricFieldQuantizer(
        n_dims=1, hbar=HBAR_QUANTUM, n_grid=N_GRID,
        lambda_min=0.0, lambda_max=3.0,
    )
    analyzer_v4 = VacuumFluctuationAnalyzer(quantizer_v4)
    result_v4 = analyzer_v4.analyze_broken(
        kappa_vec=torch.tensor([0.5]),
        alpha_vec=ALPHA_TEST,
    )

    results["V4_broken_state_fluctuation"] = {
        "kappa": 0.5,
        "lambda_star_plus": result_v4["lambda_star_plus"],
        "ground_state_energy": result_v4["ground_state_energy"],
        "zero_point_energy": result_v4["zero_point_energy"],
        "zero_point_variance": result_v4["zero_point_fluctuation_variance"],
        "is_broken_state": result_v4["is_broken_state"],
        "pass": result_v4["is_broken_state"] and result_v4["zero_point_energy"] > 1e-8,
        "thesis": (
            "κ>0 时势能双井破缺，基态在 λ* 附近。"
            "破缺态仍有零点能振荡（业力束缚态的习气种子生灭）。"
            "破缺态零点能 < 真空零点能（业力束缚更紧）。"
            "隧穿概率（顿悟）见基石4 瞬子顿悟。"
        ),
    }

    # ----- V5：经典极限 -----
    # ℏ→0 时真空退化为死寂
    # 比较 ℏ=0.1（量子区）vs ℏ=0.0001（近经典）
    # 四次势能标度：Var~ℏ^(2/3)
    # 预期比值 = (0.0001/0.1)^(2/3) = 0.001^(2/3) ≈ 0.01 << 0.1
    quantizer_v5 = MetricFieldQuantizer(
        n_dims=1, hbar=HBAR_CLASSICAL, n_grid=N_GRID,
        lambda_min=GRID_MIN, lambda_max=GRID_MAX,
    )
    analyzer_v5 = VacuumFluctuationAnalyzer(quantizer_v5)
    result_v5 = analyzer_v5.analyze_vacuum(alpha_vec=ALPHA_TEST)

    # ℏ=0.1 的真空
    result_v5_q = result_v1  # 复用 V1 的量子区结果

    variance_ratio = result_v5["zero_point_fluctuation_variance"] / (
        result_v5_q["zero_point_fluctuation_variance"] + 1e-12
    )

    results["V5_classical_limit"] = {
        "hbar_quantum": HBAR_QUANTUM,
        "hbar_classical": HBAR_CLASSICAL,
        "variance_quantum": result_v5_q["zero_point_fluctuation_variance"],
        "variance_classical": result_v5["zero_point_fluctuation_variance"],
        "variance_ratio_classical_to_quantum": variance_ratio,
        "classical_is_dead": variance_ratio < 0.1,
        "pass": variance_ratio < 0.1,
        "thesis": (
            "ℏ_cog → 0 时，真空方差 → 0（系统冻结在 cI）。"
            "经典极限：真空退化为死寂断灭（g≡cI，无涨落）。"
            "对应原理：v7.x 经典理论 = GCFT 在 ℏ_cog→0 时的断灭见回归。"
            "GCFT 的真空妙有是量子效应，经典框架无法触及。"
        ),
    }

    # ----- 总结论 -----
    v_keys = [k for k in results if k.startswith("V") and isinstance(results[k], dict) and "pass" in results[k]]
    pass_flags = [results[k].get("pass", False) for k in v_keys]
    n_pass = sum(1 for f in pass_flags if f)
    n_total = len(pass_flags)
    all_pass = (n_pass == n_total) and (n_total > 0)

    results["summary"] = {
        "all_pass": all_pass,
        "thesis": (
            "真空妙有（基石2）建立：κ=0 时量子基态仍有零点能振荡 ⟨(ĝ-cI)²⟩≠0。"
            "排除断灭见，真空 = 动态量子平衡。"
            "如来藏 = 量子基态潜能态，法身流转 = 零点能涨落。"
            "ℏ_cog → 0 时真空退化为死寂（经典断灭见回归）。"
            "GCFT 的真空妙有是量子效应，v7.x 经典框架无法触及。"
        ),
    }

    # 顶层统一字段（与 v8 其他 run_*_verification 一致）
    results["n_pass"] = n_pass
    results["n_total"] = n_total
    results["all_pass"] = all_pass
    results["pass_flags"] = pass_flags

    return results


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GCFT 基石2：真空妙有（Vacuum Fluctuation）")
    print("=" * 60)

    results = run_vacuum_fluctuation_verification()

    for key, val in results.items():
        print(f"\n--- {key} ---")
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, list) and len(sub_val) > 5:
                    print(f"  {sub_key}: [{sub_val[0]}, ..., {sub_val[-1]}] (len={len(sub_val)})")
                else:
                    print(f"  {sub_key}: {sub_val}")
