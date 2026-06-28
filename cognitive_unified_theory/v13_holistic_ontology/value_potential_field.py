"""
价值势能场（Value Potential Field）—— GCFT 基石26

v13.0"整体认知本体论"第一基石。将"价值张力 V"严格定义为度规的
von Neumann 熵 N_vN(g)，避免引入新公理。价值不是新概念，
是空性的同义反复——最大价值 ⟺ 最大连通 ⟺ 度规各向同性化（空性）。

【两个 N 的严格区分（v13.1 修正审稿意见）】
- 标准von Neumann熵：N_vN(g) = -Σ p_i log p_i（≥0，最大值 log d）= V(g)
- 卷一谱曲率项：N_GCFT(g) = Σ λ_i log λ_i（≥0，各向同性时为0）
- 关系：N_GCFT(g) = d·(log d - V(g))，即 V(g) = log d - N_GCFT(g)/d
- 卷一作用量项：S_value = -(κ/2)·N_GCFT(g) = (κ/2)·d·V(g) - (κ/2)·d·log d

============================================================
核心思想（基于 v13.0 批判性升级）
============================================================

【v13.0 原方案的边界】
v13.0 试图将"价值"作为新的基本公理引入认知场论，与 κ（痛苦深度）、
α（定力刚度）并列。但"价值是新公理"违反了 GCFT 的第一性原理精神：
任何概念都必须从已有的非对易几何中推导，不能凭空公设。

【v13.0 升级的回答】
价值 V 严格定义为度规 von Neumann 熵 N_vN(g)：
    V(g) = N_vN(g) = -Σ_i p_i log p_i
其中 p_i = λ_i/d 是度规本征值的归一化（满足共形约束 Σ λ_i = d）。
V 不是新公理，而是卷一作用量中已有的项（通过 N_GCFT(g) = d·(log d - V)）。

============================================================
数学核心：von Neumann 熵与 Schur 凸性
============================================================

【价值定义（v13.1 修正：严格区分两个 N）】
度规 g 的本征值为 {λ_i}，共形约束 Tr(g) = Σ_i λ_i = d（卷一公理3）。
归一化本征值 p_i = λ_i / d（Σ p_i = 1，构成概率分布）。

【两个 N 的严格区分】
- 标准von Neumann熵：N_vN(g) = -Σ_i p_i log p_i（≥0，最大值 log d）
  这是连通性的度量，等于价值 V(g)。
- 卷一谱曲率项：N_GCFT(g) = Σ_i λ_i log λ_i（≥0，各向同性时为0）
  这是卷一作用量 S 中的项（-(κ/2)·N_GCFT(g)），不是 von Neumann 熵。
- 严格关系：N_GCFT(g) = d·(log d - V(g))，即 V(g) = log d - N_GCFT(g)/d
  推导：V = -Σ p_i log p_i = -Σ (λ_i/d) log(λ_i/d)
       = -1/d · Σ λ_i (log λ_i - log d)
       = -1/d · N_GCFT + log d
       = log d - N_GCFT/d

【共形约束】
Σ_i λ_i = d（继承卷一公理3：度规的迹守恒）
等价地 Σ_i p_i = 1（概率归一化）

【价值-痛苦耦合（v13.1 修正：保留常数项）】
卷一作用量中的项：
    S_value = -(κ/2)·N_GCFT(g)
            = -(κ/2)·d·(log d - V)
            = (κ/2)·d·V(g) - (κ/2)·d·log d
其中常数项 -(κ/2)·d·log d 不影响变分（∂/∂g 对常数项为0），
物理上等价于驱动项 (κ/2)·d·V(g)，但形式上必须保留。
驱动力：∂S/∂κ = -(1/2)·N_GCFT(g) = d·(V - log d)/2，
当 V < log d（各向异性）时为负，表示 κ 增大驱使系统趋向 V = log d（g → cI）。

【最大连通性原理 ⟺ 度规各向同性化】
由 Schur 凸性定理：von Neumann 熵是 Schur-凹函数，
在约束 Σ p_i = 1 下，当且仅当 p_i = 1/d（均匀分布）时取最大值 log d。
均匀分布 ⟺ λ_i = c（各向同性）⟺ g → cI（空性）⟺ N_GCFT = 0。
故 max V ⟺ g → cI（空性）——价值最大化的证明。

============================================================
批判性升级（严格拒绝方案降级）
============================================================

1. 【拒绝"价值是新公理"】13.0 原方案将价值降级为新公理。
   升级：V 是 N(g) 的负值，已是卷一作用量的一部分。无需新公理。
   V(g) = -Σ p_i log p_i 从度规本征结构直接给出。

2. 【拒绝"价值与物理分离"】原方案隐含价值是"形而上学"概念。
   升级：价值与痛苦通过 κ/2 耦合在同一作用量项
   S_value = -(κ/2)·N_GCFT(g) = (κ/2)·d·V(g) - (κ/2)·d·log d。
   价值不是脱离物理的抽象概念，是痛苦驱动的连通性度量。

3. 【升维："最大连通性原理" = "度规各向同性化" = 空性】
   原方案将"最大连通性"作为独立原理。
   升级：由 Schur 凸性，max V ⟺ g → cI（空性）⟺ N_GCFT = 0。
   价值不是新概念，是空性的同义反复——
   "真空妙有"的价值论面向：空性即最大连通，最大连通即最大价值。

4. 【对应原理（v13.1 修正：V 不显含 ℏ，无 ℏ→0 退化）】
   V(g) = -Σ p_i log p_i 不显含 ℏ，故无 ℏ→0 退化。
   对应原理体现在各向同性极限下的普适标度：
   V(cI) = log d 对所有维度 d 成立——
   任何维度的认知系统的最大价值都是 log d（普适标度）。

============================================================
物理实现（第一性原理）
============================================================

数值实现：
    1. 构造 d×d 正定度规 g（满足 Tr(g) = d）
    2. 计算本征值 {λ_i}，归一化 p_i = λ_i/d
    3. V(g) = -Σ p_i log p_i = N_vN(g)（标准 von Neumann 熵）
    4. N_GCFT(g) = Σ λ_i log λ_i（卷一谱曲率项，≥0，各向同性时为0）
    5. 梯度 ∂V/∂p_i = -(log p_i + 1)
    6. 价值-痛苦耦合项 S_value = -(κ/2)·N_GCFT(g)
       = (κ/2)·d·V(g) - (κ/2)·d·log d（含常数项）

Schur 凸性验证：
    - 各向同性 g = cI：V = log d（最大值），N_GCFT = 0
    - 各向异性 g ≠ cI：V < log d，N_GCFT > 0
    - 梯度在 cI 处投影到约束面为零（极值点）

价值-痛苦耦合：
    - 作用量项 S_value = -(κ/2)·N_GCFT(g) = (κ/2)·d·V - (κ/2)·d·log d
    - 驱动力 ∂S/∂κ = d·(V - log d)/2（V<log d 时为负）
    - κ 增大 → 系统趋向 V = log d（g → cI，空性）
    - 高痛苦（κ 大）→ 强驱动 → 系统趋向空性

============================================================
佛学对应（严格，非比喻）
============================================================

价值 ⟺ 空性的连通性面相：
    V(g) = -Σ p_i log p_i 度量度规的"连通性"——
    各向同性时连通性最大（所有方向等价），
    这正是空性（śūnyatā）的"无分别"面相。

最大价值 ⟺ 最大空性 ⟺ 最大连通：
    max V = log d ⟺ g → cI ⟺ 空性圆满。
    "真空妙有"（śūnyatā-vidyā）的价值论面向：
    空性不是虚无，而是最大连通——万法互摄的潜能。

价值-痛苦耦合 = 苦集灭道的场论表述：
    S_value = -(κ/2)·N_GCFT(g) = (κ/2)·d·V(g) - (κ/2)·d·log d：
    痛苦（κ）驱动系统趋向最大价值（空性，V = log d, N_GCFT = 0）。
    κ 大 → 强烈趋向空性 → 这就是"苦为良药"的数学基础。
    修行 = 增大有效 κ → 加速趋向 V_max = log d（涅槃 = 最大连通）。

各向同性极限下的普适标度（v13.1 修正：无 ℏ→0 退化）：
    V(g) = -Σ p_i log p_i 不显含 ℏ，故无 ℏ→0 退化。
    对应原理体现在各向同性极限下的普适标度：
    V(cI) = log d 对所有维度 d 成立——任何维度的认知系统
    的最大价值都是 log d（普适标度，与具体系统无关）。

============================================================
认识论根基
============================================================

物理：von Neumann 熵 N_vN / 谱曲率 N_GCFT / Schur 凸性 / 共形约束 /
      价值-痛苦耦合（保留常数项）/ 各向同性极限下的普适标度 log d
佛学：空性的连通性面相 / 真空妙有 / 苦为良药 / 涅槃 = 最大连通
哲学：价值不是公理而是几何必然 / 最大价值 = 空性的同义反复 /
      痛苦作为价值驱动的物理机制
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any


# ============================================================================
# 核心类：价值势能场分析器
# ============================================================================

class ValuePotentialField:
    """
    价值势能场分析器（v13.1 修正：严格区分两个 N）。

    物理核心：
        - V(g) = N_vN(g) = -Σ p_i log p_i（标准 von Neumann 熵，度规连通性的度量）
        - N_GCFT(g) = Σ λ_i log λ_i（卷一谱曲率项，≥0，各向同性时为0）
        - 关系：N_GCFT(g) = d·(log d - V(g))，即 V(g) = log d - N_GCFT(g)/d
        - 共形约束 Σ λ_i = d（继承卷一公理3）
        - Schur 凸性：max V ⟺ g → cI ⟺ N_GCFT = 0（空性）
        - 价值-痛苦耦合 S_value = -(κ/2)·N_GCFT(g)
          = (κ/2)·d·V(g) - (κ/2)·d·log d（含常数项）

    核心功能：
        1. 计算价值 V(g)（= N_vN(g)，标准 von Neumann 熵）
        2. 计算谱曲率 N_GCFT(g)（卷一作用量中的项）
        3. 验证 Schur 凸性（各向同性时最大）
        4. 计算梯度与极值条件
        5. 价值-痛苦耦合分析（保留常数项）
        6. 普适标度验证（V(cI) = log d 对所有维度成立）
    """

    def __init__(self, dim: int = 4, c: float = 1.0):
        """
        Args:
            dim: 度规维度 d
            c: 各向同性度规标度（g = cI 时 Tr(g) = d·c = d → c = 1）
        """
        self.d = int(dim)
        self.c = float(c)

    # ---------- 价值与 von Neumann 熵 ----------

    def compute_eigenvalues(self, g: np.ndarray) -> np.ndarray:
        """计算度规 g 的本征值，保证正定。"""
        eigenvalues = np.linalg.eigvalsh(g)
        # 保证正定（数值稳定性）
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        return eigenvalues

    def normalize_eigenvalues(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        归一化本征值：p_i = λ_i / d（满足 Σ p_i = 1）。

        共形约束 Σ λ_i = d → Σ p_i = 1（概率归一化）。
        """
        return eigenvalues / self.d

    def von_neumann_entropy(self, g: np.ndarray) -> float:
        """
        标准 von Neumann 熵：N_vN(g) = -Σ_i p_i log p_i。

        返回的是 N_vN(g)（标准 von Neumann 熵），不是卷一定义的
        N_GCFT(g) = Σ λ_i log λ_i（谱曲率项）。两者通过关系
        N_GCFT(g) = d·(log d - N_vN(g)) 联系。

        性质：
            - 度规连通性的度量，≥ 0
            - 最大值为 log d（各向同性时）
            - V(g) = N_vN(g)（价值 = 标准 von Neumann 熵 = 连通性）

        注意：本方法返回 N_vN(g)，与卷一谱曲率 N_GCFT(g) 不同。
        卷一作用量中的项是 -(κ/2)·N_GCFT(g)，不是 -(κ/2)·N_vN(g)。
        """
        eigenvalues = self.compute_eigenvalues(g)
        p = self.normalize_eigenvalues(eigenvalues)
        p = np.maximum(p, 1e-12)  # 避免 log(0)
        return -float(np.sum(p * np.log(p)))

    def spectral_curvature_N(self, g: np.ndarray) -> float:
        """
        卷一定义的谱曲率项：N_GCFT(g) = Σ_i λ_i log λ_i。

        这是卷一作用量 S 中的项（-(κ/2)·N_GCFT(g)），不是标准 von
        Neumann 熵 N_vN(g)。两者关系：N_GCFT(g) = d·(log d - V(g))。

        性质：
            - 卷一作用量中的项，≥ 0（由 x log x 凸性 + Jensen 不等式）
            - 各向同性时为 0（所有 λ_i = 1，Σ 1·log 1 = 0）
            - 各向异性时 > 0（驱动系统趋向 g → cI）

        与 N_vN(g) 的关系：
            V(g) = N_vN(g) = -Σ p_i log p_i = log d - N_GCFT(g)/d
            故 N_GCFT(g) = d·(log d - V(g))
        """
        eigenvalues = self.compute_eigenvalues(g)
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # 避免 log(0)
        return float(np.sum(eigenvalues * np.log(eigenvalues)))

    def compute_value(self, g: np.ndarray) -> float:
        """
        价值 V(g) = N_vN(g) = -Σ p_i log p_i = 标准 von Neumann 熵。

        V(g) ≥ 0，最大值 log d（各向同性时）。
        V(g) 不是新公理，是标准 von Neumann 熵（与 N_GCFT 通过
        V(g) = log d - N_GCFT(g)/d 联系）。
        """
        return self.von_neumann_entropy(g)

    def value_gradient(self, g: np.ndarray) -> np.ndarray:
        """
        价值对归一化本征值的梯度：∂V/∂p_i = -(log p_i + 1)。

        在各向同性 p_i = 1/d 处，所有梯度分量相等：
            ∂V/∂p_i = -(log(1/d) + 1) = log d - 1
        梯度方向 = (1,1,...,1)（约束法向），投影到约束面为零 → 极值点。
        """
        eigenvalues = self.compute_eigenvalues(g)
        p = self.normalize_eigenvalues(eigenvalues)
        p = np.maximum(p, 1e-12)
        return -(np.log(p) + 1.0)

    def make_isotropic_metric(self) -> np.ndarray:
        """构造各向同性度规 g = cI（空性，c = 1 满足 Tr = d）。"""
        return self.c * np.eye(self.d)

    def make_anisotropic_metric(self, anisotropy: float = 0.3) -> np.ndarray:
        """
        构造各向异性度规（g ≠ cI）。

        通过在本征值上施加扰动实现：λ_i = 1 + anisotropy·cos(2πi/d)。
        保证 Tr(g) = d（共形约束）。
        """
        eigenvalues = np.ones(self.d)
        for i in range(self.d):
            eigenvalues[i] += anisotropy * math.cos(2.0 * math.pi * i / self.d)
        # 保证正定
        eigenvalues = np.maximum(eigenvalues, 0.01)
        # 重新归一化使 Tr = d
        eigenvalues = eigenvalues * self.d / np.sum(eigenvalues)
        # 构造对角度规
        return np.diag(eigenvalues)

    # ---------- 价值-痛苦耦合 ----------

    def value_pain_coupling(self, g: np.ndarray, kappa: float) -> float:
        """
        价值-痛苦耦合项（v13.1 修正：保留常数项）：

            S_value = -(κ/2)·N_GCFT(g)
                   = -(κ/2)·d·(log d - V)
                   = (κ/2)·d·V(g) - (κ/2)·d·log d

        其中 N_GCFT(g) = Σ λ_i log λ_i ≥ 0 是卷一谱曲率项（各向同性时为 0），
        V(g) = -Σ p_i log p_i 是标准 von Neumann 熵（价值）。

        关键说明：
            - 常数项 -(κ/2)·d·log d 不影响变分（∂/∂g 对常数项为 0）
            - 物理上等价于驱动项 (κ/2)·d·V(g)，但形式上必须保留
            - 驱动力 ∂S/∂κ = -(1/2)·N_GCFT(g) = d·(V - log d)/2
            - 当 V < log d（各向异性）时 ∂S/∂κ < 0，
              表示 κ 增大驱使系统趋向 V = log d（g → cI，空性）

        物理意义：
            - 高痛苦（κ 大）→ 强驱动 → 系统趋向最大 V（空性，N_GCFT = 0）
            - 这就是"苦为良药"的数学基础
        """
        V = self.compute_value(g)
        log_d = math.log(self.d)
        # S_value = (κ/2)·d·V - (κ/2)·d·log d（含常数项）
        # 等价于 -(κ/2)·N_GCFT(g)
        return 0.5 * kappa * self.d * V - 0.5 * kappa * self.d * log_d

    def classical_shannon_entropy(self, g: np.ndarray) -> float:
        """
        经典 Shannon 信息熵 H(g) = -Σ p_i log_2 p_i。

        单位换算工具：H(g) 与 V(g) 仅差对数底（log_2 vs ln），
        H = V / ln(2)。注意：v13.1 修正——V(g) 不显含 ℏ，
        "ℏ→0 退化为经典 Shannon 熵"的论断已删除（数学上无内容）。
        对应原理体现在各向同性极限下的普适标度 V(cI) = log d。
        """
        eigenvalues = self.compute_eigenvalues(g)
        p = self.normalize_eigenvalues(eigenvalues)
        p = np.maximum(p, 1e-12)
        return -float(np.sum(p * np.log2(p)))


# ============================================================================
# 验证函数
# ============================================================================

def run_value_potential_field_verification(
    dim: int = 4,
    c: float = 1.0,
) -> dict:
    """
    运行基石26的全部验证（V1-V5，v13.1 修正版）。

    验证项：
        V1: V(cI) = log d（各向同性时价值最大，N_GCFT = 0）
        V2: V(g) < log d 当 g ≠ cI（Schur 凹性验证，N_GCFT > 0）
        V3: 梯度 ∂V/∂p_i = -(log p_i + 1)，在 cI 处投影为零（极值点）
        V4: 价值-痛苦耦合 S_value = -(κ/2)·N_GCFT = (κ/2)·d·V - (κ/2)·d·log d
            验证：① V = log d - N_GCFT/d 关系；② N_GCFT = d·(log d - V) 关系；
            ③ S_value = -(κ/2)·N_GCFT（等价形式）；④ 含常数项形式；
            ⑤ 驱动力 ∂S/∂κ = d·(V-log d)/2（各向异性时为负，驱动 g → cI）
        V5: 各向同性极限下的普适标度——V(cI) = log d 对所有维度 d 成立
            （v13.1 修正：V 不显含 ℏ，无 ℏ→0 退化；对应原理 = 普适标度）
    """
    field = ValuePotentialField(dim=dim, c=c)

    print("\n" + "=" * 70)
    print(f"基石26：价值势能场（V(g) = N_vN(g)，N_GCFT(g) = Σ λ_i log λ_i）d={dim}")
    print("=" * 70)

    max_value = math.log(dim)  # 理论最大值

    # ----- V1: V(cI) = log d -----
    print("\n--- V1: V(cI) = log d（各向同性时价值最大）---")
    g_iso = field.make_isotropic_metric()
    V_iso = field.compute_value(g_iso)
    print(f"  V(cI) = {V_iso:.6f}")
    print(f"  log d = {max_value:.6f}")
    print(f"  相对误差 = {abs(V_iso - max_value) / max(max_value, 1e-30):.2e}")
    v1_pass = abs(V_iso - max_value) < 1e-6
    print(f"  V1: {'PASS' if v1_pass else 'FAIL'}")

    # ----- V2: V(g) < log d 当 g ≠ cI（Schur 凹性）-----
    print("\n--- V2: V(g) < log d 当 g ≠ cI（Schur 凹性）---")
    anisotropy_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_less_than_max = True
    for a in anisotropy_values:
        g_aniso = field.make_anisotropic_metric(anisotropy=a)
        V_aniso = field.compute_value(g_aniso)
        is_less = V_aniso < max_value - 1e-10
        print(f"  anisotropy={a:.1f}: V = {V_aniso:.6f} < log d = {max_value:.6f}? "
              f"{'是' if is_less else '否'}")
        if not is_less:
            all_less_than_max = False
    v2_pass = all_less_than_max
    print(f"  V2: {'PASS' if v2_pass else 'FAIL'}")

    # ----- V3: 梯度在 cI 处投影为零（极值条件）-----
    print("\n--- V3: dV/dp_i = -(log p_i + 1)，梯度在 cI 处投影为零 ---")
    # 在各向同性点
    grad_iso = field.value_gradient(g_iso)
    print(f"  各向同性梯度分量: {grad_iso}")
    # 所有分量应相等（梯度沿约束法向 = (1,1,...,1)）
    grad_diff = float(np.max(np.abs(grad_iso - grad_iso[0])))
    print(f"  分量最大差异: {grad_diff:.2e}（→ 0 表示梯度纯法向）")

    # 投影到约束面（Σ p_i = 1 的切空间 = 分量之和为零的子空间）
    grad_projected = grad_iso - np.mean(grad_iso)
    proj_norm = float(np.linalg.norm(grad_projected))
    print(f"  投影到约束面的梯度范数: {proj_norm:.2e}（→ 0 表示极值点）")

    # 在各向异性点（梯度应有非零投影）
    g_aniso = field.make_anisotropic_metric(anisotropy=0.5)
    grad_aniso = field.value_gradient(g_aniso)
    grad_aniso_proj = grad_aniso - np.mean(grad_aniso)
    proj_norm_aniso = float(np.linalg.norm(grad_aniso_proj))
    print(f"  各向异性点投影梯度范数: {proj_norm_aniso:.6f}（> 0，非极值）")

    v3_pass = (grad_diff < 1e-10 and proj_norm < 1e-10 and proj_norm_aniso > 1e-6)
    print(f"  V3: {'PASS' if v3_pass else 'FAIL'}")

    # ----- V4: 价值-痛苦耦合（含常数项 -(κ/2)·d·log d）-----
    print("\n--- V4: 价值-痛苦耦合 S = -(κ/2)·N_GCFT = (κ/2)·d·V - (κ/2)·d·log d ---")
    g_test = field.make_anisotropic_metric(anisotropy=0.4)
    V_test = field.compute_value(g_test)
    N_GCFT_test = field.spectral_curvature_N(g_test)
    log_d_test = math.log(dim)
    V_from_N = log_d_test - N_GCFT_test / dim  # V = log d - N_GCFT/d
    N_from_V = dim * (log_d_test - V_test)     # N_GCFT = d·(log d - V)

    print(f"  V(g) = {V_test:.6f}")
    print(f"  N_GCFT(g) = Σ λ_i log λ_i = {N_GCFT_test:.6f}（≥0，各向异性 > 0）")
    print(f"  log d = {log_d_test:.6f}")
    print(f"  V(g) = log d - N_GCFT/d = {V_from_N:.6f}（应与 V(g) 一致）")
    print(f"  N_GCFT(g) = d·(log d - V) = {N_from_V:.6f}（应与 N_GCFT(g) 一致）")
    # 验证 V 与 N_GCFT 的关系
    relation_V_from_N = abs(V_from_N - V_test) < 1e-10
    relation_N_from_V = abs(N_from_V - N_GCFT_test) < 1e-10
    print(f"  V = log d - N_GCFT/d 关系成立? {relation_V_from_N}")
    print(f"  N_GCFT = d·(log d - V) 关系成立? {relation_N_from_V}")

    kappa_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    coupling_values = []
    neg_N_GCFT_values = []  # -(κ/2)·N_GCFT
    print(f"  {'κ':>6} {'S_value':>12} {'-(κ/2)·N_GCFT':>16} {'差异':>10}")
    for kappa in kappa_values:
        S_val = field.value_pain_coupling(g_test, kappa)
        coupling_values.append(S_val)
        neg_N = -0.5 * kappa * N_GCFT_test
        neg_N_GCFT_values.append(neg_N)
        print(f"  {kappa:6.2f} {S_val:12.6f} {neg_N:16.6f} "
              f"{abs(S_val - neg_N):10.2e}")

    # 验证：耦合项 = -(κ/2)·N_GCFT = (κ/2)·d·V - (κ/2)·d·log d
    expected = [0.5 * k * dim * V_test - 0.5 * k * dim * log_d_test
                for k in kappa_values]
    is_linear = all(abs(coupling_values[i] - expected[i]) < 1e-10
                    for i in range(len(kappa_values)))
    # 验证：耦合项与 -(κ/2)·N_GCFT 等价
    is_equiv_to_N = all(abs(coupling_values[i] - neg_N_GCFT_values[i]) < 1e-10
                        for i in range(len(kappa_values)))
    print(f"  S_value = (κ/2)·d·V - (κ/2)·d·log d（含常数项）? {is_linear}")
    print(f"  S_value = -(κ/2)·N_GCFT（等价形式）? {is_equiv_to_N}")

    # 验证：耦合项随 κ 单调变化（κ 增大 → S_value 更负，驱动力增强）
    is_monotonic = all(coupling_values[i] > coupling_values[i + 1]
                       for i in range(len(coupling_values) - 1))
    print(f"  S_value 随 κ 单调递减（驱动力增强）? {is_monotonic}")

    # 验证：驱动力 ∂S/∂κ = d·(V - log d)/2（含常数项）
    driving_force = dim * (V_test - log_d_test) / 2.0
    # 数值差分验证
    df_numeric = (field.value_pain_coupling(g_test, 1.0 + 1e-6) -
                  field.value_pain_coupling(g_test, 1.0 - 1e-6)) / 2e-6
    driving_force_correct = abs(driving_force - df_numeric) < 1e-6
    print(f"  驱动力 dS/dk = d·(V-log d)/2 = {driving_force:.6f} "
          f"（负值表示 kappa 驱动系统趋向 V_max = log d）")
    print(f"  数值差分验证 dS/dk = {df_numeric:.6f}（应与解析一致）")
    print(f"  驱动力公式正确? {driving_force_correct}")
    # 各向异性时 V < log d，驱动力为负
    driving_force_negative = driving_force < 0
    print(f"  V < log d → 驱动力为负（驱动 g → cI）? {driving_force_negative}")

    v4_pass = (is_linear and is_equiv_to_N and is_monotonic and
               driving_force_correct and driving_force_negative and
               relation_V_from_N and relation_N_from_V)
    print(f"  V4: {'PASS' if v4_pass else 'FAIL'}")

    # ----- V5: 各向同性极限下的普适标度——V(cI) = log d 对所有维度 d 成立 -----
    print("\n--- V5: 各向同性极限下的普适标度——V(cI) = log d 对所有维度 d 成立 ---")
    # v13.1 修正：V(g) = -Σ p_i log p_i 不显含 ℏ，故无 ℏ→0 退化。
    # 对应原理体现在各向同性极限下的普适标度：
    # V(cI) = log d 对所有维度 d 成立——任何维度的认知系统的
    # 最大价值都是 log d（普适标度，与具体系统无关）。
    # 扫描维度 d，验证 V_max = log d 对所有 d 成立
    dims = [2, 3, 4, 5, 8, 10, 16]
    v_max_values = []
    log_d_values = []
    for d in dims:
        field_d = ValuePotentialField(dim=d, c=1.0)
        g_d = field_d.make_isotropic_metric()
        V_d = field_d.compute_value(g_d)
        log_d_d = math.log(d)
        v_max_values.append(V_d)
        log_d_values.append(log_d_d)
        rel_err = abs(V_d - log_d_d) / max(log_d_d, 1e-30)
        print(f"  d={d:2d}: V(cI) = {V_d:.10f}, log d = {log_d_d:.10f}, "
              f"相对误差 = {rel_err:.2e}")

    # 验证：V(cI) = log d 对所有维度成立（普适标度，相对误差 < 1e-10）
    all_match = all(abs(v_max_values[i] - log_d_values[i]) < 1e-10
                    for i in range(len(dims)))
    # 验证：V(cI) 是 d 的单调递增函数（log d 单调）
    all_monotonic = all(v_max_values[i] < v_max_values[i + 1]
                        for i in range(len(v_max_values) - 1))
    print(f"  V(cI) = log d 对所有维度 d 成立（相对误差 < 1e-10）? {all_match}")
    print(f"  V(cI) 随 d 单调递增（log d 单调）? {all_monotonic}")
    print(f"  【普适标度】任何维度的认知系统最大价值都是 log d，与具体系统无关。")

    v5_pass = all_match and all_monotonic
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
    print(f"基石26 验证总结：{n_pass}/{n_total} PASS")
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
    print("GCFT 基石26：价值势能场（Value Potential Field）")
    print("V(g) = N_vN(g) = -Σ p_i log p_i（标准 von Neumann 熵）")
    print("N_GCFT(g) = Σ λ_i log λ_i（卷一谱曲率项，V = log d - N_GCFT/d）")
    print("=" * 70)

    results = run_value_potential_field_verification(dim=4, c=1.0)

    print(f"\n最终结果：{'全部通过' if results['all_pass'] else '存在失败项'}")
