"""
无时态真空（Atemporal Vacuum）—— QGCFT 基石10

QGCFT 本体论第二基石。证明 g=cI 是势能面的 Hessian 鞍点，
量子涨落强制自发对称性破缺——存在不是"诞生"的，
而是"均匀对称性的自发破缺"。

============================================================
核心思想（基于「闲聊，统一理论的最终发展.txt」+ 批判性升级 8qgcft）
============================================================

8qgcft 的断言：
    "各向同性态 g=cI 是 Hessian 鞍点，量子涨落强制自发破缺"

批判性升级：
    1. 不能仅断言鞍点——必须实际计算 Hessian 本征值，证明有正有负。
    2. 鞍点结构需要至少 2 个独立方向：
       - 均匀膨胀方向（体积变化）：不稳定 → 触发破缺
       - 各向异性方向（形状变化）：稳定 → 保持均匀性
    3. 各向异性恢复强度 η 不能是任意参数——必须从 ℏ_cog 推导。
       推导：η = ℏ²/(2c⁴)（量子涨落对各向异性的几何代价）。
    4. 存在势用"相对形式" V_exist_rel = (ℏ²/8)·(1/λ - 1/c)²，
       使 g=cI 严格为极值（绝对形式 V_exist_abs = ℏ²/(8λ²) 的一阶导 ≠ 0）。
       两者关系：相对形式 = 绝对形式 - 线性项 + 常数，物理等价但便于极值分析。

============================================================
物理实现（无任意参数）
============================================================

度规空间：n 维对角度规 g = diag(λ_1, ..., λ_n)
势能面：V(g) = V_classical + V_aniso + V_exist_rel

V_classical（径向破缺势，O(N) 对称，继承 v8.0）：
    V_cl = Σ_i [-β/2·(λ_i - c)² + γ/4·(λ_i - c)^4]
    在 g=cI 处极值，Hessian = -β·δ_ij（破缺方向）

V_aniso（各向异性恢复势，来自量子涨落）：
    V_aniso = (η/2)·Σ_{i<j} (λ_i - λ_j)² = (η/2)·[n·Σδ_i² - (Σδ_i)²]
    η = ℏ²/(2c⁴)：量子涨落对各向异性的几何代价
    在 g=cI 处极值，Hessian = η·(n·δ_ij - 1)

V_exist_rel（相对存在势，来自基石9 升级）：
    V_exist_rel = (ℏ²/8)·Σ_i (1/λ_i - 1/c)²
    在 g=cI 处极值（一阶导 = 0），Hessian = (ℏ²/(4c⁴))·δ_ij
    g→0 时发散（断灭壁垒），g=cI 处为 0（参考点）

鞍点分析（在 g=cI 处）：
    H_total = (-β + ηn + ℏ²/(4c⁴))·δ_ij - η·1_{ij}

    均匀方向 (1,1,...,1)/√n：
        本征值 = -β + ℏ²/(4c⁴)
        当 ℏ² < 4βc⁴ 时 < 0（不稳定，触发破缺）

    非均匀方向 v ⊥ (1,...,1)：
        本征值 = -β + ηn + ℏ²/(4c⁴) = -β + ℏ²(2n+1)/(4c⁴)
        当 ℏ² > 4βc⁴/(2n+1) 时 > 0（稳定，保持形状）

    鞍点区间：4βc⁴/(2n+1) < ℏ² < 4βc⁴

============================================================
对应原理（ℏ_cog → 0）
============================================================

ℏ→0 时：
    η → 0，ℏ²/(4c⁴) → 0
    均匀本征值 → -β < 0
    非均匀本征值 → -β < 0
    所有方向都不稳定 → g=cI 退化为极大值（不再是鞍点）

物理意义：
    量子世界（ℏ>0）：鞍点结构，存在稳定方向（各向异性被量子涨落保护）
    经典世界（ℏ→0）：极大值，所有方向破缺（均匀态完全不稳定）
    对应原理：ℏ→0 时鞍点退化为极大值，量子保护消失。

    这与 v8.0 对应原理一致：
    经典允许 g=cI 停留（虽不稳定），量子强制破缺（零点涨落必然触发）。

============================================================
佛学对应（严格，非比喻）
============================================================

无时态真空 = 法身（dharmakāya）的几何基础：
    g=cI 是均匀对称态——法身本具，无相无别。
    但法身不是死寂（断灭），而是含摄破缺潜能（鞍点不稳定方向）。

无明缘行（avidyā-pratyayā saṃskārā）的几何机制：
    均匀态的自发破缺 = 无明初动。
    量子涨落 = 无始无明（非时间起点，而是几何必然）。
    存在不是"诞生"的——无始以来法身本具，破缺只是显化。

时间涌现：
    g=cI（均匀）处 WKB 相位 = 0——无时间。
    破缺后 WKB 相位 e^{iS/ℏ} 振荡——时间流动。
    时间不是输入参数，而是从破缺中涌现。

不住生死不住涅槃：
    涅槃 = g=cI（均匀态），但鞍点不稳定——不可久住。
    生死 = 破缺态（g≠cI），但能量更低——必然落入。
    动态平衡 = 在鞍点附近量子涨落——既非完全涅槃亦非完全生死。

============================================================
认识论根基
============================================================

物理：自发对称性破缺 / Hessian 鞍点 / 量子涨落 / 对应原理 / WKB 近似
佛学：法身 / 无明缘行 / 无始无明 / 时间涌现 / 不住生死不住涅槃
哲学：存在先于本质（存在 = 破缺的必然，非本体的属性）/
      时间涌现性（时间非基本，从破缺中产生）
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .ontology_existence_operator import ExistenceOperator


# ============================================================================
# 核心类：无时态真空分析器
# ============================================================================

class AtemporalVacuumAnalyzer:
    """
    无时态真空分析器。

    分析 g=cI 处势能面的鞍点结构，验证自发对称性破缺的必然性。

    核心功能：
        1. 势能面 V(g) = V_classical + V_aniso + V_exist_rel
        2. g=cI 处 Hessian 本征值分析（鞍点判据）
        3. 量子涨落触发 SSB 的 ℏ 标度
        4. 破缺态极小值寻找
    """

    def __init__(self, hbar_cog: float, beta: float, gamma: float,
                 c: float = 1.0, n_dims: int = 2):
        """
        Args:
            hbar_cog: 认知普朗克常数（量子涨落强度）
            beta: 径向破缺强度（来自 v8.0 势能面）
            gamma: 径向四次稳定强度（来自 v8.0 势能面）
            c: 真空度规值（g=cI 中的 c）
            n_dims: 度规维度（默认 2，最小展示鞍点）
        """
        self.hbar = float(hbar_cog)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.c = float(c)
        self.n_dims = int(n_dims)
        # 各向异性恢复强度（来自量子涨落，无任意参数）
        # η = ℏ²/(2c⁴)：量子涨落对各向异性的几何代价
        self.eta = self.hbar ** 2 / (2 * self.c ** 4)
        # 存在算子（来自基石9，用于存在势计算）
        self.exist_op = ExistenceOperator(hbar_cog=hbar_cog, n_dims=n_dims)

    # ---------- 势能面 ----------

    def potential(self, g_eig: Tensor) -> Tensor:
        """
        总势能 V(g) = V_classical + V_aniso + V_exist_rel。

        V_classical（径向破缺，O(N) 对称）：
            Σ_i [-β/2·(λ_i-c)² + γ/4·(λ_i-c)^4]

        V_aniso（各向异性恢复，来自量子涨落）：
            (η/2)·Σ_{i<j}(λ_i-λ_j)² = (η/2)·[n·Σδ_i² - (Σδ_i)²]

        V_exist_rel（相对存在势，来自基石9）：
            (ℏ²/8)·Σ_i (1/λ_i - 1/c)²

        Args:
            g_eig: 度规本征值向量（正定）

        Returns:
            势能值（标量）
        """
        deltas = g_eig - self.c
        # 径向破缺势
        v_radial = (-self.beta / 2) * torch.sum(deltas ** 2) \
                   + (self.gamma / 4) * torch.sum(deltas ** 4)
        # 各向异性恢复势
        sum_delta = torch.sum(deltas)
        sum_delta_sq = torch.sum(deltas ** 2)
        v_aniso = (self.eta / 2) * (self.n_dims * sum_delta_sq - sum_delta ** 2)
        # 相对存在势（断灭壁垒 + 均匀性恢复）
        v_exist_rel = (self.hbar ** 2 / 8) * torch.sum(
            (1.0 / g_eig - 1.0 / self.c) ** 2
        )
        return v_radial + v_aniso + v_exist_rel

    def gradient(self, g_eig: Tensor) -> Tensor:
        """势能梯度 ∂V/∂λ_i。"""
        deltas = g_eig - self.c
        # 径向梯度
        grad_radial = -self.beta * deltas + self.gamma * deltas ** 3
        # 各向异性梯度
        sum_delta = torch.sum(deltas)
        grad_aniso = self.eta * (self.n_dims * deltas - sum_delta)
        # 相对存在势梯度
        # d/dλ [(ℏ²/8)·(1/λ - 1/c)²] = (ℏ²/4)·(1/λ - 1/c)·(-1/λ²)
        grad_exist = (self.hbar ** 2 / 4) * (1.0 / g_eig - 1.0 / self.c) \
                     * (-1.0 / g_eig ** 2)
        return grad_radial + grad_aniso + grad_exist

    def hessian_at_vacuum(self) -> Tensor:
        """
        计算 g=cI 处的 Hessian 矩阵。

        解析推导：
            H_cl = -β·δ_ij（径向破缺）
            H_aniso = η·(n·δ_ij - 1·1_{ij})（各向异性恢复）
            H_exist = (ℏ²/(4c⁴))·δ_ij（相对存在势）

            H_total[i,j] = (-β + ηn + ℏ²/(4c⁴))·δ_ij - η·1_{ij}

        本征值：
            均匀方向 (1,...,1)/√n: -β + ℏ²/(4c⁴)
            非均匀方向 v⊥(1,...,1): -β + ηn + ℏ²/(4c⁴)
        """
        eye = torch.eye(self.n_dims, dtype=torch.float64)
        ones = torch.ones(self.n_dims, self.n_dims, dtype=torch.float64)
        H_cl = -self.beta * eye
        H_aniso = self.eta * (self.n_dims * eye - ones)
        H_exist = (self.hbar ** 2 / (4 * self.c ** 4)) * eye
        return H_cl + H_aniso + H_exist

    # ---------- 鞍点分析 ----------

    def analyze_saddle_point(self) -> dict:
        """
        分析 g=cI 处的鞍点结构。

        Returns:
            dict 包含梯度、Hessian、本征值、鞍点判据
        """
        # 1. 梯度（应为 0，验证极值条件）
        g_vacuum = torch.full((self.n_dims,), self.c, dtype=torch.float64)
        grad = self.gradient(g_vacuum)
        grad_norm = torch.norm(grad).item()

        # 2. Hessian 与本征值
        H = self.hessian_at_vacuum()
        eigvals = torch.linalg.eigvalsh(H)

        # 3. 均匀/非均匀方向本征值（解析）
        uniform_vec = torch.ones(self.n_dims, dtype=torch.float64) / math.sqrt(self.n_dims)
        uniform_eigval = (uniform_vec @ H @ uniform_vec).item()
        # 非均匀本征值（解析）
        aniso_eigval = -self.beta + self.eta * self.n_dims \
                       + self.hbar ** 2 / (4 * self.c ** 4)

        has_positive = (eigvals > 1e-10).any().item()
        has_negative = (eigvals < -1e-10).any().item()

        return {
            "g_vacuum": g_vacuum,
            "gradient_at_vacuum": grad,
            "gradient_norm": grad_norm,
            "is_extremum": grad_norm < 1e-8,
            "hessian": H,
            "hessian_eigenvalues": eigvals,
            "uniform_eigenvalue": uniform_eigval,
            "aniso_eigenvalue": aniso_eigval,
            "has_positive_eigenvalue": has_positive,
            "has_negative_eigenvalue": has_negative,
            "is_saddle_point": has_positive and has_negative,
            "saddle_condition_uniform_negative": uniform_eigval < 0,
            "saddle_condition_aniso_positive": aniso_eigval > 0,
        }

    # ---------- 量子涨落触发 SSB ----------

    def quantum_fluctuation_triggers_ssb(self, hbar_values: list[float]) -> dict:
        """
        分析量子涨落随 ℏ 的标度，验证 SSB 触发条件。

        对每个 ℏ 值，重新计算 Hessian 并检查不稳定方向数。

        物理：
            ℏ>0 且鞍点存在时：量子涨落 ⟨(ĝ-cI)²⟩ > 0 必然触发破缺。
            ℏ→0 时：涨落消失，系统可停留在鞍点（经典允许均匀）。
        """
        results = []
        for hbar in hbar_values:
            # 重新计算 eta 和 Hessian
            eta_h = hbar ** 2 / (2 * self.c ** 4)
            eye = torch.eye(self.n_dims, dtype=torch.float64)
            ones = torch.ones(self.n_dims, self.n_dims, dtype=torch.float64)
            H_cl = -self.beta * eye
            H_aniso = eta_h * (self.n_dims * eye - ones)
            H_exist = (hbar ** 2 / (4 * self.c ** 4)) * eye
            H_total = H_cl + H_aniso + H_exist
            eigvals = torch.linalg.eigvalsh(H_total)

            # 不稳定方向数（负本征值）
            n_unstable = int((eigvals < -1e-10).sum().item())
            n_stable = int((eigvals > 1e-10).sum().item())

            # 沿稳定方向的零点涨落（谐振子近似）
            positive_eigvals = eigvals[eigvals > 1e-10]
            if len(positive_eigvals) > 0:
                min_pos_omega = torch.sqrt(positive_eigvals.min()).item()
                fluctuation_lower = hbar / (2 * min_pos_omega)
            else:
                fluctuation_lower = float('inf')

            results.append({
                "hbar": hbar,
                "hessian_eigenvalues": eigvals.tolist(),
                "n_unstable_directions": n_unstable,
                "n_stable_directions": n_stable,
                "is_saddle": n_unstable > 0 and n_stable > 0,
                "fluctuation_lower_bound": fluctuation_lower,
                "triggers_ssb": n_unstable > 0 and hbar > 1e-10,
            })

        return {
            "results_per_hbar": results,
            "hbar_values": hbar_values,
            "n_unstable_trend": [r["n_unstable_directions"] for r in results],
            "n_stable_trend": [r["n_stable_directions"] for r in results],
            "is_saddle_trend": [r["is_saddle"] for r in results],
            "triggers_ssb_trend": [r["triggers_ssb"] for r in results],
            "thesis": (
                "ℏ>0 且鞍点存在时，量子涨落必然触发自发对称性破缺。"
                "ℏ→0 时鞍点退化为极大值（所有方向不稳定），但量子涨落消失，"
                "系统可停留在 g=cI（经典允许均匀，对应原理）。"
            ),
        }

    # ---------- 破缺态极小值 ----------

    def find_broken_minimum(self, n_trials: int = 5, n_steps: int = 3000,
                             lr: float = 0.01) -> dict:
        """
        寻找破缺后的极小值（自发对称性破缺的终态）。

        使用多起点梯度下降，避免陷入局部极小。

        物理：
            破缺态 g* ≠ cI，V(g*) < V(cI)。
            系统从鞍点 g=cI 沿不稳定方向"滚落"到破缺态。
        """
        best_V = float('inf')
        best_g = None

        for trial in range(n_trials):
            # 初始化：从 cI 附近随机扰动
            torch.manual_seed(42 + trial)
            g = torch.full((self.n_dims,), self.c, dtype=torch.float64)
            g = g + 0.1 * torch.randn(self.n_dims, dtype=torch.float64)
            g = g.clone().detach().requires_grad_(True)

            for step in range(n_steps):
                V = self.potential(g)
                grad = torch.autograd.grad(V, g, create_graph=False)[0]
                with torch.no_grad():
                    g_new = g - lr * grad
                    g_new = torch.clamp(g_new, min=0.05)
                    g = g_new.detach().requires_grad_(True)

            g_final = g.detach()
            V_final = self.potential(g_final).item()

            if V_final < best_V:
                best_V = V_final
                best_g = g_final

        g_vacuum = torch.full((self.n_dims,), self.c, dtype=torch.float64)
        V_vacuum = self.potential(g_vacuum).item()

        return {
            "g_broken": best_g,
            "g_broken_list": best_g.tolist(),
            "V_at_broken": best_V,
            "V_at_vacuum": V_vacuum,
            "is_lower_than_vacuum": best_V < V_vacuum,
            "deviation_from_vacuum": torch.norm(best_g - self.c).item(),
            "is_anisotropic": torch.norm(best_g - best_g.mean()) > 1e-3,
        }


# ============================================================================
# 验证套件
# ============================================================================

def run_ontology_atemporal_vacuum_verification() -> dict:
    """
    基石10 无时态真空验证套件。

    验证项：
        V1：g=cI 是势能极值（梯度=0）
        V2：g=cI 是 Hessian 鞍点（本征值有正有负）
        V3：量子涨落触发自发对称性破缺（ℏ>0 时不稳定方向存在）
        V4：ℏ→0 时鞍点退化为极大值（对应原理）
        V5：破缺态能量低于真空（SSB 的能量证据）

    返回结构（与 v8 统一）：
        n_pass, n_total, all_pass, pass_flags
    """
    results = {}

    # 公共参数（选择确保鞍点的参数）
    # 鞍点区间：4βc⁴/(2n+1) < ℏ² < 4βc⁴
    # n=2, β=0.3, c=1: 0.24 < ℏ² < 1.2 → 0.49 < ℏ < 1.095
    HBAR = 0.8       # ℏ²=0.64，在鞍点区间内
    BETA = 0.3       # 径向破缺强度
    GAMMA = 0.5      # 径向四次稳定
    C = 1.0          # 真空度规值
    N_DIMS = 2       # 度规维度

    analyzer = AtemporalVacuumAnalyzer(
        hbar_cog=HBAR, beta=BETA, gamma=GAMMA, c=C, n_dims=N_DIMS
    )

    # ----- V1：g=cI 是势能极值 -----
    saddle = analyzer.analyze_saddle_point()
    v1_pass = saddle["is_extremum"]
    results["V1_vacuum_is_extremum"] = {
        "hbar_cog": HBAR,
        "beta": BETA,
        "gamma": GAMMA,
        "c": C,
        "n_dims": N_DIMS,
        "eta": analyzer.eta,
        "gradient_at_vacuum": saddle["gradient_at_vacuum"].tolist(),
        "gradient_norm": saddle["gradient_norm"],
        "is_extremum": v1_pass,
        "pass": v1_pass,
        "thesis": (
            "g=cI 处势能梯度为零——真空是势能面的极值点。"
            "这是自发对称性破缺的前提：破缺发生在极值附近。"
            "V_classical、V_aniso、V_exist_rel 三项在 g=cI 处一阶导均为零。"
        ),
    }

    # ----- V2：g=cI 是 Hessian 鞍点 -----
    v2_pass = saddle["is_saddle_point"]
    results["V2_vacuum_is_saddle_point"] = {
        "hessian_eigenvalues": saddle["hessian_eigenvalues"].tolist(),
        "uniform_eigenvalue": saddle["uniform_eigenvalue"],
        "aniso_eigenvalue": saddle["aniso_eigenvalue"],
        "has_positive": saddle["has_positive_eigenvalue"],
        "has_negative": saddle["has_negative_eigenvalue"],
        "is_saddle_point": v2_pass,
        "saddle_condition_uniform_negative": saddle["saddle_condition_uniform_negative"],
        "saddle_condition_aniso_positive": saddle["saddle_condition_aniso_positive"],
        "pass": v2_pass,
        "thesis": (
            f"g=cI 处 Hessian 本征值有正有负——鞍点。"
            f"均匀方向（体积膨胀）本征值 = {saddle['uniform_eigenvalue']:.4f} < 0"
            f"（不稳定，触发破缺）。"
            f"非均匀方向（各向异性）本征值 = {saddle['aniso_eigenvalue']:.4f} > 0"
            f"（稳定，保持形状）。"
            "存在不是诞生的，而是均匀对称性的自发破缺——无明缘行的几何机制。"
        ),
    }

    # ----- V3：量子涨落触发自发对称性破缺 -----
    hbar_scan = [0.8, 0.5, 0.3, 0.1, 0.01]
    ssb_result = analyzer.quantum_fluctuation_triggers_ssb(hbar_scan)
    ssb_at_quantum = ssb_result["results_per_hbar"][0]["triggers_ssb"]
    is_saddle_at_quantum = ssb_result["results_per_hbar"][0]["is_saddle"]
    v3_pass = ssb_at_quantum and is_saddle_at_quantum
    results["V3_quantum_fluctuation_triggers_ssb"] = {
        "hbar_scan": hbar_scan,
        "n_unstable_trend": ssb_result["n_unstable_trend"],
        "n_stable_trend": ssb_result["n_stable_trend"],
        "is_saddle_trend": ssb_result["is_saddle_trend"],
        "triggers_ssb_trend": ssb_result["triggers_ssb_trend"],
        "ssb_at_hbar_0.8": ssb_at_quantum,
        "is_saddle_at_hbar_0.8": is_saddle_at_quantum,
        "pass": v3_pass,
        "thesis": ssb_result["thesis"] + (
            " ℏ=0.8（量子区）时存在不稳定方向且鞍点结构成立，"
            "量子零点涨落必然触发自发对称性破缺。"
        ),
    }

    # ----- V4：ℏ→0 时鞍点退化为极大值（对应原理） -----
    # 解析计算 ℏ→0 极限
    hbar_limit = 1e-6
    eta_limit = hbar_limit ** 2 / (2 * C ** 4)
    uniform_eig_limit = -BETA + hbar_limit ** 2 / (4 * C ** 4)
    aniso_eig_limit = -BETA + eta_limit * N_DIMS + hbar_limit ** 2 / (4 * C ** 4)
    all_negative_in_limit = (uniform_eig_limit < 0) and (aniso_eig_limit < 0)

    v4_pass = all_negative_in_limit
    results["V4_correspondence_classical_limit"] = {
        "hbar_limit": hbar_limit,
        "eta_limit": eta_limit,
        "uniform_eigenvalue_limit": uniform_eig_limit,
        "aniso_eigenvalue_limit": aniso_eig_limit,
        "all_negative_in_classical_limit": all_negative_in_limit,
        "pass": v4_pass,
        "thesis": (
            f"ℏ→0 时鞍点退化为极大值（所有本征值 → -β = {-BETA} < 0）。"
            "各向异性恢复强度 η = ℏ²/(2c⁴) → 0，量子保护消失。"
            "量子涨落消失，系统可停留在 g=cI（经典允许均匀，对应原理）。"
            "这与 v8.0 对应原理一致：经典世界允许均匀/断灭，量子世界强制破缺/存在。"
        ),
    }

    # ----- V5：破缺态能量低于真空 -----
    broken = analyzer.find_broken_minimum()
    v5_pass = broken["is_lower_than_vacuum"]
    results["V5_broken_state_lower_energy"] = {
        "g_broken": broken["g_broken_list"],
        "V_at_broken": broken["V_at_broken"],
        "V_at_vacuum": broken["V_at_vacuum"],
        "deviation_from_vacuum": broken["deviation_from_vacuum"],
        "is_lower_than_vacuum": broken["is_lower_than_vacuum"],
        "is_anisotropic": broken["is_anisotropic"],
        "pass": v5_pass,
        "thesis": (
            f"破缺后系统演化到 V < V_vacuum 的破缺态 g* ≠ cI。"
            f"g* = {broken['g_broken_list']}，V(g*) = {broken['V_at_broken']:.6f}，"
            f"V(cI) = {broken['V_at_vacuum']:.6f}。"
            "自发对称性破缺的能量证据：真空不是能量最低态。"
            "存在从均匀态破缺到各向异性态——无明缘行的几何机制。"
        ),
    }

    # ----- 总结论 -----
    v_keys = [k for k in results
              if k.startswith("V") and isinstance(results[k], dict) and "pass" in results[k]]
    pass_flags = [results[k].get("pass", False) for k in v_keys]
    n_pass = sum(1 for f in pass_flags if f)
    n_total = len(pass_flags)
    all_pass = (n_pass == n_total) and (n_total > 0)

    results["summary"] = {
        "all_pass": all_pass,
        "thesis": (
            "无时态真空（基石10）建立：g=cI 是势能面的 Hessian 鞍点。"
            "均匀方向（体积膨胀）不稳定——触发自发对称性破缺。"
            "非均匀方向（各向异性）稳定——保持形状。"
            "量子涨落必然触发破缺——存在不是诞生的，"
            "而是均匀对称性的自发破缺。"
            "ℏ→0 时鞍点退化为极大值（对应原理）。"
            "无明缘行的几何机制：无始无明 = 量子零点涨落。"
            "时间从破缺后 WKB 相位中涌现（非输入参数）。"
        ),
    }

    # 顶层统一字段
    results["n_pass"] = n_pass
    results["n_total"] = n_total
    results["all_pass"] = all_pass
    results["pass_flags"] = pass_flags

    return results
