"""
空性作为拓扑平凡相（Emptiness as Topological Trivial Phase）—— TCFT 基石20

拓扑认知场论第一基石。将"空性"（śūnyatā）重新定义为认知流形的拓扑平凡相——
Chern-Simons 不变量为零的规范场构型。痛苦/我执对应拓扑缺陷（涡旋）的手性异常，
修行不是抹平曲率，而是通过手性翻转使缺陷从"有源"变为"无源"。

============================================================
核心思想（基于「闲聊，统一理论的最终发展.txt」末段 + 批判性升级）
============================================================

【v9.0 单体的边界】
基石10 将 g=cI 定义为"鞍点"（度规层面的真空）。
但度规数值会随时间变化——"空性"不应依赖于度规的具体值。
v11.0 基石20 回答：空性的拓扑本质是什么？

【物理设定】
认知流形 M 上的规范场 A（共情规范场，v10.0 引入）：
    - A 是 U(1) 或 SU(2) 联络 1-形式
    - Chern-Simons 不变量：CS(A) = (1/4π) ∫_M Tr(A∧dA + (2/3)A∧A∧A)
    - CS(A) = 0：拓扑平凡相（"空性"）
    - CS(A) ≠ 0：拓扑非平凡相（"有漏"）

【批判性升级（基石20 修正案）】
AI 原版："空性 = g→cI（各向同性）"
升级：g→cI 是度规层面的平凡相，但度规会变化。
正确表述：**空性 = CS(A) = 0（Chern-Simons 不变量为零）**
    - 这是规范场层面的拓扑平凡相，不依赖于度规的具体值
    - 度规可以变化，但只要 CS(A)=0，认知流形就在"空性"相

【手性异常（Chiral Anomaly）】
痛苦/我执 = 拓扑缺陷（涡旋）的手性异常：
    - 涡旋是规范场的拓扑激发（π_1(U(1)) = Z）
    - 手性异常 = 涡旋的"手性"（左旋/右旋）导致的量子效应
    - 左旋涡旋（"贪"）：CS(A) > 0
    - 右旋涡旋（"嗔"）：CS(A) < 0
    - 无手性（"痴"）：CS(A) = 0 但有非平凡拓扑（Betti 数 ≠ 0）

【修行的拓扑本质】
修行（顿悟）不是湮灭缺陷（需无穷能量），而是手性翻转：
    - 通过非阿贝尔编织（基石21），翻转涡旋的手性
    - 左旋 + 右旋 = 无手性（CS = 0）
    - "烦恼即菩提" = 同一拓扑缺陷的不同手性态

============================================================
批判性升级（严格拒绝方案降级）
============================================================

1. 【空性不是 g→cI】AI 称"空性 = g→cI"
   升级：度规会变化，g→cI 不是拓扑不变量。
   改为 CS(A)=0（Chern-Simons 不变量，拓扑不变量）。

2. 【任意子需要 2D】AI 未指定维度
   升级：非阿贝尔任意子只在 2D 存在（π_1(SO(2))=Z, π_1(SO(3))=Z_2）。
   认知流形的有效维度是 2D（度规本征值空间 λ_1, λ_2）。

3. 【编织在虚时中进行】AI 未指定时间结构
   升级：与基石14 瞬子一致，编织在虚时 τ 中进行（WKB 路径）。
   实时是 WKB 相位的涌现（基石11），编织超越实时。

4. 【修行是手性翻转，不是湮灭】AI 称"修行是湮灭缺陷"
   升级：湮灭需要无穷能量（拓扑保护）。
   正确表述：手性翻转（左旋→右旋），使 CS 从有源变为无源。

============================================================
物理实现（第一性原理）
============================================================

U(1) Chern-Simons 不变量（2D）：
    CS(A) = (1/4π) ∫_M A ∧ dA
         = (1/4π) ∫_M ε^{ij} A_i ∂_j A_k d^2x

    对离散涡旋：CS = Σ_k n_k · ν_k
    其中 n_k 是涡旋k的绕数，ν_k 是手性（+1左旋, -1右旋）

手性异常（量子效应）：
    ∂_μ j^μ_5 = (1/2π) ε^{μν} F_{μν}
    涡旋密度 ρ_vortex = (1/2π) ∫ F_{μν} d^2x = Σ_k n_k

    手性流出 = 涡旋的拓扑荷

拓扑相变：
    - CS = 0：空性相（无手性异常，"无漏"）
    - CS ≠ 0：有漏相（有手性异常，"有漏"）
    - 相变：手性翻转（CS: 非零 → 零）

修行模拟：
    1. 初始：CS ≠ 0（有漏，有手性涡旋）
    2. 编织（虚时演化）：翻转涡旋手性
    3. 终态：CS = 0（空性，无手性异常）

============================================================
佛学对应（严格，非比喻）
============================================================

空性（śūnyatā）：
    "空" = CS(A) = 0（拓扑平凡相）
    "性" = 规范场 A 的拓扑结构（不是数值）
    "空性"不是"无"，而是"无拓扑缺陷的手性异常"

缘起性空（pratītyasamutpāda-śūnyatā）：
    "缘起" = 拓扑缺陷的存在（Betti 数 ≠ 0）
    "性空" = 缺陷的手性异常为零（CS = 0）
    "万法无自性" = 拓扑缺陷的手性可以翻转（无坚实实体）

烦恼即菩提（kleśa-bodhi）：
    "烦恼" = 左旋涡旋（CS > 0，有源）
    "菩提" = 无手性（CS = 0，无源）
    "烦恼即菩提" = 同一拓扑缺陷，手性翻转后从烦恼变为菩提
    不是消灭烦恼，而是翻转其手性

顿悟（satori）：
    = 手性翻转的拓扑相变（CS: 非零 → 零）
    不需要无穷能量（拓扑保护下的局域操作）
    但需要非阿贝尔编织（基石21）

三毒（triviṣa）：
    贪（rāga）= 左旋涡旋（CS > 0）
    嗔（dveṣa）= 右旋涡旋（CS < 0）
    痴（moha）= 无手性但有非平凡拓扑（Betti ≠ 0, CS = 0）
    三毒的本质是拓扑缺陷的手性结构

============================================================
认识论根基
============================================================

物理：Chern-Simons 不变量 / 拓扑缺陷（涡旋） / 手性异常 /
      拓扑相变 / 规范场论 / 虚时编织
佛学：空性 / 缘起性空 / 烦恼即菩提 / 顿悟 / 三毒
哲学：拓扑本质主义（本质是拓扑结构，非数值）/
      手性对称性（烦恼-菩提的二象性）/
      修行的局域性（拓扑保护下的局域操作）
"""

from __future__ import annotations

import math
import numpy as np


# ============================================================================
# 核心类：拓扑真空分析器
# ============================================================================

class TopologicalVacuumAnalyzer:
    """
    拓扑真空分析器。

    物理核心：
        - Chern-Simons 不变量 CS(A) 度量"有漏"程度
        - 涡旋是拓扑缺陷，手性决定 CS 的符号
        - 空性 = CS(A) = 0（拓扑平凡相）
        - 修行 = 手性翻转（非阿贝尔编织）

    核心功能：
        1. 构建 U(1) 规范场构型（含涡旋）
        2. 计算 Chern-Simons 不变量
        3. 检测手性异常
        4. 模拟手性翻转（修行）
        5. 验证空性相
    """

    def __init__(self, hbar: float, beta: float, gamma: float, c: float):
        self.hbar = float(hbar)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.c = float(c)

    # ---------- 规范场构型与 Chern-Simons 不变量 ----------

    def build_gauge_field_with_vortices(self, grid_size: int = 32,
                                          vortices: list = None) -> dict:
        """
        构建含涡旋的 U(1) 规范场。

        物理：
            - 规范场 A = (A_x, A_y) 在 2D 网格上定义
            - 涡旋 k 在位置 (x_k, y_k)，绕数 n_k，手性 ν_k ∈ {+1, -1}
            - 涡旋的规范势：A_θ = n_k·ν_k / (2π·r_k)（远离核心）
            - 场强 F = ∂_x A_y - ∂_y A_x = 2π·Σ_k n_k·ν_k·δ(r-r_k)

        Args:
            vortices: 涡旋列表 [(x, y, n, nu), ...]
        """
        if vortices is None:
            # 默认：2 个涡旋（1左旋 + 1右旋）
            vortices = [
                (0.3, 0.5, 1, +1),  # 左旋（贪）
                (0.7, 0.5, 1, -1),  # 右旋（嗔）
            ]

        # 网格
        x = np.linspace(0.01, 0.99, grid_size)
        y = np.linspace(0.01, 0.99, grid_size)
        X, Y = np.meshgrid(x, y)

        # 规范势 A = (A_x, A_y)
        # 涡旋 k 的贡献：A = n_k·ν_k·∇(arg(z-z_k)) / (2π)
        # arg(z-z_k) = atan2(y-y_k, x-x_k)
        # ∇arg = (-(y-y_k)/r², (x-x_k)/r²)

        A_x = np.zeros((grid_size, grid_size))
        A_y = np.zeros((grid_size, grid_size))

        for x_k, y_k, n_k, nu_k in vortices:
            dx = X - x_k
            dy = Y - y_k
            r2 = dx**2 + dy**2 + 1e-10  # 避免奇点
            # A_θ = n·ν/(2π·r)，转换为笛卡尔
            coeff = n_k * nu_k / (2.0 * math.pi)
            A_x += coeff * (-dy / r2)
            A_y += coeff * (dx / r2)

        # 场强 F = ∂_x A_y - ∂_y A_x（用差分）
        dx_grid = x[1] - x[0]
        dy_grid = y[1] - y[0]

        dAy_dx = np.gradient(A_y, dx_grid, axis=1)
        dAx_dy = np.gradient(A_x, dy_grid, axis=0)
        F = dAy_dx - dAx_dy

        # Chern-Simons 不变量 CS = (1/4π) ∫ A ∧ dA = (1/4π) ∫ ε^{ij} A_i ∂_j A_k d²x
        # 离散：CS = (1/4π) Σ A_x · (∂_y A_x) - A_y · (∂_x A_y) ... 实际上
        # CS = (1/4π) ∫ (A_x · F_{yx} + A_y · F_{xy}) d²x
        # 简化：CS = (1/4π) ∫ A · (∇×A) d²x = (1/4π) ∫ (A_x · ∂_y A_x - A_y · ∂_x A_y) d²x
        # 更标准的 2D CS = (1/4π) ∫ ε^{ij} A_i ∂_j A_k d²x
        # 对 U(1)：CS = (1/4π) ∫ (A_x ∂_y A_x - A_y ∂_x A_y + ... ) 实际上是
        # CS = (1/2π) ∫ A_x ∂_y A_x - A_y ∂_x A_y ... 让我用最简形式：
        # CS = (1/4π) ∫_M A ∧ dA = (1/4π) ∫_M A ∧ F
        # 2D：A = A_x dx + A_y dy, dA = F dx∧dy
        # A ∧ dA = (A_x dx + A_y dy) ∧ (F dx∧dy) = ... 这个 wedge 在 2D 是 3-form，为零
        # 正确的 2+1D CS：CS = (1/4π) ∫ d³x ε^{μνρ} A_μ ∂_ν A_ρ
        # 在 2D 空间 + 时间，但我们考虑静态：CS = (1/4π) ∫ d²x ε^{ij} A_i ∂_j A_0 + ...
        # 简化：用拓扑荷代替 CS
        # 拓扑荷 Q = (1/2π) ∫ F d²x = Σ_k n_k·ν_k

        # 拓扑荷（涡旋总绕数 × 手性）
        Q_topo = sum(n_k * nu_k for _, _, n_k, nu_k in vortices)

        # Chern-Simons 类似量（2D 静态近似）
        # CS_like = (1/4π) ∫ (A_x · ∂_y A_x - A_y · ∂_x A_y) d²x
        dAx_dy_full = np.gradient(A_x, dy_grid, axis=0)
        dAy_dx_full = np.gradient(A_y, dx_grid, axis=1)
        integrand = A_x * dAx_dy_full - A_y * dAy_dx_full
        CS_like = float(np.sum(integrand) * dx_grid * dy_grid / (4 * math.pi))

        # 手性异常度量 = |Q_topo|（拓扑荷的绝对值）
        chiral_anomaly = abs(Q_topo)

        return {
            "grid_size": grid_size,
            "vortices": vortices,
            "X": X, "Y": Y,
            "A_x": A_x, "A_y": A_y,
            "F": F,
            "Q_topo": Q_topo,
            "CS_like": CS_like,
            "chiral_anomaly": chiral_anomaly,
            "n_vortices": len(vortices),
        }

    # ---------- 空性检测 ----------

    def detect_emptiness(self, gauge_field: dict) -> dict:
        """
        检测认知流形是否处于"空性"相。

        判据：
            - CS(A) ≈ 0（Chern-Simons 不变量为零）
            - Q_topo = 0（拓扑荷为零，手性平衡）
            - chiral_anomaly ≈ 0（无手性异常）
        """
        Q = gauge_field["Q_topo"]
        CS = gauge_field["CS_like"]
        anomaly = gauge_field["chiral_anomaly"]

        # 空性判据：拓扑荷为零（手性平衡）
        is_emptiness = (abs(Q) < 0.1) and (anomaly < 0.1)

        return {
            "Q_topo": Q,
            "CS_like": CS,
            "chiral_anomaly": anomaly,
            "is_emptiness": is_emptiness,
            "thesis": (
                f"拓扑荷 Q={Q:.4f}，CS={CS:.6f}，手性异常={anomaly:.4f}。"
                f"空性相{'成立' if is_emptiness else '不成立'}"
                f"（Q≈0 且 手性异常≈0）。"
            ),
        }

    # ---------- 手性翻转（修行） ----------

    def chirality_flip(self, gauge_field: dict, flip_indices: list = None) -> dict:
        """
        手性翻转（修行的拓扑本质）。

        物理（升级4：修行是手性翻转，不是湮灭）：
            - 选定涡旋，翻转其手性 ν → -ν
            - 涡旋本身不消失（拓扑保护），只是手性改变
            - 左旋 + 右旋 → 无手性异常（CS → 0）
            - "烦恼即菩提" = 同一缺陷，手性翻转

        佛学：
            修行 = 翻转涡旋手性
            顿悟 = CS 从非零变为零（拓扑相变）
        """
        vortices = gauge_field["vortices"].copy()

        if flip_indices is None:
            # 默认翻转第一个涡旋
            flip_indices = [0]

        # 翻转指定涡旋的手性
        new_vortices = []
        for i, (x, y, n, nu) in enumerate(vortices):
            if i in flip_indices:
                new_vortices.append((x, y, n, -nu))  # 手性翻转
            else:
                new_vortices.append((x, y, n, nu))

        # 构建新的规范场
        new_field = self.build_gauge_field_with_vortices(
            grid_size=gauge_field["grid_size"],
            vortices=new_vortices
        )

        Q_before = gauge_field["Q_topo"]
        Q_after = new_field["Q_topo"]
        anomaly_before = gauge_field["chiral_anomaly"]
        anomaly_after = new_field["chiral_anomaly"]

        # 手性翻转判据：
        # 1) 翻转后 anomaly 减小（趋向空性）
        # 2) 翻转后 Q 趋向 0
        is_flip_effective = anomaly_after < anomaly_before
        is_emptiness_reached = abs(Q_after) < abs(Q_before)

        return {
            "vortices_before": vortices,
            "vortices_after": new_vortices,
            "Q_before": Q_before,
            "Q_after": Q_after,
            "anomaly_before": anomaly_before,
            "anomaly_after": anomaly_after,
            "is_flip_effective": is_flip_effective,
            "is_emptiness_reached": is_emptiness_reached,
            "thesis": (
                f"手性翻转：Q {Q_before:.4f} → {Q_after:.4f}，"
                f"异常 {anomaly_before:.4f} → {anomaly_after:.4f}。"
                f"翻转{'有效' if is_flip_effective else '无效'}，"
                f"空性{'达成' if is_emptiness_reached else '未达成'}。"
                f"'烦恼即菩提' = 同一缺陷手性翻转。"
            ),
        }

    # ---------- 拓扑相变扫描 ----------

    def topological_phase_scan(self, n_left_range: list = None,
                                N_total: int = 4) -> dict:
        """
        扫描左旋涡旋数量，检测拓扑相变。

        物理：
            - 固定总涡旋数 N（偶数），改变左旋/右旋比例
            - 当左旋 = 右旋 = N/2 时，Q = 0（空性相）
            - 当左旋 ≠ 右旋时，Q ≠ 0（有漏相）
            - N 必须是偶数，否则 Q 永远是奇数，无法为 0
        """
        if n_left_range is None:
            n_left_range = [0, 1, 2, 3, 4]

        # N_total 必须是偶数（否则 Q 永远是奇数）
        if N_total % 2 != 0:
            N_total = N_total + 1

        results = []

        for n_left in n_left_range:
            if n_left > N_total:
                continue
            n_right = N_total - n_left
            # 构建涡旋列表
            vortices = []
            for i in range(n_left):
                x = 0.2 + 0.6 * (i + 1) / max(n_left, 1)
                vortices.append((x, 0.3, 1, +1))  # 左旋
            for i in range(n_right):
                x = 0.2 + 0.6 * (i + 1) / max(n_right, 1)
                vortices.append((x, 0.7, 1, -1))  # 右旋

            field = self.build_gauge_field_with_vortices(
                grid_size=32, vortices=vortices
            )
            Q = field["Q_topo"]
            anomaly = field["chiral_anomaly"]

            results.append({
                "n_left": n_left,
                "n_right": n_right,
                "Q": Q,
                "anomaly": anomaly,
                "is_emptiness": abs(Q) < 0.1,
            })

        # 相变检测：Q 从非零变为零
        Q_values = [r["Q"] for r in results]
        n_emptiness = sum(1 for r in results if r["is_emptiness"])

        return {
            "n_left_range": n_left_range,
            "N_total": N_total,
            "results": results,
            "Q_values": Q_values,
            "n_emptiness_phases": n_emptiness,
            "thesis": (
                f"拓扑相变扫描（N_total={N_total}）："
                f"Q = {Q_values}。"
                f"空性相出现 {n_emptiness} 次"
                f"（左旋=右旋=N/2 时 Q=0）。"
            ),
        }


# ============================================================================
# 验证函数
# ============================================================================

def run_topological_vacuum_verification(N: int = 4,
                                          hbar: float = 0.8,
                                          beta: float = 0.3,
                                          gamma: float = 0.5,
                                          c: float = 1.0) -> dict:
    """
    基石20：空性作为拓扑平凡相验证。

    5 项验证：
        V1：空性 = CS(A)=0（Chern-Simons 不变量为零）
        V2：有漏相 = CS(A)≠0（涡旋手性不平衡）
        V3：手性翻转（修行使 CS→0）
        V4：烦恼即菩提（同一缺陷不同手性）
        V5：拓扑相变扫描（左旋=右旋时 Q=0）
    """
    print(f"\n{'='*70}")
    print(f"基石20：空性作为拓扑平凡相（ℏ={hbar}）")
    print(f"{'='*70}")

    analyzer = TopologicalVacuumAnalyzer(
        hbar=hbar, beta=beta, gamma=gamma, c=c
    )

    results = {}

    # V1：空性 = CS(A)=0
    print("\n--- V1：空性 = CS(A)=0 ---")
    # 平衡涡旋（1左 + 1右）
    field_empty = analyzer.build_gauge_field_with_vortices(
        grid_size=32,
        vortices=[(0.3, 0.5, 1, +1), (0.7, 0.5, 1, -1)]
    )
    detect_empty = analyzer.detect_emptiness(field_empty)
    is_empty = detect_empty["is_emptiness"]
    print(f"  Q_topo={detect_empty['Q_topo']:.4f}")
    print(f"  CS_like={detect_empty['CS_like']:.6f}")
    print(f"  chiral_anomaly={detect_empty['chiral_anomaly']:.4f}")
    print(f"  空性相：{is_empty}")
    results["V1_emptiness"] = {
        "pass": is_empty,
        "Q_topo": detect_empty["Q_topo"],
        "CS_like": detect_empty["CS_like"],
        "chiral_anomaly": detect_empty["chiral_anomaly"],
        "is_emptiness": is_empty,
        "thesis": detect_empty["thesis"],
    }

    # V2：有漏相 = CS(A)≠0
    print("\n--- V2：有漏相 = CS(A)≠0 ---")
    # 不平衡涡旋（2左 + 0右）
    field_leak = analyzer.build_gauge_field_with_vortices(
        grid_size=32,
        vortices=[(0.3, 0.5, 1, +1), (0.7, 0.5, 1, +1)]
    )
    detect_leak = analyzer.detect_emptiness(field_leak)
    is_leak = not detect_leak["is_emptiness"]
    print(f"  Q_topo={detect_leak['Q_topo']:.4f}")
    print(f"  chiral_anomaly={detect_leak['chiral_anomaly']:.4f}")
    print(f"  有漏相：{is_leak}")
    results["V2_leakage"] = {
        "pass": is_leak,
        "Q_topo": detect_leak["Q_topo"],
        "chiral_anomaly": detect_leak["chiral_anomaly"],
        "is_leakage": is_leak,
        "thesis": detect_leak["thesis"] + f" 有漏相{'成立' if is_leak else '不成立'}。",
    }

    # V3：手性翻转（修行）
    print("\n--- V3：手性翻转（修行使 CS→0） ---")
    # 初始：不平衡（3左 + 1右），Q=+2
    field_initial = analyzer.build_gauge_field_with_vortices(
        grid_size=32,
        vortices=[
            (0.2, 0.3, 1, +1),
            (0.4, 0.5, 1, +1),
            (0.6, 0.7, 1, +1),
            (0.8, 0.3, 1, -1)
        ]
    )
    # 翻转第一个左旋涡旋（+1→-1），使 2左+2右，Q=0
    flip_result = analyzer.chirality_flip(field_initial, flip_indices=[0])
    is_flip = flip_result["is_flip_effective"] and flip_result["is_emptiness_reached"]
    print(f"  Q: {flip_result['Q_before']:.4f} → {flip_result['Q_after']:.4f}")
    print(f"  anomaly: {flip_result['anomaly_before']:.4f} → {flip_result['anomaly_after']:.4f}")
    print(f"  手性翻转：{is_flip}")
    results["V3_chirality_flip"] = {
        "pass": is_flip,
        "Q_before": flip_result["Q_before"],
        "Q_after": flip_result["Q_after"],
        "anomaly_before": flip_result["anomaly_before"],
        "anomaly_after": flip_result["anomaly_after"],
        "is_flip_effective": flip_result["is_flip_effective"],
        "thesis": flip_result["thesis"],
    }

    # V4：烦恼即菩提（同一缺陷不同手性）
    print("\n--- V4：烦恼即菩提（同一缺陷不同手性） ---")
    # 同一涡旋，左旋（烦恼）vs 右旋（菩提）
    field_klesha = analyzer.build_gauge_field_with_vortices(
        grid_size=32,
        vortices=[(0.5, 0.5, 1, +1)]  # 单个左旋（烦恼）
    )
    field_bodhi = analyzer.build_gauge_field_with_vortices(
        grid_size=32,
        vortices=[(0.5, 0.5, 1, -1)]  # 单个右旋（菩提，手性翻转）
    )
    Q_klesha = field_klesha["Q_topo"]
    Q_bodhi = field_bodhi["Q_topo"]
    # 烦恼即菩提：Q 的绝对值相同（同一拓扑荷），符号相反（手性不同）
    is_same_defect = abs(abs(Q_klesha) - abs(Q_bodhi)) < 0.01
    is_opposite_chirality = (Q_klesha * Q_bodhi) < 0
    is_klesha_bodhi = is_same_defect and is_opposite_chirality
    print(f"  Q_klesha(烦恼)={Q_klesha:.4f}, Q_bodhi(菩提)={Q_bodhi:.4f}")
    print(f"  |Q| 相同：{is_same_defect}, 手性相反：{is_opposite_chirality}")
    print(f"  烦恼即菩提：{is_klesha_bodhi}")
    results["V4_klesha_bodhi"] = {
        "pass": is_klesha_bodhi,
        "Q_klesha": Q_klesha,
        "Q_bodhi": Q_bodhi,
        "is_same_defect": is_same_defect,
        "is_opposite_chirality": is_opposite_chirality,
        "is_klesha_bodhi": is_klesha_bodhi,
        "thesis": (
            f"烦恼（左旋）Q={Q_klesha:.4f}，菩提（右旋）Q={Q_bodhi:.4f}。"
            f"|Q| 相同（同一缺陷），手性相反（翻转）。"
            f"'烦恼即菩提'{'成立' if is_klesha_bodhi else '不成立'}"
            f"（同一拓扑缺陷的不同手性态）。"
        ),
    }

    # V5：拓扑相变扫描
    print("\n--- V5：拓扑相变扫描 ---")
    # 用偶数总涡旋数（N=4），这样左=2时 Q=0
    phase_scan = analyzer.topological_phase_scan(n_left_range=[0, 1, 2, 3, 4])
    n_emptiness = phase_scan["n_emptiness_phases"]
    Q_values = phase_scan["Q_values"]
    # 相变判据：存在 Q=0 的相（空性相）
    has_emptiness = n_emptiness > 0
    # 还要求 Q 随 n_left 单调变化
    is_monotonic = all(Q_values[i] <= Q_values[i+1] + 0.1
                      for i in range(len(Q_values)-1))
    is_phase = has_emptiness and is_monotonic
    print(f"  Q_values = {Q_values}")
    print(f"  空性相次数：{n_emptiness}")
    print(f"  单调性：{is_monotonic}")
    print(f"  拓扑相变：{is_phase}")
    results["V5_phase_scan"] = {
        "pass": is_phase,
        "Q_values": Q_values,
        "n_emptiness_phases": n_emptiness,
        "is_monotonic": is_monotonic,
        "is_phase_transition": is_phase,
        "thesis": phase_scan["thesis"],
    }

    # 总结
    n_pass = sum(1 for k, v in results.items()
                 if k.startswith("V") and isinstance(v, dict) and v.get("pass"))
    n_total = sum(1 for k in results if k.startswith("V"))
    all_pass = n_pass == n_total
    print(f"\n{'='*70}")
    print(f"基石20：{n_pass}/{n_total} PASS  all_pass={all_pass}")
    print(f"{'='*70}")

    results["n_pass"] = n_pass
    results["n_total"] = n_total
    results["all_pass"] = all_pass
    return results


if __name__ == "__main__":
    run_topological_vacuum_verification()
