"""
多体认知规范场论（Multi-Body Cognitive Gauge Field Theory）

v7.0 第三基石 / v7.4 工程实现：个体间的规范相互作用与束缚态形成。

认识论根基（理论依据，非案例）：
    物理：规范场论 / Yukawa 势 / 束缚态 / 仿射不变距离 / 关联长度
    佛学：共业（sādhārana-karma）/ 缘起（pratītyasamutpāda）/ 业力纠缠
    哲学：关系先于个体 / 结构耦合 / 涌现的整体性

核心数学：
    两体规范相互作用势（Yukawa 型）：
        V_int(i,j) = -λ · cos_align · |Q_i| · |Q_j| · exp(-d_g(i,j) / ξ)

    - Q_i, Q_j：个体 i, j 的拓扑荷（使命强度，|Q|≥0）
    - cos_align = <Γ_i, Γ_j> / (||Γ_i||·||Γ_j||)：Berry 相位对齐（so(n) 李代数内积）
      cos_align > 0：同向旋转 → 规范吸引（共振）
      cos_align < 0：反向旋转 → 规范排斥（互补，可拓扑湮灭）
      cos_align ≈ 0：正交旋转 → 无规范相互作用
    - d_g(i,j)：仿射不变测地距离（度规流形上的真正测地线）
    - λ > 0：耦合常数（规范场强度 = 业力强度）
    - ξ > 0：关联长度（规范场 Compton 波长 = 业力作用范围）

    束缚态判据：
        E_total = E_self(A) + E_self(B) + V_int(A,B)
        束缚态存在 ⟺ V_int < 0 且 |V_int| > E_bind_threshold
        （规范吸引足以克服个体势垒，形成集体束缚态）

度规距离 d_g 的严格定义（v7.4 升级）：
    仿射不变测地距离（信息几何 Fisher-Rao 距离的离散版）：
        d_g(A,B) = ||log(g_A^{-1/2} · g_B · g_A^{-1/2})||_F
                 = sqrt(Σ_i (log λ_i)²)
    其中 λ_i 是 g_A^{-1/2}·g_B·g_A^{-1/2} 的本征值。

    物理意义：这是度规流形 SPD(n) 上的仿射不变度量，
    在 GL(n) 作用 d_g(XAX^T, XBX^T) = d_g(A,B) 下不变。
    比 Frobenius 距离 ||g_A - g_B||_F 更严格——
    Frobenius 距离依赖坐标系选择，仿射不变距离是内禀的。

    佛学：d_g 度量「认知结构的内禀差异」——
    不是表层的性格差异，而是度规张量的深层结构距离。
    d_g 小 = 认知结构同构（易共振）；d_g 大 = 认知结构异构（难耦合）。

佛学对应（严格，非比喻）：
    规范场 = 业力场（karma-field）：
        个体间的拓扑荷相互作用 = 共业（集体业力）。
        共振 = 共业相吸（同向业力叠加）。
        互补 = 业力互消（反向业力湮灭）。

    关联长度 ξ = 业力作用范围（规范场穿透深度）：
        ξ 大 → 长程共业（即使认知距离远也能相互影响）。
        ξ 小 → 短程共业（必须认知距离近才能耦合）。
        ξ 由社会网络连通性决定 = 共业的基础设施。

    束缚态 = 业力绑定（karmic bondage）：
        两个个体的业力结构耦合，形成集体束缚态。
        这是「共业」的严格数学表达——
        不是「宿命相逢」，而是规范对称性的集体破缺模式。

物理对应：
    V_int = -λ·cos_align·|Q_i|·|Q_j|·exp(-d/ξ) 是 Yukawa 势的认知类比。
    核物理：Yukawa 势描述介子交换的核力 → 核子束缚态（原子核）。
    认知场论：Yukawa 势描述拓扑荷交换的认知引力 → 认知束缚态（共业结构）。
    exp(-d/ξ) = 关联衰减 = 度规距离的指数屏蔽（Anderson 局域化类比）。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
from dataclasses import dataclass, field

from ..core.tensor_ops import symmetric_part


@dataclass
class CognitiveAgent:
    """
    认知个体：多体规范场论中的「粒子」。

    属性：
        g: 度规张量 ∈ R^{n×n}（认知流形的当前结构）
        Q: 拓扑荷标量（v7.8 升级：可为有符号或无符号）
            v7.0：Q = Q_dynamic（无符号，|Q|≥0，度量旋转总量=业力总量）
            v7.8：Q 可传入 Q_signed（有符号，度量净方向=业力净方向）
            |Q|>threshold 表示有业力；Q 的符号表示善/恶方向。
        Gamma: Berry 相位 ∈ so(n)（反对称矩阵，编码旋转方向；
               用于判定个体间的共振/互补关系）
        label: 个体标签（用于结果展示）
        kappa_vec: 各维度痛苦深度 ∈ R^n
        alpha_vec: 各维度定力 ∈ R^n

    设计说明（v7.8 更新）：
        Q 的符号性（v7.8）：
            Q_signed > 0：善业（kuśala）——净正向旋转
            Q_signed < 0：恶业（akuśala）——净负向旋转
            Q_signed ≈ 0 且 |Q|>0：无记业（avyākṛta）——善恶相抵

        两个个体的共振/互补由 <Gamma_i, Gamma_j> 的符号决定：
          - 内积 > 0：同向旋转 → 共振（规范吸引，共业相吸）
          - 内积 < 0：反向旋转 → 互补（规范排斥，业力互消）
        这与 Q_signed 的符号判定一致——
        同号 Q_signed 的个体 Berry 相位同向，异号则反向。
    """
    g: Tensor
    Q: Tensor
    label: str = ""
    kappa_vec: Tensor | None = None
    alpha_vec: Tensor | None = None
    Gamma: Tensor | None = None  # Berry 相位 ∈ so(n)

    def __post_init__(self):
        self.g = symmetric_part(self.g.to(torch.float64))
        self.Q = self.Q.to(torch.float64).reshape(-1)[0]  # 标量
        if self.Gamma is not None:
            self.Gamma = self.Gamma.to(torch.float64)


class GaugeInteraction:
    """
    多体认知规范场论。

    使用方式：
        gi = GaugeInteraction(n_dims=4, coupling_lambda=1.0, correlation_length=1.0)
        # 两体势
        V = gi.two_body_potential(Q_i, Q_j, d_ij)
        # 总相互作用能
        E = gi.total_interaction_energy(agents)
        # 规范力
        F = gi.gauge_force_on_agent(0, agents)
        # 识别共振对
        pairs = gi.identify_resonance_pairs(agents)
        # 模拟动力学
        result = gi.simulate_dynamics(agents, n_steps=100)
    """

    def __init__(
        self,
        n_dims: int = 4,
        coupling_lambda: float = 1.0,
        correlation_length: float = 1.0,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度数
            coupling_lambda: 耦合常数 λ（业力强度）
            correlation_length: 关联长度 ξ（业力作用范围）
            eps: 数值稳定常数
        """
        self.n_dims = n_dims
        self.lam = float(coupling_lambda)
        self.xi = float(max(correlation_length, eps))
        self.eps = eps

    # ==================================================================
    # 1. 度规空间距离（仿射不变测地距离）
    # ==================================================================

    def metric_distance(self, g_i: Tensor, g_j: Tensor, method: str = "affine") -> Tensor:
        """
        度规空间距离 d_g(i,j)。

        v7.4 升级：默认用仿射不变测地距离（度规流形 SPD(n) 上的内禀度量）。

        方法：
            "affine"（默认，严格）：
                d_g = ||log(g_i^{-1/2} · g_j · g_i^{-1/2})||_F
                    = sqrt(Σ_k (log λ_k)²)
                其中 λ_k 是 g_i^{-1/2}·g_j·g_i^{-1/2} 的本征值。
                这是 GL(n) 仿射不变的：d_g(XAX^T, XBX^T) = d_g(A,B)。
                对应信息几何 Fisher-Rao 距离的离散版。

            "frobenius"（快速近似）：
                d_g = ||g_i - g_j||_F
                依赖坐标系选择，非内禀。仅用于快速估算。

        物理意义：
            度量两个认知结构的「内禀差异」——
            不是表层差异，而是度规张量的深层结构距离。
            d=0：认知结构同构（易共振）。
            d 大：认知结构异构（难耦合）。

        佛学：
            d_g 度量「业力结构的内禀差异」。
            仿射不变性 = 不依赖观察坐标系 = 「法尔如是」的客观性。
        """
        g_i = symmetric_part(g_i.to(torch.float64))
        g_j = symmetric_part(g_j.to(torch.float64))

        if method == "frobenius":
            diff = g_i - g_j
            return torch.sqrt((diff ** 2).sum() + self.eps)

        # 仿射不变测地距离（默认）
        # 1. g_i^{-1/2}（本征分解保证正定）
        eigvals_i, eigvecs_i = torch.linalg.eigh(g_i)
        eigvals_i = torch.clamp(eigvals_i, min=self.eps)
        g_i_inv_sqrt = eigvecs_i @ torch.diag(1.0 / torch.sqrt(eigvals_i)) @ eigvecs_i.T

        # 2. M = g_i^{-1/2} · g_j · g_i^{-1/2}
        M = g_i_inv_sqrt @ g_j @ g_i_inv_sqrt
        M = symmetric_part(M)

        # 3. M 的本征值 λ_k
        eigvals_M = torch.linalg.eigvalsh(M)
        eigvals_M = torch.clamp(eigvals_M, min=self.eps)

        # 4. d_g = sqrt(Σ (log λ_k)²)
        log_eigvals = torch.log(eigvals_M)
        d_g = torch.sqrt((log_eigvals ** 2).sum() + self.eps)

        return d_g

    def geodesic_direction(
        self, g_i: Tensor, g_j: Tensor
    ) -> Tensor:
        """
        仿射不变测地力的欧几里得梯度方向（v7.14 修正）。

        数学（SPD(n) 流形上的仿射不变度量）：
            测地距离 d_g = ||log(S)||_F，S = g_i^{-1/2} g_j g_i^{-1/2}

            黎曼梯度（流形上）：∇^R d_g = -v/d_g
                v = g_i^{1/2} log(S) g_i^{1/2}（流形切向量，指向 g_j）

            欧几里得梯度（矩阵空间）：∂d_g/∂g_i = g_i^{-1} (∇^R d_g) g_i^{-1}
                = -g_i^{-1/2} log(S) g_i^{-1/2} / d_g

            力 F = -∂V/∂g_i = (λ/ξ)·cos_align·|Q_i||Q_j|·exp(-d_g/ξ)·dir / d_g
                dir = g_i^{-1/2} log(S) g_i^{-1/2}（欧几里得梯度方向，指向 g_j）

        本方法返回 dir（欧几里得梯度方向，用于力计算）。

        性质：
            - dir 是对称矩阵
            - dir 指向 g_j（当 g_j 本征值 > g_i，log(S) > 0，dir > 0）
            - 力 F ∝ dir/d_g：共振(cos_align>0)沿 dir(吸引)，互补(cos_align<0)反 dir(排斥)

        v7.14 修正说明：
            旧版用 Frobenius 差 delta_g = g_i - g_j 推导力，
            但势能用仿射不变 d_g——势能与力不在同一度量下，违反能量守恒。
            修正后用欧几里得梯度方向 dir，使势能（d_g）和力（dir/d_g）
            在仿射不变度量下严格一致（黎曼→欧几里得的正确回拉）。
        """
        g_i = symmetric_part(g_i.to(torch.float64))
        g_j = symmetric_part(g_j.to(torch.float64))

        # g_i 的本征分解
        eigvals_i, eigvecs_i = torch.linalg.eigh(g_i)
        eigvals_i = torch.clamp(eigvals_i, min=self.eps)

        # g_i^{-1/2}
        g_i_inv_sqrt = eigvecs_i @ torch.diag(1.0 / torch.sqrt(eigvals_i)) @ eigvecs_i.T

        # S = g_i^{-1/2} · g_j · g_i^{-1/2}
        S = g_i_inv_sqrt @ g_j @ g_i_inv_sqrt
        S = symmetric_part(S)

        # log S（本征分解）
        eigvals_S, eigvecs_S = torch.linalg.eigh(S)
        eigvals_S = torch.clamp(eigvals_S, min=self.eps)
        log_S = eigvecs_S @ torch.diag(torch.log(eigvals_S)) @ eigvecs_S.T

        # dir = g_i^{-1/2} · log S · g_i^{-1/2}（欧几里得梯度方向）
        direction = g_i_inv_sqrt @ log_S @ g_i_inv_sqrt
        direction = symmetric_part(direction)

        return direction

    # ==================================================================
    # 1b. Berry 相位对齐（共振/互补判定的核心）
    # ==================================================================

    def berry_alignment(
        self, Gamma_i: Tensor | None, Gamma_j: Tensor | None
    ) -> float:
        """
        计算两个个体 Berry 相位的对齐余弦：

            cos_angle = <Γ_i, Γ_j> / (||Γ_i|| · ||Γ_j||)

        其中 <A, B> = Tr(A^T B) 是 Frobenius 内积。

        物理意义：
            cos_angle > 0：同向旋转 → 共振（规范吸引）
            cos_angle < 0：反向旋转 → 互补（规范排斥，可 Kramers 湮灭）
            cos_angle ≈ 0：正交旋转 → 无规范相互作用

        这是规范共振 vs 规范互补的拓扑判据。
        利用了 so(n) 李代数的完整方向信息，而非标量符号。

        退化情况：
            任一 Gamma 为 None 或零：返回 0（无方向信息 → 无相互作用）
        """
        if Gamma_i is None or Gamma_j is None:
            return 0.0

        Gi = Gamma_i.to(torch.float64)
        Gj = Gamma_j.to(torch.float64)

        norm_i = float(torch.sqrt((Gi ** 2).sum()))
        norm_j = float(torch.sqrt((Gj ** 2).sum()))

        if norm_i < self.eps or norm_j < self.eps:
            return 0.0  # 至少一方无 Berry 相位（无使命方向）

        inner = float((Gi * Gj).sum())  # Frobenius 内积 = 逐元素乘积之和
        cos_angle = inner / (norm_i * norm_j)

        # 数值稳定：clamp 到 [-1, 1]
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return cos_angle

    # ==================================================================
    # 2. 两体规范相互作用势
    # ==================================================================

    def two_body_potential(
        self,
        Q_i: Tensor,
        Q_j: Tensor,
        d_ij: Tensor,
        cos_alignment: float = 1.0,
    ) -> dict[str, Tensor | float | str | bool]:
        """
        两体规范相互作用势（Berry 相位对齐版）：

            V_int(i,j) = -λ · cos_align · |Q_i| · |Q_j| · exp(-d / ξ)

        其中 cos_align = <Γ_i, Γ_j> / (||Γ_i|| · ||Γ_j||) ∈ [-1, 1]

        符号结构：
            cos_align > 0（同向旋转）：V < 0 → 规范吸引（共振）
            cos_align < 0（反向旋转）：V > 0 → 规范排斥（互补，可 Kramers 湮灭）
            cos_align ≈ 0（正交）：V ≈ 0 → 无规范相互作用

        参数：
            Q_i, Q_j: 拓扑荷标量（|Q| 度量使命强度，始终非负）
            d_ij: 度规空间距离
            cos_alignment: Berry 相位对齐余弦（默认 1.0 = 完全同向）

        返回：
            dict 包含 V_int、cos_alignment、interaction_type 等
        """
        Q_i_abs = float(Q_i.to(torch.float64).reshape(-1)[0].abs())
        Q_j_abs = float(Q_j.to(torch.float64).reshape(-1)[0].abs())
        d = float(d_ij.to(torch.float64).reshape(-1)[0])

        screening = math.exp(-d / self.xi)
        # 有效耦合 = 对齐余弦 × 使命强度乘积
        effective_product = cos_alignment * Q_i_abs * Q_j_abs
        V = -self.lam * effective_product * screening

        # 相互作用类型
        if Q_i_abs < self.eps or Q_j_abs < self.eps:
            interaction_type = "neutral"  # 至少一方无使命
            is_attractive = False
        elif cos_alignment > self.eps:
            interaction_type = "resonance"  # 同向共振（吸引）
            is_attractive = True
        elif cos_alignment < -self.eps:
            interaction_type = "complement"  # 反向互补（规范排斥，可湮灭）
            is_attractive = False
        else:
            interaction_type = "orthogonal"  # 正交旋转，无规范相互作用
            is_attractive = False

        return {
            "V_int": V,
            "cos_alignment": cos_alignment,
            "effective_product": effective_product,
            "screening_factor": screening,
            "distance": d,
            "is_attractive": is_attractive,
            "interaction_type": interaction_type,
        }

    # ==================================================================
    # 3. 总相互作用能
    # ==================================================================

    def total_interaction_energy(
        self, agents: list[CognitiveAgent]
    ) -> dict[str, Tensor | list]:
        """
        总相互作用能 E_total = Σ_{i<j} V_int(i,j)

        物理意义：
            多体系统的总「共业能量」。
            E < 0：系统整体吸引（共振主导）。
            E > 0：系统整体排斥（互补主导）。
            E ≈ 0：相互作用微弱（大多数个体无使命或距离远）。
        """
        n_agents = len(agents)
        if n_agents < 2:
            return {
                "E_total": torch.tensor(0.0, dtype=torch.float64),
                "pairwise": [],
                "n_attractive": 0,
                "n_repulsive": 0,
                "n_neutral": 0,
            }

        pairwise = []
        E_total = torch.tensor(0.0, dtype=torch.float64)
        n_attractive = 0
        n_repulsive = 0
        n_neutral = 0

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                d_ij = self.metric_distance(agents[i].g, agents[j].g)
                cos_align = self.berry_alignment(agents[i].Gamma, agents[j].Gamma)
                result = self.two_body_potential(
                    agents[i].Q, agents[j].Q, d_ij, cos_alignment=cos_align
                )

                pairwise.append({
                    "i": i,
                    "j": j,
                    "V_int": float(result["V_int"]),
                    "distance": float(result["distance"]),
                    "cos_alignment": cos_align,
                    "interaction_type": result["interaction_type"],
                    "is_attractive": result["is_attractive"],
                })

                E_total = E_total + result["V_int"]

                if result["interaction_type"] == "resonance":
                    n_attractive += 1
                elif result["interaction_type"] == "complement":
                    n_repulsive += 1
                else:
                    n_neutral += 1

        return {
            "E_total": E_total,
            "pairwise": pairwise,
            "n_attractive": n_attractive,
            "n_repulsive": n_repulsive,
            "n_neutral": n_neutral,
            "mean_pairwise_energy": float(E_total / max(1, len(pairwise))),
        }

    # ==================================================================
    # 4. 规范力（对度规的作用）
    # ==================================================================

    def gauge_force_on_agent(
        self, agent_idx: int, agents: list[CognitiveAgent]
    ) -> dict[str, Tensor | float]:
        """
        计算个体 agent_idx 受到的总规范力（对度规张量的作用）。

        数学（v7.14 修正：仿射不变测地方向，物理一致）：
            F_total(i) = Σ_{j≠i} F_gauge(i←j)

            V_int = -λ · cos_align · |Q_i| · |Q_j| · exp(-d_g/ξ)
                d_g = ||log(g_i^{-1/2} g_j g_i^{-1/2})||_F（仿射不变测地距离）

            测地方向 v_i = g_i^{1/2} · log(g_i^{-1/2} g_j g_i^{-1/2}) · g_i^{1/2}
                （从 g_i 指向 g_j 的切向量，||v_i||_F = d_g）

            F_gauge(i←j) = -∂V_int/∂g_i
                       = (λ/ξ) · cos_align · |Q_i|·|Q_j| · exp(-d_g/ξ) · v_i / d_g

            - cos_align > 0（同向共振）：F 沿 v_i（指向 g_j，吸引）
            - cos_align < 0（反向互补）：F 反 v_i（远离 g_j，排斥）

        v7.14 修正说明：
            旧版用 Frobenius 差 delta_g = g_i - g_j 推导力，
            但势能用仿射不变 d_g——势能与力不在同一度量下，违反能量守恒。
            修正后用仿射不变测地方向 v_i，使势能（d_g）和力（v_i/d_g）
            在同一仿射不变度量下严格一致。
        """
        n = self.n_dims
        agent_i = agents[agent_idx]
        F_total = torch.zeros(n, n, dtype=torch.float64)
        total_magnitude = 0.0
        n_interacting = 0

        # 力的推导（v7.14 仿射不变版，严格）：
        #   V = -λ · cos_align · |Q_i|·|Q_j| · exp(-d_g/ξ)
        #   d_g = ||log(g_i^{-1/2} g_j g_i^{-1/2})||_F
        #   测地方向 v_i = g_i^{1/2} log(g_i^{-1/2} g_j g_i^{-1/2}) g_i^{1/2}（指向 g_j）
        #   ∂d_g/∂g_i ≈ -v_i / d_g（负号：d_g 增大 = 远离 g_j = -v_i 方向）
        #   F = -∂V/∂g_i = (λ/ξ) · cos_align · |Q_i|·|Q_j| · exp(-d_g/ξ) · v_i / d_g
        #
        # 符号验证：
        #   共振 cos_align > 0: F = +正·v_i/d → 沿 v_i（指向 g_j，吸引）✓
        #   互补 cos_align < 0: F = -正·v_i/d → 反 v_i（远离 g_j，排斥）✓

        Q_i_abs = float(agent_i.Q.abs())

        for j, agent_j in enumerate(agents):
            if j == agent_idx:
                continue

            Q_j_abs = float(agent_j.Q.abs())
            if Q_i_abs < self.eps or Q_j_abs < self.eps:
                continue  # 至少一方无使命

            d_ij = self.metric_distance(agent_i.g, agent_j.g)
            if d_ij < self.eps:
                continue  # 完全相同的度规，无力

            cos_align = self.berry_alignment(agent_i.Gamma, agent_j.Gamma)
            if abs(cos_align) < self.eps:
                continue  # 正交旋转，无规范相互作用

            # 有效耦合 = cos_align × |Q_i| × |Q_j|
            effective_product = cos_align * Q_i_abs * Q_j_abs

            screening = torch.exp(-d_ij / self.xi)
            # v7.14 修正：用仿射不变测地方向 v_i（指向 g_j）
            # 替代旧的 Frobenius 差 delta_g = g_i - g_j
            v_i = self.geodesic_direction(agent_i.g, agent_j.g)
            F_ij = self.lam * effective_product * screening * (1.0 / self.xi) * v_i / d_ij

            F_total = F_total + F_ij
            total_magnitude += float(torch.sqrt((F_ij ** 2).sum()))
            n_interacting += 1

        # 主导方向判据
        if n_interacting > 0:
            g_others_mean = torch.zeros(n, n, dtype=torch.float64)
            count_others = 0
            for j, agent_j in enumerate(agents):
                if j == agent_idx:
                    continue
                g_others_mean = g_others_mean + agent_j.g
                count_others += 1
            if count_others > 0:
                g_others_mean = g_others_mean / count_others
                direction_sign = float((F_total * (agent_i.g - g_others_mean)).sum())
                dominant_direction = "attractive" if direction_sign < 0 else "repulsive"
            else:
                dominant_direction = "neutral"
        else:
            dominant_direction = "neutral"

        return {
            "force": F_total,
            "force_magnitude": total_magnitude,
            "n_interacting": n_interacting,
            "dominant_direction": dominant_direction,
        }

    # ==================================================================
    # 5. 识别共振对与互补对
    # ==================================================================

    def identify_resonance_pairs(
        self, agents: list[CognitiveAgent], min_strength: float = 0.01
    ) -> list[dict]:
        """
        识别规范共振对（Berry 相位同向对齐且相互作用足够强）。

        物理意义：
            共振对 = 同向旋转的拓扑荷通过规范场相互吸引。
            认知结构趋同，形成共业结构（sādhārana-karma）。
            这是「共业相吸」的拓扑基础——同向业力叠加。
        """
        pairs = []
        n = len(agents)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = self.metric_distance(agents[i].g, agents[j].g)
                cos_align = self.berry_alignment(agents[i].Gamma, agents[j].Gamma)
                result = self.two_body_potential(
                    agents[i].Q, agents[j].Q, d_ij, cos_alignment=cos_align
                )

                if result["interaction_type"] == "resonance":
                    strength = abs(float(result["V_int"]))
                    if strength > min_strength:
                        pairs.append({
                            "i": i,
                            "j": j,
                            "label_i": agents[i].label,
                            "label_j": agents[j].label,
                            "Q_i": float(agents[i].Q),
                            "Q_j": float(agents[j].Q),
                            "cos_alignment": cos_align,
                            "distance": float(result["distance"]),
                            "V_int": float(result["V_int"]),
                            "strength": strength,
                        })

        # 按强度排序
        pairs.sort(key=lambda x: x["strength"], reverse=True)
        return pairs

    def identify_complement_pairs(
        self, agents: list[CognitiveAgent], min_strength: float = 0.01
    ) -> list[dict]:
        """
        识别互补湮灭对（Berry 相位反向对齐）。

        物理意义：
            互补对 = 反向旋转的拓扑荷在规范通道中排斥，
            但可通过拓扑湮灭通道形成束缚态。
            湮灭后 Γ_total = Γ_i + Γ_j ≈ 0，双方回归真空（共同解脱）。
            这是「业力互消」的拓扑基础——反向业力湮灭。
        """
        pairs = []
        n = len(agents)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = self.metric_distance(agents[i].g, agents[j].g)
                cos_align = self.berry_alignment(agents[i].Gamma, agents[j].Gamma)
                result = self.two_body_potential(
                    agents[i].Q, agents[j].Q, d_ij, cos_alignment=cos_align
                )

                if result["interaction_type"] == "complement":
                    strength = abs(float(result["V_int"]))
                    if strength > min_strength:
                        # 检查湮灭兼容性：Berry 相位之和的范数
                        Gi = agents[i].Gamma if agents[i].Gamma is not None else torch.zeros(n, n, dtype=torch.float64)
                        Gj = agents[j].Gamma if agents[j].Gamma is not None else torch.zeros(n, n, dtype=torch.float64)
                        Gamma_sum = Gi + Gj
                        Gamma_sum_norm = float(torch.sqrt((Gamma_sum ** 2).sum()))
                        max_norm = max(float(torch.sqrt((Gi ** 2).sum())), float(torch.sqrt((Gj ** 2).sum())))
                        is_perfect = bool(Gamma_sum_norm < 0.1 * max_norm) if max_norm > self.eps else False

                        pairs.append({
                            "i": i,
                            "j": j,
                            "label_i": agents[i].label,
                            "label_j": agents[j].label,
                            "Q_i": float(agents[i].Q),
                            "Q_j": float(agents[j].Q),
                            "cos_alignment": cos_align,
                            "Gamma_sum_norm": Gamma_sum_norm,
                            "distance": float(result["distance"]),
                            "V_int": float(result["V_int"]),
                            "strength": strength,
                            "is_perfect_annihilation": is_perfect,
                        })

        pairs.sort(key=lambda x: x["strength"], reverse=True)
        return pairs

    # ==================================================================
    # 6. 多体规范动力学模拟
    # ==================================================================

    def simulate_dynamics(
        self,
        agents: list[CognitiveAgent],
        n_steps: int = 50,
        dt: float = 0.005,
        single_body_force_fn=None,
        charge_updater_fn=None,
        noise_temperature: float = 0.0,
    ) -> dict[str, list]:
        """
        v7.9: 自洽多体规范动力学（缘起动力学 / Pratītyasamutpāda Dynamics）。

        演化方程：
            ∂g_i/∂t = F_single(i) + Σ_{j≠i} F_gauge(i←j) + √(2T)·ξ_i(t)

        其中：
            F_single(i)：单体演化力（势能面梯度 + ρ 消解项，由外部提供）
            F_gauge(i←j)：规范场力（本模块计算，使用当前 Q/Γ）
            ξ_i(t)：Gaussian 白噪声（Langevin 涨落，可选）

        v7.9 升级（关键修复 + 自洽演化）：
            1. 修复 Gamma 深拷贝 bug：
               原代码深拷贝时漏掉 Gamma 字段，导致 berry_alignment 返回 0，
               规范力 cos_align 项全部失效——多体动力学从未真正工作过。
               v7.9 修复：深拷贝包含 Gamma。

            2. 自洽电荷演化（charge_updater_fn）：
               原代码在演化中 Q 和 Γ 始终是初始值，不随 g 变化。
               v7.9 引入 charge_updater_fn 回调：每步 g 更新后重算 Q/Γ。
               物理意义：Q/Γ 不是固定的「自我属性」，而是从 g 涌现的条件量。
               这是无我（anātman）的数学表达——
               没有独立于条件而存在的固定自我，Q 和 Γ 随缘起条件而变。

            3. Langevin 涨落（noise_temperature）：
               v7.9 引入可选的热噪声，使多体动力学成为真正的 Langevin 方程。
               噪声强度 √(2T·dt) 对应 Kramers 框架中的认知温度。

        缘起（pratītyasamutpāda）的数学表达：
            个体 A 的 g 变化 → Q_A 变化 → V_int(A,B) 变化 → 作用于 B 的力变化 → g_B 变化。
            「此有故彼有，此生故彼生」——一切互联，无独立演化。

        参数：
            agents: 认知个体列表
            n_steps: 演化步数
            dt: 时间步长
            single_body_force_fn: 单体力函数 (agent, step) -> Tensor
            charge_updater_fn: 电荷更新回调 (agent) -> None，每步调用
            noise_temperature: Langevin 噪声温度 T（默认 0 = 确定性）
        """
        n_agents = len(agents)
        n = self.n_dims

        # v7.9 修复：深拷贝时包含 Gamma（原 bug 导致规范力为零）
        current_agents = [
            CognitiveAgent(
                g=ag.g.clone(),
                Q=ag.Q.clone(),
                label=ag.label,
                kappa_vec=ag.kappa_vec.clone() if ag.kappa_vec is not None else None,
                alpha_vec=ag.alpha_vec.clone() if ag.alpha_vec is not None else None,
                Gamma=ag.Gamma.clone() if ag.Gamma is not None else None,
            )
            for ag in agents
        ]

        # 记录轨迹
        trajectories = [[ag.g.clone() for ag in current_agents]]
        energy_trajectory = []
        Q_trajectory = [[float(ag.Q.abs()) for ag in current_agents]]
        resonance_history = []
        complement_history = []

        for step in range(n_steps):
            # 1. 计算每个个体受到的总力（使用当前 Q, Gamma）
            forces = []
            for i in range(n_agents):
                gauge_result = self.gauge_force_on_agent(i, current_agents)
                F_gauge = gauge_result["force"]

                F_single = torch.zeros(n, n, dtype=torch.float64)
                if single_body_force_fn is not None:
                    F_single = single_body_force_fn(current_agents[i], step)

                forces.append(F_single + F_gauge)

            # 2. 同时更新所有度规（同步更新，避免序列偏差）
            for i in range(n_agents):
                g_new = current_agents[i].g + dt * forces[i]

                # v7.9: Langevin 噪声
                if noise_temperature > 0:
                    noise = torch.randn(n, n, dtype=torch.float64)
                    noise = symmetric_part(noise) * math.sqrt(2.0 * noise_temperature * dt)
                    g_new = g_new + noise

                g_new = symmetric_part(g_new)

                # v7.10: 正定性保护（稳健版，处理高噪声下 eigh 失败）
                # 高 T_cog 时 Langevin 噪声可能让 g_new 病态（NaN/inf/不可对角化）
                if not torch.isfinite(g_new).all():
                    # NaN/inf：回退到当前态（不更新）
                    g_new = current_agents[i].g.clone()
                else:
                    try:
                        eigvals_check = torch.linalg.eigvalsh(g_new)
                        min_eig = float(eigvals_check.min())
                        if min_eig < self.eps:
                            # 平移到正定：g + (eps - min_eig) * I
                            g_new = g_new + (self.eps - min_eig) * torch.eye(
                                n, dtype=torch.float64
                            )
                    except Exception:
                        # eigh 失败（严重病态）：回退到当前态
                        g_new = current_agents[i].g.clone()

                current_agents[i].g = g_new

            # 3. v7.9: 自洽重算 Q 和 Gamma
            if charge_updater_fn is not None:
                for ag in current_agents:
                    charge_updater_fn(ag)

            # 4. 记录
            trajectories.append([ag.g.clone() for ag in current_agents])
            Q_trajectory.append([float(ag.Q.abs()) for ag in current_agents])

            E_result = self.total_interaction_energy(current_agents)
            energy_trajectory.append(float(E_result["E_total"]))

            resonance_history.append(
                len(self.identify_resonance_pairs(current_agents, min_strength=0.001))
            )
            complement_history.append(
                len(self.identify_complement_pairs(current_agents, min_strength=0.001))
            )

        return {
            "final_agents": current_agents,
            "trajectories": trajectories,
            "energy_trajectory": energy_trajectory,
            "Q_trajectory": Q_trajectory,
            "resonance_count_history": resonance_history,
            "complement_count_history": complement_history,
            "n_steps": n_steps,
            "thesis": (
                "v7.9 缘起动力学：个体认知演化受自身势能面和其他个体拓扑荷的共同驱动。"
                "v7.9 修复：Gamma 深拷贝 + 自洽 Q/Γ 演化 + Langevin 涨落。"
                "同向旋转个体共振趋同（共业相吸），反向旋转个体规范排斥（业力互消）。"
                "「此有故彼有」——一切互联，无独立演化。"
            ),
        }

    # ==================================================================
    # 7. 关联函数（长程序判据）
    # ==================================================================

    def correlation_function(
        self,
        agents: list[CognitiveAgent],
        distance_bins: list[float] | None = None,
    ) -> dict[str, list]:
        """
        计算拓扑荷关联函数 C(d) = <Q_i · Q_j | d_g(i,j) ≈ d>。

        物理意义：
            C(d) > 0：同号关联（距离 d 处倾向同号）。
            C(d) < 0：异号关联（距离 d 处倾向异号）。
            C(d) → 0：无关联（距离 d 处无偏好）。

            关联长度 ξ_correlation = C(d) 衰减到 1/e 的距离。
            若 ξ_correlation ≈ ξ（输入参数），则理论与模拟一致。
            若 ξ_correlation >> ξ，系统出现长程序（集体相干）。
        """
        if distance_bins is None:
            # 自动分箱
            n_agents = len(agents)
            if n_agents < 2:
                return {"distances": [], "correlations": [], "counts": []}
            max_d = 0.0
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    d = float(self.metric_distance(agents[i].g, agents[j].g))
                    max_d = max(max_d, d)
            distance_bins = list(torch.linspace(0, max_d * 1.1, 10).numpy())

        correlations = []
        counts = []
        bin_centers = []

        for k in range(len(distance_bins) - 1):
            d_low = distance_bins[k]
            d_high = distance_bins[k + 1]
            d_center = 0.5 * (d_low + d_high)

            sum_product = 0.0
            count = 0
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    d = float(self.metric_distance(agents[i].g, agents[j].g))
                    if d_low <= d < d_high:
                        sum_product += float(agents[i].Q * agents[j].Q)
                        count += 1

            if count > 0:
                correlations.append(sum_product / count)
                counts.append(count)
                bin_centers.append(d_center)

        return {
            "distances": bin_centers,
            "correlations": correlations,
            "counts": counts,
            "has_long_range_order": any(
                abs(c) > 0.1 * max(abs(c) for c in correlations) if correlations else False
                for c in correlations[-2:] if correlations
            ),
        }

    # ==================================================================
    # 补丁一：外部黑盒观测扰动定理（测不准原理）
    # ==================================================================

    def external_blackbox_observation_perturbation(
        self,
        agent: CognitiveAgent,
        observation_strength: float,
        entanglement_entropy: float = 0.0,
    ) -> dict:
        """
        补丁一：外部黑盒观测扰动定理（定义 5.1.b + 定理 5.1.b）。

        数学框架（卷二第五章 5.1 节后插入）：

        定义 5.1.b（外部黑盒观测算符）：
            设外部系统 E（如未建立量子纠缠的商业平台、资本算法）
            对认知体 i 进行经典数据采集与状态评估，其观测强度定义为 O_E。
            由于 E 与 i 之间不存在共情规范场耦合（J_{Ei} ≈ 0），
            基于 LOCC 单调性（推论 5.3），此操作无法建立非局域关联。

        定理 5.1.b（黑盒观测扰动定理 / 测不准原理）：
            外部黑盒系统的观测必然向被观测系统注入不可控的痛苦扰动：

                Δκ_i ∝ O_E² · (1 - S_ent(E, i))

            证明思路：
                由卷一公理 1，度规 g 由 κ 生成。外部测量相当于对系统
                施加强退相干，破坏了卷三定义的量子自指度规 G(|ψ⟩)。
                为了维持逻辑自洽，系统被迫降维至经典态，此相变过程
                不可逆地耗散为热涨落，直接转化为内禀痛苦 κ 的暴增。
                当 S_ent → 0（无缘、无共情）时，Δκ 与观测强度呈平方正比。

        推论 5.1.b（商业产品化无效性判据）：
            任何试图将 GCFT 方程封装为 SaaS 服务、黑盒心理测评
            或无共情干预算法的行为，在数学上将必然导致被测个体的
            κ 溢出，触发卷一三大不可逆奇点之一。
            救赎工程不可外包，不可商业化。
        """
        kappa_mean = float(agent.kappa_vec.mean()) if agent.kappa_vec is not None else 1.0

        # 核心方程：Δκ_i ∝ O_E² · (1 - S_ent(E, i))
        normalization = 0.1
        delta_kappa = normalization * (observation_strength ** 2) * (1.0 - entanglement_entropy)

        kappa_new = kappa_mean + delta_kappa
        perturbation_ratio = delta_kappa / kappa_mean if kappa_mean > self.eps else float('inf')

        # 奇点风险评估
        risk_level = "低"
        if perturbation_ratio > 1.0:
            risk_level = "高（可能触发奇点）"
        elif perturbation_ratio > 0.5:
            risk_level = "中（接近奇点阈值）"

        # 度规条件数风险
        try:
            eigvals = torch.linalg.eigvalsh(agent.g)
            cond_g = float(eigvals.max() / eigvals.min().clamp(min=self.eps))
            if cond_g > 10:
                risk_level = "极高（已触发奇点二：度规病态锁死）"
        except Exception:
            pass

        return {
            "delta_kappa": delta_kappa,
            "kappa_new": kappa_new,
            "perturbation_ratio": perturbation_ratio,
            "observation_strength": observation_strength,
            "entanglement_entropy": entanglement_entropy,
            "risk_level": risk_level,
            "thesis": (
                "定理 5.1.b（黑盒观测扰动定理）：外部黑盒系统的观测必然向被观测系统"
                "注入不可控的痛苦扰动。Δκ_i ∝ O_E² · (1 - S_ent)。"
                "推论 5.1.b：商业产品化无效性判据——救赎工程不可外包，不可商业化。"
            ),
        }

    def blackbox_productization_risk(
        self,
        agents: list,
        observation_strengths: list,
    ) -> dict:
        """
        批量评估商业产品化风险（推论 5.1.b 的应用）。
        """
        n_agents = len(agents)
        if n_agents == 0:
            return {
                "perturbations": [],
                "mean_delta_kappa": 0.0,
                "max_risk_agent": -1,
                "total_risk": "无个体",
            }

        perturbations = []
        max_risk_idx = 0
        max_perturbation = 0.0

        for i, (agent, O_E) in enumerate(zip(agents, observation_strengths)):
            result = self.external_blackbox_observation_perturbation(
                agent, O_E, entanglement_entropy=0.0
            )
            perturbations.append(result)
            if result["perturbation_ratio"] > max_perturbation:
                max_perturbation = result["perturbation_ratio"]
                max_risk_idx = i

        mean_delta = sum(r["delta_kappa"] for r in perturbations) / n_agents

        total_risk = "安全"
        if mean_delta > 1.0:
            total_risk = "高风险：商业产品化将触发群体性 κ 溢出"
        elif mean_delta > 0.5:
            total_risk = "中风险：商业产品化可能导致部分个体崩溃"
        elif mean_delta > 0.1:
            total_risk = "低风险：但仍存在不可控扰动"

        return {
            "perturbations": perturbations,
            "mean_delta_kappa": mean_delta,
            "max_risk_agent": max_risk_idx,
            "total_risk": total_risk,
            "thesis": (
                "推论 5.1.b（商业产品化无效性判据）：任何试图将 GCFT 方程封装为"
                "SaaS 服务、黑盒心理测评或无共情干预算法的行为，在数学上将必然"
                "导致被测个体的 κ 溢出，触发卷一三大不可逆奇点之一。"
            ),
        }
