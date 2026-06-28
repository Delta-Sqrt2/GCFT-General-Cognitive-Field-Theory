"""
任务一：多体事件图耦合与定向熵流

战略定位（v2.1 任务一）：
    从"鲁滨逊漂流记"（单体宇宙）进化为"社会宇宙"（多体耦合）。
    社会不是个体的集合，而是事件关系的网络。
    人际关系的本质是自由能的定向流动。

物理与哲学直觉：
    - 物理：社会是巨大的能量交换网络。剥削（薅羊毛）是热力学定律——
            高熵系统（混乱者）必然从低熵系统（有序者）吸取负熵。
    - 哲学：剥削不是道德概念，是能量流散度的符号。
    - 工程：全局事件图 + 跨图因果耦合，全张量运算。

数学定义（严格可微，无降级）：
    步骤 A：全局事件图
        废除"个体流形"概念。全局事件节点 E = {e_1, e_2, ...}，
        每个节点携带 owner_id（归属于谁）。
        全局因果邻接 C_global 包含内部边和跨图边。

    步骤 B：跨图因果耦合 J
        J_mn^{AB} = sim(e_m^A, e_n^B) · exp(-Δt/τ)
        描述 A 的事件 m 对 B 的事件 n 的因果压力。

    步骤 C：定向熵流散度
        ∇·J_A = Σ_{B≠A} Σ_{m∈A, n∈B} J_mn^{AB} · (H(e_m) - H(e_n))
        ∇·J_A < 0：A 在被吸血（熵增危机）
        ∇·J_A > 0：A 在吸取能量（寄生成功）

工程铁律（v2.1 专属）：
    1. 严禁 individual_manifold / tensor_product（预设流形复辟）
    2. 严禁 torch.gradient / 连续导数（连续场论复辟）
    3. 严禁向量平均替代微观流动（算力爆炸降级）
    4. 严禁标量规范场（规范场标量化）
    5. 全部张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor


class MultiBodyCoupling:
    """
    多体事件图耦合：全局事件图 + 跨图因果耦合 + 定向熵流。

    使用方式：
        coupling = MultiBodyCoupling(tau_causal=2.0)
        # 添加多个个体的事件
        coupling.add_event(feature=..., owner_id="A", timestamp=0.0)
        coupling.add_event(feature=..., owner_id="B", timestamp=1.0)
        # 计算定向熵流
        divergence = coupling.entropy_flow_divergence()
        # divergence["A"] < 0: A 在被吸血
        # divergence["B"] > 0: B 在吸取能量

    白盒保证：
        - 无个体流形（全局事件图）
        - 无连续导数（离散求和）
        - 无向量平均（每条跨图边微观流动求和）
        - 全张量运算，可微
    """

    def __init__(self, tau_causal: float = 2.0, eps: float = 1e-12):
        """
        参数：
            tau_causal: 因果时间核衰减常数
            eps: 数值稳定常数
        """
        self.tau_causal = float(tau_causal)
        self.eps = float(eps)

        # 全局事件存储
        self._features: list[Tensor] = []
        self._owners: list[str] = []
        self._timestamps: list[float] = []
        self._force_types: list[str] = []

        # 缓存
        self._feature_matrix_cache: Tensor | None = None
        self._owner_mask_cache: dict[str, Tensor] | None = None
        self._global_causal_cache: Tensor | None = None
        self._cross_graph_coupling_cache: Tensor | None = None

    def add_event(
        self,
        feature: Tensor,
        owner_id: str,
        timestamp: float,
        force_type: str = "UNKNOWN",
    ) -> None:
        """
        添加事件到全局事件图。

        参数：
            feature: 事件特征向量 φ_i ∈ R^d
            owner_id: 归属个体标识（如 "A", "B"）
            timestamp: 事件时间
            force_type: 力类型
        """
        self._features.append(feature.to(torch.float64))
        self._owners.append(owner_id)
        self._timestamps.append(float(timestamp))
        self._force_types.append(force_type)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._feature_matrix_cache = None
        self._owner_mask_cache = None
        self._global_causal_cache = None
        self._cross_graph_coupling_cache = None

    @property
    def n_events(self) -> int:
        return len(self._features)

    @property
    def owners(self) -> list[str]:
        return list(dict.fromkeys(self._owners))  # 保序去重

    def feature_matrix(self) -> Tensor:
        """全局特征矩阵 Φ ∈ R^{N×d}。"""
        if self._feature_matrix_cache is None and self._features:
            self._feature_matrix_cache = torch.stack(self._features, dim=0)
        return self._feature_matrix_cache if self._feature_matrix_cache is not None else torch.zeros(0, 0, dtype=torch.float64)

    def owner_masks(self) -> dict[str, Tensor]:
        """
        个体归属掩码：mask_A[i] = 1 if event i belongs to A.

        返回 dict[owner_id, Tensor(N,)]。
        """
        if self._owner_mask_cache is None:
            N = self.n_events
            masks = {owner: torch.zeros(N, dtype=torch.float64) for owner in self.owners}
            for i, owner in enumerate(self._owners):
                masks[owner][i] = 1.0
            self._owner_mask_cache = masks
        return self._owner_mask_cache

    def global_causal_adjacency(self) -> Tensor:
        """
        全局因果邻接矩阵 C_global ∈ R^{N×N}。

        包含内部边（自己的事件）和跨图边（人际因果）。
        C_ij = sim(φ_i, φ_j) · exp(-|Δt|/τ) · sigmoid(Δt/τ_soft)

        全张量运算（无 for 循环）。
        """
        if self._global_causal_cache is not None:
            return self._global_causal_cache

        Phi = self.feature_matrix()
        if Phi.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        N = Phi.shape[0]

        # 语义相似度（余弦）
        norms = Phi.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        Phi_norm = Phi / norms
        S = Phi_norm @ Phi_norm.transpose(-1, -2)
        S = torch.clamp(S, min=-1.0 + self.eps, max=1.0 - self.eps)

        # 时间核
        t = torch.tensor(self._timestamps, dtype=torch.float64)
        DeltaT = t.unsqueeze(0) - t.unsqueeze(1)  # ΔT_ij = t_j - t_i
        tau_soft = self.tau_causal * 0.1
        H_soft = torch.sigmoid(DeltaT / tau_soft)
        K = torch.exp(-DeltaT.abs() / self.tau_causal) * H_soft

        # 全局因果邻接
        C = S * K
        # 对角线置零
        C = C * (1.0 - torch.eye(N, dtype=torch.float64))

        self._global_causal_cache = C
        return C

    def cross_graph_coupling(self) -> Tensor:
        """
        跨图因果耦合矩阵 J ∈ R^{N×N}。

        J_mn^{AB} = sim(e_m^A, e_n^B) · exp(-Δt/τ)
        仅保留跨个体边（owner 不同）。

        全张量运算（无 for 循环）：
            构建跨个体掩码 cross_mask[i,j] = 1 if owner_i != owner_j
            J = C_global * cross_mask
        """
        if self._cross_graph_coupling_cache is not None:
            return self._cross_graph_coupling_cache

        C = self.global_causal_adjacency()
        if C.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        N = self.n_events
        # 跨个体掩码（全张量运算）
        owner_idx = torch.tensor([hash(o) % (2**31) for o in self._owners], dtype=torch.float64)
        # owner_i != owner_j 的掩码
        cross_mask = (owner_idx.unsqueeze(0) != owner_idx.unsqueeze(1)).to(torch.float64)

        # 跨图耦合 = 全局因果 × 跨个体掩码
        J = C * cross_mask
        self._cross_graph_coupling_cache = J
        return J

    def event_entropy(self) -> Tensor:
        """
        事件信息熵 H(e_i) = -Σ_d p_d · log(p_d)。
        复用 v2.0 DiscreteAction 的熵计算。
        """
        Phi = self.feature_matrix()
        if Phi.shape[0] == 0:
            return torch.zeros(0, dtype=torch.float64)
        Phi_abs = Phi.abs()
        total = Phi_abs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = Phi_abs / total
        H = -(p * torch.log(p.clamp(min=self.eps))).sum(dim=-1)
        return H

    def entropy_flow_divergence(self) -> dict[str, Tensor]:
        """
        定向熵流散度 ∇·J_A（核心算符）。

        ∇·J_A = Σ_{B≠A} Σ_{m∈A, n∈B} J_mn^{AB} · (H(e_m) - H(e_n))

        判据：
            ∇·J_A < 0：A 在被吸血（熵增危机，能量流出）
            ∇·J_A > 0：A 在吸取能量（寄生成功，能量流入）

        物理意义：
            这是"剥削"的涌现定义——非道德概念，是热力学定律。
            高熵系统（混乱者）从低熵系统（有序者）吸取负熵。

        全张量运算（无 for 循环，无向量平均）：
            1. 熵差矩阵 ΔH_mn = H_m - H_n
            2. 微观流动矩阵 flow_mn = J_mn · ΔH_mn
            3. 个体散度 = Σ_{m∈A, n∉A} flow_mn（按 owner 掩码求和）

        严禁：
            - 向量平均替代微观流动（算力爆炸降级）
            - 连续导数（连续场论复辟）
            - 个体流形（预设流形复辟）
        """
        J = self.cross_graph_coupling()
        H = self.event_entropy()

        if J.shape[0] == 0:
            return {owner: torch.tensor(0.0, dtype=torch.float64) for owner in self.owners}

        N = self.n_events

        # 熵差矩阵 ΔH_mn = H_m - H_n（全张量运算）
        DeltaH = H.unsqueeze(1) - H.unsqueeze(0)  # (N, N)

        # 微观流动矩阵 flow_mn = J_mn · ΔH_mn
        flow = J * DeltaH  # (N, N)

        # 个体散度：对每个 owner A，Σ_{m∈A, n∉A} flow_mn
        masks = self.owner_masks()
        divergence = {}
        for owner, mask in masks.items():
            # m ∈ A 的掩码（行掩码）
            mask_row = mask.unsqueeze(1)  # (N, 1)
            # n ∉ A 的掩码（列掩码）
            mask_col = (1.0 - mask).unsqueeze(0)  # (1, N)
            # 组合掩码：m∈A 且 n∉A
            combined = mask_row * mask_col  # (N, N)
            # 散度 = Σ flow_mn * combined
            div_A = (flow * combined).sum()
            divergence[owner] = div_A

        return divergence

    def parasitism_asymmetry(self) -> dict[str, Tensor]:
        """
        寄生不对称性：量化"薅羊毛"关系。

        Parasitism(A→B) = (flow_A_out - flow_A_in) / (flow_A_out + flow_A_in)

        正值：A 向 B 输送能量（A 被吸血）
        负值：A 从 B 吸取能量（A 寄生）

        物理意义：
            有向因果边的不对称性量化寄生关系。
        """
        J = self.cross_graph_coupling()
        H = self.event_entropy()

        if J.shape[0] == 0:
            return {owner: torch.tensor(0.0, dtype=torch.float64) for owner in self.owners}

        DeltaH = H.unsqueeze(1) - H.unsqueeze(0)
        flow = J * DeltaH

        masks = self.owner_masks()
        asymmetry = {}
        for owner, mask in masks.items():
            mask_row = mask.unsqueeze(1)
            mask_col = (1.0 - mask).unsqueeze(0)
            combined = mask_row * mask_col
            # A 流出的能量（m∈A, n∉A）
            flow_out = (flow * combined).sum()
            # A 流入的能量（m∉A, n∈A）
            flow_in = (flow * combined.transpose(-1, -2)).sum()
            total = flow_out.abs() + flow_in.abs() + self.eps
            asymmetry[owner] = (flow_out - flow_in) / total

        return asymmetry

    def summary(self) -> dict[str, Tensor | float]:
        """多体耦合摘要（仅供审计）。"""
        div = self.entropy_flow_divergence()
        return {
            "n_events": self.n_events,
            "n_owners": len(self.owners),
            "owners": self.owners,
            "divergence": div,
            "global_causal_density": float(self.global_causal_adjacency().mean()),
            "cross_graph_density": float(self.cross_graph_coupling().mean()),
        }
