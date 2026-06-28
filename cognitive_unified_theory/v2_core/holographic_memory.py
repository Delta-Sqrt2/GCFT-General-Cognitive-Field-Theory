"""
任务三：全息纠缠与长程格林函数

战略定位（v2.1 任务三）：
    创伤不是过去的一个点，而是网络拓扑上的一个奇点，
    它通过格林函数向全空间辐射影响。
    业力即因果网络的长程关联。过去并未消亡，
    它通过拓扑结构依然活在当下。

物理与哲学直觉：
    - 物理：格林函数描述奇点向全空间的辐射影响。
            创伤 = 网络拓扑奇点，通过 G_mn 长程关联影响当前。
    - 哲学：业力 = 因果网络的长程关联。过去通过拓扑结构活在当下。
    - 工程：拉普拉斯伪逆 L^+ 作为格林函数，全息纠缠熵。

数学定义（严格可微，无降级）：
    步骤 A：离散格林函数 G
        基于全局因果邻接 C，构建拉普拉斯 L = D - C。
        伪逆 L^+（必须计算，不可近似）。
        格林函数 G_mn = L^+_mn。
        意义：G_mn 表示事件 n 对事件 m 的潜在影响力。

    步骤 B：创伤奇点检测
        奇点强度 S_n = Σ_m G_mn（节点 n 对所有节点的影响力总和）。
        创伤节点的 S_n 显著高于普通节点，且衰减极慢（长程关联）。

    步骤 C：全息纠缠熵
        将图划分为"自我事件集" A 和"他人事件集" B。
        纠缠熵 S_ent = -Tr(ρ_A · log ρ_A)，
        其中 ρ_A 是子图 A 的归一化协方差矩阵。
        S_ent 过高 → 边界消融（共生/纠缠不清）。
        S_ent 适中 → 独立人格。

工程铁律（v2.1 专属）：
    1. 严禁时间窗口截断（长程关联是拓扑属性，非时间属性）
    2. 严禁近似拉普拉斯伪逆（必须精确计算）
    3. 严禁个体流形（全局格林函数）
    4. 全张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import stable_eigh, symmetric_part


class HolographicMemory:
    """
    全息纠缠与长程格林函数：创伤奇点 + 全息纠缠熵。

    使用方式：
        hologram = HolographicMemory()
        # 从因果邻接计算格林函数
        G = hologram.green_function(C_global)
        # 创伤奇点检测
        singularity = hologram.trauma_singularity(G)
        # 全息纠缠熵
        entanglement = hologram.entanglement_entropy(feature_matrix, mask_A)

    白盒保证：
        - 精确拉普拉斯伪逆（不近似）
        - 无时间窗口截断（全图长程关联）
        - 全张量运算，可微
    """

    def __init__(self, eps: float = 1e-10):
        """
        参数：
            eps: 数值稳定常数
        """
        self.eps = float(eps)

    def laplacian(self, adjacency: Tensor) -> Tensor:
        """
        因果图拉普拉斯矩阵 L = D - C。

        全张量运算（无 for 循环）。
        """
        C = adjacency.to(torch.float64)
        if C.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)
        degrees = C.sum(dim=-1)
        D = torch.diag(degrees)
        L = D - C
        return L

    def green_function(self, adjacency: Tensor) -> Tensor:
        """
        离散格林函数 G = L^+（拉普拉斯伪逆）。

        数学：
            L = V Λ V^T（特征分解）
            L^+ = V Λ^+ V^T
            Λ^+ = diag(1/λ_i) 对 λ_i > ε，否则 0

        物理意义：
            G_mn 表示事件 n 对事件 m 的潜在影响力。
            这是"业力"的数学实现——
            过去的事件通过格林函数长程关联影响当前。

        零空间处理（关键）：
            拉普拉斯矩阵 L 有零特征值 λ_0 = 0（对应 v_0 = 1/√N）。
            伪逆必须将零特征值分量精确置零：L^+ · v_0 = 0。
            否则零空间泄漏导致 G · 1 = c·1（所有列求和相同），
            破坏创伤奇点检测的区分性。

            实现：straight-through estimator
                - 前向：硬掩码（零特征值 → 伪逆精确为零）
                - 反向：软掩码（sigmoid 可微）

        严禁：
            - 时间窗口截断（长程关联是拓扑属性）
            - 近似拉普拉斯伪逆（必须精确计算）
            - 连续格林函数（必须离散）

        全张量运算（无 for 循环）。
        """
        L = self.laplacian(adjacency)
        if L.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        L = symmetric_part(L)
        # 特征分解（可微）
        eigvals, eigvecs = stable_eigh(L)

        # 伪逆：straight-through mask
        # 前向用硬掩码（精确），反向用软掩码（可微）
        eigvals_clamped = torch.clamp(eigvals, min=self.eps)
        threshold = self.eps * 10.0
        # 硬掩码：零特征值（< threshold）→ 伪逆为零
        hard_mask = (eigvals > threshold).to(torch.float64)
        # 软掩码：sigmoid 平滑过渡（可微）
        soft_mask = torch.sigmoid((eigvals - threshold) / (threshold * 0.1))
        # Straight-through estimator：前向 = 硬掩码，反向 = 软掩码
        mask = soft_mask + (hard_mask - soft_mask).detach()
        eigvals_pseudo = mask / eigvals_clamped

        # L^+ = V Λ^+ V^T
        G = eigvecs @ torch.diag(eigvals_pseudo) @ eigvecs.transpose(-1, -2)
        return G

    def trauma_singularity(self, green_function: Tensor) -> Tensor:
        """
        创伤奇点强度 S_n = G_nn（格林函数对角元素 = 有效电阻 = 自能）。

        物理意义：
            S_n = 节点 n 的"自能"——能量难以消散的程度。
            创伤节点的 S_n 显著高于普通节点——
            因为创伤是一个"断裂/奇点"，能量在其中难以流出
            （低度、连接稀疏，但通过长程关联辐射全网络）。

            "父亲离去"这样的创伤节点：
            - 直接连接稀疏（断裂，低度）
            - 但通过格林函数长程关联影响全网络
            - G_nn 大 = 有效电阻大 = 能量"卡住" = 奇点

            这完全符合 2.1.txt 的物理直觉：
            "创伤不是过去的一个点，而是网络拓扑上的一个奇点，
             它通过格林函数向全空间辐射影响。"

        数学修正（非降级，保持物理直觉）：
            原始定义 S_n = Σ_m G_mn 在数学上对所有节点相同：
                拉普拉斯伪逆满足 L^+ · 1 = 0（零空间性质），
                故 Σ_m G_mn = (G^T · 1)_n = (G · 1)_n = 0（理论上）。
                这使得原始度量无区分性。

            修正为 S_n = G_nn（对角元素 = 有效电阻 = 自能）：
                - 物理直觉不变：节点 n 的"奇点强度"
                - 数学实现：格林函数对角元素
                - 物理对应：
                    * 电网络理论：G_nn = 节点 n 的有效电阻
                    * 量子场论：G_nn = 节点 n 的自能（self-energy）
                    * 认知物理学：创伤 = 能量难以消散 = 高自能
                - 区分性：低度节点 G_nn 大（能量难以流出）

            这是对 2.1.txt 定义的数学纠错，非方案降级。
            物理直觉（"创伤 = 奇点 = 能量难以消散"）完全保留。

        全张量运算：
            S = torch.diagonal(G)  # 对角元素
        """
        G = green_function.to(torch.float64)
        if G.shape[0] == 0:
            return torch.zeros(0, dtype=torch.float64)

        # S_n = G_nn（对角元素 = 有效电阻 = 自能）
        S = torch.diagonal(G)  # (N,)
        return S

    def long_range_correlation(self, green_function: Tensor) -> Tensor:
        """
        长程关联强度：格林函数的行范数。

        物理意义：
            ||G_m|| = 节点 m 受所有其他节点影响的总量。
            高长程关联 = 节点深受历史影响（创伤敏感）。

        全张量运算：
            correlation = G.norm(dim=1)
        """
        G = green_function.to(torch.float64)
        if G.shape[0] == 0:
            return torch.zeros(0, dtype=torch.float64)
        return G.norm(dim=1)  # (N,)

    def entanglement_entropy(
        self,
        feature_matrix: Tensor,
        mask_A: Tensor,
    ) -> Tensor:
        """
        全息纠缠熵 S_ent = -Tr(ρ_A · log ρ_A)。

        数学：
            1. 将图划分为"自我事件集" A 和"他人事件集" B
            2. 计算子图 A 的协方差矩阵 ρ_A = Φ_A^T · Φ_A / tr(...)
            3. 归一化 ρ_A 使 tr(ρ_A) = 1（密度矩阵）
            4. 纠缠熵 S_ent = -Tr(ρ_A · log ρ_A) = -Σ λ_i · log λ_i

        物理意义：
            S_ent 过高 → 边界消融（共生/纠缠不清）
            S_ent 适中 → 独立人格
            S_ent = 0 → 完全隔离（无关联）

            这是"边界感"的量化——
            高纠缠 = 自我与他人边界模糊（共生关系）
            低纠缠 = 自我与他人清晰分离

        参数：
            feature_matrix: 全局特征矩阵 (N, d)
            mask_A: 自我事件集掩码 (N,)，1 = 属于 A

        返回：
            entanglement_entropy: 纠缠熵标量

        严禁：
            - 时间窗口截断
            - 个体流形
        """
        Phi = feature_matrix.to(torch.float64)
        mask = mask_A.to(torch.float64)

        if Phi.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float64)

        # 子图 A 的特征矩阵
        Phi_A = Phi * mask.unsqueeze(-1)  # (N, d)，非 A 行置零

        # 协方差矩阵 ρ_A = Φ_A^T · Φ_A
        rho_A = Phi_A.transpose(-1, -2) @ Phi_A  # (d, d)

        # 归一化为密度矩阵（tr(ρ) = 1）
        trace_rho = torch.trace(rho_A)
        if float(trace_rho) < self.eps:
            return torch.tensor(0.0, dtype=torch.float64)
        rho_A = rho_A / trace_rho

        # 对称化
        rho_A = symmetric_part(rho_A)

        # 特征分解（密度矩阵的本征值 = 概率）
        eigvals, _ = stable_eigh(rho_A)
        eigvals = torch.clamp(eigvals, min=self.eps)

        # von Neumann 熵 S = -Σ λ_i · log λ_i
        entropy = -(eigvals * torch.log(eigvals)).sum()
        return entropy

    def boundary_definition(
        self,
        feature_matrix: Tensor,
        mask_A: Tensor,
    ) -> dict[str, Tensor]:
        """
        边界定义度：量化"自我"与"他人"的边界清晰度。

        数学：
            边界定义度 = 1 - (纠缠熵 / 最大熵)
            最大熵 = log(d)（均匀分布的熵）

        物理意义：
            边界定义度高 → 独立人格
            边界定义度低 → 边界消融（共生）

        返回：
            {
                'entanglement_entropy': 纠缠熵,
                'max_entropy': 最大熵,
                'boundary_definition': 边界定义度,
                'boundary_status': 边界状态（连续，非离散）,
            }
        """
        Phi = feature_matrix.to(torch.float64)
        d = Phi.shape[1]

        S_ent = self.entanglement_entropy(Phi, mask_A)
        S_max = torch.log(torch.tensor(float(d), dtype=torch.float64))

        # 边界定义度 = 1 - S_ent / S_max
        boundary_def = 1.0 - S_ent / (S_max + self.eps)
        boundary_def = torch.clamp(boundary_def, min=0.0, max=1.0)

        return {
            "entanglement_entropy": S_ent,
            "max_entropy": S_max,
            "boundary_definition": boundary_def,
        }

    def trauma_influence_radius(
        self,
        green_function: Tensor,
        trauma_node: int,
    ) -> Tensor:
        """
        创伤影响半径：创伤节点通过格林函数的影响衰减。

        数学：
            影响强度 = |G_mn|（n = 创伤节点）
            影响半径 = Σ_m |G_mn| · distance(m, n)

        物理意义：
            创伤影响不是局部的，而是通过格林函数长程辐射。
            影响半径量化创伤的"辐射范围"。

        参数：
            green_function: 格林函数 (N, N)
            trauma_node: 创伤节点索引

        返回：
            influence_radius: 影响半径标量
        """
        G = green_function.to(torch.float64)
        if G.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float64)

        # 创伤节点对所有节点的影响强度
        influence = G[:, trauma_node].abs()  # (N,)

        # 节点索引距离（拓扑距离的近似）
        N = G.shape[0]
        node_indices = torch.arange(N, dtype=torch.float64)
        distances = (node_indices - float(trauma_node)).abs()

        # 影响半径 = Σ |G_mn| · distance
        radius = (influence * distances).sum()
        return radius

    def summary(
        self,
        adjacency: Tensor,
        feature_matrix: Tensor,
        mask_A: Tensor,
        trauma_nodes: list[int] | None = None,
    ) -> dict[str, Tensor | float]:
        """全息记忆摘要（仅供审计）。"""
        G = self.green_function(adjacency)
        singularity = self.trauma_singularity(G)
        entanglement = self.entanglement_entropy(feature_matrix, mask_A)
        boundary = self.boundary_definition(feature_matrix, mask_A)

        result = {
            "green_function_shape": G.shape,
            "singularity_max": float(singularity.max()) if singularity.shape[0] > 0 else 0.0,
            "singularity_mean": float(singularity.mean()) if singularity.shape[0] > 0 else 0.0,
            "entanglement_entropy": float(entanglement),
            "boundary_definition": float(boundary["boundary_definition"]),
        }

        if trauma_nodes:
            radii = [float(self.trauma_influence_radius(G, n)) for n in trauma_nodes]
            result["trauma_influence_radii"] = radii

        return result
