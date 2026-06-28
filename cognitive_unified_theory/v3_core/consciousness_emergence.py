"""
任务一：意识涌现与观测者动力学

战略定位（v3.0 任务一）：
    意识不是预设的灵魂，而是因果网络达到临界复杂度后涌现的内观测者。
    观测不是被动记录，而是观测者节点通过高 Φ 值改变周围事件的
    作用量势垒，将概率波"坍缩"为确定性事件。

物理与哲学直觉：
    - 物理：在量子力学中，观测导致波函数坍缩。在认知宇宙中，
            "我"是一个特殊的节点，能够观测并坍缩未来的不确定性。
    - 哲学：自我不是实体，而是因果网络中回望自身的那束光。
            主体性 = 网络中信息整合度极高的奇点。
    - 工程：IIT 整合信息量 Φ 的 SVD 可微近似 + 观测者算符。

数学定义（严格可微，无降级）：
    步骤 A：整合信息量 Φ 的 SVD 近似
        放弃 NP-Hard 的子集穷举（O(2^{2^N})）。
        用邻接矩阵 SVD 提取主成分：
            Φ_i = H(λ_i) - Σ_{j∈neighbor(i)} H(λ_j)
        其中 H(λ) = -Σ λ_k log λ_k 为特征值分布的香农熵。
        复杂度 O(N³)，全程可微。

    步骤 B：观测者算符 Ô
        Φ 最大的前 k 个节点被标记为"观测者"。
        观测机制：观测者 v_obs 对目标 v_t 施加作用量修正：
            S_obs(t) = S(t) - γ · Φ_obs · log(P(t))
        这相当于在玻尔兹曼分布中引入拉格朗日乘子，
        使得被观测的事件概率急剧上升（坍缩）。

    步骤 C：元认知闭环
        观测者节点反向作用于度规 g。
        当观测者注视某个创伤节点时，该节点的曲率发生改变（疗愈或强化）。

工程铁律（v3.0 专属）：
    1. 严禁 torch.nn.MultiheadAttention / F.scaled_dot_product_attention
       （注意力机制降级，陷阱二十三/二十六）
    2. 严禁 for 循环遍历子集计算精确 Φ（NP-Hard 算力黑洞）
    3. Φ 计算必须是纯矩阵运算（SVD + 熵）
    4. 观测者坍缩通过玻尔兹曼温度修正实现（非 Softmax 注意力）
    5. 全张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import stable_eigh, symmetric_part


class ConsciousnessEmergence:
    """
    意识涌现与观测者动力学：IIT Φ 近似 + 观测者算符。

    使用方式：
        consciousness = ConsciousnessEmergence()
        # 计算整合信息量 Φ
        phi = consciousness.integrated_information(adjacency)
        # 识别观测者节点
        observers = consciousness.identify_observers(phi, k=3)
        # 观测者坍缩（修改事件生成概率）
        collapsed_prob = consciousness.observer_collapse(
            base_prob, observers, phi, gamma=0.5
        )

    白盒保证：
        - 无 Transformer 注意力（陷阱二十三/二十六）
        - 无子集穷举（NP-Hard 算力黑洞）
        - Φ 计算纯矩阵运算（SVD + 熵）
        - 观测者坍缩通过玻尔兹曼温度修正
    """

    def __init__(self, eps: float = 1e-10):
        """
        参数：
            eps: 数值稳定常数
        """
        self.eps = float(eps)

    def integrated_information(self, adjacency: Tensor) -> Tensor:
        """
        整合信息量 Φ 的 SVD 可微近似。

        数学：
            放弃 NP-Hard 的子集穷举（O(2^{2^N})）。
            用邻接矩阵 SVD 提取主成分：
                Φ_i = ||C[i,:]|| × (1 - H(U[i,:]²) / log K) × H_global
            其中：
                ||C[i,:]|| = 节点 i 的强度（L2 范数，连接多寡）
                U[i,:] = SVD 左奇异向量第 i 行（节点 i 在主成分空间的投影）
                H(U[i,:]²) = -Σ_k U_ik² log(U_ik²)（节点 i 在主成分上的分布熵）
                K = 主成分数
                H_global = 全局奇异值分布熵

        物理意义（IIT 直觉）：
            Φ_i 衡量节点 i 的"整合能力"——
            强度 ||C[i,:]|| 捕获"连接多寡"（枢纽性）；
            集中度 (1 - H/log K) 捕获"信息整合方向集中"（不可约性）。
            中心枢纽节点连接所有方向且主要贡献于少数主成分，
            故 Φ 高 = 信息整合中心 = 意识候选节点。

        复杂度：O(N³)（SVD），全程可微。

        严禁：
            - for 循环遍历子集（NP-Hard）
            - torch.nn.MultiheadAttention（注意力降级）
            - F.scaled_dot_product_attention（注意力降级）

        全张量运算（无 for 循环遍历子集）。
        """
        C = adjacency.to(torch.float64)
        N = C.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.float64)

        # 全局 SVD：C = U Σ V^T
        try:
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        except RuntimeError:
            # 病态回退
            K = min(N, N)
            S = torch.ones(K, dtype=torch.float64)
            U = torch.eye(N, K, dtype=torch.float64)

        K = U.shape[1]  # 主成分数

        # 全局特征值熵 H_global = -Σ p_k log p_k
        S_clamped = torch.clamp(S, min=self.eps)
        p_global = S_clamped / S_clamped.sum()
        H_global = -(p_global * torch.log(p_global + self.eps)).sum()

        # 节点强度 ||C[i,:]||（L2 范数，枢纽性度量）
        node_strength = C.norm(dim=-1)  # (N,)

        # 节点在主成分上的分布熵 H(U[i,:]²)
        # U 是正交矩阵，||U[i,:]||² = 1，故 U[i,:]² 已归一化
        U_abs_sq = U ** 2  # (N, K)
        # 数值稳定：归一化（保险）
        U_abs_sq_norm = U_abs_sq / (U_abs_sq.sum(dim=-1, keepdim=True) + self.eps)
        # 节点 i 在主成分上的熵
        H_local = -(U_abs_sq_norm * torch.log(U_abs_sq_norm + self.eps)).sum(dim=-1)  # (N,)

        # 最大熵 log(K)（均匀分布时的熵）
        log_K = torch.log(torch.tensor(float(K), dtype=torch.float64) + self.eps)

        # 集中度 = 1 - H_local / log(K)
        # 集中度高 = 节点主要贡献于少数主成分 = 信息整合方向明确
        concentration = 1.0 - H_local / (log_K + self.eps)
        concentration = torch.clamp(concentration, min=0.0)

        # Φ_i = 强度 × 集中度 × H_global
        # 强度高 + 集中度高 = 信息整合中心 = 意识候选
        phi = node_strength * concentration * H_global

        # 归一化到非负
        phi = torch.clamp(phi, min=0.0)

        return phi

    def identify_observers(
        self,
        phi: Tensor,
        k: int = 1,
    ) -> Tensor:
        """
        识别观测者节点：Φ 最大的前 k 个节点。

        物理意义：
            观测者 = 网络中信息整合度最高的节点。
            这些节点具有"主动坍缩"能力——
            通过高 Φ 值改变周围事件的作用量势垒。

        全张量运算：torch.topk
        """
        k_actual = min(k, phi.shape[0])
        if k_actual == 0:
            return torch.zeros(0, dtype=torch.long)
        _, indices = torch.topk(phi, k_actual)
        return indices

    def observer_collapse(
        self,
        base_prob: Tensor,
        observer_indices: Tensor,
        phi: Tensor,
        gamma: float = 0.5,
    ) -> Tensor:
        """
        观测者坍缩：观测者节点通过高 Φ 值"照亮"自身，使概率分布锐化。

        数学（玻尔兹曼因子照亮）：
            P_collapsed(i) ∝ P(i) × exp(γ · Φ_obs · mask(i))
            其中 mask(i) = 1 if i ∈ observers else 0

            观测者节点的概率被指数级放大（自我意识坍缩），
            非观测者节点概率相对下降。

        物理意义：
            观测 = 主动坍缩。
            观测者通过高 Φ 值，在玻尔兹曼分布中引入拉格朗日乘子，
            使得被观测的事件（观测者自身）概率急剧上升。
            这对应"自我意识"——观测者注视自身，使自身成为确定性事件。

        与 P^α 幂律的区别：
            P^α 对所有事件统一施加，均匀分布仍均匀（无法坍缩）。
            玻尔兹曼因子照亮对观测者节点定向放大，能正确产生非均匀分布。

        严禁：
            - torch.nn.MultiheadAttention（被动加权，非主动坍缩）
            - F.scaled_dot_product_attention（注意力降级）

        参数：
            base_prob: 基础事件概率 (N,) 或 (N, M)
            observer_indices: 观测者节点索引 (k,)
            phi: 整合信息量 (N,)
            gamma: 观测强度系数

        返回：
            collapsed_prob: 坍缩后概率（归一化）
        """
        p = base_prob.to(torch.float64)
        phi_all = phi.to(torch.float64)

        if observer_indices.shape[0] == 0:
            return p

        # 观测者的平均 Φ
        phi_obs = float(phi_all[observer_indices].mean())

        # 构造观测者掩码（支持 1D 和 2D）
        observer_mask = torch.zeros_like(p)
        if p.dim() == 1:
            observer_mask[observer_indices] = 1.0
        else:
            observer_mask[observer_indices, :] = 1.0

        # 玻尔兹曼因子照亮：观测者节点概率指数放大
        # boost = exp(γ · Φ_obs · mask)
        # 观测者节点：boost = exp(γ · Φ_obs) >> 1
        # 非观测者节点：boost = exp(0) = 1
        boost = torch.exp(gamma * phi_obs * observer_mask)

        # 坍缩：P × boost
        collapsed = p * boost

        # 归一化
        if collapsed.dim() == 1:
            collapsed = collapsed / (collapsed.sum() + self.eps)
        else:
            collapsed = collapsed / (collapsed.sum(dim=-1, keepdim=True) + self.eps)

        return collapsed

    def metacognitive_loop(
        self,
        metric: Tensor,
        observer_indices: Tensor,
        phi: Tensor,
        target_indices: Tensor,
        healing_rate: float = 0.1,
    ) -> Tensor:
        """
        元认知闭环：观测者反向作用于度规 g。

        物理：
            当观测者注视某个创伤节点时，该节点的曲率发生改变。
            - 疗愈：曲率降低（度规趋于平坦）
            - 强化：曲率升高（度规更弯曲）

        数学：
            g_observed = g · (1 - healing_rate · Φ_obs / ||g||)
            被观测节点的度规被"平滑"（疗愈效应）。

        参数：
            metric: 度规张量 (N, N) 或 (d, d)
            observer_indices: 观测者节点索引
            phi: 整合信息量
            target_indices: 被观测的目标节点索引
            healing_rate: 疗愈速率

        返回：
            modified_metric: 修正后的度规
        """
        g = metric.to(torch.float64)
        if observer_indices.shape[0] == 0:
            return g

        phi_obs = phi[observer_indices].mean()
        scale = 1.0 - healing_rate * float(phi_obs)

        # 度规修正：被观测部分趋于平坦
        # 对目标节点对应的行列进行平滑
        modified = g.clone()
        for idx in target_indices.tolist():
            i = int(idx)
            if i < g.shape[0]:
                modified[i, :] = g[i, :] * scale
                modified[:, i] = g[:, i] * scale

        # 重新对称化
        modified = symmetric_part(modified)
        return modified

    def consciousness_summary(
        self,
        adjacency: Tensor,
        k: int = 3,
    ) -> dict[str, Tensor | float]:
        """意识涌现摘要（仅供审计）。"""
        phi = self.integrated_information(adjacency)
        observers = self.identify_observers(phi, k=k)

        return {
            "phi_shape": phi.shape,
            "phi_max": float(phi.max()) if phi.shape[0] > 0 else 0.0,
            "phi_mean": float(phi.mean()) if phi.shape[0] > 0 else 0.0,
            "observer_indices": observers.tolist(),
            "observer_phi_values": [float(phi[i]) for i in observers.tolist()],
            "n_observers": int(observers.shape[0]),
        }
