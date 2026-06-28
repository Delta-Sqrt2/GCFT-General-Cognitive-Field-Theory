"""
L2 离散作用量层 —— 事件图泛函 S[G] 与作用量极小化动力学

战略定位（v2.0 任务三）：
    废除 v1.x 的连续拉格朗日量 ∫L dV dt。
    系统不追求连续时间积分极小，而是追求整个事件图的作用量泛函 S[G] 极小。
    系统倾向于生成那些能降低整体因果张力的事件。

物理与哲学直觉：
    - 物理：路径积分量子场论——系统沿作用量极小化的路径演化
    - 哲学：最小作用量原理——"命运"是作用量极小化的数学解
    - 工程：离散求和 Σ 替代连续积分 ∫，可微且可算

数学定义（严格可微，无连续场论复辟）：
    事件图泛函：
        S[G] = α · Σ_i Complexity(e_i) + β · Σ_{i,j} Tension(C_ij)

    其中：
        Complexity(e_i) = -Σ_d p_d · log(p_d)    事件信息熵
            p_d = |φ_i,d| / Σ_d |φ_i,d|          特征维度的概率分布
            物理意义：事件的信息复杂度

        Tension(C_ij) = C_ij · (H(e_i) - H(e_j))  因果链张力
            H(e_i) = Complexity(e_i)               事件熵
            物理意义：高熵事件流向低熵事件产生的自由能梯度
                      这就是"痛苦"的涌现定义——非预设概念

    事件生成概率（玻尔兹曼分布）：
        P(e_{t+1} | G_t) ∝ exp(-ΔS / κ)
        ΔS = S[G_t ∪ {e_{t+1}}] - S[G_t]
        κ = 1/(n_dims+2) 结构常数（与 v1.2 一致）

    命运的数学解：
        系统沿作用量极小化路径演化。
        高概率事件 = 降低系统总张力的事件。
        这就是"为什么人会重复某些行为"的物理本质——
        重复行为是作用量极小化的吸引子。

工程铁律（v2.0 专属）：
    1. 严禁 ∫dVdt 连续积分：必须用 Σ 离散求和
    2. 严禁 detach()：作用量对事件特征可微
    3. 严禁 for 循环遍历节点：全部矩阵运算
    4. 严禁预设"痛苦分数"：痛苦由张力涌现
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import symmetric_part


class DiscreteAction:
    """
    离散作用量层：事件图泛函 S[G] 与作用量极小化动力学。

    使用方式：
        action = DiscreteAction(n_dims=8)
        S = action.compute_action(ontology)         # 当前事件图的作用量
        tension = action.tension_matrix(ontology)    # 因果张力矩阵
        probs = action.event_generation_prob(ontology, candidate_features)  # 候选事件生成概率

    白盒保证：
        - 作用量由离散求和 Σ 计算（非连续积分 ∫）
        - 痛苦由因果张力涌现（非预设分数）
        - 全程可微（无 detach）
        - 全张量运算（无 for 循环）
    """

    def __init__(self, n_dims: int, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-12):
        """
        参数：
            n_dims: 认知维度
            alpha: 复杂度项系数（事件内禀信息权重）
            beta: 张力项系数（因果链自由能权重）
            eps: 数值稳定常数

        物理意义：
            α/β 比例决定系统偏好"信息丰富"还是"因果和谐"。
            默认 α=β=1（等权），由理论对称性推导。
        """
        self.n_dims = int(n_dims)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        # 玻尔兹曼常数 κ = 1/(n+2)（结构常数，与 v1.2 度规耦合一致）
        self.kappa = torch.tensor(1.0 / (n_dims + 2), dtype=torch.float64)

    def event_entropy(self, feature_matrix: Tensor) -> Tensor:
        """
        事件信息熵 H(e_i) = -Σ_d p_d · log(p_d)。

        数学：
            p_d = |φ_i,d| / Σ_d |φ_i,d|    特征维度的概率分布
            H(e_i) = -Σ_d p_d · log(p_d)   Shannon 熵

        物理意义：
            事件特征分布越均匀，信息熵越高（事件越复杂）。
            事件特征集中（少数维度主导），熵低（事件简单/僵化）。

        全张量运算（无 for 循环）：
            P = |Φ| / |Φ|.sum(dim=-1, keepdim=True)
            H = -(P * log(P)).sum(dim=-1)
        """
        Phi = feature_matrix.to(torch.float64)
        # 绝对值（特征强度）
        Phi_abs = Phi.abs()
        # 归一化为概率分布（沿特征维度）
        total = Phi_abs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = Phi_abs / total
        # Shannon 熵
        H = -(p * torch.log(p.clamp(min=self.eps))).sum(dim=-1)
        return H

    def complexity_term(self, feature_matrix: Tensor) -> Tensor:
        """
        复杂度项 Σ_i Complexity(e_i)。

        数学：
            Σ_i H(e_i)    所有事件熵的总和

        物理意义：
            整个事件图的信息复杂度。
            高复杂度 = 系统经历多样化事件。
        """
        H = self.event_entropy(feature_matrix)
        return H.sum()

    def tension_matrix(self, ontology) -> Tensor:
        """
        因果张力矩阵 T ∈ R^{N×N}。

        T_ij = C_ij · (H(e_i) - H(e_j))

        物理意义：
            高熵事件流向低熵事件产生正张力（自由能释放）。
            低熵事件流向高熵事件产生负张力（自由能消耗）。
            张力总和 = 系统的"痛苦势能"涌现定义。

        全张量运算（无 for 循环）：
            H = event_entropy(Φ)              (N,)
            ΔH = H.unsqueeze(1) - H.unsqueeze(0)   (N, N)
            T = C * ΔH
        """
        C = ontology.causal_adjacency()
        Phi = ontology.feature_matrix()
        if C.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        H = self.event_entropy(Phi)  # (N,)

        # 熵差矩阵 ΔH_ij = H_i - H_j（全张量运算）
        DeltaH = H.unsqueeze(1) - H.unsqueeze(0)  # (N, N)

        # 张力 = 因果邻接 × 熵差
        T = C * DeltaH
        return T

    def tension_term(self, ontology) -> Tensor:
        """
        张力项 Σ_{i,j} Tension(C_ij)。

        数学：
            Σ_{i,j} C_ij · (H(e_i) - H(e_j))

        物理意义：
            整个因果网络的总自由能梯度。
            这就是"痛苦"的涌现定义——非预设概念，由因果结构涌现。

        注：由于 ΔH_ij = -ΔH_ji，且 C_ij = C_ji（无向），
            理论上对称部分相消。但因果邻接 C 是有向的（时间因果），
            因此张力项非零，反映因果方向上的自由能流动。
        """
        T = self.tension_matrix(ontology)
        # 取正部分（自由能释放，即"痛苦"）
        # 软 ReLU（保持可微）：positive_tension = softplus(T)
        positive_tension = torch.nn.functional.softplus(T)
        return positive_tension.sum()

    def compute_action(self, ontology) -> dict[str, Tensor]:
        """
        计算事件图泛函 S[G] = α·Σ Complexity + β·Σ Tension。

        参数：
            ontology: EventOntology 实例

        返回：
            {
                'action': 总作用量 S[G],
                'complexity': 复杂度项 Σ H(e_i),
                'tension': 张力项 Σ Tension(C_ij),
                'entropy_vector': 各事件熵 H(e_i),
                'tension_matrix': 因果张力矩阵 T_ij,
            }

        物理意义：
            作用量极小化 = 系统倾向降低总张力（减少痛苦）
            + 平衡信息复杂度（保持认知丰富性）。
            这就是"命运"的数学解：作用量极小化的路径。

        严禁：
            - ∫dVdt 连续积分（必须用 Σ 离散求和）
            - detach() 断裂梯度
            - 预设"痛苦分数"（痛苦由张力涌现）
        """
        Phi = ontology.feature_matrix()
        if Phi.shape[0] == 0:
            zero = torch.tensor(0.0, dtype=torch.float64)
            return {
                "action": zero,
                "complexity": zero,
                "tension": zero,
                "entropy_vector": torch.zeros(0, dtype=torch.float64),
                "tension_matrix": torch.zeros(0, 0, dtype=torch.float64),
            }

        # 复杂度项
        complexity = self.complexity_term(Phi)

        # 张力项
        tension = self.tension_term(ontology)

        # 总作用量
        action = self.alpha * complexity + self.beta * tension

        # 熵向量
        entropy_vector = self.event_entropy(Phi)

        # 张力矩阵
        tension_mat = self.tension_matrix(ontology)

        return {
            "action": action,
            "complexity": complexity,
            "tension": tension,
            "entropy_vector": entropy_vector,
            "tension_matrix": tension_mat,
        }

    def event_generation_prob(
        self,
        ontology,
        candidate_features: Tensor,
    ) -> Tensor:
        """
        候选事件生成概率（玻尔兹曼分布）。

        P(e_{t+1} | G_t) ∝ exp(-ΔS / κ)

        其中：
            ΔS = S[G_t ∪ {e_{t+1}}] - S[G_t]
            κ = 1/(n_dims+2) 结构常数

        参数：
            ontology: 当前事件图
            candidate_features: 候选事件特征矩阵 (M, d)，M 个候选

        返回：
            probs: 候选事件生成概率 (M,)，归一化

        物理意义：
            高概率事件 = 降低系统总作用量的事件。
            系统沿作用量极小化路径演化。
            这就是"为什么人会重复某些行为"的物理本质——
            重复行为是作用量极小化的吸引子。

        全张量运算（无 for 循环）：
            通过批量扩展候选事件，一次性计算所有 ΔS。
        """
        candidate_features = candidate_features.to(torch.float64)
        M = candidate_features.shape[0]

        # 当前作用量
        current_action = self.compute_action(ontology)["action"]

        # 批量计算加入每个候选事件后的作用量
        # 策略：将候选事件作为新行追加到特征矩阵，重新计算作用量
        # 为保持全张量运算，使用批量扩展
        Phi_current = ontology.feature_matrix()  # (N, d)
        N = Phi_current.shape[0]
        d = Phi_current.shape[1]

        # 扩展：对每个候选，构造 (N+1, d) 的特征矩阵
        # 批量化：(M, N+1, d)
        Phi_expanded = torch.zeros(M, N + 1, d, dtype=torch.float64)
        Phi_expanded[:, :N, :] = Phi_current.unsqueeze(0).expand(M, -1, -1)
        Phi_expanded[:, N, :] = candidate_features

        # 批量计算复杂度项（每个候选的）
        # H(e_i) 对每个事件独立，可批量计算
        Phi_abs = Phi_expanded.abs()
        total = Phi_abs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = Phi_abs / total
        H = -(p * torch.log(p.clamp(min=self.eps))).sum(dim=-1)  # (M, N+1)
        complexity_per_candidate = H.sum(dim=-1)  # (M,)

        # 批量计算张力项
        # 需要扩展因果邻接矩阵（加入新节点）
        C_current = ontology.causal_adjacency()  # (N, N)
        t_current = ontology.timestamp_vector()  # (N,)
        tau = ontology.tau_causal

        # 新节点的时间戳 = 当前最大时间 + 1
        t_new = t_current.max() + 1.0 if N > 0 else 0.0

        # 扩展时间向量 (M, N+1)
        t_expanded = torch.zeros(M, N + 1, dtype=torch.float64)
        t_expanded[:, :N] = t_current.unsqueeze(0).expand(M, -1)
        t_expanded[:, N] = t_new

        # 扩展因果邻接 (M, N+1, N+1)
        C_expanded = torch.zeros(M, N + 1, N + 1, dtype=torch.float64)
        C_expanded[:, :N, :N] = C_current.unsqueeze(0).expand(M, -1, -1)

        # 新节点与现有节点的因果连接
        # 语义相似度：cos(φ_new, φ_existing)
        Phi_current_norm = Phi_current / Phi_current.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        candidate_norm = candidate_features / candidate_features.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sim_new_to_existing = candidate_norm @ Phi_current_norm.transpose(-1, -2)  # (M, N)

        # 时间核：新节点在所有现有节点之后
        DeltaT_new = t_new - t_current  # (N,) > 0
        K_new = torch.exp(-DeltaT_new.abs() / tau) * torch.sigmoid(DeltaT_new / (tau * 0.1))  # (N,)
        K_new = K_new.unsqueeze(0).expand(M, -1)  # (M, N)

        # 新节点的因果邻接
        C_expanded[:, N, :N] = sim_new_to_existing * K_new  # 新节点依赖现有节点

        # 批量计算张力
        # ΔH_ij = H_i - H_j
        DeltaH = H.unsqueeze(2) - H.unsqueeze(1)  # (M, N+1, N+1)
        T = C_expanded * DeltaH  # (M, N+1, N+1)
        positive_tension = torch.nn.functional.softplus(T)
        tension_per_candidate = positive_tension.sum(dim=(-1, -2))  # (M,)

        # 批量作用量
        action_per_candidate = self.alpha * complexity_per_candidate + self.beta * tension_per_candidate

        # ΔS = S[G ∪ {e}] - S[G]
        delta_S = action_per_candidate - current_action

        # 玻尔兹曼概率 P ∝ exp(-ΔS/κ)
        logits = -delta_S / self.kappa
        probs = torch.softmax(logits, dim=0)  # 归一化

        return probs

    def action_gradient(self, ontology) -> Tensor:
        """
        作用量对事件特征的梯度 ∂S/∂Φ。

        物理意义：
            梯度方向 = 作用量上升最快的方向。
            负梯度 = 作用量极小化方向 = 系统演化方向。

        工程：
            利用 PyTorch autograd 自动微分。
            事件特征 requires_grad=True，作用量 backward()。
        """
        Phi = ontology.feature_matrix()
        if not Phi.requires_grad:
            Phi.requires_grad_(True)

        action_result = self.compute_action(ontology)
        action = action_result["action"]

        if Phi.grad is not None:
            Phi.grad.zero_()
        action.backward()
        return Phi.grad.clone() if Phi.grad is not None else torch.zeros_like(Phi)

    def summary(self, ontology) -> dict[str, Tensor | float]:
        """作用量摘要（仅供审计）。"""
        result = self.compute_action(ontology)
        return {
            "action": float(result["action"]),
            "complexity": float(result["complexity"]),
            "tension": float(result["tension"]),
            "mean_entropy": float(result["entropy_vector"].mean()) if result["entropy_vector"].shape[0] > 0 else 0.0,
            "max_tension": float(result["tension_matrix"].max()) if result["tension_matrix"].shape[0] > 0 else 0.0,
            "kappa": float(self.kappa),
        }
