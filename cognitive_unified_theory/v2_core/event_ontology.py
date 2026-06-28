"""
L0 事件本体层 —— 世界的基本单元是事件，不是状态

战略定位（v2.0 任务一）：
    废除 v1.x 的连续状态向量 S 作为基本单元。
    基本实体是事件节点 e_i = (主体, 动作, 客体, 语境, 时间)。
    事件之间通过因果和语义连接，构成现实的底层图谱。

物理与哲学直觉：
    - 物理：圈量子引力/CDT——时空从离散量子事件中涌现，而非背景存在
    - 哲学：过程哲学（怀特海）——现实实体是事件，不是物质点
    - 工程：事件图比连续场更可算，可用张量矩阵运算替代偏微分方程

数学定义：
    事件节点 e_i 的特征向量 φ_i ∈ R^d（由 v1.1 解析器的力向量+结构特征构成）
    因果邻接矩阵：
        C_ij = sim_semantic(e_i, e_j) · K_temporal(t_j - t_i)
    其中：
        sim_semantic = cos(φ_i, φ_j)         语义相似度（余弦）
        K_temporal(Δt) = exp(-|Δt|/τ) · H(Δt)  因果时间核（仅当 j 在 i 之后）
        H(Δt) = (1 + sign(Δt))/2            因果方向阶跃（未来依赖过去）

    语义相似度矩阵 S_ij = cos(φ_i, φ_j)（无时间约束）

工程铁律（v2.0 专属）：
    1. 严禁 torch_geometric：纯 PyTorch 张量运算
    2. 严禁 for 循环遍历节点：全部矩阵乘法
    3. 严禁 detach()：保留 requires_grad=True 全链路可微
    4. 严禁预先初始化度规 g_μν：度规由 L1 涌现
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


class EventNode:
    """
    事件节点：v2.0 的基本实体。

    属性：
        feature: 事件特征向量 φ_i ∈ R^d（由 v1.1 解析器输出构成）
        timestamp: 事件发生时间 t_i
        force_type: 力类型（IMPACT/CONSTRAINT/SELF_STATE/POTENTIAL，来自 v1.1）
        source_text: 原始文本（仅审计，不参与计算）

    物理意义：
        一个事件节点不"在"时空中，它定义了局部时空。
        事件的特征向量是其在认知空间中的"内禀性质"，
        因果邻接关系由特征相似度与时间顺序共同决定。
    """

    def __init__(
        self,
        feature: Tensor,
        timestamp: float,
        force_type: str = "UNKNOWN",
        source_text: str = "",
    ):
        self.feature = feature.to(torch.float64)
        self.timestamp = float(timestamp)
        self.force_type = force_type
        self.source_text = source_text

    def __repr__(self) -> str:
        return (
            f"EventNode(t={self.timestamp:.2f}, "
            f"type={self.force_type}, "
            f"|φ|={float(self.feature.norm()):.4f})"
        )


class EventOntology:
    """
    事件本体层：构建事件图、因果邻接矩阵、语义相似度矩阵。

    使用方式：
        ontology = EventOntology(tau_causal=1.0)
        ontology.add_event(feature=force_vector, timestamp=0.0, force_type="IMPACT")
        C = ontology.causal_adjacency()   # 因果邻接矩阵
        S = ontology.semantic_similarity()  # 语义相似度矩阵
        Phi = ontology.feature_matrix()   # 事件特征矩阵 (N, d)

    白盒保证：
        - 全部张量运算，无 torch_geometric
        - 因果邻接由语义相似度 × 时间核推导（非硬编码）
        - 保留 requires_grad=True，全链路可微
        - 严禁预先初始化度规（度规由 L1 涌现）
    """

    def __init__(self, tau_causal: float = 1.0, eps: float = 1e-12):
        """
        参数：
            tau_causal: 因果时间核的衰减常数 τ
                       物理意义：事件影响力随时间衰减的特征时间
                       由事件序列的时间尺度推导（非硬编码经验值）
            eps: 数值稳定常数（仅防奇异，不参与物理量定义）
        """
        self.tau_causal = float(tau_causal)
        self.eps = float(eps)

        # 事件存储（按时间顺序）
        self._events: list[EventNode] = []

        # 缓存（事件变更时清空）
        self._feature_matrix_cache: Tensor | None = None
        self._timestamp_vector_cache: Tensor | None = None
        self._causal_cache: Tensor | None = None
        self._semantic_cache: Tensor | None = None

    def add_event(
        self,
        feature: Tensor,
        timestamp: float,
        force_type: str = "UNKNOWN",
        source_text: str = "",
    ) -> EventNode:
        """
        添加事件节点。

        参数：
            feature: 事件特征向量 φ_i ∈ R^d
                     （通常来自 v1.1 解析器的 force_vector）
            timestamp: 事件发生时间
            force_type: 力类型（来自 v1.1 解析器）
            source_text: 原始文本（仅审计）
        """
        node = EventNode(feature, timestamp, force_type, source_text)
        self._events.append(node)
        self._invalidate_cache()
        return node

    def add_events_from_parser(self, parsed_events: list) -> None:
        """
        从 v1.1 解析器输出批量添加事件。

        参数：
            parsed_events: NarrativeParser.parse() 返回的事件列表
                          每个 event 含 .force_vector, .time_step, .force_type, .source_text
        """
        for event in parsed_events:
            self.add_event(
                feature=event.force_vector,
                timestamp=event.time_step,
                force_type=event.force_type.name if hasattr(event.force_type, 'name') else str(event.force_type),
                source_text=getattr(event, 'source_text', ''),
            )

    def __len__(self) -> int:
        return len(self._events)

    @property
    def n_events(self) -> int:
        return len(self._events)

    @property
    def feature_dim(self) -> int:
        if self._events:
            return self._events[0].feature.shape[0]
        return 0

    def _invalidate_cache(self) -> None:
        self._feature_matrix_cache = None
        self._timestamp_vector_cache = None
        self._causal_cache = None
        self._semantic_cache = None

    def feature_matrix(self) -> Tensor:
        """
        事件特征矩阵 Φ ∈ R^{N×d}。
        每行是一个事件的特征向量 φ_i。

        严禁 for 循环遍历节点：使用 torch.stack 一次性构建。
        """
        if self._feature_matrix_cache is None:
            if not self._events:
                return torch.zeros(0, 0, dtype=torch.float64)
            self._feature_matrix_cache = torch.stack(
                [e.feature for e in self._events], dim=0
            )
        return self._feature_matrix_cache

    def timestamp_vector(self) -> Tensor:
        """
        时间戳向量 t ∈ R^N。
        """
        if self._timestamp_vector_cache is None:
            self._timestamp_vector_cache = torch.tensor(
                [e.timestamp for e in self._events], dtype=torch.float64
            )
        return self._timestamp_vector_cache

    def semantic_similarity(self) -> Tensor:
        """
        语义相似度矩阵 S ∈ R^{N×N}。

        S_ij = cos(φ_i, φ_j) = (φ_i · φ_j) / (||φ_i|| ||φ_j||)

        物理意义：
            两个事件在认知特征空间中的方向一致性。
            不含时间信息，纯语义关联。

        全张量运算（无 for 循环）：
            S = Φ_norm @ Φ_norm^T
        """
        if self._semantic_cache is not None:
            return self._semantic_cache

        Phi = self.feature_matrix()
        if Phi.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        # L2 归一化（沿特征维度）
        norms = Phi.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        Phi_norm = Phi / norms

        # 余弦相似度矩阵（全张量运算）
        S = Phi_norm @ Phi_norm.transpose(-1, -2)
        # 数值稳定：截断到 [-1, 1]
        S = torch.clamp(S, min=-1.0 + self.eps, max=1.0 - self.eps)

        self._semantic_cache = S
        return S

    def causal_adjacency(self) -> Tensor:
        """
        因果邻接矩阵 C ∈ R^{N×N}。

        C_ij = sim_semantic(e_i, e_j) · K_temporal(t_j - t_i)

        其中：
            sim_semantic = cos(φ_i, φ_j)           语义相似度
            K_temporal(Δt) = exp(-|Δt|/τ) · H(Δt)  因果时间核
            H(Δt) = (1 + sign(Δt))/2               因果方向阶跃
                                                   （仅当 j 在 i 之后，即 Δt > 0）

        物理意义：
            因果邻接 = 语义相关 × 时间因果方向
            未来事件依赖过去事件，反之不成立（因果不可逆）

        全张量运算（无 for 循环）：
            ΔT = t_j - t_i（时间差矩阵）
            K = exp(-|ΔT|/τ) · H(ΔT)
            C = S ⊙ K（Hadamard 积）
        """
        if self._causal_cache is not None:
            return self._causal_cache

        S = self.semantic_similarity()
        if S.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        t = self.timestamp_vector()
        # 时间差矩阵 ΔT_ij = t_j - t_i（全张量运算）
        # ΔT = t[None, :] - t[:, None]
        DeltaT = t.unsqueeze(0) - t.unsqueeze(1)  # (N, N)

        # 因果时间核 K(Δt) = exp(-|Δt|/τ) · H(Δt)
        # H(Δt) = (1 + sign(Δt))/2，仅当 Δt > 0（j 在 i 之后）为 1
        # 使用 sigmoid 软化阶跃（保持可微，避免离散跳变）
        # 软阶跃：H_soft(Δt) = sigmoid(Δt / τ_soft)，τ_soft 由因果尺度推导
        tau_soft = self.tau_causal * 0.1  # 软阶跃尺度（由因果尺度推导）
        H_soft = torch.sigmoid(DeltaT / tau_soft)

        # 时间衰减
        K_temporal = torch.exp(-DeltaT.abs() / self.tau_causal) * H_soft

        # 因果邻接 = 语义相似度 ⊙ 时间核
        C = S * K_temporal

        # 对角线置零（事件不自因果）
        N = C.shape[0]
        C = C * (1.0 - torch.eye(N, dtype=torch.float64))

        self._causal_cache = C
        return C

    def causal_laplacian(self) -> Tensor:
        """
        因果图拉普拉斯矩阵 L ∈ R^{N×N}。

        L = D - C
        其中 D = diag(Σ_j C_ij)（度矩阵）

        物理意义：
            拉普拉斯矩阵编码图的扩散性质。
            其伪逆 L^+ 给出节点间的扩散距离（L1 层使用）。

        全张量运算（无 for 循环）：
            D = diag(C.sum(dim=-1))
            L = D - C
        """
        C = self.causal_adjacency()
        if C.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.float64)

        # 度矩阵（对角）
        degrees = C.sum(dim=-1)
        D = torch.diag(degrees)
        L = D - C
        return L

    def summary(self) -> dict[str, Tensor | int | float]:
        """事件图摘要统计（仅供审计）。"""
        if self.n_events == 0:
            return {"n_events": 0}

        C = self.causal_adjacency()
        S = self.semantic_similarity()
        return {
            "n_events": self.n_events,
            "feature_dim": self.feature_dim,
            "causal_density": float(C.mean()),
            "causal_max": float(C.max()),
            "semantic_mean": float(S.mean()),
            "time_span": float(self.timestamp_vector().max() - self.timestamp_vector().min()),
        }
