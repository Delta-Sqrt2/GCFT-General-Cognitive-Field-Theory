"""
任务二：认知格点规范场

战略定位（v2.1 任务二）：
    道德/法律是时空的内禀属性，类似于晶格中的缺陷结构。
    违背道德产生的痛苦，是几何回路无法闭合产生的张力。
    自由不是随心所欲，而是寻找满足规范对称性的路径。
    越界 = 产生几何曲率 = 痛苦。

物理与哲学直觉：
    - 物理：格点规范场论（Wilson 格点规范）。规范场 A 是切空间的联络，
            必须是矩阵，用于旋转状态向量的相位。标量只能缩放能量，
            无法描述"方向性的扭曲"（如道德冲突）。
    - 哲学：自由 = 寻找满足规范对称性的路径（"从心所欲不逾矩"）。
    - 工程：离散规范联络 A_mn ∈ R^{d×d}，Wilson 环曲率，协变差分。

数学定义（严格可微，无降级）：
    步骤 A：离散规范联络 A_mn
        在事件图每条边 (m→n) 上定义矩阵 A_mn ∈ R^{d×d}。
        初始化：A_mn ≈ I（单位矩阵，无扭曲）。
        学习机制：A_mn 由社会共识事件通过反向传播调整。

    步骤 B：格点曲率张量 F_mnp
        寻找闭合三角回路 (m→n→p→m)。
        Wilson 环：W_mnp = A_mn · A_np · A_pm
        曲率：F_mnp = W_mnp - I
        痛苦势能：V = Tr(F_mnp^T · F_mnp)

    步骤 C：离散协变差分
        Δφ = φ_n - A_mn · φ_m
        A_mn 旋转了参考系。如果"知法犯法"（A_mn 指向错误），
        Δφ 会产生巨大的投影分量，导致系统动荡。

工程铁律（v2.1 专属）：
    1. 严禁规范场标量化：A 必须是 (d, d) 矩阵，非标量
    2. 严禁连续导数：离散协变差分，非 ∂_μ
    3. 严禁个体流形：规范场定义在全局事件图边上
    4. 全张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor


class GaugeField:
    """
    认知格点规范场：离散规范联络 + Wilson 环曲率 + 协变差分。

    使用方式：
        gauge = GaugeField(n_dims=8)
        # 在事件图边上初始化规范联络
        gauge.initialize_from_adjacency(C_global, n_events=N)
        # 计算曲率（痛苦势能）
        curvature = gauge.compute_curvature(triangles)
        pain_potential = gauge.pain_potential(triangles)
        # 协变差分
        delta_phi = gauge.covariant_difference(phi_m, phi_n, m, n)

    白盒保证：
        - 规范场 A 是 (d, d) 矩阵（非标量）
        - 离散协变差分（非连续导数）
        - Wilson 环曲率（非连续曲率张量）
        - 全张量运算，可微
    """

    def __init__(self, n_dims: int, eps: float = 1e-10):
        """
        参数：
            n_dims: 认知维度 d（规范场 A_mn ∈ R^{d×d}）
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.eps = float(eps)

        # 规范联络存储：A[m, n] = (d, d) 矩阵
        # 用 (N, N, d, d) 张量存储，全张量运算
        self._gauge_links: Tensor | None = None  # (N, N, d, d)
        self._n_events: int = 0

    def initialize_from_adjacency(
        self,
        adjacency: Tensor,
        n_events: int,
        init_scale: float = 0.1,
    ) -> Tensor:
        """
        从因果邻接矩阵初始化规范联络。

        A_mn = I + ε · noise · adjacency_mn
        其中 ε = init_scale（小扰动，保持接近单位矩阵）。

        物理意义：
            初始规范场接近单位矩阵（无扭曲）。
            社会共识事件通过反向传播调整 A_mn，
            使符合共识的事件序列作用量最小。

        参数：
            adjacency: 因果邻接矩阵 (N, N)
            n_events: 事件数 N
            init_scale: 初始扰动幅度（小值，保持稳定性）

        返回：
            gauge_links: 规范联络张量 (N, N, d, d)

        严禁：
            - 标量规范场（必须是 d×d 矩阵）
            - 连续导数初始化
        """
        N = int(n_events)
        d = self.n_dims
        adj = adjacency.to(torch.float64)

        # 单位矩阵基底 (d, d)
        I = torch.eye(d, dtype=torch.float64)

        # 初始扰动：小随机矩阵 × 邻接强度
        # 使用确定性初始化（非随机，保证可复现）
        # 扰动 = sin(m·n) 的小幅度振荡（可微，确定性）
        m_idx = torch.arange(N, dtype=torch.float64).unsqueeze(1)
        n_idx = torch.arange(N, dtype=torch.float64).unsqueeze(0)
        perturbation_pattern = torch.sin(m_idx * n_idx * 0.1) * init_scale  # (N, N)

        # 规范联络 A_mn = I + perturbation · (adjacency_mn) · skew_matrix
        # 使用反对称矩阵作为扰动（保持 A 接近正交，模拟旋转）
        # skew = [[0, -θ], [θ, 0]] 形式（2D 旋转的推广）
        # 对每个 (m,n)，扰动 = θ_mn · J_d，其中 J_d 是反对称生成元
        J_generator = self._skew_generator(d)  # (d, d) 反对称

        # A_mn = I + θ_mn · J · adjacency_mn
        # 形状 (N, N, d, d)
        gauge = (
            I.unsqueeze(0).unsqueeze(0).expand(N, N, d, d)
            + perturbation_pattern.unsqueeze(-1).unsqueeze(-1)
            * J_generator.unsqueeze(0).unsqueeze(0)
            * adj.unsqueeze(-1).unsqueeze(-1)
        )

        self._gauge_links = gauge
        self._n_events = N
        return gauge

    def _skew_generator(self, d: int) -> Tensor:
        """
        构造 d×d 反对称生成元矩阵 J。
        J[i, i+1] = 1, J[i+1, i] = -1（循环反对称）。

        物理意义：
            反对称生成元对应旋转的无穷小生成元。
            规范场 A = I + θ·J 描述小角度旋转。
        """
        J = torch.zeros(d, d, dtype=torch.float64)
        for i in range(d - 1):
            J[i, i + 1] = 1.0
            J[i + 1, i] = -1.0
        # 如果 d 是奇数，最后一个维度无配对，保持 0
        return J

    def wilson_loop(self, m: int, n: int, p: int) -> Tensor:
        """
        计算 Wilson 环 W_mnp = A_mn · A_np · A_pm。

        物理意义：
            闭合三角回路的规范场乘积。
            如果规范场可积（无曲率），W = I。
            如果 W ≠ I，说明回路有曲率（道德冲突/越界）。

        参数：
            m, n, p: 三角回路的事件索引

        返回：
            W: Wilson 环矩阵 (d, d)
        """
        if self._gauge_links is None:
            raise RuntimeError("规范场未初始化，请先调用 initialize_from_adjacency")

        A_mn = self._gauge_links[m, n]  # (d, d)
        A_np = self._gauge_links[n, p]
        A_pm = self._gauge_links[p, m]

        # Wilson 环 = 矩阵乘积链
        W = A_mn @ A_np @ A_pm
        return W

    def compute_curvature(self, triangles: list[tuple[int, int, int]]) -> Tensor:
        """
        计算格点曲率张量 F_mnp = W_mnp - I。

        参数：
            triangles: 三角回路列表 [(m, n, p), ...]

        返回：
            curvatures: 曲率张量 (T, d, d)，T = 三角形数

        物理意义：
            曲率 = Wilson 环偏离单位矩阵的程度。
            非零曲率 = 道德冲突/越界 = 痛苦。
        """
        if self._gauge_links is None:
            raise RuntimeError("规范场未初始化")

        d = self.n_dims
        I = torch.eye(d, dtype=torch.float64)

        curvatures = []
        for m, n, p in triangles:
            W = self.wilson_loop(m, n, p)
            F = W - I
            curvatures.append(F)

        if not curvatures:
            return torch.zeros(0, d, d, dtype=torch.float64)

        return torch.stack(curvatures, dim=0)  # (T, d, d)

    def pain_potential(self, triangles: list[tuple[int, int, int]]) -> Tensor:
        """
        痛苦势能 V = Σ Tr(F_mnp^T · F_mnp)。

        物理意义：
            痛苦 = 曲率的 Frobenius 范数平方和。
            道德冲突越强，曲率越大，痛苦势能越高。
            这是"内疚"、"羞耻"的涌现定义。

        数学：
            V = Σ_mnp ||F_mnp||_F^2 = Σ_mnp Tr(F^T · F)

        严禁：
            - 标量规范场（无法计算矩阵迹）
            - 连续曲率（必须离散 Wilson 环）
        """
        curvatures = self.compute_curvature(triangles)  # (T, d, d)
        if curvatures.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float64)

        # Tr(F^T · F) = ||F||_F^2（全张量运算）
        pain_per_triangle = (curvatures ** 2).sum(dim=(-1, -2))  # (T,)
        total_pain = pain_per_triangle.sum()
        return total_pain

    def covariant_difference(self, phi_m: Tensor, phi_n: Tensor, m: int, n: int) -> Tensor:
        """
        离散协变差分 Δφ = φ_n - A_mn · φ_m。

        物理意义：
            A_mn 旋转了参考系。如果"知法犯法"（A_mn 指向错误），
            Δφ 会产生巨大的投影分量，导致系统动荡。
            这是"越界产生痛苦"的数学实现。

        参数：
            phi_m: 事件 m 的特征向量 (d,)
            phi_n: 事件 n 的特征向量 (d,)
            m, n: 事件索引

        返回：
            delta_phi: 协变差分 (d,)

        严禁：
            - 标量缩放（必须矩阵乘法）
            - 连续导数（必须离散差分）
        """
        if self._gauge_links is None:
            raise RuntimeError("规范场未初始化")

        A_mn = self._gauge_links[m, n]  # (d, d)
        phi_m = phi_m.to(torch.float64)
        phi_n = phi_n.to(torch.float64)

        # 协变差分 = φ_n - A_mn · φ_m
        delta_phi = phi_n - A_mn @ phi_m
        return delta_phi

    def batch_covariant_difference(
        self,
        feature_matrix: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """
        批量协变差分（全张量运算）。

        对所有边 (m→n) 计算 Δφ_mn = φ_n - A_mn · φ_m。

        参数：
            feature_matrix: 特征矩阵 (N, d)
            adjacency: 邻接矩阵 (N, N)

        返回：
            delta_phi: 协变差分张量 (N, N, d)

        全张量运算（无 for 循环）：
            A·φ = einsum('mnij,nj->mni', A, φ)
        """
        if self._gauge_links is None:
            raise RuntimeError("规范场未初始化")

        Phi = feature_matrix.to(torch.float64)  # (N, d)
        N = Phi.shape[0]

        # A_mn · φ_m：对每条边 (m,n)，计算 A_mn @ phi_m
        # einsum: A(m,n,i,j) * phi(m,j) -> (m,n,i)
        # 注意：A_mn 作用在 phi_m 上（源节点）
        A_phi = torch.einsum('mnij,mj->mni', self._gauge_links, Phi)  # (N, N, d)

        # φ_n 扩展到 (N, N, d)
        Phi_expanded = Phi.unsqueeze(0).expand(N, N, -1)  # (N, N, d) = phi_n

        # 协变差分 = φ_n - A_mn · φ_m
        delta_phi = Phi_expanded - A_phi  # (N, N, d)

        # 用邻接矩阵掩码（仅保留有因果连接的边）
        delta_phi = delta_phi * adjacency.unsqueeze(-1)

        return delta_phi

    def gauge_invariant_action(
        self,
        feature_matrix: Tensor,
        adjacency: Tensor,
        triangles: list[tuple[int, int, int]],
    ) -> Tensor:
        """
        规范不变作用量 S_gauge。

        S_gauge = Σ_{m→n} ||Δφ_mn||^2 + λ · Σ_{mnp} Tr(F_mnp^T · F_mnp)

        物理意义：
            第一项：协变差分的能量（越界代价）
            第二项：曲率痛苦势能（道德冲突）
            λ：曲率权重（由理论推导）

        严禁：
            - 标量规范场（无法保证规范不变性）
            - 连续作用量（必须离散求和）
        """
        # 协变差分能量
        delta_phi = self.batch_covariant_difference(feature_matrix, adjacency)  # (N, N, d)
        kinetic = (delta_phi ** 2).sum()  # 标量

        # 曲率痛苦势能
        pain = self.pain_potential(triangles)

        # λ 由维数推导（结构常数）
        lam = 1.0 / (self.n_dims + 2)

        # 规范不变作用量
        S = kinetic + lam * pain
        return S

    def learn_gauge_from_consensus(
        self,
        feature_matrix: Tensor,
        adjacency: Tensor,
        triangles: list[tuple[int, int, int]],
        n_iterations: int = 10,
        lr: float = 0.01,
    ) -> list[Tensor]:
        """
        从社会共识事件学习规范场。

        学习机制：
            通过反向传播调整 A_mn，使得符合共识的事件序列作用量最小。
            这是"道德内化"的数学实现——
            规范场通过社会共识事件训练，形成"良知"。

        参数：
            feature_matrix: 社会共识事件特征 (N, d)
            adjacency: 因果邻接 (N, N)
            triangles: 三角回路
            n_iterations: 训练迭代数
            lr: 学习率

        返回：
            loss_history: 作用量损失历史
        """
        if self._gauge_links is None:
            self.initialize_from_adjacency(adjacency, feature_matrix.shape[0])

        # 确保规范联络可微
        self._gauge_links.requires_grad_(True)

        loss_history = []
        optimizer = torch.optim.Adam([self._gauge_links], lr=lr)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            S = self.gauge_invariant_action(feature_matrix, adjacency, triangles)
            S.backward()
            optimizer.step()
            loss_history.append(float(S))

        return loss_history

    def summary(self, triangles: list[tuple[int, int, int]]) -> dict[str, Tensor | float]:
        """规范场摘要（仅供审计）。"""
        if self._gauge_links is None:
            return {"status": "未初始化"}

        pain = self.pain_potential(triangles)
        return {
            "gauge_shape": self._gauge_links.shape,
            "n_triangles": len(triangles),
            "pain_potential": float(pain),
            "gauge_deviation_from_identity": float(
                (self._gauge_links - torch.eye(self.n_dims, dtype=torch.float64).unsqueeze(0).unsqueeze(0)).norm()
            ),
        }
