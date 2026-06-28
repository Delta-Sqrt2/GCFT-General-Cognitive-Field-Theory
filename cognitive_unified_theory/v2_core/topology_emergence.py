"""
L1 拓扑涌现层 —— 度规 g_μν 从因果距离矩阵涌现

战略定位（v2.0 任务二）：
    废除 v1.x 中预先给定的度规 g_μν。
    度规必须由事件因果网络的距离结构涌现出来。
    这是"时空从事件涌现"的数学实现。

物理与哲学直觉：
    - 物理：圈量子引力——时空度规从离散因果网络涌现
    - 哲学：莱布尼茨关系主义——空间不是容器，是物体间的关系网络
    - 工程：拉普拉斯伪逆 + MDS 都是可微的矩阵运算，保持 autograd

数学管道（严格可微）：
    1. 因果距离矩阵 D_ij：
       拉普拉斯矩阵 L = diag(Σ C) - C
       伪逆 L^+（通过特征分解实现，可微）
       扩散距离 D_ij = √(L^+_ii + L^+_jj - 2·L^+_ij)

    2. 可微多维缩放（MDS）：
       双中心化距离平方矩阵 B = -1/2 · J · D² · J
       其中 J = I - 1/N · 11^T（中心化矩阵）
       特征分解 B = V Λ V^T
       取前 n_dims 个特征向量作为切空间基底 Ξ
       度规 g_μν = Ξ^T · B · Ξ / N（局部内积矩阵）

    3. 拓扑相变检测：
       当因果网络关键节点断开时，C 变化 → D 突变 → g 特征值结构改变
       这就是"创伤改变时空本身"的数学过程

工程铁律（v2.0 专属）：
    1. 严禁预先初始化度规 g_μν：必须由因果距离涌现
    2. 严禁 detach()：MDS 全程可微
    3. 严禁 for 循环遍历节点：全部矩阵运算
    4. 严禁连续场论 ∫dVdt：用 Σ 离散求和
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import stable_eigh, symmetric_part


class TopologyEmergence:
    """
    拓扑涌现层：从因果距离矩阵涌现度规 g_μν。

    使用方式：
        topology = TopologyEmergence(n_dims=8)
        g = topology.emerge_metric(ontology)
        # g: 涌现的度规张量 (n_dims, n_dims)，由因果网络决定

        # 验证拓扑相变：断开关键节点
        g_after = topology.emerge_metric(ontology_after_perturbation)
        phase_shift = topology.phase_shift(g, g_after)

    白盒保证：
        - 度规由因果距离涌现（非预先给定）
        - 全程可微（无 detach）
        - 全张量运算（无 for 循环）
        - 拓扑相变可量化
    """

    def __init__(self, n_dims: int, eps: float = 1e-10):
        """
        参数：
            n_dims: 涌现度规的维度（认知空间的内禀维度）
                   物理意义：认知流形的内禀维度，由理论假设给定
                   （类似弦理论选择紧致化维度）
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.eps = float(eps)

        # 缓存上一次涌现的度规（用于相变检测）
        self._last_metric: Tensor | None = None
        self._last_eigvals: Tensor | None = None

    def laplacian_pseudoinverse(self, L: Tensor) -> Tensor:
        """
        拉普拉斯矩阵的伪逆 L^+（可微实现）。

        数学：
            L = V Λ V^T（特征分解）
            L^+ = V Λ^+ V^T
            其中 Λ^+ = diag(1/λ_i) 对 λ_i > ε，否则 0

        物理意义：
            拉普拉斯伪逆编码图的扩散性质。
            L^+_ij 给出节点 i, j 之间的扩散核，
            反映信息/影响在因果网络上的传播难度。

        可微性：
            使用 torch.linalg.eigh（可微），严禁 detach。
        """
        L = symmetric_part(L.to(torch.float64))
        N = L.shape[0]

        # 特征分解（可微）
        eigvals, eigvecs = stable_eigh(L)

        # 伪逆：对非零特征值取倒数，零特征值保持零
        # 软阈值（保持可微，避免离散 if）
        # 1/λ 软化：1/(λ + ε·exp(-λ/ε))，当 λ 大时趋近 1/λ，λ 小时趋近 0
        soft_inverse = 1.0 / (eigvals + self.eps * torch.exp(-eigvals / self.eps))
        # 但更稳定的方式：直接对大于阈值的特征值取倒数
        # 使用软掩码：mask = sigmoid((λ - ε_threshold) / ε_soft)
        eigvals_clamped = torch.clamp(eigvals, min=self.eps)
        # 软掩码：特征值越大，伪逆贡献越大
        threshold = self.eps * 10.0
        soft_mask = torch.sigmoid((eigvals - threshold) / (threshold * 0.1))
        eigvals_pseudo = soft_mask / eigvals_clamped

        # L^+ = V Λ^+ V^T
        L_plus = eigvecs @ torch.diag(eigvals_pseudo) @ eigvecs.transpose(-1, -2)
        return L_plus

    def diffusion_distance(self, L: Tensor) -> Tensor:
        """
        因果扩散距离矩阵 D ∈ R^{N×N}。

        D_ij = √(L^+_ii + L^+_jj - 2·L^+_ij)

        物理意义：
            两个事件在因果网络上的"扩散距离"。
            因果联系越强（影响传播越容易），距离越近。
            "父亲离去"这样的关键节点断开后，
            相关事件的扩散距离突变，反映时空拓扑改变。

        全张量运算（无 for 循环）：
            D² = diag(L^+)·1^T + 1·diag(L^+)^T - 2·L^+
            D = √(D²)
        """
        L_plus = self.laplacian_pseudoinverse(L)
        N = L_plus.shape[0]

        # 提取对角元素（扩散距离公式）
        diag_L_plus = torch.diagonal(L_plus)  # (N,)

        # D²_ij = L^+_ii + L^+_jj - 2·L^+_ij（全张量运算）
        D_sq = (
            diag_L_plus.unsqueeze(1)  # L^+_ii 列扩展
            + diag_L_plus.unsqueeze(0)  # L^+_jj 行扩展
            - 2.0 * L_plus
        )
        # 数值稳定：截断非负
        D_sq = torch.clamp(D_sq, min=0.0)

        # D = √(D²)
        D = torch.sqrt(D_sq + self.eps)
        return D

    def _double_centering(self, D: Tensor) -> Tensor:
        """
        双中心化变换 B = -1/2 · J · D² · J。

        数学：
            J = I - (1/N)·11^T（中心化矩阵）
            B = -1/2 · J · D² · J

        物理意义：
            将距离矩阵转换为内积矩阵。
            B 的特征向量给出嵌入空间的坐标，
            B 本身就是涌现度规的"种子"。

        全张量运算（无 for 循环）。
        """
        N = D.shape[0]
        D_sq = D ** 2

        # 中心化矩阵 J = I - (1/N)·11^T
        J = torch.eye(N, dtype=torch.float64) - 1.0 / N * torch.ones(N, N, dtype=torch.float64)

        # B = -1/2 · J · D² · J
        B = -0.5 * J @ D_sq @ J
        # 对称化（消除数值误差）
        B = symmetric_part(B)
        return B

    def emerge_metric(self, L: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        从因果拉普拉斯矩阵涌现度规 g_μν。

        完整管道（严格可微）：
            1. 因果距离 D = diffusion_distance(L)
            2. 双中心化 B = -1/2 · J · D² · J
            3. 特征分解 B = V Λ V^T
            4. 取前 n_dims 个特征向量作为切空间基底 Ξ
            5. 度规 g_μν = Ξ^T · B · Ξ / N（局部内积矩阵）

        参数：
            L: 因果拉普拉斯矩阵 (N, N)

        返回：
            metric: 涌现的度规张量 (n_dims, n_dims)
            basis: 切空间基底 (N, n_dims)
            eigvals: B 的特征值（降序，用于分析拓扑结构）

        物理意义：
            度规不再是"预先给定的舞台"，而是从事件因果网络涌现的几何。
            当因果网络变化（如创伤事件加入），度规可微地变化。
            这就是"时空从事件涌现"的数学实现。

        严禁：
            - 预先初始化度规 g_μν
            - detach() 断裂梯度
            - for 循环遍历节点
        """
        L = L.to(torch.float64)
        N = L.shape[0]

        # 1. 因果距离矩阵
        D = self.diffusion_distance(L)

        # 2. 双中心化
        B = self._double_centering(D)

        # 3. 特征分解（可微）
        eigvals, eigvecs = stable_eigh(B)
        # 特征值升序，翻转得到降序
        eigvals = torch.flip(eigvals, dims=[0])
        eigvecs = torch.flip(eigvecs, dims=[1])

        # 4. 取前 n_dims 个特征向量作为切空间基底
        n = min(self.n_dims, N)
        basis = eigvecs[:, :n]  # (N, n)

        # 5. 涌现度规 g_μν = Ξ^T · B · Ξ / N
        # 物理意义：局部切空间上的内积矩阵
        # 归一化因子 N 由事件数推导（非硬编码）
        metric = basis.transpose(-1, -2) @ B @ basis / float(N)
        metric = symmetric_part(metric)

        # 度规正定性保证：对角加载（保持可微）
        # 正则化强度由最小特征值推导（非硬编码）
        metric_eigvals, _ = stable_eigh(metric)
        min_eig = metric_eigvals.min()
        # 软正则化：当最小特征值接近 0 时，增加正则化
        reg = torch.clamp(-min_eig + self.eps, min=0.0)
        metric = metric + reg * torch.eye(n, dtype=torch.float64)

        # 迹归一化：trace(g) = n（与 v1.2 度规归一化一致）
        # 物理意义：度规的迹 = 认知空间的总"体积尺度"
        # 归一化保证不同事件图涌现的度规可比较
        trace_g = torch.trace(metric)
        if float(trace_g) > self.eps:
            metric = metric * (float(n) / trace_g)

        # 缓存（用于相变检测）
        self._last_metric = metric
        self._last_eigvals = eigvals

        return metric, basis, eigvals

    def emerge_from_ontology(self, ontology) -> tuple[Tensor, Tensor, Tensor]:
        """
        从 EventOntology 涌现度规（便捷接口）。

        参数：
            ontology: EventOntology 实例

        返回：
            metric, basis, eigvals（同 emerge_metric）
        """
        L = ontology.causal_laplacian()
        return self.emerge_metric(L)

    def phase_shift(self, g_before: Tensor, eigvals_before: Tensor, g_after: Tensor, eigvals_after: Tensor) -> Tensor:
        """
        拓扑相变量化：两次度规涌现的差异。

        数学：
            由于扰动前后事件数可能不同，度规维度可能不同，
            无法直接矩阵相减。改用标量不变量比较：
                Δ_curvature = |κ_after - κ_before| / |κ_before|
                Δ_eff_dim = |d_eff_after - d_eff_before| / |d_eff_before|
                phase_shift = sqrt(Δ_curvature² + Δ_eff_dim²)

        物理意义：
            当因果网络关键节点断开（如"父亲离去"），
            度规发生拓扑相变。相变量化这种"时空重构"的强度。
            曲率变化 = 认知空间弯曲程度改变
            有效维度变化 = 认知空间维度改变（降维=僵化，升维=开放）

        应用：
            创伤 = 拓扑相变。创伤事件加入因果网络后，
            度规的相变量反映创伤对认知时空的重构程度。
        """
        # 曲率强度变化
        kappa_before = self.curvature_intensity(g_before)
        kappa_after = self.curvature_intensity(g_after)
        delta_kappa = (kappa_after - kappa_before).abs() / (kappa_before.abs() + self.eps)

        # 有效维度变化
        d_eff_before = self.effective_dimension(eigvals_before)
        d_eff_after = self.effective_dimension(eigvals_after)
        delta_dim = (d_eff_after - d_eff_before).abs() / (d_eff_before.abs() + self.eps)

        # 综合相变量（欧氏范数）
        phase_shift = torch.sqrt(delta_kappa ** 2 + delta_dim ** 2)
        return phase_shift

    def curvature_intensity(self, metric: Tensor) -> Tensor:
        """
        度规曲率强度（复用 v1.2 标量曲率概念，但度规是涌现的）。

        数学：
            曲率强度 = ||g - I||_F / ||I||_F
            （度规偏离欧氏基准的程度）

        物理意义：
            度规越偏离单位矩阵，认知空间越弯曲。
            高曲率 = 创伤聚集区 = "引力井"。
        """
        g = symmetric_part(metric.to(torch.float64))
        n = g.shape[0]
        I = torch.eye(n, dtype=torch.float64)
        return (g - I).norm() / I.norm()

    def effective_dimension(self, eigvals: Tensor) -> Tensor:
        """
        涌现度规的有效维度（信息论推导，非硬编码）。

        数学：
            有效维度 = exp(H(p))，其中 p_i = λ_i / Σλ_i
            H(p) = -Σ p_i · log(p_i)（Shannon 熵）

        物理意义：
            度规特征值分布越均匀，有效维度越高（认知空间越丰富）。
            度规特征值集中（少数维度主导），有效维度低（认知僵化）。
        """
        eigvals = torch.clamp(eigvals, min=self.eps)
        n = min(self.n_dims, eigvals.shape[0])
        eigvals_top = eigvals[:n]

        # 归一化为概率分布
        total = eigvals_top.sum()
        p = eigvals_top / total

        # Shannon 熵
        H = -(p * torch.log(p)).sum()

        # 有效维度 = exp(H)
        return torch.exp(H)

    def summary(self, metric: Tensor, eigvals: Tensor) -> dict[str, Tensor]:
        """涌现度规摘要（仅供审计）。"""
        return {
            "metric_shape": metric.shape,
            "curvature_intensity": self.curvature_intensity(metric),
            "effective_dimension": self.effective_dimension(eigvals),
            "top_eigvals": eigvals[:min(5, eigvals.shape[0])],
            "metric_trace": torch.trace(metric),
            "metric_det": torch.det(metric + self.eps * torch.eye(metric.shape[0], dtype=torch.float64)),
        }
