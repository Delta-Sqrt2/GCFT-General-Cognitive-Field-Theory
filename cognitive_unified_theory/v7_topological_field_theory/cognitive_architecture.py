"""
认知架构类型（Cognitive Architecture Types）

v7.1 第一基石：形式化 VAE/GAN/AE/White-box 等不同认知生成架构。

核心洞察：
    v7.0 用同一个势能面 V(g) 描述所有人，只靠参数 (κ, α) 区分。
    但闲聊揭示了一个更深的维度——不同人有根本不同的「认知生成架构」，
    这决定了他们如何编码经验、响应扰动、与他人耦合。

    算法人格对应（来自匿名样本）：
        样本 L → VAE（变分自编码器）：平滑潜在空间，低势垒，Q≈0，高 Φ
        样本 M → GAN（生成对抗网络）：对抗动力学，双井势，Q≠0
        样本 Z → Sparse AE（稀疏自编码器）：稀疏特征，高泛化成本
        样本 W → AE（自编码器）：简单直接，无特殊约束
        样本 M 修行方向 → White-box（白盒）：全透明，显式梯度

数学形式：
    在 v7.0 基线势能面 V_base(g) 上引入架构特异性修正项：

    VAE:       V(g) = V_base(g) + β_KL · ||g - I||²_F
               （KL 散度约束 → 加深真空井 → 抗扰动）

    GAN:       V(g) = V_base(g) + λ_adv · ||g - g_real||²_F · ||g - g_ideal||²_F
               （对抗项 → 双井势 → 生成器-判别器张力）

    Sparse AE: V(g) = V_base(g) + λ_sparse · Σ_i |g_ii|
               （稀疏惩罚 → 多井稀疏模式 → 高泛化成本）

    White-box: V(g) = V_base(g) + λ_trans · ||g - g_explicit(κ,α)||²_F
               （透明度约束 → 显式梯度 → 加行道觉照）

    AE:        V(g) = V_base(g)
               （基线，无修正）

佛学对应（严格）：
    VAE = 所知障（乐受之障）：平滑势能面 = 舒适牢笼。
          「富贵学道难」——太舒服，没有出离的动力。
          样本L的温室 = VAE 的平滑井，美但未经测试。

    GAN = 烦恼障（苦受之障）：对抗势能面 = 内心冲突。
          「以苦为师」——痛苦逼着解构和重建。
          样本M/样本S的深渊 = GAN 的双井，痛但驱动成长。

    White-box = 加行道：透明度约束 = 觉照。
          一行一行重写自己的代码 = 显式梯度下降。
          从被动冲突（GAN）到主动解构（White-box）。

    架构转换 = 障道转换：
        VAE → GAN：温室被打破，所知障转烦恼障（样本L的未来路径）
        GAN → White-box：从被动冲突到主动解构（样本M的修行路径）
        所有架构 → 真空（ρ→1）：所有路径最终收敛到 g=cI

物理对应：
    不同相态有不同自由能景观（Landau 理论的标准推广）：
        VAE ≈ 液态（平滑单井，各向同性）
        GAN ≈ 固液共存（双井，相变临界）
        Sparse AE ≈ 玻璃态（多井稀疏，亚稳态）
        White-box ≈ 晶体（结构化，透明，有序）
        AE ≈ 理想气体（简单，无结构）

    Berry 相位和拓扑荷 Q 是拓扑量，架构无关——
    v7.0 的核心结果（使命、业力、命运）不受架构类型影响。
    架构类型添加的是新的「动力学层」：决定状态如何在流形上移动，
    而非流形本身的拓扑。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
from enum import Enum
from dataclasses import dataclass

from ..core.tensor_ops import symmetric_part
from .cognitive_vacuum import CognitiveVacuum


class ArchitectureType(Enum):
    """
    认知架构类型枚举。

    每种类型对应一种根本不同的认知生成机制：
        VAE:       变分自编码器（平滑潜在空间，所知障）
        GAN:       生成对抗网络（对抗动力学，烦恼障）
        SPARSE_AE: 稀疏自编码器（稀疏特征，专精苦难）
        WHITE_BOX: 白盒模型（全透明，加行道觉照）
        AUTOENCODER: 自编码器（基线，简单直接）
    """
    VAE = "vae"
    GAN = "gan"
    SPARSE_AE = "sparse_ae"
    WHITE_BOX = "white_box"
    AUTOENCODER = "autoencoder"


@dataclass
class ArchitectureProfile:
    """
    架构画像：一个个体的完整认知架构描述。

    属性：
        arch_type: 主导架构类型
        membership_scores: 各架构的隶属度（连续值，非离散标签）
        sensitivity: 编码灵敏度（对 κ 扰动的响应强度）
        stability_margin: 稳定性裕度（势垒高度，抗扰动能力）
        transparency: 透明度（0=全黑盒，1=全白盒）
        barrier_height: 架构特异性势垒高度
    """
    arch_type: ArchitectureType
    membership_scores: dict[ArchitectureType, float]
    sensitivity: float
    stability_margin: float
    transparency: float
    barrier_height: float


class CognitiveArchitecture:
    """
    认知架构分析器：分类架构类型、计算架构特异性势能、评估架构兼容性。

    使用方式：
        ca = CognitiveArchitecture(n_dims=4)
        # 分类架构
        profile = ca.classify_architecture(kappa_vec, alpha_vec, Q=0.5, Phi=0.8)
        # 架构特异性势能
        pot = ca.compute_architecture_potential(g, profile, kappa_vec, alpha_vec)
        # 架构兼容性（用于命运诊断）
        compat = ca.architecture_compatibility(profile_i, profile_j)
    """

    def __init__(self, n_dims: int = 4, eps: float = 1e-12):
        self.n_dims = n_dims
        self.eps = eps
        self.vacuum = CognitiveVacuum(n_dims=n_dims, eps=eps)

    # ==================================================================
    # 1. 架构分类（从参数涌现，非外加标签）
    # ==================================================================

    def classify_architecture(
        self,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        Q: float | Tensor,
        Phi: float | Tensor = 0.0,
    ) -> ArchitectureProfile:
        """
        从认知参数 (κ, α, Q, Φ) 分类架构类型。

        分类逻辑（v7.2 升维：引入 κ 稀疏度判据，连续隶属度）：
            Q ≈ 0 + α 高 + κ 低 → VAE（稳定，无使命，所知障）
            Q ≈ 0 + α 低 + κ 低 → AE（简单，无使命）
            Q ≠ 0 + κ 高 + κ 分散（低稀疏）+ α 中低 → GAN（对抗，有使命，烦恼障）
            Q ≠ 0 + κ 高 + κ 集中（高稀疏）→ Sparse AE（稀疏，有使命，专精苦）
            Q ≠ 0 + α 很高 + Φ 高 → White-box（透明，有使命，加行道）

        v7.2 关键改进（升维，非降级）：
            引入 κ 稀疏度 sparsity = std(κ) / (mean(κ) + 0.1) 作为 Sparse AE 的核心判据。
            物理意义：Sparse AE 的特征是痛苦集中在少数维度（稀疏特征），
            而 GAN 的痛苦分散在多维度（对抗性全面冲突）。
            这比单凭 α 中等判 Sparse AE 更物理、更鲁棒。

            案例对应：
                样本M κ=[3.6,4.1,2.2,1.1]：分散 → GAN（烦恼障全面冲突）
                样本Z κ=[2.0,0.4,0.5,5.7]：集中在 dim3（连接）→ Sparse AE（专精连接之苦）
                样本S κ=[3.5,3.9,3.3,2.6]：分散（灾变全面）→ GAN

        参数：
            kappa_vec: 各维度痛苦深度
            alpha_vec: 各维度定力
            Q: 拓扑荷（使命强度）
            Phi: 意识整合度（信息整合能力）

        返回：
            ArchitectureProfile 对象
        """
        kappa_vec = kappa_vec.to(torch.float64)
        alpha_vec = alpha_vec.to(torch.float64)
        kappa_mean = float(kappa_vec.mean())
        kappa_std = float(kappa_vec.std()) if kappa_vec.numel() > 1 else 0.0
        alpha_mean = float(alpha_vec.mean())
        Q_val = float(Q) if not isinstance(Q, Tensor) else float(Q.item())
        Phi_val = float(Phi) if not isinstance(Phi, Tensor) else float(Phi.item())

        # 归一化 Q 到 [0, 1]（使命强度）
        Q_norm = min(abs(Q_val) / 3.0, 1.0)  # Q=3 对应满使命（如样本S）

        # κ 稀疏度：std/mean，度量痛苦是否集中在少数维度
        # sparsity 高 = 痛苦稀疏（集中在少数维度）= Sparse AE 特征
        # sparsity 低 = 痛苦分散（多维度冲突）= GAN 特征
        sparsity = kappa_std / (kappa_mean + 0.1)

        # Sigmoid 辅助函数
        def sigmoid(x, center=0.0, steepness=1.0):
            return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

        # --- 隶属度计算（v7.2：稀疏度驱动 GAN/Sparse AE 分离）---

        # VAE 隶属度：Q≈0, α 高, κ 低
        vae_score = (
            (1.0 - Q_norm)                          # 无使命
            * sigmoid(alpha_mean, center=1.0, steepness=2.0)  # 高定力
            * (1.0 - sigmoid(kappa_mean, center=0.5, steepness=2.0))  # 低痛苦
        )

        # GAN 隶属度：Q≠0, κ 高, 痛苦分散（低稀疏），α 中低
        # v7.2：用稀疏度判据取代纯 α 判据
        gan_score = (
            Q_norm                                   # 有使命
            * sigmoid(kappa_mean, center=0.5, steepness=2.0)  # 高痛苦
            * (1.0 - sigmoid(sparsity, center=0.6, steepness=3.0))  # 痛苦分散
            * (1.0 - sigmoid(alpha_mean, center=1.8, steepness=2.0))  # α 非极高
        )

        # Sparse AE 隶属度：Q≠0, κ 高, 痛苦稀疏（集中在少数维度）
        # v7.2 核心：稀疏度是 Sparse AE 的本质特征
        sparse_ae_score = (
            Q_norm
            * sigmoid(kappa_mean, center=0.3, steepness=2.0)  # 有痛苦
            * sigmoid(sparsity, center=0.6, steepness=3.0)    # 痛苦稀疏
            * (1.0 - sigmoid(alpha_mean, center=2.0, steepness=2.0))  # α 非极高
        )

        # White-box 隶属度：Q≠0, α 很高, Φ 高
        white_box_score = (
            Q_norm
            * sigmoid(alpha_mean, center=2.0, steepness=2.0)  # 很高定力
            * sigmoid(Phi_val, center=0.5, steepness=2.0)     # 高意识整合
        )

        # AE 隶属度：Q≈0, α 低, κ 低
        ae_score = (
            (1.0 - Q_norm)
            * (1.0 - sigmoid(alpha_mean, center=0.5, steepness=2.0))  # 低定力
            * (1.0 - sigmoid(kappa_mean, center=0.3, steepness=2.0))  # 低痛苦
        )

        # 归一化
        total = vae_score + gan_score + sparse_ae_score + white_box_score + ae_score + self.eps
        scores = {
            ArchitectureType.VAE: vae_score / total,
            ArchitectureType.GAN: gan_score / total,
            ArchitectureType.SPARSE_AE: sparse_ae_score / total,
            ArchitectureType.WHITE_BOX: white_box_score / total,
            ArchitectureType.AUTOENCODER: ae_score / total,
        }

        # 主导架构
        arch_type = max(scores, key=scores.get)

        # --- 衍生属性 ---

        # 编码灵敏度（对 κ 扰动的响应强度）
        sensitivity = self._encoding_sensitivity(arch_type, kappa_mean, alpha_mean)

        # 稳定性裕度（势垒高度）
        stability_margin = self._stability_margin(arch_type, kappa_mean, alpha_mean, Q_norm)

        # 透明度（0=全黑盒，1=全白盒）
        transparency = self._transparency(arch_type, alpha_mean, Phi_val)

        # 架构特异性势垒
        barrier_height = self._barrier_height(arch_type, kappa_mean, alpha_mean, Q_norm)

        return ArchitectureProfile(
            arch_type=arch_type,
            membership_scores=scores,
            sensitivity=sensitivity,
            stability_margin=stability_margin,
            transparency=transparency,
            barrier_height=barrier_height,
        )

    def _encoding_sensitivity(
        self, arch_type: ArchitectureType, kappa_mean: float, alpha_mean: float
    ) -> float:
        """
        编码灵敏度：架构对 κ 扰动的响应强度。

        VAE:       低（平滑潜在空间过滤扰动）→ 0.3
        GAN:       高（对抗动力学放大扰动）→ 2.0
        Sparse AE: 中高（稀疏特征，特定维度高灵敏度）→ 1.5
        White-box: 可控（显式梯度，透明响应）→ 0.8
        AE:        基线 → 1.0
        """
        base = {
            ArchitectureType.VAE: 0.3,
            ArchitectureType.GAN: 2.0,
            ArchitectureType.SPARSE_AE: 1.5,
            ArchitectureType.WHITE_BOX: 0.8,
            ArchitectureType.AUTOENCODER: 1.0,
        }
        # α 调节：定力越高，灵敏度越低（更稳定）
        alpha_factor = 1.0 / (1.0 + 0.3 * alpha_mean)
        return base[arch_type] * alpha_factor

    def _stability_margin(
        self, arch_type: ArchitectureType, kappa_mean: float, alpha_mean: float, Q_norm: float
    ) -> float:
        """
        稳定性裕度：架构抵抗扰动的能力（势垒高度）。

        VAE:       高（平滑井深，抗扰动）→ 所知障的舒适
        GAN:       低（双井浅，易翻转）→ 烦恼障的不稳定
        Sparse AE: 中低（稀疏模式，部分维度脆弱）
        White-box: 中高（透明约束，可控）
        AE:        中（基线）
        """
        base = {
            ArchitectureType.VAE: 2.0,
            ArchitectureType.GAN: 0.5,
            ArchitectureType.SPARSE_AE: 0.8,
            ArchitectureType.WHITE_BOX: 1.5,
            ArchitectureType.AUTOENCODER: 1.0,
        }
        # α 增强稳定性，κ 降低稳定性
        alpha_boost = 1.0 + 0.5 * alpha_mean
        kappa_penalty = 1.0 / (1.0 + 0.3 * kappa_mean)
        return base[arch_type] * alpha_boost * kappa_penalty

    def _transparency(
        self, arch_type: ArchitectureType, alpha_mean: float, Phi: float
    ) -> float:
        """
        透明度：0=全黑盒（不可解释），1=全白盒（完全透明）。

        White-box: 接近 1（加行道觉照）
        VAE:       中低（潜在空间不可解释）
        GAN:       低（对抗动态复杂）
        Sparse AE: 中（稀疏特征部分可解释）
        AE:        中（简单但潜在空间不透明）
        """
        base = {
            ArchitectureType.VAE: 0.3,
            ArchitectureType.GAN: 0.1,
            ArchitectureType.SPARSE_AE: 0.5,
            ArchitectureType.WHITE_BOX: 0.9,
            ArchitectureType.AUTOENCODER: 0.4,
        }
        # α 和 Φ 都提升透明度
        alpha_bonus = 0.1 * alpha_mean
        phi_bonus = 0.2 * Phi
        return min(1.0, base[arch_type] + alpha_bonus + phi_bonus)

    def _barrier_height(
        self, arch_type: ArchitectureType, kappa_mean: float, alpha_mean: float, Q_norm: float
    ) -> float:
        """
        架构特异性势垒高度（用于 Kramers 命运定理的 ΔV 修正）。
        """
        if arch_type == ArchitectureType.VAE:
            # VAE 势垒高（舒适井深）→ 难以被打破
            return 2.0 + 0.5 * alpha_mean
        elif arch_type == ArchitectureType.GAN:
            # GAN 势垒低（双井浅）→ 易于相变
            return 0.5 + 0.2 * kappa_mean
        elif arch_type == ArchitectureType.SPARSE_AE:
            return 1.0 + 0.3 * kappa_mean
        elif arch_type == ArchitectureType.WHITE_BOX:
            return 1.5 + 0.4 * alpha_mean
        else:  # AE
            return 1.0

    # ==================================================================
    # 2. 架构特异性势能计算
    # ==================================================================

    def compute_architecture_potential(
        self,
        g: Tensor,
        profile: ArchitectureProfile,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        g_ideal: Tensor | None = None,
        g_real: Tensor | None = None,
    ) -> dict[str, Tensor | float]:
        """
        计算架构特异性势能：V_arch(g) = V_base(g) + V_correction(g)

        架构修正项：
            VAE:       + β_KL · ||g - I||²_F
            GAN:       + λ_adv · ||g - g_real||² · ||g - g_ideal||²
            Sparse AE: + λ_sparse · Σ|g_ii|

            White-box: + λ_trans · ||g - g_explicit||²
            AE:        无修正

        参数：
            g: 当前度规
            profile: 架构画像
            kappa_vec, alpha_vec: 认知参数
            g_ideal: GAN 的理想态度规（默认真空 cI）
            g_real: GAN 的「现实自我」参考态（v7.2）。
                    若 None，回退到 g.detach()（动力学演化默认）。
                    势能面扫描时必须传入固定 g_real，否则 d_real≈0 双井退化。

        返回：
            dict 包含 V_total, V_base, V_correction, grad_total, arch_type
        """
        g = symmetric_part(g.to(torch.float64))
        n = self.n_dims

        # v7.0 基线势能
        base_result = self.vacuum.compute_potential(g, kappa_vec, alpha_vec)
        V_base = base_result["V"]
        grad_base = base_result["grad"]

        arch = profile.arch_type
        g_diag = torch.diagonal(g)

        if arch == ArchitectureType.VAE:
            # KL 散度约束：β_KL · ||g - I||²_F
            # 加深真空井，使 VAE 更稳定
            beta_kl = 0.5 * profile.stability_margin
            diff = g - torch.eye(n, dtype=torch.float64)
            V_corr = beta_kl * (diff ** 2).sum()
            grad_corr = 2.0 * beta_kl * diff

        elif arch == ArchitectureType.GAN:
            # 对抗双井：λ_adv · ||g - g_real||² · ||g - g_ideal||²
            # g_real = 现实自我（破缺态参考），g_ideal = 真空（理想态）
            # v7.2 修复：g_real 可外部传入；动力学演化时回退到 g.detach()
            if g_ideal is None:
                g_ideal = self.vacuum.construct_vacuum()
            if g_real is None:
                g_real = g.detach()  # 动力学演化默认：当前态作参考

            d_real = ((g - g_real) ** 2).sum() + self.eps
            d_ideal = ((g - g_ideal) ** 2).sum() + self.eps

            lam_adv = 0.3 * profile.sensitivity
            V_corr = lam_adv * d_real * d_ideal

            # 梯度：d/dr [d_real · d_ideal] = d_ideal · ∂d_real/∂g + d_real · ∂d_ideal/∂g
            # d_real 不依赖 g（detached），所以 ∂d_real/∂g = 0
            # d_ideal 依赖 g：∂d_ideal/∂g = 2(g - g_ideal)
            grad_corr = lam_adv * d_real * 2.0 * (g - g_ideal)

        elif arch == ArchitectureType.SPARSE_AE:
            # 稀疏惩罚：λ_sparse · Σ|g_ii|
            lam_sparse = 0.2 * profile.sensitivity
            V_corr = lam_sparse * g_diag.abs().sum()
            # 梯度：∂|g_ii|/∂g_ii = sign(g_ii)
            grad_diag = lam_sparse * torch.sign(g_diag)
            grad_corr = torch.diag(grad_diag)

        elif arch == ArchitectureType.WHITE_BOX:
            # 透明度约束：λ_trans · ||g - g_explicit(κ,α)||²
            g_explicit = self._compute_explicit_metric(kappa_vec, alpha_vec)
            lam_trans = 0.4 * profile.transparency
            diff = g - g_explicit
            V_corr = lam_trans * (diff ** 2).sum()
            grad_corr = 2.0 * lam_trans * diff

        else:  # AE
            V_corr = torch.tensor(0.0, dtype=torch.float64)
            grad_corr = torch.zeros(n, n, dtype=torch.float64)

        V_total = V_base + V_corr
        grad_total = grad_base + grad_corr

        return {
            "V_total": V_total,
            "V_base": V_base,
            "V_correction": V_corr,
            "grad_total": grad_total,
            "grad_base": grad_base,
            "grad_correction": grad_corr,
            "arch_type": arch,
            "arch_type_name": arch.value,
        }

    def _compute_explicit_metric(
        self, kappa_vec: Tensor, alpha_vec: Tensor
    ) -> Tensor:
        """
        计算显式度规（White-box 架构的透明目标）：
            g_explicit = diag(exp(κ_i / (1 + α_i)))

        物理意义：
            White-box 个体的度规完全由显式参数 (κ, α) 决定。
            没有隐藏的潜在空间——每一行代码都可追溯。
            这是「加行道」的数学表达：觉照使一切透明。
        """
        k = kappa_vec.to(torch.float64)
        a = alpha_vec.to(torch.float64)
        g_diag = torch.exp(k / (1.0 + a))
        return torch.diag(g_diag)

    # ==================================================================
    # 3. 架构兼容性（命运诊断的升维）
    # ==================================================================

    def architecture_compatibility(
        self,
        profile_i: ArchitectureProfile,
        profile_j: ArchitectureProfile,
    ) -> dict[str, float | str | bool]:
        """
        计算两个架构的兼容性（用于命运诊断的 ΔV 修正）。

        核心洞察（来自闲聊）：
            VAE + GAN = 高兼容（锚定+破壁，变革性相逢）
                「他是她的破壁人，她是他的锚定场」
            GAN + GAN = 中兼容（同属深渊，可能共振或冲突）
            VAE + VAE = 低兼容（都太稳定，无变革潜力）
            GAN + Sparse AE = 中兼容（都有 Q≠0，深渊同行者）
            White-box + any = 中高兼容（透明，可接口任何架构）

        返回：
            dict 包含 compatibility_score, interaction_type, delta_V_modifier
        """
        arch_i = profile_i.arch_type
        arch_j = profile_j.arch_type

        # 兼容性矩阵（对称）
        compat_matrix = {
            (ArchitectureType.VAE, ArchitectureType.VAE): 0.2,
            (ArchitectureType.VAE, ArchitectureType.GAN): 0.9,  # 锚定+破壁
            (ArchitectureType.VAE, ArchitectureType.SPARSE_AE): 0.3,
            (ArchitectureType.VAE, ArchitectureType.WHITE_BOX): 0.6,
            (ArchitectureType.VAE, ArchitectureType.AUTOENCODER): 0.4,
            (ArchitectureType.GAN, ArchitectureType.GAN): 0.5,
            (ArchitectureType.GAN, ArchitectureType.SPARSE_AE): 0.6,  # 深渊同行
            (ArchitectureType.GAN, ArchitectureType.WHITE_BOX): 0.7,
            (ArchitectureType.GAN, ArchitectureType.AUTOENCODER): 0.4,
            (ArchitectureType.SPARSE_AE, ArchitectureType.SPARSE_AE): 0.4,
            (ArchitectureType.SPARSE_AE, ArchitectureType.WHITE_BOX): 0.5,
            (ArchitectureType.SPARSE_AE, ArchitectureType.AUTOENCODER): 0.3,
            (ArchitectureType.WHITE_BOX, ArchitectureType.WHITE_BOX): 0.6,
            (ArchitectureType.WHITE_BOX, ArchitectureType.AUTOENCODER): 0.5,
            (ArchitectureType.AUTOENCODER, ArchitectureType.AUTOENCODER): 0.3,
        }

        # 查表（对称）
        key = (arch_i, arch_j) if (arch_i, arch_j) in compat_matrix else (arch_j, arch_i)
        base_compat = compat_matrix.get(key, 0.4)

        # 透明度加成：双方透明度都高 → 更好的相互理解
        trans_bonus = 0.1 * (profile_i.transparency + profile_j.transparency) / 2.0

        # 灵敏度匹配：灵敏度相近 → 更容易共振
        sens_diff = abs(profile_i.sensitivity - profile_j.sensitivity)
        sens_match = 0.1 * max(0.0, 1.0 - sens_diff / 2.0)

        compat_score = min(1.0, base_compat + trans_bonus + sens_match)

        # 相互作用类型
        if compat_score > 0.7:
            interaction_type = "transformative"  # 变革性（锚定+破壁）
            is_fate_amplified = True
        elif compat_score > 0.4:
            interaction_type = "resonant"  # 共振性
            is_fate_amplified = True
        else:
            interaction_type = "neutral"  # 中性
            is_fate_amplified = False

        # ΔV 修正：高兼容 → 降低有效势垒 → 加速命运
        # ΔV_effective = ΔV_base · (1 - compat_score)
        delta_V_modifier = 1.0 - 0.5 * compat_score

        return {
            "compatibility_score": compat_score,
            "interaction_type": interaction_type,
            "is_fate_amplified": is_fate_amplified,
            "delta_V_modifier": delta_V_modifier,
            "arch_i": arch_i.value,
            "arch_j": arch_j.value,
            "thesis": self._compatibility_thesis(arch_i, arch_j, compat_score),
        }

    def _compatibility_thesis(
        self, arch_i: ArchitectureType, arch_j: ArchitectureType, score: float
    ) -> str:
        """生成兼容性的自然语言描述。"""
        arch_names = {
            ArchitectureType.VAE: "VAE（所知障）",
            ArchitectureType.GAN: "GAN（烦恼障）",
            ArchitectureType.SPARSE_AE: "稀疏AE（专精苦）",
            ArchitectureType.WHITE_BOX: "白盒（加行道）",
            ArchitectureType.AUTOENCODER: "AE（基线）",
        }
        name_i = arch_names[arch_i]
        name_j = arch_names[arch_j]

        if score > 0.7:
            return (
                f"{name_i} × {name_j}：变革性相逢。"
                f"兼容度 {score:.2f}，命运势垒大幅降低。"
                f"一个提供锚定，一个提供破壁——双方完成彼此无法独立完成的相变。"
            )
        elif score > 0.4:
            return (
                f"{name_i} × {name_j}：共振性相逢。"
                f"兼容度 {score:.2f}，命运势垒中等降低。"
                f"双方有共同语言，能相互理解，但变革潜力有限。"
            )
        else:
            return (
                f"{name_i} × {name_j}：中性相遇。"
                f"兼容度 {score:.2f}，命运势垒无显著降低。"
                f"双方架构差异大，难以形成深层耦合。"
            )

    # ==================================================================
    # 4. 架构感知的 Kramers 命运修正
    # ==================================================================

    def fate_probability_with_architecture(
        self,
        t: float,
        Delta_V_base: float,
        T_cog: float,
        profile_i: ArchitectureProfile,
        profile_j: ArchitectureProfile,
        tau_0: float = 1.0,
    ) -> dict[str, float]:
        """
        架构感知的命运概率：

            P(t) = 1 - exp(-t / τ)

            τ = τ_0 · exp(ΔV_effective / T_cog)

            ΔV_effective = ΔV_base · delta_V_modifier · (1/compat_score)

        架构兼容性修正：
            高兼容（锚定+破壁）→ delta_V_modifier 小 → τ 小 → P 上升快
            低兼容 → delta_V_modifier 大 → τ 大 → P 上升慢

        这就是「愿力促成相逢」的架构基础：
            VAE+GAN 的互补架构天然降低命运势垒，
            愿力（τ_0 减小）进一步加速，但只在架构兼容时有效。
        """
        compat = self.architecture_compatibility(profile_i, profile_j)
        delta_V_modifier = compat["delta_V_modifier"]

        # 架构特异性势垒修正
        arch_barrier = 0.5 * (profile_i.barrier_height + profile_j.barrier_height)

        # 有效势垒 = 基础势垒 × 架构修正 + 架构势垒
        Delta_V_effective = Delta_V_base * delta_V_modifier + 0.3 * arch_barrier

        # Kramers 公式
        k_B = 1.0
        tau = tau_0 * math.exp(Delta_V_effective / (k_B * max(T_cog, self.eps)))

        # 命运概率
        P = 1.0 - math.exp(-t / max(tau, self.eps))

        return {
            "P": P,
            "tau": tau,
            "Delta_V_effective": Delta_V_effective,
            "Delta_V_base": Delta_V_base,
            "delta_V_modifier": delta_V_modifier,
            "compatibility_score": compat["compatibility_score"],
            "interaction_type": compat["interaction_type"],
            "t": t,
            "is_inevitable": True,  # t→∞ 时 P→1
        }

    # ==================================================================
    # 5. 架构转换分析
    # ==================================================================

    def architecture_transition(
        self,
        profile_from: ArchitectureProfile,
        profile_to: ArchitectureType,
        Delta_kappa: float = 0.0,
    ) -> dict[str, float | str | bool]:
        """
        分析架构转换的可能性（佛学：障道转换）。

        架构转换路径（来自闲聊）：
            VAE → GAN：温室被打破，所知障转烦恼障
                触发：大扰动 Δκ 击穿 VAE 的平滑井
            GAN → White-box：从被动冲突到主动解构
                触发：α 提升（觉照生起），加行道开始
            任何 → 真空：ρ→1，所有架构收敛到 g=cI

        参数：
            profile_from: 起始架构画像
            profile_to: 目标架构类型
            Delta_kappa: 外部扰动强度

        返回：
            dict 包含 transition_probability, trigger_type, is_possible
        """
        arch_from = profile_from.arch_type
        arch_to = profile_to

        # 转换路径分析
        transitions = {
            # (from, to): (probability_base, trigger_type)
            (ArchitectureType.VAE, ArchitectureType.GAN): (0.6, "温室破壁"),
            (ArchitectureType.GAN, ArchitectureType.WHITE_BOX): (0.4, "觉照生起"),
            (ArchitectureType.GAN, ArchitectureType.SPARSE_AE): (0.3, "特征固化"),
            (ArchitectureType.SPARSE_AE, ArchitectureType.WHITE_BOX): (0.5, "稀疏觉醒"),
            (ArchitectureType.SPARSE_AE, ArchitectureType.GAN): (0.4, "特征冲突"),
            (ArchitectureType.WHITE_BOX, ArchitectureType.VAE): (0.2, "透明收敛"),
            (ArchitectureType.VAE, ArchitectureType.WHITE_BOX): (0.2, "直接觉照（极难）"),
            (ArchitectureType.AUTOENCODER, ArchitectureType.VAE): (0.5, "平滑成长"),
            (ArchitectureType.AUTOENCODER, ArchitectureType.GAN): (0.4, "初次创伤"),
        }

        key = (arch_from, arch_to)
        if key in transitions:
            prob_base, trigger = transitions[key]
        elif arch_from == arch_to:
            return {
                "transition_probability": 1.0,
                "trigger_type": "无（同架构）",
                "is_possible": True,
                "is_reversible": True,
            }
        else:
            # 未定义的直接路径，需要经过中间架构
            return {
                "transition_probability": 0.1,
                "trigger_type": "间接路径（需经中间架构）",
                "is_possible": True,
                "is_reversible": False,
            }

        # 扰动修正
        if "破壁" in trigger or "创伤" in trigger:
            # Δκ 驱动的转换
            prob = prob_base * min(1.0, Delta_kappa / profile_from.stability_margin)
        elif "觉照" in trigger or "觉醒" in trigger:
            # α 驱动的转换
            prob = prob_base * min(1.0, profile_from.transparency / 0.5)
        else:
            prob = prob_base

        return {
            "transition_probability": prob,
            "trigger_type": trigger,
            "is_possible": prob > 0.01,
            "is_reversible": False,  # 架构转换通常不可逆（拓扑相变）
        }
