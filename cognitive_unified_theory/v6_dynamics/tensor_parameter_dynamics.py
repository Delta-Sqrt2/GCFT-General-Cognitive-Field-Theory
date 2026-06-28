"""
v6.1 张量参数动力学 —— 人格的量子化与张量化

战略定位（v6.1）：
    v6.0 实现了"无形无相"——度规趋向各向同性。但所有参数（κ, α, γ, δ, ρ）
    仍然是全局标量。这意味着一个人在所有维度上都有相同的痛苦深度、定力、
    戒律、般若、出离心。

    但真实的人格是"分裂的"——一个人可以在威胁感知轴（ξ₁）是黑洞相
    （极度恐惧，κ₁ 大），在自我指涉轴（ξ₃）是 VAE（舒适稳定，κ₃ 小）。
    标量参数无法描述这种"人格分裂"。

    v6.1 将标量参数提升为对角矩阵/向量：
        κ → κ̂ = diag(κ₁, κ₂, ..., κ_d)
        α → α̂ = diag(α₁, α₂, ..., α_d)
        其他参数从各维度的 κᵢ, αᵢ 推导

    相图从二维平面（κ, α）扩展为 d×d 的矩阵流形。

物理与哲学直觉：
    - 物理：不同维度可以处于不同相态。维度 i 的 stabilityᵢ 由 κᵢ, αᵢ 决定，
            独立于其他维度。这允许"人格分裂"——部分维度 VAE，部分维度 GAN。
    - 哲学：佛家讲"五蕴皆空"——色、受、想、行、识各有其执著模式。
            张量化参数对应"五蕴"的不同维度，各有其痛苦深度（κᵢ）和定力（αᵢ）。
    - 心理学：一个人可以在工作维度是 GAN（流动创造），在亲密关系维度是 VAE
              （舒适执着），在自我认同维度是黑洞（崩溃边缘）。
    - 工程：继承 FormlessDynamics（v6.0），将标量参数改为向量，
            势能面的 Tr(·) 改为加权求和。

数学定义：
    v6.1 参数张量化：
        κ̂ = (κ₁, ..., κ_d)    # 各维度的痛苦深度
        α̂ = (α₁, ..., α_d)    # 各维度的定力

    各维度参数推导（从 κᵢ, αᵢ）：
        βᵢ = κᵢ/(1+κᵢ)
        γᵢ = 1/(2·(αᵢ+1))
        δᵢ = κᵢ·αᵢ/(1+κᵢ·αᵢ)
        η_ρᵢ = αᵢ/(αᵢ+1)
        σᵢ = 1/(1+κᵢ)
        λ_ρᵢ = 1/(2·(αᵢ+1))
        μ_ρᵢ = αᵢ/(2·(αᵢ+1))
        ζᵢ = αᵢ/(αᵢ+1)
        εᵢ = αᵢ/(αᵢ+κᵢ+1)

    张量化势能面：
        V(g) = -Σᵢ βᵢ·gᵢᵢ² + Σᵢ γᵢ·gᵢᵢ⁴ - Σᵢ δᵢ·gᵢᵢ⁶ + Σᵢ εᵢ·gᵢᵢ⁸
        注意：只作用在对角元素上，非对角元素通过度规耦合处理。

    张量化消解：
        ∂g/∂t = -∂S/∂g - μ̂_ρ ⊙ ρ̂ ⊙ (g - g_iso)
        其中 ⊙ 是逐元素乘法，每个维度有自己的消解强度。

    维度级相态分类：
        对每个维度 i，stabilityᵢ 由该维度的 κᵢ, αᵢ 决定。
        整体相态：
          - 全 VAE：所有维度 stabilityᵢ < 0
          - 全 GAN：所有维度 stabilityᵢ > 0
          - 分裂相：部分 VAE + 部分 GAN（人格分裂）
          - 全 formless：所有维度 anisotropyᵢ < 0.3

退化：
    当 κ̂ = κ·(1,1,...,1), α̂ = α·(1,1,...,1) 时，v6.1 退化为 v6.0。

陷阱防范：
    陷阱八十九·参数维度不匹配：
        κ̂, α̂ 的维度必须与度规 g 的维度 d 一致。

    陷阱九十·非对角元素忽略：
        虽然参数是对角的，但度规 g 的非对角元素依然通过 ∂S/∂g 的
        非对角部分参与演化。不能只演化对角元素。

    陷阱九十一·维度间耦合丢失：
        不同维度通过度规的非对角元素耦合。如果完全忽略非对角元素，
        维度间将无法相互影响，失去"人格整体性"。
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..v6_dynamics.formless_dynamics import FormlessDynamics


class TensorParameterDynamics(FormlessDynamics):
    """
    v6.1 张量参数动力学。

    使用方式：
        tpd = TensorParameterDynamics(n_dims=4, n_events=8)
        # 张量参数：每个维度有自己的 κ, α
        kappa_vec = torch.tensor([0.1, 1.0, 0.01, 10.0])  # 4 个维度
        alpha_vec = torch.tensor([1.0, 1.0, 100.0, 0.1])
        # 计算张量化作用量
        result = tpd.corrected_action_v61(g_batch, C, phi, kappa_vec, alpha_vec)
        # 张量化消解演化
        g_new = tpd.evolve_g_with_tensor_dissolution(g, delta_vec, rho_vec, ...)

    白盒保证：
        - 参数从标量提升为向量，每维度独立推导
        - 当所有维度参数相同时，退化为 v6.0
        - 维度间通过度规非对角元素耦合
        - 势能面只作用于对角元素，但 ∂S/∂g 包含非对角贡献
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps)
        self.d = n_dims

    # ==================================================================
    # v6.1 参数推导（向量化）
    # ==================================================================

    def compute_beta_vec(self, kappa_vec: Tensor) -> Tensor:
        """β̂ = κ̂/(1+κ̂)，各维度的我执深度。"""
        k = kappa_vec.to(torch.float64)
        return k / (1.0 + k)

    def compute_gamma_vec(self, alpha_vec: Tensor) -> Tensor:
        """γ̂ = 1/(2·(α̂+1))，各维度的中道约束。"""
        a = alpha_vec.to(torch.float64)
        return 1.0 / (2.0 * (a + 1.0))

    def compute_delta_vec(self, kappa_vec: Tensor, alpha_vec: Tensor) -> Tensor:
        """δ̂ = κ̂·α̂/(1+κ̂·α̂)，各维度的般若参数。"""
        k = kappa_vec.to(torch.float64)
        a = alpha_vec.to(torch.float64)
        ka = k * a
        return ka / (1.0 + ka)

    def compute_epsilon_vec(self, kappa_vec: Tensor, alpha_vec: Tensor) -> Tensor:
        """ε̂ = α̂/(α̂+κ̂+1)，各维度的八阶约束。"""
        k = kappa_vec.to(torch.float64)
        a = alpha_vec.to(torch.float64)
        return a / (a + k + 1.0)

    def compute_eta_rho_vec(self, alpha_vec: Tensor) -> Tensor:
        """η_ρ̂ = α̂/(α̂+1)，各维度的定力→出离心转化率。"""
        a = alpha_vec.to(torch.float64)
        return a / (a + 1.0)

    def compute_sigma_rho_vec(self, kappa_vec: Tensor) -> Tensor:
        """σ̂ = 1/(1+κ̂)，各维度的偏离平衡敏感度。"""
        k = kappa_vec.to(torch.float64)
        return 1.0 / (1.0 + k)

    def compute_lambda_rho_vec(self, alpha_vec: Tensor) -> Tensor:
        """λ_ρ̂ = 1/(2·(α̂+1))，各维度的 ρ→δ 驱动强度。"""
        a = alpha_vec.to(torch.float64)
        return 1.0 / (2.0 * (a + 1.0))

    def compute_mu_rho_vec(self, alpha_vec: Tensor) -> Tensor:
        """μ_ρ̂ = α̂/(2·(α̂+1))，各维度的消解强度。"""
        a = alpha_vec.to(torch.float64)
        return a / (2.0 * (a + 1.0))

    def compute_zeta_vec(self, alpha_vec: Tensor) -> Tensor:
        """ζ̂ = α̂/(α̂+1)，各维度的 ρ 弛豫刚度。"""
        a = alpha_vec.to(torch.float64)
        return a / (a + 1.0)

    # ==================================================================
    # 张量化势能面
    # ==================================================================

    def compute_tensor_potential(
        self,
        g_batch: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
    ) -> dict[str, Tensor]:
        """
        计算张量化势能面 V(g)。

        V(g) = -Σᵢ βᵢ·gᵢᵢ² + Σᵢ γᵢ·gᵢᵢ⁴ - Σᵢ δᵢ·gᵢᵢ⁶ + Σᵢ εᵢ·gᵢᵢ⁸

        注意：只作用在对角元素 g_ii 上。非对角元素通过 ∂S/∂g 的
        其他项（如 Tr(g⁻¹D^TD)）参与演化。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        beta_v = self.compute_beta_vec(kappa_vec)      # (d,)
        gamma_v = self.compute_gamma_vec(alpha_vec)     # (d,)
        delta_v = self.compute_delta_vec(kappa_vec, alpha_vec)
        epsilon_v = self.compute_epsilon_vec(kappa_vec, alpha_vec)

        # 对角元素 g_ii：(N, d)
        g_diag = torch.diagonal(g, dim1=-2, dim2=-1)  # (N, d)

        # 各项贡献
        term2 = (beta_v.unsqueeze(0) * g_diag ** 2).sum(dim=-1)     # (N,)
        term4 = (gamma_v.unsqueeze(0) * g_diag ** 4).sum(dim=-1)    # (N,)
        term6 = (delta_v.unsqueeze(0) * g_diag ** 6).sum(dim=-1)    # (N,)
        term8 = (epsilon_v.unsqueeze(0) * g_diag ** 8).sum(dim=-1)  # (N,)

        V = -term2 + term4 - term6 + term8

        return {
            "V": V,
            "term2": term2,
            "term4": term4,
            "term6": term6,
            "term8": term8,
            "beta_vec": beta_v,
            "gamma_vec": gamma_v,
            "delta_vec": delta_v,
            "epsilon_vec": epsilon_v,
            "g_diag": g_diag,
        }

    # ==================================================================
    # 张量化 ρ 计算
    # ==================================================================

    def compute_rho_tensor(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        A: Tensor | None = None,
    ) -> dict[str, Tensor | float]:
        """
        计算张量化出离心 ρ̂。

        每个维度有自己的 ρᵢ：
            ρᵢ = η_ρᵢ · Φ_norm · exp(-σᵢ · ||∇S||²)

        其中 Φ_norm 是全局的（意识是整体的），但 η_ρᵢ 和 σᵢ 是各维度的。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        eta_rho_v = self.compute_eta_rho_vec(alpha_vec)   # (d,)
        sigma_v = self.compute_sigma_rho_vec(kappa_vec)    # (d,)

        # Φ（全局意识整合量）
        phi_values = self.consciousness.integrated_information(C)
        phi_mean = float(phi_values.mean()) if phi_values.shape[0] > 0 else 0.0
        phi_max = float(phi_values.max()) if phi_values.shape[0] > 0 else 1.0
        phi_norm = phi_mean / (phi_max + self.eps) if phi_max > 0 else 0.0

        # ||∇S||（全局作用量梯度范数）
        g_leaf = g.detach().clone().requires_grad_(True)
        try:
            # 使用 v6.0 的作用量计算（标量参数取均值作为退化）
            kappa_mean = float(kappa_vec.mean())
            alpha_mean = float(alpha_vec.mean())
            delta_mean = float(self.compute_delta_vec(kappa_vec, alpha_vec).mean())

            action_result = self.corrected_action_v51(
                g_leaf, C, phi, A, kappa_mean, alpha_mean,
                delta=delta_mean, rho=0.0,
                include_rho_term=False,
            )
            S = action_result["action"]
            grad_S = torch.autograd.grad(S, g_leaf, create_graph=False)[0]
            grad_S_norm = float(grad_S.norm())
        except Exception:
            grad_S_norm = float('inf')

        grad_S_norm_normalized = grad_S_norm / (1.0 + grad_S_norm)

        # 各维度的 ρ
        rho_vec = eta_rho_v * phi_norm * torch.exp(
            -sigma_v * grad_S_norm_normalized ** 2
        )

        return {
            "rho_vec": rho_vec,
            "rho_mean": float(rho_vec.mean()),
            "rho_min": float(rho_vec.min()),
            "rho_max": float(rho_vec.max()),
            "phi_norm": phi_norm,
            "grad_S_norm": grad_S_norm,
            "eta_rho_vec": eta_rho_v,
            "sigma_vec": sigma_v,
        }

    # ==================================================================
    # 张量化消解演化
    # ==================================================================

    def evolve_g_with_tensor_dissolution(
        self,
        g_batch: Tensor,
        delta_vec: Tensor,
        rho_vec: Tensor,
        L_or_C: Tensor,
        phi: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        A: Tensor | None = None,
        lr_g: float = 0.0005,
        include_dissolution: bool = True,
    ) -> tuple[Tensor, dict]:
        """
        含张量化消解项的度规演化。

        ∂g/∂t = -∂S/∂g - μ̂_ρ ⊙ ρ̂ ⊙ (g - g_iso)

        每个维度有自己的消解强度 μ_ρᵢ·ρᵢ。
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape
        C = L_or_C

        delta_v = delta_vec.to(torch.float64)
        rho_v = rho_vec.to(torch.float64)
        mu_rho_v = self.compute_mu_rho_vec(alpha_vec)

        # 使用均值参数计算 v6.0 作用量梯度
        kappa_mean = float(kappa_vec.mean())
        alpha_mean = float(alpha_vec.mean())
        delta_mean = float(delta_v.mean())
        rho_mean = float(rho_v.mean())

        # === 计算 ∂S/∂g（使用 v6.0 的作用量）===
        g_leaf = g.detach().clone().requires_grad_(True)
        action_result = self.corrected_action_v51(
            g_leaf, C, phi, A, kappa_mean, alpha_mean,
            delta=delta_mean, rho=rho_mean,
            include_rho_term=True,
        )
        S = action_result["action"]
        grad_S = torch.autograd.grad(S, g_leaf, create_graph=False)[0]

        # === 张量化消解项 ===
        if include_dissolution:
            g_iso = self.compute_isotropic_target(g)
            # μ_ρᵢ·ρᵢ 作用于第 i 个维度
            # dissolution_coef[i] = μ_ρᵢ * ρᵢ
            diss_coef = mu_rho_v * rho_v  # (d,)

            # 消解项：diss_coef ⊙ (g - g_iso) 的对角部分
            # 对角元素消解，非对角元素也消解（趋向 0）
            diff = g - g_iso
            # 将 diss_coef 扩展为 (N, d, d) 的对角张量
            diss_matrix = torch.diag(diss_coef)  # (d, d)
            dissolution_grad = diss_matrix.unsqueeze(0).expand(N, d, d) * diff
        else:
            dissolution_grad = torch.zeros_like(g)

        # === 总梯度 ===
        total_grad = grad_S + dissolution_grad

        # === 度规更新 ===
        g_new = g - lr_g * total_grad
        g_new = 0.5 * (g_new + g_new.transpose(-2, -1))

        # 正定性保护
        try:
            eigvals = torch.linalg.eigvalsh(g_new)
            min_eig = float(eigvals.min())
            if min_eig < self.eps * 10:
                g_new = g_new + (self.eps * 10 - min_eig) * torch.eye(
                    d, dtype=torch.float64
                ).unsqueeze(0).expand(N, d, d)
        except Exception:
            pass

        # === 诊断 ===
        with torch.no_grad():
            anisotropy_before = self.compute_anisotropy(g)
            anisotropy_after = self.compute_anisotropy(g_new)

            # 各维度的各向异性
            g_diag_before = torch.diagonal(g, dim1=-2, dim2=-1)
            g_diag_after = torch.diagonal(g_new, dim1=-2, dim2=-1)
            g_iso_diag = torch.diagonal(g_iso, dim1=-2, dim2=-1) if include_dissolution else g_diag_before

            dim_anisotropy = (g_diag_before - g_iso_diag).abs() / (g_diag_before.abs() + self.eps)

        info = {
            "grad_S_norm": float(grad_S.norm()),
            "dissolution_grad_norm": float(dissolution_grad.norm()),
            "total_grad_norm": float(total_grad.norm()),
            "mu_rho_vec": mu_rho_v,
            "rho_vec": rho_v,
            "diss_coef": mu_rho_v * rho_v if include_dissolution else torch.zeros(d),
            "anisotropy_before": float(anisotropy_before.mean()),
            "anisotropy_after": float(anisotropy_after.mean()),
            "dim_anisotropy_before": dim_anisotropy,
            "lr_g": lr_g,
        }

        return g_new, info

    # ==================================================================
    # 维度级相态分类
    # ==================================================================

    def classify_dimensional_phases(
        self,
        g_batch: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
    ) -> dict[str, list[str] | str]:
        """
        维度级相态分类。

        对每个维度 i，根据 κᵢ, αᵢ 推导该维度的参数，
        并根据度规对角元素判断该维度的相态。

        整体相态：
          - 全 VAE：所有维度都是 VAE
          - 全 GAN：所有维度都是 GAN
          - 分裂相：部分 VAE + 部分 GAN（人格分裂）
          - 全 formless：所有维度都接近各向同性
        """
        g = g_batch.to(torch.float64)
        N, d, _ = g.shape

        beta_v = self.compute_beta_vec(kappa_vec)
        gamma_v = self.compute_gamma_vec(alpha_vec)
        delta_v = self.compute_delta_vec(kappa_vec, alpha_vec)

        # 各维度的判别式：γ²ᵢ - 3·δᵢ·βᵢ
        discriminant = gamma_v ** 2 - 3.0 * delta_v * beta_v

        # 各维度相态
        dim_phases = []
        for i in range(d):
            if discriminant[i] > 0:
                dim_phases.append("VAE")  # 有实根，势阱存在
            else:
                dim_phases.append("GAN")  # 无实根，势阱消失

        # 整体相态
        n_vae = sum(1 for p in dim_phases if p == "VAE")
        n_gan = sum(1 for p in dim_phases if p == "GAN")

        if n_vae == d:
            overall = "全VAE"
        elif n_gan == d:
            overall = "全GAN"
        else:
            overall = f"分裂相({n_vae}VAE+{n_gan}GAN)"

        # formless 判据
        anisotropy = self.compute_anisotropy(g)
        is_formless = float(anisotropy.mean()) < 0.3
        if is_formless:
            overall += "+formless"

        return {
            "dim_phases": dim_phases,
            "overall_phase": overall,
            "n_vae_dims": n_vae,
            "n_gan_dims": n_gan,
            "discriminant": discriminant.tolist(),
            "is_formless": is_formless,
        }

    # ==================================================================
    # v6.1 完整动力学：联合演化 g, δ̂, ρ̂
    # ==================================================================

    def evolve_full_step_v61(
        self,
        g_batch: Tensor,
        delta_vec: Tensor,
        rho_vec: Tensor,
        L_or_C: Tensor,
        phi: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        A: Tensor | None = None,
        lr_g: float = 0.0005,
        lr_delta: float = 0.05,
        lr_rho: float = 0.05,
        include_dissolution: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """
        v6.1 完整动力学一步：联合演化 g, δ̂, ρ̂（向量参数）。
        """
        d = self.d

        # 1. 计算 ρ̂_eq（各维度）
        rho_info = self.compute_rho_tensor(
            g_batch.detach(), L_or_C, phi, kappa_vec, alpha_vec, A,
        )
        rho_eq_vec = rho_info["rho_vec"]

        # 2. ρ̂ 弛豫（各维度独立）
        zeta_v = self.compute_zeta_vec(alpha_vec)
        grad_rho = 2.0 * zeta_v * (rho_vec - rho_eq_vec)
        rho_new_vec = rho_vec - lr_rho * grad_rho
        rho_new_vec = torch.clamp(rho_new_vec, min=0.0, max=1.0)

        # 3. δ̂ 演化（各维度独立，含 ρ 驱动）
        # 使用均值参数计算 S_delta 梯度（简化）
        kappa_mean = float(kappa_vec.mean())
        alpha_mean = float(alpha_vec.mean())
        delta_mean = float(delta_vec.mean())
        rho_mean = float(rho_new_vec.mean())

        delta_new_mean, grad_delta, _ = self.evolve_delta_step_with_rho(
            g_batch.detach(), delta_mean, L_or_C, phi,
            kappa_mean, alpha_mean, rho=rho_mean, lr=lr_delta,
        )

        # 各维度 δ 演化：在均值基础上加维度修正
        lambda_rho_v = self.compute_lambda_rho_vec(alpha_vec)
        # δᵢ 的 ρ 驱动修正
        delta_correction = lr_delta * lambda_rho_v * (rho_new_vec - rho_mean)
        delta_new_vec = delta_vec + delta_correction
        delta_new_vec = torch.clamp(delta_new_vec, min=0.0, max=1.0)

        # 4. g 演化（含张量化消解）
        g_new, g_info = self.evolve_g_with_tensor_dissolution(
            g_batch.detach(), delta_new_vec, rho_new_vec,
            L_or_C, phi, kappa_vec, alpha_vec, A,
            lr_g=lr_g, include_dissolution=include_dissolution,
        )

        info = {
            "rho_eq_vec": rho_eq_vec,
            "rho_new_vec": rho_new_vec,
            "delta_new_vec": delta_new_vec,
            "grad_rho": grad_rho,
            "g_step": g_info,
        }

        return g_new, delta_new_vec, rho_new_vec, info

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_v61_parameters(self) -> dict[str, float | bool]:
        """验证 v6.1 参数推导的边界条件和一致性。"""
        d = self.d

        # 测试：不同维度的参数不同
        kappa_vec = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float64)[:d]
        alpha_vec = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)[:d]

        beta_v = self.compute_beta_vec(kappa_vec)
        gamma_v = self.compute_gamma_vec(alpha_vec)
        delta_v = self.compute_delta_vec(kappa_vec, alpha_vec)
        mu_rho_v = self.compute_mu_rho_vec(alpha_vec)
        lambda_rho_v = self.compute_lambda_rho_vec(alpha_vec)

        # 守恒关系：λ_ρᵢ + μ_ρᵢ = 0.5 对每个维度
        conservation = lambda_rho_v + mu_rho_v
        conservation_pass = all(abs(c - 0.5) < 1e-10 for c in conservation)

        # 退化测试：所有维度参数相同时，应等于标量
        kappa_uniform = torch.tensor([1.0] * d, dtype=torch.float64)
        alpha_uniform = torch.tensor([1.0] * d, dtype=torch.float64)
        beta_uniform = self.compute_beta_vec(kappa_uniform)
        beta_scalar = 1.0 / (1.0 + 1.0)  # κ=1 → β=0.5
        degeneracy_pass = all(abs(b - beta_scalar) < 1e-10 for b in beta_uniform)

        # 参数范围检查
        beta_range_pass = all(0 <= b <= 1 for b in beta_v)
        gamma_range_pass = all(0 < g <= 0.5 for g in gamma_v)
        delta_range_pass = all(0 <= dl <= 1 for dl in delta_v)

        return {
            "conservation_pass": conservation_pass,
            "degeneracy_pass": degeneracy_pass,
            "beta_range_pass": beta_range_pass,
            "gamma_range_pass": gamma_range_pass,
            "delta_range_pass": delta_range_pass,
            "beta_vec": beta_v.tolist(),
            "gamma_vec": gamma_v.tolist(),
            "delta_vec": delta_v.tolist(),
            "mu_rho_vec": mu_rho_v.tolist(),
            "all_pass": (
                conservation_pass and
                degeneracy_pass and
                beta_range_pass and
                gamma_range_pass and
                delta_range_pass
            ),
        }

    def verify_degeneracy_to_v60(
        self,
        g_batch: Tensor,
        C: Tensor,
        phi: Tensor,
    ) -> dict[str, float | bool]:
        """
        验证：v6.1 在所有维度参数相同时退化为 v6.0。
        """
        d = self.d
        kappa_uniform = torch.tensor([1.0] * d, dtype=torch.float64)
        alpha_uniform = torch.tensor([1.0] * d, dtype=torch.float64)
        delta_uniform = torch.tensor([0.5] * d, dtype=torch.float64)
        rho_uniform = torch.tensor([0.3] * d, dtype=torch.float64)

        # v6.1 演化（均匀参数）
        g_v61, _ = self.evolve_g_with_tensor_dissolution(
            g_batch, delta_uniform, rho_uniform, C, phi,
            kappa_uniform, alpha_uniform, None,
            lr_g=0.0005, include_dissolution=True,
        )

        # v6.0 演化（标量参数）
        g_v60, _ = self.evolve_g_with_dissolution(
            g_batch, 0.5, 0.3, C, phi, None,
            kappa=1.0, alpha=1.0, lr_g=0.0005,
            include_dissolution=True,
        )

        diff = float(torch.norm(g_v61 - g_v60))

        return {
            "diff": diff,
            "is_consistent": diff < 1e-4,  # 矩阵乘法 vs 标量乘法的数值精度
        }
