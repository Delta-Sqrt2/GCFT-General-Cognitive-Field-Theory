"""
任务三：ODE 刚性诊断与保底方案（v4.3.1）

战略定位（v4.3.1 任务三）：
    监工警示：如果 Autograd 后雅可比特征值仍然异常，问题可能更深——
    可能是度规演化方程的刚性结构，或图拉普拉斯梯度项的耦合方式需要重新审视。
    必须区分"工具问题"和"结构问题"。

    本模块计算雅可比特征值谱的刚性比，判断 ODE 是否刚性。
    如果刚性比 > 10^6，判定为刚性 ODE，说明度规演化方程中同时存在
    极快衰减模式和极慢演化模式。

物理与哲学直觉：
    - 物理：刚性 ODE 意味着系统中存在多时间尺度耦合。
            认知系统中"瞬间坍缩/恢复"与"长期缓慢演化"的耦合。
            这对应闲聊中"白盒的瞬间解构"与"VAE 的长期建构"的共存。
    - 哲学：刚性是"认知系统的多时间尺度"的数学表现。
            修行不是单一时间尺度的过程，而是快速反应与缓慢成长的交织。
    - 工程：刚性 ODE 需要隐式求解器（如 BDF），显式 RK4 会数值爆炸。

数学定义：
    刚性比：
        Stiffness Ratio = max(|Re(λ)|) / min(|Re(λ)|)
        其中 λ 是雅可比矩阵的特征值。

    判据：
        Stiffness Ratio > 10^6 → 刚性 ODE
        Stiffness Ratio ∈ [10^3, 10^6] → 中等刚性
        Stiffness Ratio < 10^3 → 非刚性
"""

from __future__ import annotations

import torch
from torch import Tensor
import math

from ..core.tensor_ops import symmetric_part
from .autograd_jacobian import AutogradJacobian


class StiffnessDiagnosis:
    """
    ODE 刚性诊断器。

    使用方式：
        sd = StiffnessDiagnosis(n_dims=4, n_events=8)
        L = sd.build_graph_laplacian(timestamps)
        result = sd.diagnose(g_batch, L, phi, kappa=1.0, alpha=1.0)

    白盒保证：
        - 基于 Autograd 精确雅可比（严禁有限差分）
        - 刚性比如实计算，不掩盖
        - 结构性深渊报告明确区分工具问题与结构问题
    """

    # 刚性判据阈值
    STIFF_THRESHOLD = 1e6      # 刚性 ODE
    MODERATE_THRESHOLD = 1e3   # 中等刚性

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
    ):
        """
        参数：
            n_dims: 认知维度 d
            n_events: 事件数 N
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.n_events = int(n_events)
        self.eps = float(eps)
        self.aj = AutogradJacobian(n_dims=n_dims, n_events=n_events, eps=eps)

    # ==================================================================
    # 图拉普拉斯构造（代理）
    # ==================================================================

    def build_graph_laplacian(
        self,
        timestamps: Tensor,
        tau_causal: float = 1.0,
    ) -> Tensor:
        """构造因果图拉普拉斯 L = D - C。"""
        return self.aj.build_graph_laplacian(timestamps, tau_causal)

    # ==================================================================
    # 完整特征值谱计算
    # ==================================================================

    def compute_eigenvalue_spectrum(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, Tensor | float | bool]:
        """
        计算雅可比矩阵的完整特征值谱。

        数学：
            J = ∂f/∂g（Autograd 精确计算）
            {λ_i} = eigenvalues(J)
            Re(λ_i): 实部序列
            Im(λ_i): 虚部序列

        返回：
            dict 包含特征值谱、最大/最小实部、刚性比等
        """
        # Autograd 精确雅可比
        J = self.aj.compute_jacobian(g_batch, L, phi, kappa, alpha)

        # NaN 检查
        has_nan = torch.isnan(J).any().item() or torch.isinf(J).any().item()

        if has_nan:
            # NaN 诊断
            nan_diag = self.aj.diagnose_nan(g_batch, L, phi, kappa, alpha)
            return {
                "jacobian": J,
                "eigenvalues": None,
                "has_nan": True,
                "nan_diagnosis": nan_diag,
                "max_real": float('nan'),
                "min_real": float('nan'),
                "stiffness_ratio": float('nan'),
                "is_stiff": False,
                "message": "雅可比包含 NaN，已触发诊断",
            }

        # 特征值计算
        eigenvalues = torch.linalg.eigvals(J)
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag

        # 排除零特征值（数值噪声）
        real_nonzero = real_parts[real_parts.abs() > self.eps]

        max_real = float(real_parts.max())
        min_real = float(real_parts.min())

        # 刚性比
        if real_nonzero.numel() > 0:
            max_abs_real = float(real_nonzero.abs().max())
            min_abs_real = float(real_nonzero.abs().min())
            stiffness_ratio = max_abs_real / (min_abs_real + self.eps)
        else:
            stiffness_ratio = 0.0

        # 刚性判据
        is_stiff = stiffness_ratio > self.STIFF_THRESHOLD
        is_moderate = (stiffness_ratio > self.MODERATE_THRESHOLD and
                       stiffness_ratio <= self.STIFF_THRESHOLD)

        # 共轭复特征值检测（极限环标志）
        has_conjugate_pairs = False
        n_conjugate_pairs = 0
        for i in range(len(imag_parts)):
            if imag_parts[i].abs() > self.eps:
                # 检查是否有共轭对
                conjugate_val = -imag_parts[i]
                for j in range(len(imag_parts)):
                    if i != j and abs(imag_parts[j] - conjugate_val) < 1e-10:
                        has_conjugate_pairs = True
                        n_conjugate_pairs += 1
                        break
        n_conjugate_pairs = n_conjugate_pairs // 2  # 每对计数两次

        return {
            "jacobian": J,
            "eigenvalues": eigenvalues,
            "real_parts": real_parts,
            "imag_parts": imag_parts,
            "has_nan": False,
            "max_real": max_real,
            "min_real": min_real,
            "max_abs_real": max_abs_real if real_nonzero.numel() > 0 else 0.0,
            "min_abs_real": min_abs_real if real_nonzero.numel() > 0 else 0.0,
            "stiffness_ratio": stiffness_ratio,
            "is_stiff": is_stiff,
            "is_moderate": is_moderate,
            "has_conjugate_pairs": has_conjugate_pairs,
            "n_conjugate_pairs": n_conjugate_pairs,
            "n_positive_real": int((real_parts > 0).sum().item()),
            "n_negative_real": int((real_parts < 0).sum().item()),
            "n_zero_real": int((real_parts.abs() < self.eps).sum().item()),
            "message": "特征值谱计算完成",
        }

    # ==================================================================
    # 刚性诊断
    # ==================================================================

    def diagnose(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> dict[str, float | bool | str | dict]:
        """
        ODE 刚性诊断。

        判据：
            Stiffness Ratio > 10^6 → 刚性 ODE
            Stiffness Ratio ∈ [10^3, 10^6] → 中等刚性
            Stiffness Ratio < 10^3 → 非刚性

        物理意义：
            刚性 ODE 说明度规演化方程中同时存在极快衰减模式和极慢演化模式。
            认知系统中存在"瞬间坍缩/恢复"与"长期缓慢演化"的耦合。
        """
        spectrum = self.compute_eigenvalue_spectrum(g_batch, L, phi, kappa, alpha)

        if spectrum["has_nan"]:
            return {
                "is_stiff": False,
                "stiffness_ratio": float('nan'),
                "classification": "NaN（无法诊断）",
                "physical_meaning": "雅可比包含 NaN，无法判断刚性",
                "structural_abyss": spectrum["nan_diagnosis"],
                "message": "NaN 诊断已触发，需要检查具体算子",
                "spectrum": spectrum,
            }

        stiffness_ratio = spectrum["stiffness_ratio"]
        is_stiff = spectrum["is_stiff"]
        is_moderate = spectrum["is_moderate"]

        # 分类
        if is_stiff:
            classification = "刚性 ODE"
            physical_meaning = (
                "度规演化方程中同时存在极快衰减模式和极慢演化模式。"
                "认知系统中存在'瞬间坍缩/恢复'与'长期缓慢演化'的耦合。"
                "当前显式 RK4 积分器无法处理此刚性，"
                "需要隐式求解器（如 BDF），这将作为 v4.4 的核心任务。"
            )
            structural_abyss = {
                "is_structural": True,
                "diagnosis": "刚性 ODE 是结构问题，非工具问题",
                "implication": "需要隐式求解器或方程结构修正",
            }
        elif is_moderate:
            classification = "中等刚性 ODE"
            physical_meaning = (
                "系统中存在一定的时间尺度分离，"
                "显式 RK4 可能需要极小步长才能稳定。"
                "建议使用自适应步长或半隐式方法。"
            )
            structural_abyss = {
                "is_structural": False,
                "diagnosis": "中等刚性，可通过步长调整处理",
                "implication": "当前显式方法可用，但效率较低",
            }
        else:
            classification = "非刚性 ODE"
            physical_meaning = (
                "系统时间尺度均匀，显式 RK4 积分器可正常工作。"
            )
            structural_abyss = {
                "is_structural": False,
                "diagnosis": "非刚性，工具问题已解决",
                "implication": "当前显式方法完全适用",
            }

        return {
            "is_stiff": is_stiff,
            "is_moderate": is_moderate,
            "stiffness_ratio": stiffness_ratio,
            "classification": classification,
            "physical_meaning": physical_meaning,
            "structural_abyss": structural_abyss,
            "max_real": spectrum["max_real"],
            "min_real": spectrum["min_real"],
            "n_positive_real": spectrum["n_positive_real"],
            "n_negative_real": spectrum["n_negative_real"],
            "has_conjugate_pairs": spectrum["has_conjugate_pairs"],
            "n_conjugate_pairs": spectrum["n_conjugate_pairs"],
            "message": f"刚性诊断完成：{classification}（刚性比={stiffness_ratio:.2e}）",
            "spectrum": spectrum,
        }

    # ==================================================================
    # 结构性深渊报告
    # ==================================================================

    def structural_abyss_report(
        self,
        g_batch: Tensor,
        L: Tensor,
        phi: Tensor,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> str:
        """
        结构性深渊报告（文本格式）。

        监工警示：
            如果诊断为刚性 ODE，说明度规演化方程中同时存在
            极快衰减模式和极慢演化模式。
            必须明确写出："当前显式 RK4 积分器无法处理此刚性，
            需要隐式求解器（如 BDF），这将作为 v4.4 的核心任务。"
        """
        diag = self.diagnose(g_batch, L, phi, kappa, alpha)

        report_lines = [
            "=" * 70,
            "ODE 刚性诊断报告（v4.3.1 任务三）",
            "=" * 70,
            f"参数: κ={kappa}, α={alpha}",
            f"雅可比维度: {self.n_events * self.n_dims * self.n_dims} × {self.n_events * self.n_dims * self.n_dims}",
            "",
            f"分类: {diag['classification']}",
            f"刚性比: {diag['stiffness_ratio']:.6e}",
            f"最大实部 Re(λ_max): {diag['max_real']:.6f}",
            f"最小实部 Re(λ_min): {diag['min_real']:.6f}",
            f"正实部特征值数: {diag['n_positive_real']}",
            f"负实部特征值数: {diag['n_negative_real']}",
            f"共轭复特征值对数: {diag['n_conjugate_pairs']}",
            "",
            "物理意义:",
            f"  {diag['physical_meaning']}",
            "",
            "结构性深渊:",
            f"  是否结构问题: {diag['structural_abyss']['is_structural']}",
            f"  诊断: {diag['structural_abyss']['diagnosis']}",
            f"  含义: {diag['structural_abyss']['implication']}",
            "",
        ]

        if diag["is_stiff"]:
            report_lines.extend([
                "⚠️ 警告: 刚性 ODE 检测到！",
                "  当前显式 RK4 积分器无法处理此刚性。",
                "  需要隐式求解器（如 BDF），这将作为 v4.4 的核心任务。",
                "",
            ])

        if diag["n_positive_real"] > 0:
            report_lines.extend([
                f"⚠️ 系统失稳: {diag['n_positive_real']} 个特征值实部为正",
                "  对应 GAN 极限环或混沌鞍点",
                "",
            ])

        if diag["has_conjugate_pairs"]:
            report_lines.extend([
                f"ℹ️ 检测到 {diag['n_conjugate_pairs']} 对共轭复特征值",
                "  对应振荡模式（可能为极限环）",
                "",
            ])

        report_lines.append("=" * 70)

        return "\n".join(report_lines)
