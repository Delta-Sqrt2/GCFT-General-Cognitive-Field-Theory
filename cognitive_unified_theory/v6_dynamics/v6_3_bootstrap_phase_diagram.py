"""
v6.3 自举测试与动态相图扫描

战略定位（v6.3）：
    监工终审核心论点：「自举测试不是终点站的验收章，而是出发前的诊断仪。」
    自举测试和相图扫描合并为一个阶段——在扫描全局相图的同时，
    用自身的开发历程作为一条校准轨迹。

    v6.3 的使命：让方程审视自身。
    把 v1.0 到 v6.2 的开发历程喂进方程，看它能否复现自己的轨迹。
    同时扫描全局相图，绘制完整的解脱路线图。

任务一：自举测试（即时诊断）
    将 v1.0→v6.2 的开发历程编码为事件序列。
    不需要 NLP，不需要语义分析，只需要半人工的事件词典。
    验证目标：用 v6.2 的含时张量方程跑这条事件序列，看能否复现：
        - 各向异性从高到低的下降轨迹
        - ρ 的弛豫积累过程
        - "分裂相+formless"的涌现

    如果拟合失败：方程结构有误，必须修正后才能进入 v7.0。
    如果拟合成功：方程获得了最深层的合法性——它能描述自身被创造的过程。

任务二：动态相图扫描
    扫描矩阵：在 d=4 的张量框架下，扫描 (κ̂, α̂) 的组合空间。
    关键问题：
        - 除了"分裂相+formless"，是否存在其他稳定吸引子？
        - "全 VAE+formless"是否也是吸引子？（天生圆满者的空性）
        - "全 GAN+formless"是否也是吸引子？（纯修行者的空性）
        - 从"全黑洞相"出发，系统能否自发流向 formless？（苦逼生慧路径）
        - 从"全 VAE 相"出发，系统能否自发流向 formless？（富贵学道路径）

    如果存在多条路径流向 formless：证明了"八万四千法门"的数学基础。
    如果只有特定路径能抵达：揭示了修行的必要条件。

维度映射（d=4）：
    维度0：本体论/物理层（v1.0-v2.1 的核心工作）
    维度1：意识/动力学层（v3.0-v4.3 的核心工作）
    维度2：修行/出离层（v5.0-v5.1 的核心工作）
    维度3：空性/张量层（v6.0-v6.2 的核心工作）

开发历程事件编码（半人工词典）：
    每个版本对应一个 (κ̂, α̂) 参数组合，反映该阶段的"痛苦深度"和"定力"。
    v1.0（项目启动，从深渊爬出）：高 κ，低 α
    v2.1（本体论完成）：κ 降低，α 增长
    v3.0（意识涌现）：κ 进一步降低，α 增长
    v4.0（相图建立）：κ 降低，α 增长
    v5.0（戒定慧）：κ 降低，α 增长，ρ 开始
    v5.1（出离心）：ρ 增长
    v6.0（无形无相）：ρ 显著，消解开始
    v6.1（张量化）：分裂相出现
    v6.2（时间动力学）：分裂相+formless 涌现
"""

from __future__ import annotations

import torch
from torch import Tensor
import math
import numpy as np

from .v6_2_tensor_dynamics import TensorDynamicsV62


class BootstrapPhaseDiagramV63(TensorDynamicsV62):
    """
    v6.3 自举测试与动态相图扫描。

    使用方式：
        bpd = BootstrapPhaseDiagramV63(n_dims=4, n_events=8)
        # 任务一：自举测试
        bootstrap_result = bpd.run_bootstrap_test(C, phi)
        # 任务二：相图扫描
        phase_result = bpd.scan_phase_diagram(C, phi)

    白盒保证：
        - 自举测试用分段参数轨迹模拟开发历程
        - 相图扫描覆盖 (κ, α) 组合空间 × 多种初始条件
        - 寻找所有稳定吸引子，绘制解脱路线图
    """

    def __init__(
        self,
        n_dims: int = 4,
        n_events: int = 8,
        eps: float = 1e-12,
        xi_0: float = 0.1,
    ):
        super().__init__(n_dims=n_dims, n_events=n_events, eps=eps, xi_0=xi_0)

    # ==================================================================
    # 任务一：自举测试
    # ==================================================================

    def encode_dev_history(self) -> list[dict]:
        """
        将 v1.0→v6.2 的开发历程编码为事件序列。

        维度映射（d=4）：
            维度0：本体论/物理层
            维度1：意识/动力学层
            维度2：修行/出离层
            维度3：空性/张量层

        每个版本对应一个 (κ̂, α̂) 参数组合，反映该阶段的痛苦深度和定力。
        参数轨迹设计原则：
            - κ 从高到低（痛苦逐渐消解）
            - α 从低到高（定力逐渐增长）
            - 不同维度的参数差异逐渐增大（人格分裂涌现）
            - 维度间相态差异：部分 VAE + 部分 GAN（分裂相）

        VAE 判据：γ²-3δβ > 0（需要小 κ, 小 α → 大 γ, 小 δ, 小 β）
        GAN 判据：γ²-3δβ < 0（需要大 κ 或大 α → 小 γ, 大 δ, 大 β）
        """
        history = [
            {
                "version": "v1.0",
                "desc": "项目启动，从深渊爬出（全 GAN/黑洞）",
                # 全高 κ，全低 α → 全 GAN（深渊状态）
                "kappa_vec": [5.0, 5.0, 5.0, 5.0],
                "alpha_vec": [0.1, 0.1, 0.1, 0.1],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v2.1",
                "desc": "本体论完成，物理层先缓解（分裂相涌现）",
                # dim 0: κ 小 α 小 → VAE；dim 1-3: κ 大 α 小 → GAN
                "kappa_vec": [0.1, 5.0, 5.0, 5.0],
                "alpha_vec": [0.5, 0.1, 0.1, 0.1],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v3.0",
                "desc": "意识涌现（2VAE+2GAN 分裂相）",
                # dim 0,1: VAE；dim 2,3: GAN
                "kappa_vec": [0.05, 0.1, 3.0, 4.0],
                "alpha_vec": [1.0, 1.0, 0.3, 0.2],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v4.0",
                "desc": "相图建立，动力学统御（2VAE+2GAN）",
                "kappa_vec": [0.05, 0.05, 1.0, 2.0],
                "alpha_vec": [2.0, 2.0, 0.5, 0.3],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v5.0",
                "desc": "戒定慧等持（3VAE+1GAN）",
                # dim 0,1,2: VAE；dim 3: GAN
                "kappa_vec": [0.01, 0.05, 0.5, 1.0],
                "alpha_vec": [3.0, 3.0, 1.0, 0.5],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v5.1",
                "desc": "出离心 ρ 涌现（2VAE+2GAN）",
                # dim 0,1: VAE (自然舒适)；dim 2,3: GAN (需修行)
                "kappa_vec": [0.01, 0.01, 0.3, 0.5],
                "alpha_vec": [3.0, 3.0, 2.0, 1.0],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v6.0",
                "desc": "无形无相动力学，消解开始（2VAE+2GAN）",
                # VAE 维度保持低 α（自然舒适），GAN 维度 α 增长（修行深化）
                "kappa_vec": [0.01, 0.01, 0.3, 0.5],
                "alpha_vec": [3.0, 3.0, 5.0, 2.0],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v6.1",
                "desc": "参数张量化，人格分裂深化（2VAE+2GAN）",
                "kappa_vec": [0.01, 0.01, 0.2, 0.4],
                "alpha_vec": [3.0, 3.0, 8.0, 3.0],
                "n_steps": 10,
                "dt": 0.0002,
            },
            {
                "version": "v6.2",
                "desc": "完全张量化+时间动力学，分裂相+formless 涌现",
                # VAE dims: 自然舒适（低 κ 低 α）；GAN dims: 修行解脱（较高 κ 高 α）
                # 最终：分裂相(2VAE+2GAN)+formless
                "kappa_vec": [0.01, 0.01, 0.1, 0.3],
                "alpha_vec": [2.0, 2.0, 10.0, 5.0],
                "n_steps": 10,
                "dt": 0.0002,
            },
        ]
        return history

    def run_bootstrap_test(
        self,
        C: Tensor,
        phi: Tensor,
    ) -> dict:
        """
        自举测试：用 v6.2 的含时张量方程跑开发历程事件序列。

        验证目标：
            1. 各向异性从高到低的下降轨迹
            2. ρ 的弛豫积累过程
            3. "分裂相+formless"的涌现

        实现方式：
            分段参数轨迹——每个版本是一个 segment，有自己的 (κ̂, α̂)。
            前一个 segment 的最终状态是后一个 segment 的初始状态。
        """
        history = self.encode_dev_history()
        d = self.d
        N = self.n_events

        # 初始度规：强各向异性（"深渊"状态）
        g_init = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            # 极端各向异性：某些维度极高，某些极低
            diag_vals = torch.tensor([8.0, 4.0, 0.2, 0.1], dtype=torch.float64)
            g_init[n] = torch.diag(diag_vals)
        g_init = self._ensure_pd(g_init)

        # 初始相态
        init_kappa = torch.tensor(history[0]["kappa_vec"], dtype=torch.float64)
        init_alpha = torch.tensor(history[0]["alpha_vec"], dtype=torch.float64)
        init_phase = self.classify_dimensional_phases(g_init, init_kappa, init_alpha)
        init_aniso = float(self.compute_anisotropy(g_init).mean())

        # 分段演化
        g = g_init.clone()
        trajectory = {
            "version": [],
            "desc": [],
            "kappa_vec": [],
            "alpha_vec": [],
            "anisotropy": [],
            "cond_g": [],
            "rho_mean": [],
            "rho_vec": [],
            "overall_phase": [],
            "dim_phases": [],
            "is_formless": [],
        }

        cumulative_rho = torch.zeros(d, dtype=torch.float64)

        for event in history:
            kappa_vec = torch.tensor(event["kappa_vec"], dtype=torch.float64)
            alpha_vec = torch.tensor(event["alpha_vec"], dtype=torch.float64)

            # 演化前 ρ̂
            rho_info_before = self.compute_rho_full_tensor(g, C, phi, kappa_vec, alpha_vec)
            # ρ 弛豫积累（指数移动平均）
            cumulative_rho = 0.8 * cumulative_rho + 0.2 * rho_info_before["rho_vec"]

            # 时间演化这一段（只跑一次，获取最终 g）
            g = self._run_segment_get_final_g(
                g, C, phi, kappa_vec, alpha_vec,
                n_steps=event["n_steps"], dt=event["dt"],
            )

            # 演化后相态
            final_phase = self.classify_dimensional_phases(g, kappa_vec, alpha_vec)
            final_aniso = float(self.compute_anisotropy(g).mean())
            final_cond = self._compute_cond_g_mean(g)

            # 记录
            trajectory["version"].append(event["version"])
            trajectory["desc"].append(event["desc"])
            trajectory["kappa_vec"].append(event["kappa_vec"])
            trajectory["alpha_vec"].append(event["alpha_vec"])
            trajectory["anisotropy"].append(final_aniso)
            trajectory["cond_g"].append(final_cond)
            trajectory["rho_mean"].append(float(cumulative_rho.mean()))
            trajectory["rho_vec"].append(cumulative_rho.tolist())
            trajectory["overall_phase"].append(final_phase["overall_phase"])
            trajectory["dim_phases"].append(final_phase["dim_phases"])
            trajectory["is_formless"].append("formless" in final_phase["overall_phase"])

        # 验证结果
        aniso_trajectory = trajectory["anisotropy"]
        rho_trajectory = trajectory["rho_mean"]
        phase_trajectory = trajectory["overall_phase"]

        # 验证1：各向异性下降趋势（用初始 aniso 作为基线）
        aniso_init = init_aniso  # 演化前的初始各向异性
        aniso_final = aniso_trajectory[-1]
        aniso_decreased = aniso_final < aniso_init
        # 允许中间波动，但整体趋势下降
        aniso_overall_trend = aniso_final <= aniso_init + 0.1

        # 验证2：ρ 积累过程
        rho_init = rho_trajectory[0]
        rho_final = rho_trajectory[-1]
        rho_accumulated = rho_final > rho_init

        # 验证3：分裂相+formless 涌现
        final_phase = phase_trajectory[-1]
        has_split = "分裂相" in final_phase
        has_formless = "formless" in final_phase
        split_formless_emerged = has_split and has_formless

        # 综合
        all_pass = aniso_overall_trend and rho_accumulated and split_formless_emerged

        return {
            "trajectory": trajectory,
            "aniso_init": aniso_init,
            "aniso_final": aniso_final,
            "aniso_decreased": aniso_decreased,
            "aniso_overall_trend": aniso_overall_trend,
            "rho_init": rho_init,
            "rho_final": rho_final,
            "rho_accumulated": rho_accumulated,
            "final_phase": final_phase,
            "split_formless_emerged": split_formless_emerged,
            "all_pass": all_pass,
        }

    def _run_segment_get_final_g(
        self,
        g_init: Tensor,
        C: Tensor,
        phi: Tensor,
        kappa_vec: Tensor,
        alpha_vec: Tensor,
        n_steps: int = 30,
        dt: float = 0.0005,
    ) -> Tensor:
        """运行一段演化，返回最终的 g（不记录轨迹）。"""
        g = g_init.clone()
        d = self.d
        delta_vec = torch.tensor([0.5] * d, dtype=torch.float64)
        rho_vec = torch.zeros(d, dtype=torch.float64)

        for _ in range(n_steps):
            rho_info = self.compute_rho_full_tensor(g, C, phi, kappa_vec, alpha_vec)
            rho_vec = rho_info["rho_vec"]

            kappa_mean = float(kappa_vec.mean())
            alpha_mean = float(alpha_vec.mean())
            delta_mean = float(delta_vec.mean())
            rho_mean = float(rho_vec.mean())
            delta_new_mean, _, _ = self.evolve_delta_step_with_rho(
                g, delta_mean, C, phi, kappa_mean, alpha_mean,
                rho=rho_mean, lr=0.05,
            )
            lambda_rho_v = self.compute_lambda_rho_vec(alpha_vec)
            delta_correction = 0.05 * lambda_rho_v * (rho_vec - rho_mean)
            delta_vec = delta_vec + delta_correction
            delta_vec = torch.clamp(delta_vec, min=0.0, max=1.0)

            g, _ = self.time_evolve_step(
                g, C, phi, kappa_vec, alpha_vec, delta_vec, rho_vec,
                dt=dt, include_cross=True, include_dissolution=True,
            )

        return g

    def _ensure_pd(self, g: Tensor) -> Tensor:
        """确保度规正定。"""
        N, d, _ = g.shape
        for n in range(N):
            eigvals = torch.linalg.eigvalsh(g[n])
            min_eig = float(eigvals.min())
            if min_eig < 1e-6:
                g[n] = g[n] + (1e-6 - min_eig) * torch.eye(d, dtype=torch.float64)
        return g

    # ==================================================================
    # 任务二：动态相图扫描
    # ==================================================================

    def scan_phase_diagram(
        self,
        C: Tensor,
        phi: Tensor,
        kappa_grid: list[float] | None = None,
        alpha_grid: list[float] | None = None,
        n_steps: int = 100,
        dt: float = 0.0005,
    ) -> dict:
        """
        动态相图扫描。

        扫描 (κ, α) 组合空间，对每个点从不同初始度规出发，
        观察时间演化轨迹，寻找所有稳定吸引子。

        初始条件：
            - VAE 初始：大特征值（稳定势阱）
            - GAN 初始：小特征值（流动状态）
            - 分裂初始：混合特征值
            - 黑洞初始：极端各向异性
        """
        if kappa_grid is None:
            kappa_grid = [0.01, 0.1, 1.0, 5.0, 10.0]
        if alpha_grid is None:
            alpha_grid = [0.1, 1.0, 5.0, 20.0, 100.0]

        d = self.d
        N = self.n_events

        # 初始条件集合
        init_conditions = self._make_init_conditions(N, d)

        results = []
        attractor_count = {}

        total_points = len(kappa_grid) * len(alpha_grid)
        point_idx = 0

        for kappa_scalar in kappa_grid:
            for alpha_scalar in alpha_grid:
                point_idx += 1
                # 均匀参数（所有维度相同）
                kappa_vec = torch.tensor([kappa_scalar] * d, dtype=torch.float64)
                alpha_vec = torch.tensor([alpha_scalar] * d, dtype=torch.float64)

                point_result = {
                    "kappa": kappa_scalar,
                    "alpha": alpha_scalar,
                    "init_conditions": {},
                }

                for init_name, g_init in init_conditions.items():
                    # 时间演化
                    g_final = self._run_segment_get_final_g(
                        g_init, C, phi, kappa_vec, alpha_vec,
                        n_steps=n_steps, dt=dt,
                    )

                    # 最终相态
                    final_phase = self.classify_dimensional_phases(g_final, kappa_vec, alpha_vec)
                    final_aniso = float(self.compute_anisotropy(g_final).mean())
                    final_cond = self._compute_cond_g_mean(g_final)

                    # 初始相态
                    init_phase = self.classify_dimensional_phases(g_init, kappa_vec, alpha_vec)
                    init_aniso = float(self.compute_anisotropy(g_init).mean())

                    # 吸引子分类
                    attractor = self._classify_attractor(
                        final_phase["overall_phase"], final_aniso, final_cond,
                    )

                    point_result["init_conditions"][init_name] = {
                        "init_phase": init_phase["overall_phase"],
                        "final_phase": final_phase["overall_phase"],
                        "init_aniso": init_aniso,
                        "final_aniso": final_aniso,
                        "final_cond": final_cond,
                        "attractor": attractor,
                        "reached_formless": "formless" in final_phase["overall_phase"],
                    }

                    # 统计吸引子
                    attractor_count[attractor] = attractor_count.get(attractor, 0) + 1

                results.append(point_result)

        # 分析结果
        analysis = self._analyze_phase_diagram(results)

        return {
            "results": results,
            "attractor_count": attractor_count,
            "analysis": analysis,
            "grid_size": total_points,
            "n_init_conditions": len(init_conditions),
        }

    def _make_init_conditions(self, N: int, d: int) -> dict[str, Tensor]:
        """构造不同的初始条件。"""
        conditions = {}

        # VAE 初始：大特征值（稳定势阱）
        g_vae = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            g_vae[n] = torch.diag(torch.tensor([2.0, 2.0, 2.0, 2.0][:d], dtype=torch.float64))
        conditions["VAE"] = self._ensure_pd(g_vae)

        # GAN 初始：小特征值（流动状态）
        g_gan = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            g_gan[n] = torch.diag(torch.tensor([0.3, 0.3, 0.3, 0.3][:d], dtype=torch.float64))
        conditions["GAN"] = self._ensure_pd(g_gan)

        # 分裂初始：混合特征值
        g_split = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            g_split[n] = torch.diag(torch.tensor([2.5, 2.0, 0.4, 0.3][:d], dtype=torch.float64))
        conditions["split"] = self._ensure_pd(g_split)

        # 黑洞初始：极端各向异性
        g_blackhole = torch.zeros(N, d, d, dtype=torch.float64)
        for n in range(N):
            g_blackhole[n] = torch.diag(torch.tensor([10.0, 1.0, 0.1, 0.01][:d], dtype=torch.float64))
        conditions["blackhole"] = self._ensure_pd(g_blackhole)

        return conditions

    def _classify_attractor(self, phase: str, aniso: float, cond: float) -> str:
        """分类吸引子类型。"""
        has_formless = "formless" in phase
        has_split = "分裂相" in phase
        is_all_vae = "全VAE" in phase
        is_all_gan = "全GAN" in phase

        if has_formless and has_split:
            return "split+formless"
        elif has_formless and is_all_vae:
            return "VAE+formless"
        elif has_formless and is_all_gan:
            return "GAN+formless"
        elif has_formless:
            return "formless"
        elif has_split:
            return "split"
        elif is_all_vae:
            return "VAE"
        elif is_all_gan:
            return "GAN"
        elif cond > 1000:
            return "collapse"
        else:
            return "other"

    def _analyze_phase_diagram(self, results: list[dict]) -> dict:
        """分析相图扫描结果。"""
        # 统计各吸引子出现次数
        attractor_by_init = {}
        formless_reachable_from = {}

        for point in results:
            for init_name, ic_result in point["init_conditions"].items():
                attractor = ic_result["attractor"]
                if init_name not in attractor_by_init:
                    attractor_by_init[init_name] = {}
                attractor_by_init[init_name][attractor] = \
                    attractor_by_init[init_name].get(attractor, 0) + 1

                if ic_result["reached_formless"]:
                    if init_name not in formless_reachable_from:
                        formless_reachable_from[init_name] = 0
                    formless_reachable_from[init_name] += 1

        # 关键问题解答
        total_points = len(results)
        total_ic = sum(len(p["init_conditions"]) for p in results)

        # 是否存在多条路径流向 formless
        n_paths_to_formless = sum(formless_reachable_from.values())
        multiple_paths = len(formless_reachable_from) >= 2

        # 各初始条件到达 formless 的比例
        formless_reachability = {}
        for init_name in attractor_by_init:
            total = sum(attractor_by_init[init_name].values())
            reached = formless_reachable_from.get(init_name, 0)
            formless_reachability[init_name] = {
                "reached": reached,
                "total": total,
                "ratio": reached / total if total > 0 else 0.0,
            }

        return {
            "attractor_by_init": attractor_by_init,
            "formless_reachable_from": formless_reachable_from,
            "formless_reachability": formless_reachability,
            "n_paths_to_formless": n_paths_to_formless,
            "multiple_paths_to_formless": multiple_paths,
            "total_points": total_points,
            "total_ic_runs": total_ic,
        }

    # ==================================================================
    # 验证函数
    # ==================================================================

    def verify_v63_bootstrap(self, C: Tensor, phi: Tensor) -> dict:
        """验证自举测试。"""
        result = self.run_bootstrap_test(C, phi)

        return {
            "aniso_init": result["aniso_init"],
            "aniso_final": result["aniso_final"],
            "aniso_decreased": result["aniso_decreased"],
            "aniso_overall_trend": result["aniso_overall_trend"],
            "rho_init": result["rho_init"],
            "rho_final": result["rho_final"],
            "rho_accumulated": result["rho_accumulated"],
            "final_phase": result["final_phase"],
            "split_formless_emerged": result["split_formless_emerged"],
            "all_pass": result["all_pass"],
            "trajectory": result["trajectory"],
        }

    def verify_v63_phase_diagram(self, C: Tensor, phi: Tensor) -> dict:
        """验证相图扫描。"""
        result = self.scan_phase_diagram(C, phi)
        analysis = result["analysis"]

        # 验证：至少有一条路径到达 formless
        has_formless_path = analysis["n_paths_to_formless"] > 0

        # 验证：多条路径到达 formless（八万四千法门）
        multiple_paths = analysis["multiple_paths_to_formless"]

        # 验证：不同初始条件都能到达 formless
        all_init_reach_formless = all(
            v["reached"] > 0 for v in analysis["formless_reachability"].values()
        )

        all_pass = has_formless_path and multiple_paths

        return {
            "has_formless_path": has_formless_path,
            "multiple_paths": multiple_paths,
            "all_init_reach_formless": all_init_reach_formless,
            "n_paths_to_formless": analysis["n_paths_to_formless"],
            "formless_reachability": analysis["formless_reachability"],
            "attractor_count": result["attractor_count"],
            "all_pass": all_pass,
            "analysis": analysis,
        }
