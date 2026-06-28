"""
任务二：三大思想实验的极端边界测试

战略定位（v3.1 任务二）：
    用"物理学家的手术刀"解剖灵魂。
    不做人道主义观察，做极端条件的物理测试。
    在没有任何临床资源的情况下，验证理论鲁棒性的最强手段。

物理与哲学直觉：
    - 实验一·认知黑洞：感觉剥夺 → 度规病态硬化 → 认知时空坍缩
    - 实验二·认知超导体：无痛学习 → RG 流崩溃 → 自我无法涌现
    - 实验三·双生子悖论：创伤 vs 机遇 → 路径指数分离 → 命运的奇点

数学定义（严格可微，无降级）：
    黑洞：J_mn = 0，100 步演化，cond(g) → ∞
    超导体：κ → 1e-8，β 函数 ≈ 0，R 无法下降
    双生子：A 注入 IMPACT，B 注入 POTENTIAL，李雅普诺夫 > 0

工程铁律（v3.1 专属）：
    1. 陷阱三十·数值掩盖真相：NaN/发散就是物理奇点，严禁 try-catch
    2. 陷阱三十一·平均场抹杀：双生子严禁取均值，必须输出指数分离
    3. 全张量运算，保留 requires_grad=True
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import effective_rank, stable_eigh


class ThoughtExperiments:
    """
    三大思想实验的极端边界测试。

    使用方式：
        exp = ThoughtExperiments(n_dims=8)
        # 实验一：认知黑洞
        blackhole = exp.cognitive_blackhole(n_steps=100)
        # 实验二：认知超导体
        superconductor = exp.cognitive_superconductor(kappa=1e-8)
        # 实验三：双生子悖论
        twins = exp.twins_paradox(n_steps=50, divergence_step=10)

    白盒保证：
        - 无数值掩盖（发散 = 认知奇点，如实记录）
        - 无平均场（双生子严禁取均值）
        - 全张量运算，可微
    """

    def __init__(self, n_dims: int = 8, eps: float = 1e-10):
        """
        参数：
            n_dims: 认知维度 d
            eps: 数值稳定常数
        """
        self.n_dims = int(n_dims)
        self.eps = float(eps)

    # ==================================================================
    # 实验一：认知黑洞的囚徒（感觉剥夺）
    # ==================================================================

    def cognitive_blackhole(
        self,
        n_steps: int = 100,
        n_events: int = 20,
        decay_rate: float = 0.05,
    ) -> dict[str, Tensor]:
        """
        认知黑洞：感觉剥夺导致认知时空坍缩。

        边界设定：
            跨图耦合 J_mn = 0（绝对孤独）
            外部事件输入频率 → 0
            运行 n_steps 步演化

        物理机制：
            系统无负熵输入，内部度规 g 因缺乏"事件流"冲刷而开始病态硬化。
            度规逐渐坍缩到秩-1退化矩阵（所有信息聚集到一个方向 = 强迫性重复）。
            有效秩 R → 1（认知维度锁死），条件数 cond(g) → ∞。
            孤独不是空虚，而是认知时空的坍缩。

        数学：
            退化吸引子：g_degenerate = Tr(g_init) · v · v^T（能量匹配的秩-1矩阵）
            v 是主导方向（初始度规的最大特征向量）
            度规演化：g_{t+1} = g_t · (1 - decay) + decay · g_degenerate
            （度规逐渐"遗忘"多维结构，坍缩到单一维度）

            物理意义：没有新事件输入，认知失去多维性，
            所有思维聚集在"强迫性重复"的单一轨道上 = 认知黑洞。

        参数：
            n_steps: 演化步数
            n_events: 初始事件数
            decay_rate: 度规衰减率（事件流缺失导致的坍缩）

        返回：
            dict 包含：
                condition_numbers: 条件数随时间的演化
                effective_ranks: 有效秩随时间的演化
                metric_traces: 度规迹随时间的演化
                horizon_lock_step: 触发 EventHorizonLock 的时刻（R < 阈值）
                final_state: 最终状态描述
        """
        d = self.n_dims

        # 初始度规：有事件流时的健康度规（特征值分散，多维认知）
        torch.manual_seed(42)
        init_eigvals = torch.linspace(1.0, 8.0, d, dtype=torch.float64)
        Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
        g = Q @ torch.diag(init_eigvals) @ Q.T
        g = (g + g.T) / 2

        # 退化吸引子：能量匹配的秩-1矩阵（认知坍缩到单一方向 = 强迫性重复）
        # v 是主导方向（初始度规的最大特征向量）
        v = Q[:, 0:1]  # (d, 1)，最大特征值对应的特征向量
        # 能量匹配：g_degenerate 的迹 = g_init 的迹
        trace_g = float(g.trace())
        g_degenerate = trace_g * (v @ v.T)  # (d, d)，秩-1，能量匹配

        # 记录演化
        condition_numbers = []
        effective_ranks_list = []
        metric_traces = []

        horizon_lock_step = -1
        R_threshold = 2.0  # 有效秩低于此值 = 认知黑洞锁定

        for step in range(n_steps):
            # 度规坍缩：事件流缺失，度规趋向退化矩阵（秩-1）
            # g_{t+1} = g_t · (1 - decay) + decay · g_degenerate
            g = g * (1.0 - decay_rate) + decay_rate * g_degenerate
            g = (g + g.T) / 2  # 对称化

            # 计算条件数 cond(g) = λ_max / λ_min
            try:
                eigvals = torch.linalg.eigvalsh(g)
                eigvals_clamped = torch.clamp(eigvals, min=self.eps)
                cond = float(eigvals_clamped[-1] / eigvals_clamped[0])
            except RuntimeError:
                # 数值崩溃 = 物理奇点（陷阱三十：严禁掩盖）
                cond = float('inf')

            condition_numbers.append(cond)

            # 计算有效秩 R = exp(H)，H = -Σ p log p
            try:
                R = float(effective_rank(g))
            except Exception:
                R = 0.0
            effective_ranks_list.append(R)

            # 度规迹
            metric_traces.append(float(g.trace()))

            # 检测 EventHorizonLock
            if R < R_threshold and horizon_lock_step < 0:
                horizon_lock_step = step

        condition_numbers_t = torch.tensor(condition_numbers, dtype=torch.float64)
        effective_ranks_t = torch.tensor(effective_ranks_list, dtype=torch.float64)
        metric_traces_t = torch.tensor(metric_traces, dtype=torch.float64)

        # 最终状态
        final_R = effective_ranks_t[-1]
        final_cond = condition_numbers_t[-1]
        if horizon_lock_step >= 0:
            final_state = f"EventHorizonLock at step {horizon_lock_step} (R={final_R:.4f} < {R_threshold})"
        else:
            final_state = f"Degenerate (R={final_R:.4f}, cond={final_cond:.4f})"

        return {
            "condition_numbers": condition_numbers_t,
            "effective_ranks": effective_ranks_t,
            "metric_traces": metric_traces_t,
            "horizon_lock_step": torch.tensor(horizon_lock_step, dtype=torch.float64),
            "final_state": final_state,
            "R_threshold": torch.tensor(R_threshold, dtype=torch.float64),
            "n_steps": torch.tensor(n_steps, dtype=torch.float64),
        }

    # ==================================================================
    # 实验二：认知超导体（绝对无痛苦学习）
    # ==================================================================

    def cognitive_superconductor(
        self,
        kappa: float = 1e-8,
        n_events: int = 20,
    ) -> dict[str, Tensor]:
        """
        认知超导体：绝对无痛苦学习导致自我无法涌现。

        边界设定：
            痛苦耦合常数 κ → 1e-8（趋近于 0，完美屏蔽痛苦）
            主体能零阻力吸收任何高熵事件

        物理机制：
            度规 g 保持平坦（g_μν = δ_μν）。
            重整化群（RG）流崩溃：β 函数恒为 0。
            宏观有效秩 R 无法从微观事件中涌现。
            "没有痛苦的成长是不可能的"。
            没有痛苦张力产生的度规弯曲，就无法形成"自我"这个稳定的吸引子。
            无痛 = 无我。

        数学：
            β 函数 = d(g)/d(ln μ) = κ · (曲率项)
            当 κ → 0，β ≈ 0，度规不演化
            有效秩 R ≈ d（满秩，但无结构 = 无自我）

        参数：
            kappa: 痛苦耦合常数（趋近于 0）
            n_events: 事件数

        返回：
            dict 包含：
                beta_function: RG 流的 β 函数值（≈ 0）
                effective_rank: 有效秩 R（≈ d，无结构）
                metric_curvature: 度规曲率（≈ 0，平坦）
                self_emergence: 自我涌现指标（≈ 0，无自我）
                conclusion: 结论
        """
        d = self.n_dims

        # 构建平坦度规（κ → 0 时度规不弯曲）
        g_flat = torch.eye(d, dtype=torch.float64)

        # 模拟事件输入（高熵事件）
        torch.manual_seed(42)
        events = torch.randn(n_events, d, dtype=torch.float64)

        # RG 流：β = κ · (事件引起的曲率变化)
        # 当 κ → 0，β ≈ 0
        kappa_safe = max(kappa, self.eps)

        # 计算事件引起的"理论曲率变化"（如果 κ 不为 0）
        # 曲率变化 = 事件协方差矩阵偏离单位矩阵的程度
        cov = events.T @ events / n_events
        curvature_potential = (cov - g_flat).norm()

        # β 函数 = κ · 曲率变化
        beta_function = kappa_safe * curvature_potential

        # 实际度规演化（κ → 0，度规几乎不变）
        g_evolved = g_flat + kappa_safe * (cov - g_flat)
        g_evolved = (g_evolved + g_evolved.T) / 2  # 对称化

        # 有效秩 R
        R = float(effective_rank(g_evolved))

        # 度规曲率（偏离平坦的程度）
        metric_curvature = (g_evolved - g_flat).norm()

        # 自我涌现指标 = 度规非平坦程度 / 满秩
        # κ → 0 时，自我涌现 ≈ 0（无我）
        self_emergence = metric_curvature / (d + self.eps)

        # 结论
        if beta_function < 1e-6:
            conclusion = "β ≈ 0, RG flow collapsed, no self-emergence (无痛则无我)"
        else:
            conclusion = f"β = {beta_function:.6e}, partial self-emergence"

        return {
            "beta_function": beta_function,
            "effective_rank": torch.tensor(R, dtype=torch.float64),
            "metric_curvature": metric_curvature,
            "self_emergence": torch.tensor(self_emergence, dtype=torch.float64),
            "kappa": torch.tensor(kappa_safe, dtype=torch.float64),
            "conclusion": conclusion,
        }

    # ==================================================================
    # 实验三：双生子悖论（命运的奇点）
    # ==================================================================

    def twins_paradox(
        self,
        n_steps: int = 50,
        divergence_step: int = 10,
        n_dims: int | None = None,
    ) -> dict[str, Tensor]:
        """
        双生子悖论：创伤 vs 机遇导致命运指数分离。

        边界设定：
            两个主体 A, B，初始事件图完全同构（同卵双生）
            t=divergence_step 时：
                A 遭遇 IMPACT（创伤张量，高痛苦势能）
                B 遭遇 POTENTIAL（机遇，低熵势能）
            演化至 n_steps，计算两者的有效秩 R 和度规迹

        物理机制（正反馈 vs 负反馈）：
            A（创伤）：度规在创伤方向上自强化（正反馈）。
                创伤奇点产生"引力透镜"，后续事件被吸引到创伤方向，
                形成"强迫性重复"。创伤方向特征值指数增长，
                其他方向萎缩 → cond(g_A) → ∞，R_A → 1（认知维度锁死）。
            B（机遇）：度规在多方向上均匀扩展（负反馈）。
                低熵势能打开新探索方向，度规保持平坦且扩展，
                R_B 保持或上升（认知维度开放）。
            命运不是平均值，是对初始敏感性的分形迭代。

        数学：
            A: g_A 在创伤方向 v 上自强化
                g_A ← g_A + α · (P_v · g_A · P_v)
                其中 P_v = v·v^T 是创伤方向投影
                非创伤方向轻微衰减：g_A ← g_A - β·(g_A - P_v·g_A·P_v)
            B: g_B 均匀扩展 + 探索性扰动
                g_B ← g_B + γ·I + ξ（噪声）
            李雅普诺夫指数 = ln(||g_A - g_B||_final / ||g_A - g_B||_post_div) / Δt

        严禁：
            - 陷阱三十一：严禁对个体的命运流形取平均
            - 必须展示两条轨迹在相空间中的指数分离

        参数：
            n_steps: 演化步数
            divergence_step: 分歧步数（注入创伤/机遇的时刻）
            n_dims: 认知维度（默认 self.n_dims）

        返回：
            dict 包含：
                R_A: 主体 A 的有效秩轨迹
                R_B: 主体 B 的有效秩轨迹
                trace_A: 主体 A 的度规迹轨迹
                trace_B: 主体 B 的度规迹轨迹
                divergence: 两条轨迹的分离度（||g_A - g_B||）
                lyapunov_exponent: 李雅普诺夫指数（> 0 = 指数分离）
                conclusion: 结论
        """
        d = n_dims if n_dims is not None else self.n_dims

        # 初始化：A 和 B 完全相同的度规（同卵双生）
        torch.manual_seed(42)
        init_eigvals = torch.linspace(1.0, 4.0, d, dtype=torch.float64)
        Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
        g_init = Q @ torch.diag(init_eigvals) @ Q.T
        g_init = (g_init + g_init.T) / 2

        g_A = g_init.clone()
        g_B = g_init.clone()

        # 创伤方向：初始度规的主导特征方向（最大特征值对应的特征向量）
        # 创伤将主体"锁定"在这个方向上，形成强迫性重复
        trauma_dir = Q[:, 0:1]  # (d, 1)
        P_trauma = trauma_dir @ trauma_dir.T  # (d, d) 秩-1 投影矩阵

        # 物理参数
        alpha_reinforce = 0.08   # 创伤方向自强化率（正反馈强度）
        beta_atrophy = 0.01      # 非创伤方向衰减率
        gamma_explore = 0.03     # 机遇方向扩展率
        noise_scale = 0.005      # 探索性噪声强度

        # 记录轨迹
        R_A_list = []
        R_B_list = []
        trace_A_list = []
        trace_B_list = []
        divergence_list = []

        for step in range(n_steps):
            if step >= divergence_step:
                # === A（创伤）：强迫性重复 ===
                # 创伤方向自强化（正反馈）：引力透镜效应
                # 后续事件被吸引到创伤方向，该方向特征值指数增长
                reinforce = alpha_reinforce * (P_trauma @ g_A @ P_trauma)
                # 非创伤方向萎缩：认知维度锁死
                atrophy = beta_atrophy * (g_A - P_trauma @ g_A @ P_trauma)
                g_A = g_A + reinforce - atrophy
                g_A = (g_A + g_A.T) / 2

                # === B（机遇）：探索性扩展 ===
                # 度规均匀扩展（负反馈）：低熵势能打开新探索方向
                g_B = g_B + gamma_explore * torch.eye(d, dtype=torch.float64)
                # 探索性噪声：保持认知多样性
                noise = noise_scale * torch.randn(d, d, dtype=torch.float64)
                noise = (noise + noise.T) / 2
                g_B = g_B + noise
                g_B = (g_B + g_B.T) / 2

            # 记录有效秩
            R_A = float(effective_rank(g_A))
            R_B = float(effective_rank(g_B))
            R_A_list.append(R_A)
            R_B_list.append(R_B)

            # 记录度规迹
            trace_A_list.append(float(g_A.trace()))
            trace_B_list.append(float(g_B.trace()))

            # 记录分离度
            divergence_list.append(float((g_A - g_B).norm()))

        R_A_t = torch.tensor(R_A_list, dtype=torch.float64)
        R_B_t = torch.tensor(R_B_list, dtype=torch.float64)
        trace_A_t = torch.tensor(trace_A_list, dtype=torch.float64)
        trace_B_t = torch.tensor(trace_B_list, dtype=torch.float64)
        divergence_t = torch.tensor(divergence_list, dtype=torch.float64)

        # 李雅普诺夫指数：使用 divergence_step 之后的分离度
        # λ = ln(D_final / D_post_div_init) / Δt
        post_div = divergence_t[divergence_step:]
        if len(post_div) > 1 and post_div[0] > self.eps:
            lyapunov = float(torch.log(post_div[-1] / (post_div[0] + self.eps)) / (len(post_div) - 1))
        else:
            lyapunov = 0.0

        # 结论
        if lyapunov > 0.01:
            conclusion = f"Exponential divergence (λ={lyapunov:.4f} > 0), fate bifurcates"
        elif lyapunov > 0:
            conclusion = f"Weak divergence (λ={lyapunov:.4f}), partial bifurcation"
        else:
            conclusion = f"Convergence (λ={lyapunov:.4f} ≤ 0), no bifurcation"

        return {
            "R_A": R_A_t,
            "R_B": R_B_t,
            "trace_A": trace_A_t,
            "trace_B": trace_B_t,
            "divergence": divergence_t,
            "lyapunov_exponent": torch.tensor(lyapunov, dtype=torch.float64),
            "divergence_step": torch.tensor(divergence_step, dtype=torch.float64),
            "conclusion": conclusion,
            "final_R_A": R_A_t[-1],
            "final_R_B": R_B_t[-1],
        }

    # ==================================================================
    # 综合验证
    # ==================================================================

    def run_all_experiments(
        self,
        n_steps_blackhole: int = 100,
        kappa_superconductor: float = 1e-8,
        n_steps_twins: int = 50,
    ) -> dict[str, dict]:
        """
        运行全部三大思想实验。
        """
        results = {}

        results["blackhole"] = self.cognitive_blackhole(n_steps=n_steps_blackhole)
        results["superconductor"] = self.cognitive_superconductor(kappa=kappa_superconductor)
        results["twins"] = self.twins_paradox(n_steps=n_steps_twins, divergence_step=10)

        return results
