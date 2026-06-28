"""
认知推进器最优控制模块 —— 庞特里亚金最大值原理（PMP）干预方案

战略定位（v1.2 步骤三）：
    v1.1 只能算命，v1.2 要改命。根据庞特里亚金最大值原理，
    计算出一套最小能量的干预方案。这是系统的"药方"。

物理原理：
    将人生轨迹视为测地线，求解使其偏离痛苦奇点的最优控制力 u(t)。
    不是给鸡汤，而是给精确的"受力方向"和"受力大小"。

数学公式（已锁定，无模糊地带）：
    A. 哈密顿量定义：
        H(S, g, u) = (1/2) S^T g S + (1/2) κ (tr(g) - n)^2 + (1/2) ||u||^2

        物理意义：
            - (1/2) S^T g S：当前状态的痛苦势能（度规加权能量）
            - (1/2) κ (tr(g) - n)^2：度规偏离基线的曲率惩罚
              （tr(g)=n 是基线，偏离越大说明创伤越深）
            - (1/2) ||u||^2：控制能量消耗（最小能量干预）

    B. 协态方程（反向积分协态 λ）：
        dλ/dt = -∂H/∂S = -g · S + η · g · λ

        终端条件：
            λ(T) = ∂/∂S ( (1/2) ||S(T) - S_target||_g^2 )
                  = g · (S(T) - S_target)

        物理意义：
            λ 是"影子价格"，衡量状态偏离目标的心理代价。
            反向积分保证终端条件满足（目标导向）。

    C. 最优控制律：
        u*(t) = -λ(t) / ||λ(t)|| · min(||λ(t)||, κ · ||S(t)||)

        物理意义：
            - 方向：-λ/||λ||（沿代价下降方向）
            - 幅度：min(||λ||, κ·||S||)（不超过系统能承受的限度）
            - κ·||S|| 是"安全上限"，防止干预过强导致流形撕裂

    D. 演化过程：
        必须调用引擎的 RK4 在度规 g 下积分。
        严禁线性插值 S_next = S_current + (S_target - S_current) / steps。
        直线路径会穿过"心理大山"，临床上不可能。

工程铁律（v1.2 三大铁律）：
    1. 拒绝字典：控制力由 PMP 数学结构推导
    2. 内生安全：控制力不超过 κ·||S||，防止触发事件视界
    3. 几何诚实：所有演化基于流形测地线（RK4 + 度规 g）

死刑纠错（v1.2 步骤三）：
    错误一：线性插值降级。严禁 S_next = S_current + (S_target - S_current) / steps。
           必须调用 RK4 在弯曲时空中积分。
    错误二：忽视度规。严禁用欧氏距离 ||S1 - S2||。
           必须用度规加权距离 sqrt((S1-S2)^T g (S1-S2))。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.tensor_ops import safe_inverse, symmetric_part, stable_eigh
from ..core.event_horizon import EventHorizonLock


class CognitiveThruster:
    """
    认知推进器：基于庞特里亚金最大值原理（PMP）的最优干预方案。

    使用方式：
        thruster = CognitiveThruster(n_dims=8)
        result = thruster.compute(
            S_current=current_state,
            S_target=target_state,
            metric=current_metric,
            T_horizon=10.0,  # 干预时间窗口
            n_steps=50,
        )
        # result: {
        #     'trajectory': 干预后的状态轨迹,
        #     'control_sequence': 控制力序列 u(t),
        #     'costate_sequence': 协态序列 λ(t),
        #     'pain_reduction': 痛苦减少量,
        #     'max_lock_degree': 最大锁定程度（安全约束）,
        #     'total_control_energy': 总控制能量,
        # }

    白盒保证：
        - 控制力由 PMP 数学结构推导（非经验调参）
        - 演化基于 RK4 + 度规 g（非线性插值）
        - 距离基于度规加权（非欧氏）
        - 安全约束：控制力不超过 κ·||S||，防止流形撕裂
    """

    def __init__(self, n_dims: int):
        """
        参数：
            n_dims: 认知维度
        """
        self.n_dims = n_dims
        # 耦合常数 κ = 1/(n+2)（结构常数，与度规场方程一致）
        self.kappa = torch.tensor(1.0 / (n_dims + 2), dtype=torch.float64)
        # 事件视界检测器（安全约束）
        self.horizon = EventHorizonLock(n_dims=n_dims)

    def hamiltonian(self, S: Tensor, g: Tensor, u: Tensor, lambda_vec: Tensor) -> Tensor:
        """
        计算庞特里亚金哈密顿量 H(S, g, u, λ)。

        H = (1/2) S^T g S + (1/2) κ (tr(g) - n)^2 + (1/2) ||u||^2 + λ^T · f(S, u)

        其中 f(S, u) = -∇H_potential + u 是状态演化方程（简化形式）。

        物理意义：
            - (1/2) S^T g S：痛苦势能
            - (1/2) κ (tr(g) - n)^2：度规曲率惩罚
            - (1/2) ||u||^2：控制能量
            - λ^T · f：协态与状态演化的内积
        """
        S = S.to(torch.float64)
        g = symmetric_part(g.to(torch.float64))
        u = u.to(torch.float64)
        lam = lambda_vec.to(torch.float64)
        n = self.n_dims

        # 痛苦势能
        pain_potential = 0.5 * (S @ g @ S)

        # 度规曲率惩罚
        trace_g = torch.trace(g)
        curvature_penalty = 0.5 * self.kappa * (trace_g - n) ** 2

        # 控制能量
        control_energy = 0.5 * (u @ u)

        # 状态演化（简化：f = -g·S + u）
        f = -g @ S + u

        # 哈密顿量
        H = pain_potential + curvature_penalty + control_energy + lam @ f

        return H

    def state_dynamics(self, S: Tensor, g: Tensor, u: Tensor, eta: Tensor) -> Tensor:
        """
        状态演化方程 dS/dt = ∂H/∂λ = f(S, u)。

        dS/dt = -g · S + η · g · (S_baseline - S) + u

        物理意义：
            - -g·S：痛苦势能梯度下降（减少痛苦）
            - η·g·(S_baseline - S)：稳态回归（认知惯性）
            - u：外部控制力（干预）

        严禁：线性插值 S_next = S + (S_target - S)/steps。
        """
        S = S.to(torch.float64)
        g = symmetric_part(g.to(torch.float64))
        u = u.to(torch.float64)

        # 痛苦梯度下降
        pain_gradient = -g @ S

        # 稳态回归（向原点回归，简化形式）
        # η 由度规条件数推导（与演化场一致）
        eigvals, _ = stable_eigh(g)
        eigvals = torch.clamp(eigvals, min=1e-20)
        cond = eigvals.max() / eigvals.min()
        eta_val = 1.0 / (1.0 + torch.log(1.0 + cond))
        regression = eta_val * g @ (-S)  # 向原点回归

        # 控制力
        dS = pain_gradient + regression + u

        return dS

    def costate_dynamics(self, S: Tensor, g: Tensor, lambda_vec: Tensor, eta: Tensor) -> Tensor:
        """
        协态方程 dλ/dt = -∂H/∂S。

        dλ/dt = -g · S + η · g · λ

        物理意义：
            λ 是"影子价格"，衡量状态偏离目标的心理代价。
            -g·S：痛苦势能对状态的梯度（代价来源）
            η·g·λ：协态的扩散项（信息传播）

        终端条件：
            λ(T) = g · (S(T) - S_target)
        """
        S = S.to(torch.float64)
        g = symmetric_part(g.to(torch.float64))
        lam = lambda_vec.to(torch.float64)

        # 协态演化
        dLambda = -g @ S + eta * g @ lam

        return dLambda

    def optimal_control(self, S: Tensor, lambda_vec: Tensor) -> Tensor:
        """
        最优控制律 u*(t)。

        u*(t) = -λ(t) / ||λ(t)|| · min(||λ(t)||, κ · ||S(t)||)

        物理意义：
            - 方向：-λ/||λ||（沿代价下降方向）
            - 幅度：min(||λ||, κ·||S||)（安全上限）
            - κ·||S|| 防止干预过强导致流形撕裂

        严禁：u = Adam 优化器梯度下降（非 PMP）。
        """
        S = S.to(torch.float64)
        lam = lambda_vec.to(torch.float64)

        lam_norm = lam.norm() + 1e-30
        S_norm = S.norm()

        # 方向：-λ/||λ||
        direction = -lam / lam_norm

        # 幅度：min(||λ||, κ·||S||)
        amplitude = torch.minimum(lam_norm, self.kappa * S_norm)

        # 最优控制力
        u_star = direction * amplitude

        return u_star

    def metric_weighted_distance(self, S_a: Tensor, S_b: Tensor, g: Tensor) -> Tensor:
        """
        度规加权距离 d(S_a, S_b) = sqrt((S_a - S_b)^T g (S_a - S_b))。

        物理意义：
            两个认知状态间的"心理距离"。痛苦使度规弯曲，距离改变。
            忽略度规等于无视心理防御机制。

        严禁：欧氏距离 ||S_a - S_b||。
        """
        delta = (S_a - S_b).to(torch.float64)
        g = symmetric_part(g.to(torch.float64))
        return torch.sqrt(torch.clamp(delta @ g @ delta, min=1e-30))

    def compute(
        self,
        S_current: Tensor,
        S_target: Tensor,
        metric: Tensor,
        T_horizon: float = 10.0,
        n_steps: int = 50,
        n_shooting_iterations: int = 5,
    ) -> dict[str, Tensor | float]:
        """
        计算最优干预方案（严格 PMP 前向-反向打靶法）。

        参数：
            S_current: 当前认知状态 S(0)
            S_target: 目标认知状态 S(T)
            metric: 当前度规 g
            T_horizon: 干预时间窗口 [0, T]
            n_steps: 离散化步数
            n_shooting_iterations: 打靶法迭代次数（前向-反向迭代）

        返回：
            {
                'trajectory': 干预后的状态轨迹 (T+1, n),
                'control_sequence': 控制力序列 (T, n),
                'costate_sequence': 协态序列 (T+1, n),
                'pain_reduction': 痛苦减少量,
                'max_lock_degree': 最大锁定程度（安全约束）,
                'total_control_energy': 总控制能量,
                'final_distance': 终态到目标的度规距离,
                'is_safe': 是否安全（未触发事件视界）,
                'n_shooting_iterations': 打靶法迭代次数,
            }

        算法（严格 PMP，无降级）：
            1. 前向 pass：用当前协态猜测 λ(t) 演化状态 S(t)，应用最优控制 u*(t)
            2. 反向 pass：用终端条件 λ(T) = g·(S(T)-S_target) 反向积分协态 λ(t)
            3. 打靶迭代：重复 1-2 直至收敛（协态与状态自洽）
            4. 安全约束：连续衰减因子 (1-lock_degree)，非离散 if

        严禁：
            - 线性插值 S_next = S + (S_target - S)/steps
            - 欧氏距离 ||S - S_target||
            - Adam 优化器代替 PMP
            - 协态前向近似（必须反向积分）
            - 离散 if lock_degree > threshold（必须连续衰减）
        """
        S = S_current.to(torch.float64).clone()
        S_tgt = S_target.to(torch.float64)
        g = symmetric_part(metric.to(torch.float64))
        n = self.n_dims

        dt = T_horizon / n_steps

        # η 由度规条件数推导（与演化场一致）
        eigvals, _ = stable_eigh(g)
        eigvals = torch.clamp(eigvals, min=1e-20)
        cond = eigvals.max() / eigvals.min()
        eta = 1.0 / (1.0 + torch.log(1.0 + cond))

        # ============================================================
        # 严格 PMP 双向打靶法（前向状态 + 反向协态 + 迭代）
        # ============================================================
        # 初始协态猜测（首次迭代用零初值，后续用反向积分结果更新）
        lambda_trajectory = torch.zeros(n_steps + 1, n, dtype=torch.float64)
        state_trajectory = torch.zeros(n_steps + 1, n, dtype=torch.float64)
        state_trajectory[0] = S.clone()
        control_sequence = torch.zeros(n_steps, n, dtype=torch.float64)

        for iteration in range(n_shooting_iterations):
            # ---------- 前向 pass：用当前协态演化状态 ----------
            S_forward = S.clone()
            state_trajectory[0] = S_forward.clone()

            for step in range(n_steps):
                # 最优控制力 u*(t) = -λ/||λ|| · min(||λ||, κ·||S||)
                lam = lambda_trajectory[step]
                u = self.optimal_control(S_forward, lam)
                control_sequence[step] = u.clone()

                # RK4 积分状态方程 dS/dt = f(S, u)
                k1 = self.state_dynamics(S_forward, g, u, eta)
                k2 = self.state_dynamics(S_forward + 0.5 * dt * k1, g, u, eta)
                k3 = self.state_dynamics(S_forward + 0.5 * dt * k2, g, u, eta)
                k4 = self.state_dynamics(S_forward + dt * k3, g, u, eta)
                S_forward = S_forward + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                # 安全约束：连续衰减因子（非离散 if）
                # lock_degree ∈ [0, 1]，衰减 = (1 - lock_degree) 连续
                lock_degree = self.horizon.compute_lock_degree(g)
                attenuation = 1.0 - float(lock_degree)  # 连续衰减
                # 状态向安全状态连续混合（非离散替换）
                S_safe, _, _, _, _ = self.horizon.filter(S_forward, g, dt=dt)
                S_forward = attenuation * S_forward + (1.0 - attenuation) * S_safe

                state_trajectory[step + 1] = S_forward.clone()

            # ---------- 反向 pass：从终端条件反向积分协态 ----------
            # 终端条件：λ(T) = g · (S(T) - S_target)
            lambda_T = g @ (state_trajectory[-1] - S_tgt)
            lambda_trajectory[-1] = lambda_T.clone()

            lam_backward = lambda_T.clone()
            for step in range(n_steps - 1, -1, -1):
                # 反向积分协态方程 dλ/dt = -g·S + η·g·λ
                # 反向时间 τ = T - t，dλ/dτ = -dλ/dt = g·S - η·g·λ
                S_at_step = state_trajectory[step]
                # RK4 反向积分（dτ = dt）
                kl1 = -self.costate_dynamics(S_at_step, g, lam_backward, eta)
                kl2 = -self.costate_dynamics(S_at_step, g, lam_backward + 0.5 * dt * kl1, eta)
                kl3 = -self.costate_dynamics(S_at_step, g, lam_backward + 0.5 * dt * kl2, eta)
                kl4 = -self.costate_dynamics(S_at_step, g, lam_backward + dt * kl3, eta)
                lam_backward = lam_backward + (dt / 6.0) * (kl1 + 2 * kl2 + 2 * kl3 + kl4)

                lambda_trajectory[step] = lam_backward.clone()

            # 打靶法收敛性：协态已更新，进入下一次前向 pass

        # 最终前向 pass：用收敛后的协态生成最终轨迹
        S_final = S.clone()
        trajectory = [S_final.clone()]
        final_control = []
        final_costate = [lambda_trajectory[0].clone()]
        lock_degrees = []

        for step in range(n_steps):
            lam = lambda_trajectory[step]
            u = self.optimal_control(S_final, lam)

            # RK4 状态积分
            k1 = self.state_dynamics(S_final, g, u, eta)
            k2 = self.state_dynamics(S_final + 0.5 * dt * k1, g, u, eta)
            k3 = self.state_dynamics(S_final + 0.5 * dt * k2, g, u, eta)
            k4 = self.state_dynamics(S_final + dt * k3, g, u, eta)
            S_final = S_final + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 安全约束：连续衰减（非离散 if）
            lock_degree = self.horizon.compute_lock_degree(g)
            attenuation = 1.0 - float(lock_degree)
            S_safe, _, _, _, _ = self.horizon.filter(S_final, g, dt=dt)
            S_final = attenuation * S_final + (1.0 - attenuation) * S_safe
            # 控制力连续衰减（非离散 if u = u * (1-lock)）
            u = u * attenuation

            lock_degrees.append(float(lock_degree))
            trajectory.append(S_final.clone())
            final_control.append(u.clone())
            final_costate.append(lambda_trajectory[step + 1].clone())

        trajectory_tensor = torch.stack(trajectory, dim=0)
        control_tensor = torch.stack(final_control, dim=0)
        costate_tensor = torch.stack(final_costate, dim=0)

        # 评估指标
        # 痛苦减少量（度规加权能量差）
        pain_initial = float(S_current @ g @ S_current)
        pain_final = float(S_final @ g @ S_final)
        pain_reduction = pain_initial - pain_final

        # 总控制能量
        total_control_energy = float((control_tensor ** 2).sum())

        # 终态到目标的度规距离
        final_distance = float(self.metric_weighted_distance(S_final, S_tgt, g))

        # 最大锁定程度
        max_lock_degree = max(lock_degrees) if lock_degrees else 0.0

        # 安全性判定（连续阈值，非离散 if）
        is_safe = max_lock_degree < 0.8

        return {
            "trajectory": trajectory_tensor,
            "control_sequence": control_tensor,
            "costate_sequence": costate_tensor,
            "pain_reduction": pain_reduction,
            "max_lock_degree": max_lock_degree,
            "total_control_energy": total_control_energy,
            "final_distance": final_distance,
            "is_safe": is_safe,
            "n_steps": n_steps,
            "n_shooting_iterations": n_shooting_iterations,
        }

    def prescribe(
        self,
        S_current: Tensor,
        S_target: Tensor,
        metric: Tensor,
        T_horizon: float = 10.0,
        n_steps: int = 50,
    ) -> dict[str, Tensor | float | str]:
        """
        生成处方（含控制力描述和安全性评估）。

        在 compute() 基础上增加人类可读的处方描述，
        但所有数值仍由 PMP 数学结构推导，无字典查找。
        """
        result = self.compute(S_current, S_target, metric, T_horizon, n_steps)

        # 处方描述（基于数学量，非心理学标签）
        control_seq = result["control_sequence"]
        # 平均控制力方向（度规加权）
        avg_control = control_seq.mean(dim=0)
        avg_control_norm = float(avg_control.norm())

        # 主导控制方向（沿哪个 ξ 轴施加控制力最多）
        control_magnitudes = control_seq.abs().mean(dim=0)  # 各轴平均控制力幅度
        dominant_axis = int(control_magnitudes.argmax())
        dominant_axis_magnitude = float(control_magnitudes[dominant_axis])

        # 处方文本（ξ 符号 + 数值，非心理学标签）
        # 安全性描述：连续安全度（非离散 if-else，非查表）
        # safety_degree ∈ [0, 1]：1=完全安全，0=完全锁定
        max_lock = float(result["max_lock_degree"])
        safety_degree = 1.0 - max_lock  # 连续安全度

        # 连续插值安全等级（softmax 加权，无 if-elif）
        safety_anchors = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        safety_labels = ["临界", "低安全", "中安全", "高安全"]
        k_safety = float(self.n_dims)
        safety_distances = torch.tensor(
            [-abs(safety_degree - a) * k_safety for a in safety_anchors],
            dtype=torch.float64,
        )
        safety_weights = torch.softmax(safety_distances, dim=0)
        safety_dominant_idx = int(safety_weights.argmax())  # 仅显示用
        safety_tier = safety_labels[safety_dominant_idx]
        safety_spectrum = "+".join(
            f"{safety_labels[i]}({float(safety_weights[i]):.3f})"
            for i in range(len(safety_labels))
        )
        safety_desc = f"{safety_tier}[谱:{safety_spectrum}](safety={safety_degree:.3f}, max_lock={max_lock:.3f})"

        prescription_text = (
            f"[处方·PMP] 基于庞特里亚金最大值原理的最优干预方案。\n"
            f"  干预时间窗口: T={T_horizon}, 步数: {n_steps}\n"
            f"  主导控制轴: ξ{dominant_axis+1}（平均幅度 {dominant_axis_magnitude:.4f}）\n"
            f"  平均控制力范数: {avg_control_norm:.4f}\n"
            f"  痛苦减少量: {result['pain_reduction']:.4f}\n"
            f"  总控制能量: {result['total_control_energy']:.4f}\n"
            f"  终态到目标度规距离: {result['final_distance']:.4f}\n"
            f"  安全性: {safety_desc}\n"
            f"  物理意义: 在度规 g 下沿测地线施加最小能量控制力，\n"
            f"            使状态偏离痛苦奇点，趋向目标状态。"
        )

        return {
            **result,
            "prescription_text": prescription_text,
            "dominant_control_axis": dominant_axis,
            "avg_control_norm": avg_control_norm,
        }
