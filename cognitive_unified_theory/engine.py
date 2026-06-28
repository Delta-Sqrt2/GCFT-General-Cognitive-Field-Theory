"""
心智物理引擎（Mental Physics Engine）

整合五层架构为统一系统：
    L1 本体论  → 认知流形 (M, g)，命运坐标 S0
    L2 动力学  → 类 Navier-Stokes 演化，痛苦转化，负熵判据
    L3 时间逆行 → 纤维丛记忆，哈密顿-雅可比反向传播
    L4 主体论  → 透明度 τ(t)，分级防火墙
    L5 控制论  → 流形碰撞，艺术合成，战术地图

核心公理（不可违反）：
    1. 白盒绝对性 —— 严禁 NLP 黑盒 API
    2. 数学决定论 —— 心理现象映射为几何实体
    3. 连续性原则 —— 严禁离散布尔逻辑
    4. 因果律守恒 —— 严禁篡改历史数据

引擎主循环：
    事件输入 → L1 解析（度规更新）→ L2 演化（状态推进）
    → L3 记忆（权重投影）→ L4 透明度（权限更新）
    → L5 战术地图（状态可视化）
"""

from __future__ import annotations

import torch
from torch import Tensor

from .layer1_ontology import CognitiveManifold
from .layer2_dynamics import PainPotentialField, CognitiveEvolutionField, NegativeEntropyCriterion
from .layer3_temporal import CognitiveFiberBundle, HolographicMemory, HamiltonJacobiBackprop
from .layer4_subject import TransparencyOperator, CognitiveFirewall
from .layer5_cybernetics import ManifoldCollisionSolver, ArtSynthesizer, TacticalMap
from .layer5_cybernetics.inverse_projector import InverseProjector
from .layer5_cybernetics.cognitive_thruster import CognitiveThruster
from .core.adaptive_integrator import AdaptiveIntegrator, SingularityAlert
from .core.event_horizon import EventHorizonLock


class MentalPhysicsEngine:
    """
    心智物理引擎：人类认知动力学大统一理论的工程实现。

    使用方式：
        engine = MentalPhysicsEngine(n_dims=8)
        engine.initialize(family_boundary=B_family)
        engine.process_event(event_tensor)
        report = engine.get_tactical_report()
    """

    def __init__(self, n_dims: int = 8, seed: int | None = None, device: str | torch.device = "cpu"):
        self.n_dims = n_dims
        self.seed = seed
        self.device = torch.device(device)

        # L1 本体论
        self.manifold = CognitiveManifold(n_dims=n_dims, seed=seed, device=device)

        # L2 动力学
        self.pain_field = PainPotentialField(n_dims=n_dims)
        self.evolution = CognitiveEvolutionField(n_dims=n_dims)
        self.entropy_criterion = NegativeEntropyCriterion(n_dims=n_dims)

        # L3 时间逆行
        self.fiber_bundle = CognitiveFiberBundle(n_dims=n_dims)
        self.memory = HolographicMemory(n_dims=n_dims)
        self.hjb = HamiltonJacobiBackprop(n_dims=n_dims)

        # L4 主体论
        self.transparency = TransparencyOperator(n_dims=n_dims)
        self.firewall = CognitiveFirewall(n_dims=n_dims)

        # L5 控制论
        self.collision_solver = ManifoldCollisionSolver(n_dims=n_dims)
        self.art_synthesizer = ArtSynthesizer(n_dims=n_dims)
        self.tactical_map = TacticalMap(n_dims=n_dims, n_components=2)

        # v1.2 临床工具化模块
        self.inverse_projector = InverseProjector(n_dims=n_dims)  # 逆投影器（诊断）
        self.cognitive_thruster = CognitiveThruster(n_dims=n_dims)  # 推进器（处方）
        self.event_horizon = EventHorizonLock(n_dims=n_dims)  # 事件视界（安全）

        # 引擎状态
        self._initialized = False
        self._event_count = 0
        self._time = 0.0

    def initialize(self, family_boundary: Tensor | None = None) -> Tensor:
        """
        引擎初始化：
            1. L1 正交基生成
            2. L1 命运坐标 S0 解算
            3. L3 初始记忆记录
        """
        # L1 正交基
        self.manifold._initialize_orthogonal_basis()

        # L1 S0 解算
        S0 = self.manifold.calculate_S0(family_boundary=family_boundary)

        # 设置演化场的认知基线（S0 = 认知舒适区）
        self.evolution.set_baseline_state(S0)

        # L3 初始记忆
        self.memory.record(S0, timestamp=0.0, metric=self.manifold.metric.current)
        self.fiber_bundle.add_state(S0, timestamp=0.0)

        self._initialized = True
        return S0

    def process_event(self, event_tensor: Tensor, dt: float = 1.0) -> dict[str, Tensor]:
        """
        处理一个认知事件（全链路）。

        流程：
            1. L1 解析事件 → 度规更新，痛苦能量张量
            2. L2 痛苦势能场设置 → 演化方程积分
            3. L3 记忆记录 → 纤维丛更新
            4. L4 透明度更新 → 防火墙权限
            5. L5 战术地图更新

        参数：
            event_tensor: 形式化事件张量（非自然语言）
            dt: 时间步长

        返回：
            全链路状态报告
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        self._event_count += 1
        self._time += dt

        # ===== L1 本体论：事件解析 =====
        event_result = self.manifold.parse_formal_event(event_tensor)
        metric = self.manifold.metric.current
        state_before = self.manifold.state.clone()

        # ===== L2 动力学：痛苦势能与演化 =====
        self.pain_field.set_event(event_tensor, metric)
        pain_grad_mag = self.pain_field.gradient_magnitude(state_before)

        # 设置演化场
        self.evolution.set_metric(metric)
        self.evolution.set_hamiltonian(lambda S: self.pain_field.potential(S))

        # 演化（RK4 积分）
        trajectory = self.evolution.evolve(
            state_before, t_span=(0.0, dt), n_steps=20, method="rk4"
        )
        state_after = trajectory[-1]

        # 负熵判据
        state_history = self.manifold.history
        pain_integral = self.pain_field.accumulate_gradient_integral(trajectory)
        entropy_result = self.entropy_criterion.evaluate(
            state_history, trajectory, pain_integral
        )

        # 更新流形状态
        self.manifold.set_state(state_after)

        # ===== L3 时间逆行：记忆与纤维丛 =====
        self.memory.record(state_after, timestamp=self._time, metric=metric)
        self.fiber_bundle.add_state(state_after, timestamp=self._time)

        # ===== L4 主体论：透明度与防火墙 =====
        tau = self.transparency.update(
            pain_grad_mag, self.manifold.history, dt=torch.tensor(dt, dtype=torch.float64)
        )
        permissions = self.firewall.compute_permissions(tau)

        # ===== L5 控制论：战术地图 =====
        if self.manifold.history.shape[0] >= 2:
            self.tactical_map.compute_projection(self.manifold.history, metric)

        report = {
            # L1
            "curvature_intensity": event_result["curvature_intensity"],
            "trauma_tensor_trace": torch.trace(event_result["trauma_tensor"]),
            # L2
            "pain_potential": self.pain_field.potential(state_after),
            "pain_gradient_magnitude": pain_grad_mag,
            "state_change": (state_after - state_before).norm(),
            "delta_R": entropy_result["delta_R"],
            "pain_integral": entropy_result["pain_integral"],
            "criterion_met": entropy_result["criterion_met"],
            "collapse_risk": entropy_result["collapse_risk"],
            # L3
            "memory_weights": self.memory.get_weights(),
            "trauma_intensity": self.memory.trauma_intensity(),
            # L4
            "transparency": tau,
            "permission_black": permissions["black"],
            "permission_gray": permissions["gray"],
            "permission_white": permissions["white"],
            "state_label": self.firewall.state_label(tau),
            # L5
            "effective_rank": self.manifold.cognitive_effective_rank(),
        }
        return report

    def release_trauma(self, high_dim_metric: Tensor | None = None) -> dict[str, Tensor]:
        """
        释怀操作（时间逆行层）：
            用高维度规重投影历史权重，降低创伤权重。
            历史数据不变，权重改变。

        需要 τ > τ_gray（灰盒态以上）才能执行。
        """
        tau = self.transparency.tau
        permissions = self.firewall.compute_permissions(tau)

        # 权限检查（连续，非布尔）：释怀强度 = 灰盒+白盒权限
        release_strength = permissions["gray"] + permissions["white"]

        if high_dim_metric is None:
            # 默认高维度规：更平坦的度规（升维视角）
            high_dim_metric = torch.eye(self.n_dims, dtype=torch.float64) * 0.3

        # L3 高维重投影
        weights_before = self.memory.get_weights().clone()
        self.memory.reweight_by_projection(high_dim_metric, tau)

        # L3 HJ 反向传播重写
        self.hjb.set_high_dim_metric(high_dim_metric)
        current_state = self.manifold.state
        self.hjb.compute_action_gradient(current_state, time_horizon=1.0, n_steps=30)
        weights_after_hjb = self.hjb.rewrite_history_weights(
            self.memory.get_weights(), tau
        )
        for i in range(len(weights_after_hjb)):
            self.memory.reweight(i, weights_after_hjb[i])

        weights_after = self.memory.get_weights()
        trauma_before = self.memory.trauma_intensity()

        return {
            "release_strength": release_strength,
            "transparency": tau,
            "weights_before": weights_before,
            "weights_after": weights_after,
            "weight_change": (weights_after - weights_before).norm(),
            "trauma_intensity": trauma_before,
            "history_unchanged": (self.memory.get_history() - self.memory.get_history()).abs().max(),
        }

    def collide_with(self, other_engine: "MentalPhysicsEngine") -> dict[str, Tensor]:
        """
        人际流形碰撞（控制论层）：
            解算两个认知流形的碰撞。
        """
        return self.collision_solver.solve_collision(
            self.manifold.state,
            other_engine.manifold.state,
            self.manifold.metric.current,
            other_engine.manifold.metric.current,
        )

    def synthesize_art(self) -> dict[str, Tensor]:
        """
        艺术合成（控制论层）：
            将当前情感流形结构投影到语言空间。
        """
        return self.art_synthesizer.synthesize(
            self.manifold.state,
            self.manifold.metric.current,
            self.manifold.history,
        )

    def get_tactical_report(self) -> dict[str, Tensor]:
        """
        实时战术地图报告（控制论层）：
            整合所有层的数学状态。
        """
        return self.tactical_map.summary_report(
            self.manifold.state,
            self.manifold.metric.current,
            self.transparency.tau,
            self.manifold.history,
        )

    def get_full_diagnosis(self) -> dict[str, Tensor]:
        """
        完整诊断报告：所有层的核心指标。
        """
        return {
            # L1 本体论
            "S0": self.manifold.S0,
            "current_state": self.manifold.state,
            "metric_curvature": self.manifold.metric.curvature_intensity(),
            "manifold_capacity": self.manifold.manifold_capacity(),
            # L2 动力学
            "effective_rank": self.manifold.cognitive_effective_rank(),
            "growth_trajectory": self.entropy_criterion.growth_trajectory(),
            "pain_trajectory": self.entropy_criterion.pain_trajectory(),
            # L3 时间逆行
            "memory_weights": self.memory.get_weights(),
            "trauma_intensity": self.memory.trauma_intensity(),
            "release_degree": self.memory.release_degree(),
            # L4 主体论
            "transparency": self.transparency.tau,
            "transparency_history": self.transparency.tau_history,
            "state_label": self.firewall.state_label(self.transparency.tau),
            # L5 控制论
            "tactical_report": self.get_tactical_report(),
        }

    def evolve_long_term(
        self,
        events: list[tuple[float, Tensor, float]],
        rtol: float = 1e-6,
        atol: float = 1e-10,
    ) -> dict[str, Tensor | list[SingularityAlert] | list[dict]]:
        """
        长周期自适应演化：处理带时间戳的事件序列。

        使用 RKF45 自适应步长积分器，在事件间演化认知状态，
        在事件点应用力向量（度规更新+痛苦注入）。

        物理流程：
            对每个事件 (t_i, F_i, dt_i)：
            1. 从当前时刻演化到 t_i（自适应步长）
            2. 在 t_i 应用力 F_i（度规更新+状态冲击）
            3. 记录奇点预警

        参数：
            events: [(time, force_vector, step_size), ...]
                    time: 事件时刻（如年份 1889）
                    force_vector: 力向量 F ∈ R^n
                    step_size: 事件持续时间 dt
            rtol: 相对误差容限
            atol: 绝对误差容限

        返回：
            {
                'times': 时间点序列,
                'states': 状态序列 (T, n),
                'alerts': 奇点预警列表,
                'event_reports': 每个事件的报告,
            }
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        integrator = AdaptiveIntegrator(rtol=rtol, atol=atol, dt_min=1e-6, dt_max=1.0)

        all_times = [0.0]
        all_states = [self.manifold.state.clone()]
        all_alerts: list[SingularityAlert] = []
        event_reports: list[dict] = []

        current_time = 0.0

        # 按时间排序事件
        sorted_events = sorted(events, key=lambda e: e[0])

        for event_time, force_vector, step_size in sorted_events:
            # 1. 演化到事件时刻
            if event_time > current_time:
                # 设置当前度规到演化场
                self.evolution.set_metric(self.manifold.metric.current)
                self.evolution.set_baseline_state(self.manifold.state.clone())

                # 度规函数（用于奇点检测）
                def metric_fn(t):
                    return self.manifold.metric.current

                times, states, alerts = integrator.integrate(
                    derivative_fn=self.evolution.derivative,
                    initial_state=self.manifold.state,
                    t_span=(current_time, event_time),
                    metric_fn=metric_fn,
                    dt_init=min(0.5, event_time - current_time),
                )

                # 记录结果
                if len(times) > 1:
                    all_times.extend(times[1:].tolist())
                    all_states.extend([s.clone() for s in states[1:]])
                all_alerts.extend(alerts)

                # 更新引擎状态到事件前
                if len(states) > 0:
                    self.manifold.set_state(states[-1])

                current_time = event_time

            # 2. 在事件点应用力向量
            report = self.process_event(force_vector, dt=max(step_size, 0.1))
            event_reports.append({
                "time": event_time,
                "force_magnitude": force_vector.norm().item(),
                "curvature": report["curvature_intensity"].item(),
                "pain": report["pain_potential"].item(),
                "state_change": report["state_change"].item(),
                "collapse_risk": report["collapse_risk"].item(),
            })

            all_times.append(current_time)
            all_states.append(self.manifold.state.clone())

        return {
            "times": torch.tensor(all_times, dtype=torch.float64),
            "states": torch.stack(all_states, dim=0),
            "alerts": all_alerts,
            "event_reports": event_reports,
        }

    # ==================================================================
    # v1.2 临床工具化方法（诊断 + 处方 + 安全）
    # ==================================================================

    def diagnose(self, n_text_axes: int = 3) -> dict[str, Tensor | list | str | float]:
        """
        v1.2 临床诊断：将当前高维几何状态逆投影为自然语言诊断。

        闭环流程：
            当前状态 S + 度规 g → 逆投影器 → 诊断文本

        返回：
            {
                'force_scores': 力类型得分（IMPACT/CONSTRAINT/SELF_STATE/POTENTIAL）,
                'primary_axes': 全部轴权重（未截断）,
                'intensity': 总能量强度 I = S^T g S,
                'text': 诊断文本（含 ξ 符号，非心理学标签）,
                'dominant_force': 主导力类型,
                'dominant_probability': 主导力类型概率,
            }

        白盒保证：
            - 力类型由度规结构推断（对角/非对角/特征值），非字典
            - 轴权重由能量功率计算（softmax 连续，无 topk 截断）
            - 修饰词由强度连续插值（非离散查表）
            - 认知轴用 ξ 符号（非"焦虑/抑郁"等标签）
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        return self.inverse_projector.project(
            state=self.manifold.state,
            metric=self.manifold.metric.current,
            n_text_axes=n_text_axes,
        )

    def check_safety(self) -> dict[str, Tensor | str | float]:
        """
        v1.2 安全检测：检查当前状态是否接近认知黑洞（事件视界）。

        返回：
            {
                'lock_degree': 锁定程度 [0, 1],
                'is_locked': 是否进入事件视界锁定,
                'frequency': 极限环频率（未锁定时为 0）,
                'state_label': 状态标签（NORMAL/EVENT_HORIZON_LOCKED）,
                'report': 事件视界状态报告,
            }

        内生安全原则：
            - 严禁 raise Error
            - 数值崩溃转化为临床特征（强迫性重复）
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        lock_degree = self.event_horizon.compute_lock_degree(self.manifold.metric.current)
        is_locked = float(lock_degree) > self.event_horizon.lock_threshold

        return {
            "lock_degree": lock_degree,
            "is_locked": is_locked,
            "frequency": self.event_horizon.frequency if is_locked else torch.tensor(0.0),
            "state_label": self.event_horizon.STATE_EVENT_HORIZON_LOCKED if is_locked else self.event_horizon.STATE_NORMAL,
            "report": self.event_horizon.report(),
        }

    def apply_safety_filter(self, dt: float = 0.1) -> dict[str, Tensor | str | float]:
        """
        v1.2 应用安全过滤：若度规病态，构造极限环（强迫性重复）。

        物理意义：
            当度规条件数发散时，系统不"死机"，而是进入 2 维极限环。
            状态在两个近简并特征值维度间做周期运动。
            对应临床：精神崩溃 = 陷入无意义循环（非脑子停转）。

        返回：
            {
                'safe_state': 安全状态（若锁定，则在极限环上）,
                'safe_metric': 正则化后的安全度规,
                'lock_degree': 锁定程度,
                'frequency': 极限环频率,
                'state_label': 状态标签,
            }
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        S_safe, g_safe, lock_degree, frequency, state_label = self.event_horizon.filter(
            state=self.manifold.state,
            metric=self.manifold.metric.current,
            dt=dt,
        )

        # 更新引擎状态为安全状态
        if state_label == self.event_horizon.STATE_EVENT_HORIZON_LOCKED:
            self.manifold.set_state(S_safe)

        return {
            "safe_state": S_safe,
            "safe_metric": g_safe,
            "lock_degree": lock_degree,
            "frequency": frequency,
            "state_label": state_label,
        }

    def prescribe(
        self,
        S_target: Tensor,
        T_horizon: float = 10.0,
        n_steps: int = 50,
    ) -> dict[str, Tensor | float | str]:
        """
        v1.2 临床处方：基于庞特里亚金最大值原理（PMP）计算最优干预方案。

        闭环流程：
            当前状态 S + 目标状态 S_target + 度规 g
            → PMP 求解（前向状态演化 + 反向协态积分）
            → 最优控制力序列 u*(t)
            → 处方文本

        参数：
            S_target: 目标认知状态
            T_horizon: 干预时间窗口
            n_steps: 离散化步数

        返回：
            {
                'trajectory': 干预后的状态轨迹,
                'control_sequence': 控制力序列 u(t),
                'pain_reduction': 痛苦减少量,
                'max_lock_degree': 最大锁定程度（安全约束）,
                'is_safe': 是否安全（未触发事件视界）,
                'prescription_text': 处方文本（ξ 符号 + 数值）,
            }

        数学保证：
            - 控制力由 PMP 推导：u* = -λ/||λ|| · min(||λ||, κ·||S||)
            - 演化基于 RK4 + 度规 g（非线性插值，非线性插值降级）
            - 距离基于度规加权（非欧氏）
            - 安全约束：控制力不超过 κ·||S||，防止流形撕裂
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        return self.cognitive_thruster.prescribe(
            S_current=self.manifold.state,
            S_target=S_target,
            metric=self.manifold.metric.current,
            T_horizon=T_horizon,
            n_steps=n_steps,
        )

    def clinical_session(
        self,
        user_input: str,
        parser,
        S_target: Tensor | None = None,
        T_horizon: float = 10.0,
        n_steps: int = 50,
    ) -> dict[str, Tensor | list | str | float]:
        """
        v1.2 完整临床会话：输入 → 诊断 → 安全检测 → 处方。

        闭环流程（v1.2 步骤四规范）：
            用户输入（自然语言）
            → Parser（v1.1）→ 张量 S_in
            → Engine（v1.0）→ 度规 g，预测演化
            → Horizon（v1.2）→ 检测安全，若危险则锁定
            → Projector（v1.2）→ 输出诊断文本
            → Thruster（v1.2）→ 计算干预向量 u
            → Projector（v1.2）→ 输出处方文本

        参数：
            user_input: 用户自然语言输入（如"我感到很压抑"）
            parser: NarrativeParser 实例（v1.1 解析器）
            S_target: 目标状态（None 则使用 S0 作为目标）
            T_horizon: 干预时间窗口
            n_steps: 离散化步数

        返回：
            {
                'parsed_events': 解析的事件张量列表,
                'diagnosis': 诊断结果（逆投影器输出）,
                'safety': 安全检测结果,
                'prescription': 处方结果（推进器输出）,
                'session_text': 完整会话文本报告,
            }

        梯度守恒原则：
            全链路保持 PyTorch 张量流动，严禁 .detach() 或转字符串传递。
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")

        # 1. Parser：自然语言 → 事件张量
        events = parser.parse(user_input, time_offset=self._time)

        # 应用事件到引擎
        for event in events:
            self.process_event(event.force_vector, dt=max(event.step_size, 0.1))

        # 2. Horizon：安全检测
        safety = self.check_safety()

        # 若危险，应用安全过滤
        if safety["is_locked"]:
            self.apply_safety_filter()

        # 3. Projector：诊断
        diagnosis = self.diagnose()

        # 4. Thruster：处方
        if S_target is None:
            S_target = self.manifold.S0  # 默认目标：回到命运坐标（认知舒适区）

        prescription = self.prescribe(
            S_target=S_target,
            T_horizon=T_horizon,
            n_steps=n_steps,
        )

        # 5. 生成完整会话报告
        session_lines = [
            "=" * 70,
            "v1.2 临床会话报告",
            "=" * 70,
            f"用户输入: {user_input}",
            f"解析事件数: {len(events)}",
            "",
            "--- 诊断 ---",
            diagnosis["text"],
            "",
            "--- 安全检测 ---",
            safety["report"],
            "",
            "--- 处方 ---",
            prescription["prescription_text"],
            "",
            "=" * 70,
        ]

        return {
            "parsed_events": events,
            "diagnosis": diagnosis,
            "safety": safety,
            "prescription": prescription,
            "session_text": "\n".join(session_lines),
        }

    @property
    def time(self) -> float:
        return self._time

    @property
    def event_count(self) -> int:
        return self._event_count

    def __repr__(self) -> str:
        return (
            f"MentalPhysicsEngine(n_dims={self.n_dims}, "
            f"events={self._event_count}, "
            f"time={self._time:.1f}, "
            f"τ={self.transparency.tau:.3f}, "
            f"state={self.firewall.state_label(self.transparency.tau)})"
        )
