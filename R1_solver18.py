import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd
import pickle
import os
import time


class SupplyChainOptimizationModel:

    def __init__(self, network_params, pomdp_params, prediction_params):
        self.network_params = network_params
        self.pomdp_params = pomdp_params
        self.prediction_params = prediction_params

        # 基础参数
        self.num_nodes = None  # |\mathcal{K}|
        self.num_periods = None  # |\mathcal{T}|
        self.num_states = None  # |\mathcal{R}_{k^t}|
        self.num_actions = None  # |\mathcal{A}_{k^t}|
        self.num_obs = None  # |\mathcal{O}_{k^t}|

        # Gurobi模型
        self.model = None

        # 网络数据
        self.network_data = None
        self.parent_node_dic = {}  # \Theta_{k^t}
        self.G_dic = {}
        self.C_dic = {}
        self.independent_nodes = []
        self.other_nodes = []

        # 模型参数

        self.budget = 100  # B
        self.cost = {}  # c_{k^ta^t}
        self.gamma = 0.9  # \gamma

        # POMDP概率参数
        self.P_transition = {}  # P_{k^t}(r^{t+1}|r^t, a^t)
        self.P_observation = {}  # P_{k^t}(o^t|r^t, a^{t-1})

        # 观测和动作参数
        self.o_hat = {}  # \hat{o}_{k^t}
        self.a_hat_0 = {}  # \hat{a}_{k^0}
        self.u_hat_0 = {}  # \hat{u}_{k^0}(r^0)
        self.g_hat_0 = {}  # \hat{g}_{k^0}^{j}(r^0)

        # 决策变量
        self.x = {}  # x_{k^ta^t\hat{o}^t}
        self.u = {}  # u_{k^t}(r^t)
        self.G = {}  # G_{k^t}^{j}(r^t)

        # 集合参数
        self.sets = {}

        # 时间记录
        self.start_time = time.time()
        self.time_used = 0

    def initialize_components(self):
        """初始化各个组件并生成基础数据"""
        print("🔧 初始化组件...")

        try:
            # 导入并调用网络生成函数
            from R1_network_generate3 import generate_supply_chain_network

            network_results = generate_supply_chain_network(
                num_suppliers=self.network_params['num_suppliers'],
                num_manufacturers=self.network_params['num_manufacturers'],
                num_periods=self.prediction_params['num_periods'],
                num_states=self.prediction_params['num_states'],
                connection_density=self.network_params['connection_density'],
                seed=self.network_params['seed'],
                verbose=False
            )

            (self.network, self.layer_info, self.temporal_network,
             self.temporal_node_info, self.parent_dict, self.independent_nodes,
             self.other_nodes, self.parent_node_dic, self.C_dic, self.G_dic) = network_results

            print("    ✓ 网络生成成功")

        except ImportError as e:
            print(f"网络生成模块不可用: {e}")

        # 设置基本参数
        if hasattr(self, 'layer_info'):
            self.num_nodes = self.layer_info['num_nodes']
        else:
            self.num_nodes = self.network_params.get('num_suppliers', 2) + self.network_params.get('num_manufacturers',
                                                                                                   2) + 1

        self.num_periods = self.prediction_params.get('num_periods', 3)
        self.num_states = self.prediction_params.get('num_states', 2)
        self.num_actions = self.pomdp_params.get('action_space_size', 3)
        self.num_obs = self.num_states  # 观测数等于状态数
        self.gamma = self.pomdp_params.get('discount_factor', 0.9)

        # 初始化其他组件
        self.initialize_other_components()
        self.create_sets()
        self.initialize_parameters()

        print("组件初始化完成")

    def initialize_other_components(self):
        """ 初始化POMDP和预测组件，传递observed_data"""
        try:
            from R1_para_POMDP3 import POMDPParametersGenerator

            self.pomdp_generator = POMDPParametersGenerator(
                network_data=(self.network, self.layer_info, self.temporal_network,
                              self.temporal_node_info, self.parent_dict, self.independent_nodes,
                              self.other_nodes, self.parent_node_dic, self.C_dic, self.G_dic),
                num_states=self.num_states,
                num_actions=self.num_actions,
                seed=self.network_params.get('seed', 42)
            )

            self.pomdp_data, _ = self.pomdp_generator.generate_complete_pomdp_parameters(export_excel=False)
            print("POMDP参数生成成功")

        except Exception as e:
            print(f"POMDP参数生成失败: {e}")
            self.pomdp_data = {}

        try:
            from R1_prediction_inputDBN12 import ImprovedBalancedBayesianPredictor

            self.predictor = ImprovedBalancedBayesianPredictor(
                network_data=(
                    self.network, self.layer_info, self.temporal_network,
                    self.temporal_node_info, self.parent_dict, self.independent_nodes,
                    self.other_nodes, self.parent_node_dic, self.C_dic, self.G_dic
                ),
                num_states=self.num_states,
                num_periods=self.num_periods,
                disruption_level=self.prediction_params.get('disruption_level'),
                observed_data=self.prediction_params.get('observed_data', None),
                mcmc_samples=self.prediction_params.get('mcmc_samples', 1000),
                mc_samples=self.prediction_params.get('mc_samples', 1000),
                seed=self.network_params.get('seed', 42)
            )

            self.prediction_data = self.predictor.run()
            print("预测模型初始化成功")

        except Exception as e:
            print(f"预测模型初始化失败: {e}")
            self.prediction_data = {}



    def create_sets(self):
        print("📊 创建集合参数...")

        # \mathcal{K}: 供应链合作伙伴集合
        self.sets['K'] = list(range(self.num_nodes))

        # \mathcal{T}: 时间周期集合
        self.sets['T'] = list(range(self.num_periods))

        # \mathcal{R}_{k^t}: 状态空间
        self.sets['R_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.sets['R_kt'][(k, t)] = list(range(self.num_states))

        # \mathcal{A}_{k^t}: 动作空间 (修改为[0,1,2]与POMDP一致)
        self.sets['A_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T'][:-1]:  # t \in \mathcal{T}\backslash\{|\mathcal{T}|\}
                self.sets['A_kt'][(k, t)] = list(range(self.num_actions))  # [0, 1, 2]

        # \mathcal{O}_{k^t}: 观测空间
        self.sets['O_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.sets['O_kt'][(k, t)] = list(range(self.num_obs))

        # \Theta_{k^t}: 父节点集合
        self.sets['Theta_kt'] = {}
        for t in self.sets['T']:  # t ≥ 0
            for k in self.sets['K']:
                parents = []

                # 1. 时间父节点：节点自身在前一时期 (k, t-1)
                if t == 0:
                    parents.append((k, -1))  # 虚拟时间-1
                else:
                    parents.append((k, t - 1))  # 实际前一时期

                # 2. 空间父节点：网络中的父节点在当前时期 (parent_k, t)
                if hasattr(self, 'parent_node_dic') and k in self.parent_node_dic:
                    for parent_k in self.parent_node_dic[k]:
                        parents.append((parent_k, t))  # 当前时期t

                self.sets['Theta_kt'][(k, t)] = parents

        # \delta_{k^t}: 父节点状态组合索引集合
        self.sets['delta_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                if k in self.G_dic:
                    self.sets['delta_kt'][(k, t)] = list(range(len(self.G_dic[k])))
                else:
                    self.sets['delta_kt'][(k, t)] = [0]  # 默认单个组合

        print(f"  - 节点数 |\mathcal{{K}}|: {len(self.sets['K'])}")
        print(f"  - 时间周期数 |\mathcal{{T}}|: {len(self.sets['T'])}")
        print(f"  - 状态数: {self.num_states}")
        print(f"  - 动作数: {self.num_actions}")

    def initialize_parameters(self):
        """初始化模型参数 - 严格按照数学符号"""
        print("初始化模型参数...")

        self.cost = {}
        resource_cost_multipliers = {
            1: 1.0,  # 基础资源
            2: 1.3,  # 中等重要性资源
            3: 1.6,  # 高重要性资源
            4: 2.0,  # 关键资源
            5: 1.2,  # 特殊资源类型
        }

        '''case_cost_set'''
        base_action_costs = {
            0: 0,  # 无动作
            1: 80,  # mild
            2: 200  # intense
        }

        # 为每个资源-时间-动作组合分配成本
        for k in self.sets['K']:
            multiplier = resource_cost_multipliers.get(k, 1.0)
            for t in self.sets['T'][:-1]:  # t \in \mathcal{T}\backslash\{|\mathcal{T}|\}
                for a in self.sets['A_kt'][(k, t)]:
                    # 成本 = 基础动作成本 × 资源类型乘子
                    self.cost[(k, t, a)] = base_action_costs[a] * multiplier

        '''为example设置cost'''
        for t in self.sets['T'][:-1]:  # t \in \mathcal{T}\backslash\{|\mathcal{T}|\}
            self.cost[(0, t, 1)] = 12
            self.cost[(0, t, 2)] = 24
            self.cost[(1, t, 1)] = 5
            self.cost[(1, t, 2)] = 10
            self.cost[(2, t, 1)] = 9
            self.cost[(2, t, 2)] = 18
            self.cost[(3, t, 1)] = 39
            self.cost[(3, t, 2)] = 78
            self.cost[(4, t, 1)] = 36
            self.cost[(4, t, 2)] = 72


        #从预测数据中提取观测状态
        self._extract_observations_from_prediction()

        # \hat{a}_{k^0}: t=0时的已知动作
        self.a_hat_0 = {}
        last_node = max(self.sets['K'])
        for k in self.sets['K']:
            if k == last_node:
                self.a_hat_0[k] = 0  # 最后节点：无动作
            else:
                self.a_hat_0[k] = np.random.choice([0, 1, 2])

        '''为example设置初始信念状态'''
        self.u_hat_0 = {}
        self.u_hat_0[(0, 0)] = 0.3
        self.u_hat_0[(0, 1)] = 0.7
        self.u_hat_0[(1, 0)] = 0.4
        self.u_hat_0[(1, 1)] = 0.6
        self.u_hat_0[(2, 0)] = 0.2
        self.u_hat_0[(2, 1)] = 0.8
        self.u_hat_0[(3, 0)] = 0.4
        self.u_hat_0[(3, 1)] = 0.6
        self.u_hat_0[(4, 0)] = 0.5
        self.u_hat_0[(4, 1)] = 0.5
        self.u_hat_0[(5, 0)] = 0.3
        self.u_hat_0[(5, 1)] = 0.7

        '''为example设置初始信念cpt'''
        self.g_hat_0 = {}

        # CPT矩阵定义 - 按照图片中的格式
        cpt_matrices = {
            # 节点1,2: CPT₁₂ = CPT₂₂ = CPT₃₂ (2x2矩阵)
            0: np.array([[0.7, 0.3],
                         [0.3, 0.7]]),
            1: np.array([[0.7, 0.3],
                         [0.3, 0.7]]),
            2: np.array([[0.7, 0.3],
                         [0.3, 0.7]]),

            # 节点3: CPT₄₂ (2x8矩阵)
            3: np.array([[0.6, 0.5, 0.5, 0.3, 0.5, 0.3, 0.2, 0.1],
                         [0.4, 0.5, 0.5, 0.7, 0.5, 0.7, 0.8, 0.9]]),

            # 节点4: CPT₅₂ (2x4矩阵)
            4: np.array([[0.8, 0.6, 0.6, 0.2],
                         [0.2, 0.4, 0.4, 0.8]]),

            # 节点5: CPT₆₂ (2x8矩阵)
            5: np.array([[0.9, 0.7, 0.6, 0.3, 0.6, 0.3, 0.5, 0.2],
                         [0.1, 0.3, 0.4, 0.7, 0.4, 0.7, 0.5, 0.8]])
        }

        for k in self.sets['K']:
            if (k, 0) in self.sets['delta_kt']:
                for j in self.sets['delta_kt'][(k, 0)]:

                    #使用预定义的CPT矩阵
                    if k in cpt_matrices:
                        cpt_matrix = cpt_matrices[k]

                        # 从CPT矩阵的第j列获取概率分布
                        probs = cpt_matrix[:, j]

                        # 设置概率值
                        for r in range(len(probs)):
                            if r < self.num_states:
                                self.g_hat_0[(k, j, r)] = float(probs[r])
                                print(f"g_hat_0[({k}, {j}, {r})] = {float(probs[r]):.3f}")

                    else:
                        # 对于节点0（独立节点），使用均匀分布
                        uniform_prob = 1.0 / self.num_states
                        for r in self.sets['R_kt'][(k, 0)]:
                            self.g_hat_0[(k, j, r)] = uniform_prob
                            print(f"g_hat_0[({k}, {j}, {r})] = {uniform_prob:.3f}")

        print(" 初始信念条件概率设置完成")


        # P_{k^t}(o^t|r^t, a^{t-1}): 观测概率
        print("从POMDP数据中提取观测概率...")
        # 初始化概率字典
        self.P_transition = {}  # P_transition[(k, t, r_next, r_curr, a)]
        self.P_observation = {}  # P_observation[(k, t, o, r, a_prev)]

        if not hasattr(self, 'pomdp_data') or not self.pomdp_data:
            print("POMDP数据未找到，使用默认概率")
            return

        # 获取POMDP数据
        transition_probs = self.pomdp_data.get('transition_probabilities', {})
        observation_probs = self.pomdp_data.get('observation_probabilities', {})

        if not transition_probs or not observation_probs:
            print("POMDP概率矩阵为空，使用默认概率")
            return

        # 初始化转移概率 P(r^{t+1} | r^t, a^t)
        print("初始化转移概率矩阵...")
        for k in range(self.num_nodes):
            if k not in transition_probs:
                print(f"节点 {k} 的转移概率未找到")
                continue

            # transition_probs[k] 的维度是 [current_state, action, next_state]
            trans_matrix = transition_probs[k]

            for t in range(self.num_periods):  # POMDP概率是时间不变的
                for r_curr in range(self.num_states):
                    for a in range(self.num_actions):
                        for r_next in range(self.num_states):
                            # 从矩阵中获取概率值
                            prob = trans_matrix[r_curr, a, r_next]
                            self.P_transition[(k, t, r_next, r_curr, a)] = float(prob)

        # 初始化观测概率 P(o^t | r^t, a^{t-1})
        print("初始化观测概率矩阵...")
        for k in range(self.num_nodes):
            if k not in observation_probs:
                print(f"节点 {k} 的观测概率未找到")
                continue

            # observation_probs[k] 的维度是 [current_state, previous_action, observation]
            obs_matrix = observation_probs[k]

            for t in range(self.num_periods):  # POMDP概率是时间不变的
                for r in range(self.num_states):
                    for a_prev in range(self.num_actions):
                        for o in range(self.num_obs):
                            # 从矩阵中获取概率值
                            prob = obs_matrix[r, a_prev, o]
                            self.P_observation[(k, t, o, r, a_prev)] = float(prob)

        print(f"POMDP概率矩阵初始化完成")
        print(f"转移概率条目数: {len(self.P_transition)}")
        print(f"观测概率条目数: {len(self.P_observation)}")

    def _extract_observations_from_prediction(self):
        print("从预测数据中提取观测状态...")

        # 初始化所有观测数据为默认值
        self.o_hat = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.o_hat[(k, t)] = 0

        #从预测数据中提取观测状态
        if hasattr(self, 'prediction_data') and isinstance(self.prediction_data, dict):
            print(f"预测数据包含的周期: {list(self.prediction_data.keys())}")

            extraction_successful = 0
            total_extractions = 0

            for k in self.sets['K']:
                for t in self.sets['T']:
                    total_extractions += 1
                    period_key = f'period_{t}'

                    if period_key in self.prediction_data:
                        period_data = self.prediction_data[period_key]

                        #关键：直接使用预测器确定的observed_state
                        if 'observed_state' in period_data and k in period_data['observed_state']:
                            observed_state = int(period_data['observed_state'][k])
                            self.o_hat[(k, t)] = observed_state
                            extraction_successful += 1
                            print(f"周期 {t} 节点 {k}: 观测状态 {observed_state}")
                        else:
                            print(f"周期 {t} 节点 {k}: 未找到observed_state，使用默认值0")
                    else:
                        print(f"周期 {t}: 未找到数据，节点 {k} 使用默认值0")

            print(f"观测状态提取统计: {extraction_successful}/{total_extractions} 成功")

            # 显示最终的观测状态分布
            print(" 最终观测状态分布:")
            for t in self.sets['T']:
                states = [self.o_hat[(k, t)] for k in self.sets['K']]
                print(f"周期 {t}: {states}")

        else:
            print("未找到预测数据，使用默认观测状态")

        print("观测数据提取完成")

    def model_building(self):
        """构建Gurobi模型 - 严格按照数学公式"""
        print('-----------------------------------------------------')
        print('model building')
        self.model = Model("SupplyChainResilience")

        # 创建决策变量
        self.create_decision_variables()

        # 添加约束
        self.add_constraints()

        # 设置目标函数
        self.set_objective()

        # 模型参数设置
        self.model.setParam('OutputFlag', 1)
        self.model.setParam('NonConvex', 2)
        self.model.setParam('TimeLimit', 3600)
        self.model.setParam('MIPGap', 0.001)

        self.model.update()
        print("模型构建完成")

    def create_decision_variables(self):
        """创建决策变量 - 严格按照数学符号"""
        print("创建决策变量...")

        # x_{k^ta^t\hat{o}^t}: 决策变量 (公式\ref{cons6})
        self.x = {}
        for k in self.sets['K']:
            for t in self.sets['T'][1:-1]:  # t \in \mathcal{T}\backslash\{|\mathcal{T}|\}
                o_hat = self.o_hat[(k, t)]
                for a in self.sets['A_kt'][(k, t)]:
                    self.x[(k, t, a, o_hat)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{k}^{t}_{a}_{o_hat}"
                    )

        # u_{k^t}(r^t): 信念状态概率 (公式\ref{cons7})
        self.u = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                for r in self.sets['R_kt'][(k, t)]:
                    self.u[(k, t, r)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0, ub=1.0,
                        name=f"u_{k}^{t}_{r}"
                    )

        # G_{k^t}^{j}(r^t): 条件信念概率 (公式\ref{cons7})
        self.G = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                if self.sets['Theta_kt'][(k, t)]:  # \Theta_{k^t} \neq \emptyset
                    for j in self.sets['delta_kt'][(k, t)]:
                        for r in self.sets['R_kt'][(k, t)]:
                            self.G[(k, t, j, r)] = self.model.addVar(
                                vtype=GRB.CONTINUOUS,
                                lb=0.0, ub=1.0,
                                name=f"G_{k}^{t}_{j}_{r}"
                            )

        print(f"  - 决策变量 x: {len(self.x)} 个")
        print(f"  - 信念状态变量 u: {len(self.u)} 个")
        print(f"  - 条件信念变量 G: {len(self.G)} 个")

    def add_constraints(self):
        print("🔧 添加约束条件...")

        constraint_count = 0

        # 约束 (1): 动作选择约束 - ∑_{a^t ∈ A_{k^t}} x_{k^ta^t\hat{o}^t} = 1
        print(" 添加约束 \\ref{cons1}: 动作选择约束...")
        for k in self.sets['K']:
            for t in self.sets['T'][1:-1]:  # t ∈ T\{|T|}
                o_hat = self.o_hat[(k, t)]
                expr = quicksum(self.x[(k, t, a, o_hat)] for a in self.sets['A_kt'][(k, t)])
                self.model.addConstr(expr == 1, name=f"cons1_{k}_{t}")
                constraint_count += 1

        # 约束 (1.2): 最后节点固定选择动作1 - x_{|K|^t1\hat{o}^t} = 1
        print("添加约束 \\ref{cons1.2}: 最后节点固定约束...")
        last_node = max(self.sets['K'])  # |K|
        for t in self.sets['T'][1:-1]:
            o_hat = self.o_hat[(last_node, t)]
            self.model.addConstr(
                self.x[(last_node, t, 0, o_hat)] == 1,
                name=f"cons1_2_{t}"
            )
            constraint_count += 1

        # 约束 (1.3): 预算约束 - ∑∑∑ x_{k^ta^t\hat{o}^t} · c_{k^ta^t} ≤ B
        print(" 添加约束 \\ref{cons1.3}: 预算约束...")
        budget_expr = quicksum(
            self.x[(k, t, a, self.o_hat[(k, t)])] * self.cost[(k, t, a)]
            for k in self.sets['K']
            for t in self.sets['T'][1:-1]
            for a in self.sets['A_kt'][(k, t)]
        )
        self.model.addConstr(budget_expr <= self.budget, name="cons1_3_budget")
        constraint_count += 1

        # ===========================================
        # 初始条件设置 (t=0) - 这些是已知值，不是约束
        # ===========================================
        print("    设置初始条件 (t=0)...")

        # 设置初始信念状态 u_{k^0}(r^0) = \hat{u}_{k^0}(r^0)
        for k in self.sets['K']:
            for r in self.sets['R_kt'][(k, 0)]:
                initial_prob = self.u_hat_0.get((k, r))
                self.model.addConstr(
                    self.u[(k, 0, r)] == initial_prob,
                    name=f"initial_u_{k}_{r}"
                )
                constraint_count += 1

        # 设置初始信念条件概率 G_{k^0}^{j}(r^0) = \hat{g}_{k^0}^{j}(r^0)
        for k in self.sets['K']:
            if self.sets['Theta_kt'][(k, 0)]:  # 如果t=0时有父节点定义
                for j in self.sets['delta_kt'][(k, 0)]:
                    for r in self.sets['R_kt'][(k, 0)]:
                        if (k, 0, j, r) in self.G:
                            initial_g_prob = self.g_hat_0.get((k, j, r))
                            self.model.addConstr(
                                self.G[(k, 0, j, r)] == initial_g_prob,
                                name=f"initial_G_{k}_{j}_{r}"
                            )
                            constraint_count += 1

        # ===========================================
        # 约束 (3): t=1时的条件信念概率更新
        # ===========================================
        print("    添加约束 \\ref{cons3}: t=1条件信念概率约束...")
        if 1 in self.sets['T']:
            for k in self.sets['K']:
                if self.sets['Theta_kt'][(k, 1)]:  # Θ_{k^1} ≠ ∅
                    for j in self.sets['delta_kt'][(k, 1)]:
                        for r in self.sets['R_kt'][(k, 1)]:
                            if (k, 1, j, r) not in self.G:
                                continue

                            # 实现公式 (3)
                            o_hat_1 = self.o_hat[(k, 1)]
                            a_hat_0 = self.a_hat_0[k]

                            # 分子: P(o^1|r^1,a^0) * Σ P(r^1|r^0,a^0) * ĝ_{k^0}^j(r^0)
                            p_obs = self.P_observation[(k, 1, o_hat_1, r, a_hat_0)]
                            numerator_sum = sum(
                                self.P_transition[(k, 0, r, r0, a_hat_0)] * self.g_hat_0[(k, j, r0)]
                                for r0 in self.sets['R_kt'][(k, 0)]
                            )
                            numerator = p_obs * numerator_sum

                            # 分母: Σ P(o^1|r̃^1,a^0) * Σ P(r̃^1|r^0,a^0) * ĝ_{k^0}^j(r^0)
                            denominator = 0.0
                            for r_tilde in self.sets['R_kt'][(k, 1)]:
                                p_obs_tilde = self.P_observation[(k, 1, o_hat_1, r_tilde, a_hat_0)]
                                denominator_sum = sum(
                                    self.P_transition[(k, 0, r_tilde, r0, a_hat_0)] * self.g_hat_0[(k, j, r0)]
                                    for r0 in self.sets['R_kt'][(k, 0)]
                                )
                                denominator += p_obs_tilde * denominator_sum

                            # 避免除零
                            if denominator < 1e-8:
                                denominator = 1e-8

                            # 添加约束: G_{k^1}^j(r^1) = 计算值
                            calculated_prob = numerator / denominator
                            self.model.addConstr(
                                self.G[(k, 1, j, r)] == calculated_prob,
                                name=f"cons3_{k}_{j}_{r}"
                            )
                            constraint_count += 1

        # ===========================================
        # 约束 (5): t≥2时的条件信念概率更新
        # ===========================================
        print("    添加约束 \\ref{cons5}: t≥2条件信念概率约束...")
        for t in range(2, len(self.sets['T'])):  # t ≥ 2
            if t not in self.sets['T']:
                continue

            for k in self.sets['K']:
                if not self.sets['Theta_kt'][(k, t)]:  # 跳过没有父节点的情况
                    continue

                for j in self.sets['delta_kt'][(k, t)]:
                    for r in self.sets['R_kt'][(k, t)]:
                        if (k, t, j, r) not in self.G:
                            continue


                        o_hat_t = self.o_hat[(k, t)]
                        o_hat_prev = self.o_hat[(k, t - 1)]

                        # 创建辅助变量来处理分子和分母
                        numerator_aux_vars = []
                        denominator_aux_vars = []

                        for a in self.sets['A_kt'][(k, t - 1)]:
                            # 分子辅助变量
                            aux_num = self.model.addVar(
                                vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                                name=f"aux_num_{k}_{t}_{j}_{r}_{a}"
                            )

                            p_obs = self.P_observation[(k, t, o_hat_t, r, a)]
                            inner_sum = quicksum(
                                self.x[(k, t - 1, a, o_hat_prev)] *
                                self.P_transition[(k, t - 1, r, r_prev, a)] *
                                self.G[(k, t - 1, j, r_prev)]
                                for r_prev in self.sets['R_kt'][(k, t - 1)]
                            )
                            self.model.addConstr(
                                aux_num == p_obs * inner_sum,
                                name=f"aux_num_def_{k}_{t}_{j}_{r}_{a}"
                            )
                            numerator_aux_vars.append(aux_num)

                            # 分母辅助变量
                            aux_den = self.model.addVar(
                                vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                                name=f"aux_den_{k}_{t}_{j}_{r}_{a}"
                            )

                            action_denominator_terms = []
                            for r_tilde in self.sets['R_kt'][(k, t)]:
                                p_obs_tilde = self.P_observation[(k, t, o_hat_t, r_tilde, a)]
                                inner_sum_den = quicksum(
                                    self.x[(k, t - 1, a, o_hat_prev)] *
                                    self.P_transition[(k, t - 1, r_tilde, r_prev, a)] *
                                    self.G[(k, t - 1, j, r_prev)]
                                    for r_prev in self.sets['R_kt'][(k, t - 1)]
                                )
                                action_denominator_terms.append(p_obs_tilde * inner_sum_den)

                            self.model.addConstr(
                                aux_den == quicksum(action_denominator_terms),
                                name=f"aux_den_def_{k}_{t}_{j}_{r}_{a}"
                            )
                            denominator_aux_vars.append(aux_den)

                            constraint_count += 2

                        # 主约束：G * 总分母 = 总分子
                        total_numerator = quicksum(numerator_aux_vars)
                        total_denominator = quicksum(denominator_aux_vars)

                        self.model.addConstr(
                            self.G[(k, t, j, r)] * total_denominator == total_numerator,
                            name=f"cons5_{k}_{t}_{j}_{r}"
                        )
                        constraint_count += 1

        # 约束 (prob2): t≥1时的DBN信念状态递推
        # ===========================================
        print("    添加约束 \\ref{prob2}: DBN信念状态递推...")
        for t in range(1, len(self.sets['T'])):  # t≥1
            for k in self.sets['K']:
                if not self.sets['Theta_kt'][(k, t)]:  # 跳过没有父节点的情况
                    continue

                for r in self.sets['R_kt'][(k, t)]:
                    if (k, t, r) not in self.u:
                        continue

                    # 实现公式(prob2): u_{k^t}(r^t) = Σ_j G_{k^t}^j(r^t) * Π_{θ∈Θ_{k^t}} u_θ(C^{-1}_θ(j))
                    belief_expr = QuadExpr()

                    for j in self.sets['delta_kt'][(k, t)]:
                        if (k, t, j, r) not in self.G:
                            continue

                        # 创建乘积辅助变量
                        product_aux_var = self.model.addVar(
                            lb=0, ub=1, vtype=GRB.CONTINUOUS,
                            name=f"product_aux_{k}_{t}_{j}_{r}"
                        )

                        parent_set = self.sets['Theta_kt'][(k, t)]

                        if len(parent_set) == 1:
                            # 单个父节点情况
                            parent_k, parent_t = parent_set[0]
                            parent_state = self._get_parent_state(k, j, 0)

                            if (parent_k, parent_t, parent_state) in self.u:
                                self.model.addConstr(
                                    product_aux_var == self.u[(parent_k, parent_t, parent_state)],
                                    name=f"single_parent_{k}_{t}_{j}_{r}"
                                )
                                constraint_count += 1
                        else:
                            # 多父节点情况 - 递推构建乘积
                            temp_aux_vars = []

                            for parent_idx, (parent_k_i, parent_t_i) in enumerate(parent_set):
                                parent_state_i = self._get_parent_state(k, j, parent_idx)

                                if (parent_k_i, parent_t_i, parent_state_i) not in self.u:
                                    continue

                                if parent_idx == 0:
                                    temp_aux_vars.append(self.u[(parent_k_i, parent_t_i, parent_state_i)])
                                else:
                                    intermediate_product = self.model.addVar(
                                        lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                        name=f"intermediate_{k}_{t}_{j}_{parent_idx}"
                                    )

                                    if parent_idx == 1:
                                        self.model.addQConstr(
                                            intermediate_product ==
                                            temp_aux_vars[0] * self.u[(parent_k_i, parent_t_i, parent_state_i)],
                                            name=f"product_step_{k}_{t}_{j}_{parent_idx}"
                                        )
                                    else:
                                        self.model.addQConstr(
                                            intermediate_product ==
                                            temp_aux_vars[-1] * self.u[(parent_k_i, parent_t_i, parent_state_i)],
                                            name=f"product_step_{k}_{t}_{j}_{parent_idx}"
                                        )

                                    temp_aux_vars.append(intermediate_product)
                                    constraint_count += 1

                            if temp_aux_vars:
                                self.model.addConstr(
                                    product_aux_var == temp_aux_vars[-1],
                                    name=f"final_product_{k}_{t}_{j}_{r}"
                                )
                                constraint_count += 1

                        # G * 乘积项
                        term_aux_var = self.model.addVar(
                            lb=0, ub=1, vtype=GRB.CONTINUOUS,
                            name=f"term_aux_{k}_{t}_{j}_{r}"
                        )

                        self.model.addQConstr(
                            term_aux_var == self.G[(k, t, j, r)] * product_aux_var,
                            name=f"term_product_{k}_{t}_{j}_{r}"
                        )
                        constraint_count += 1

                        belief_expr.addTerms(1, term_aux_var)

                    self.model.addQConstr(
                        self.u[(k, t, r)] == belief_expr,
                        name=f'dbn_belief_update_{k}_{t}_{r}'
                    )
                    constraint_count += 1

        print(f"约束条件添加完成，总计: {constraint_count} 个约束")

    def _get_parent_state(self, k, j, parent_idx):
        """从组合索引j中提取第parent_idx个父节点的状态"""
        if k in self.G_dic and j < len(self.G_dic[k]):
            combination = self.G_dic[k][j]
            if parent_idx < len(combination):
                if parent_idx < len(combination) - 1:
                    return combination[parent_idx]  # 空间父节点状态
                else:
                    return combination[-1]  # 时间父节点状态（前一时期状态）
        return 0  # 默认状态

    def set_objective(self):
        """设置目标函数 - 公式 \\ref{OBJ}"""
        print("设置目标函数...")

        # min \sum_{t=1}^{|\mathcal{T}|} \gamma^{t} u_{|\mathcal{K}|^t}(|\mathcal{R}_{|\mathcal{K}|^t}|)
        last_node = max(self.sets['K'])  # |\mathcal{K}|
        worst_state = 0  # 最差状态是状态0，不是最大编号状态

        obj_expr = LinExpr(0)
        for t in range(1, len(self.sets['T'])):  # t=1 to |\mathcal{T}|
            if (last_node, t, worst_state) in self.u:
                coeff = self.gamma ** t
                obj_expr.addTerms(coeff, self.u[(last_node, t, worst_state)])

        self.model.setObjective(obj_expr, sense=GRB.MINIMIZE)
        print("目标函数设置完成")

    def solve_model(self):
        """求解模型并处理结果"""
        print('-----------------------------------------------------')
        print('开始求解模型...')

        try:
            self.model.optimize()

            # 处理求解结果
            if self.model.status == GRB.Status.OPTIMAL:
                print('✅ 模型求解成功 - 找到最优解')
                self.extract_solution()

            elif self.model.status == GRB.Status.INFEASIBLE:
                print('模型不可行')
                print('计算不可行子系统 (IIS)...')
                self.model.computeIIS()
                self.model.write("infeasible_model.ilp")
                print('IIS已保存到 infeasible_model.ilp')

            elif self.model.status == GRB.Status.UNBOUNDED:
                print('模型无界')

            elif self.model.status == GRB.Status.INF_OR_UNBD:
                print('模型不可行或无界')

            elif self.model.status == GRB.Status.TIME_LIMIT:
                print('达到时间限制')
                if self.model.SolCount > 0:
                    print('提取当前最优解...')
                    self.extract_solution()

            else:
                print(f'求解状态: {self.model.status}')

        except Exception as e:
            print(f'求解过程中出现错误: {str(e)}')
            import traceback
            traceback.print_exc()

        self.time_used = time.time() - self.start_time
        print(f'总用时: {self.time_used:.2f} 秒')

    def extract_solution(self):
        """提取并分析解"""
        print('提取解信息...')

        if hasattr(self.model, 'objVal'):
            print(f'目标函数值: {self.model.objVal:.6f}')
            print('*' * 50)

        # 提取决策变量解
        print('📋 决策变量 x_{k^ta^t\\hat{o}^t}:')
        x_solution = {}
        for key, var in self.x.items():
            if hasattr(var, 'x') and var.x > 0.5:  # 二进制变量阈值
                k, t, a, o_hat = key
                x_solution[key] = var.x
                action_name = {0: "无动作", 1: "mild intervention", 2: "intense intervention"}
                print(f'  节点 {k}, 时期 {t}, 动作 {a}({action_name.get(a, "未知")}), 观测 {o_hat}: {var.x:.0f}')

        print('*' * 50)

        # 提取信念状态解
        print('信念状态 u_{k^t}(r^t):')
        u_solution = {}
        for key, var in self.u.items():
            if hasattr(var, 'x'):
                k, t, r = key
                u_solution[key] = var.x
                if var.x > 0.001:  # 只显示非零概率
                    print(f'  节点 {k}, 时期 {t}, 状态 {r}: {var.x:.4f}')

        print('*' * 50)

        # 提取条件信念概率解
        if self.G:
            print('条件信念概率 G_{k^t}^{j}(r^t):')
            g_solution = {}
            for key, var in self.G.items():
                if hasattr(var, 'x'):
                    k, t, j, r = key
                    g_solution[key] = var.x
                    if var.x > 0.001:  # 只显示非零概率
                        print(f'  节点 {k}, 时期 {t}, 组合 {j}, 状态 {r}: {var.x:.4f}')

            print('*' * 50)

        # 分析解的质量
        self.analyze_solution_quality(x_solution, u_solution)

        return x_solution, u_solution

    def analyze_solution_quality(self, x_solution, u_solution):
        """分析解的质量"""
        print('解质量分析:')

        #  最后节点的风险状态概率
        last_node = max(self.sets['K'])
        print(f'最后节点 ({last_node}) 各时期风险状态概率:')

        for t in self.sets['T']:
            worst_state = 0
            prob = u_solution.get((last_node, t, worst_state), 0)
            print(f'    时期 {t}: {prob:.4f}')

        print('*' * 50)

    def export_to_excel(self, filename=None):
        """导出结果到Excel文件"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"B{self.budget}-supply_chain_results_{self.prediction_params['disruption_level']}.xlsx"

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 决策变量表
                if hasattr(self, 'x') and self.x:
                    x_data = []
                    for (k, t, a, o_hat), var in self.x.items():
                        if hasattr(var, 'x'):
                            action_name = {0: "无动作", 1: "mild intervention", 2: "intense intervention"}
                            x_data.append({
                                '节点': k,
                                '时期': t,
                                '动作': a,
                                '动作名称': action_name.get(a, "未知"),
                                '观测': o_hat,
                                '决策值': var.x,
                                '成本': self.cost.get((k, t, a), 0)
                            })

                    df_x = pd.DataFrame(x_data)
                    df_x.to_excel(writer, sheet_name='决策变量', index=False)

                # 信念状态表
                if hasattr(self, 'u') and self.u:
                    u_data = []
                    for (k, t, r), var in self.u.items():
                        if hasattr(var, 'x'):
                            u_data.append({
                                '节点': k,
                                '时期': t,
                                '状态': r,
                                '概率': var.x
                            })

                    df_u = pd.DataFrame(u_data)
                    df_u.to_excel(writer, sheet_name='信念状态', index=False)

                # 参数汇总表
                params_data = {
                    '参数名': ['节点数', '时期数', '状态数', '动作数', '预算', '折现因子', '求解时间(秒)',
                               '目标函数值'],
                    '值': [
                        self.num_nodes,
                        self.num_periods,
                        self.num_states,
                        self.num_actions,
                        self.budget,
                        self.gamma,
                        round(self.time_used, 2),
                        self.model.objVal if hasattr(self.model, 'objVal') else 'N/A'
                    ]
                }
                df_params = pd.DataFrame(params_data)
                df_params.to_excel(writer, sheet_name='参数汇总', index=False)

            print(f'结果已导出到Excel: {filename}')
            return filename

        except Exception as e:
            print(f'Excel导出失败: {str(e)}')
            return None

    def run_optimization(self, time_limit=3600, save_results=True, export_excel=True):
        """运行完整优化流程"""
        print("开始供应链韧性优化...")
        print("=" * 60)

        try:
            # 初始化组件
            self.initialize_components()

            # 构建模型
            self.model_building()

            # 设置时间限制
            if time_limit:
                self.model.setParam('TimeLimit', time_limit)

            # 求解模型
            self.solve_model()

            # 保存和导出结果
            if save_results and self.model and hasattr(self.model, 'objVal'):
                # self.save_results()

                if export_excel:
                    self.export_to_excel()

            print("\n 优化流程完成！")
            print("=" * 60)

            return self.model.Status if self.model else None

        except Exception as e:
            print(f" 优化过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def print_model_summary(self):
        """打印模型摘要信息"""
        if not self.model:
            print("模型尚未构建")
            return

        print("📋 模型摘要:")
        print(f"  优化方向: 最小化")
        print(f"  变量总数: {self.model.NumVars}")
        print(f"  约束总数: {self.model.NumConstrs}")
        print(f"  二进制变量: {self.model.NumBinVars}")
        print(f"  连续变量: {self.model.NumVars - self.model.NumBinVars}")
        print(f"  非零系数: {self.model.NumNZs}")

        if hasattr(self.model, 'objVal'):
            print(f"  目标函数值: {self.model.objVal:.6f}")
            print(f"  求解时间: {self.time_used:.2f} 秒")


def get_observed_data(disruption_level):
    """根据disruption级别返回观测数据 - 体现供应链层级抗风险能力差异"""

    if disruption_level.lower() == 'light':
        return {
            1: {  # Period 1 - 只有suppliers受到显著影响
                # Suppliers (最脆弱) - 履约率0.45-0.50，容易进入状态0
                0: {'D_obs': 90, 'SD_obs': 38},   # φ = 0.45
                1: {'D_obs': 88, 'SD_obs': 42},   # φ = 0.50
                2: {'D_obs': 92, 'SD_obs': 48},   # φ = 0.50
                # Manufacturers (中等抗风险) - 履约率0.75-0.80，主要保持状态1
                3: {'D_obs': 45, 'SD_obs': 33},  # φ = 0.75
                4: {'D_obs': 47, 'SD_obs': 38},  # φ = 0.80
                # Retailer (最强抗风险) - 履约率0.85，稳定在状态1
                5: {'D_obs': 45, 'SD_obs': 39}    # φ = 0.85
            }
        }

    elif disruption_level.lower() == 'moderate':
        return {
            1: {  # Period 1 - suppliers和manufacturers都受影响
                # Suppliers (严重受影响) - 履约率0.35-0.40，高概率状态0
                0: {'D_obs': 100, 'SD_obs': 24},   # φ = 0.35
                1: {'D_obs': 98, 'SD_obs': 27},   # φ = 0.40
                2: {'D_obs': 102, 'SD_obs': 33},   # φ = 0.40
                # Manufacturers (中度受影响) - 履约率0.45-0.50，可能状态0
                3: {'D_obs': 55, 'SD_obs': 21},  # φ = 0.45
                4: {'D_obs': 52, 'SD_obs': 24},  # φ = 0.50
                # Retailer (轻度受影响) - 履约率0.70，主要状态1
                5: {'D_obs': 50, 'SD_obs': 31}    # φ = 0.70
            }
        }

    elif disruption_level.lower() == 'severe':
        return {
            1: {  # Period 1 - 所有节点都受到严重影响
                # Suppliers (极度受影响) - 履约率0.20-0.25，高概率状态0
                0: {'D_obs': 110, 'SD_obs': 15},   # φ = 0.20
                1: {'D_obs': 105, 'SD_obs': 19},   # φ = 0.25
                2: {'D_obs': 108, 'SD_obs': 23},   # φ = 0.25
                # Manufacturers (严重受影响) - 履约率0.30-0.35，高概率状态0
                3: {'D_obs': 60, 'SD_obs': 6},   # φ = 0.30
                4: {'D_obs': 58, 'SD_obs': 18},   # φ = 0.35
                # Retailer (中度受影响) - 履约率0.45，可能状态0或1
                5: {'D_obs': 55, 'SD_obs': 22}    # φ = 0.45
            }
        }

    else:
        return None  # 正常情况使用自动生成

def main():
    print("基于POMDP和动态贝叶斯网络的数学规划模型")
    print("=" * 60)

    # 手动输入disruption级别
    print("请选择Disruption级别:")
    print("  Light    - 轻微disruption")
    print("  Moderate - 中等disruption")
    print("  Severe   - 严重disruption")


    while True:
        disruption_input = input("\n请输入级别 (Light/Moderate/Severe): ").strip()

        if disruption_input.lower() in ['light', 'moderate', 'severe']:
            disruption_level = disruption_input.lower()
            break
        else:
            print(" 无效输入！请输入 Light, Moderate, Severe")

    print(f"\n✅ 已选择: {disruption_level.upper()}")

    # 基础参数配置
    network_params = {
        'num_suppliers': 3,
        'num_manufacturers': 2,
        'connection_density': 0.7,
        'seed': 21
    }

    pomdp_params = {
        'discount_factor': 0.9,
        'action_space_size': 3
    }

    # 🔧 修正：根据选择的级别设置预测参数，确保observed_data正确传递
    prediction_params = {
        'num_periods': 4,
        'num_states': 2,
        'mcmc_samples': 1000,
        'mc_samples': 1000,
        'disruption_level': disruption_level,                    # ✅ 添加disruption_level
        'observed_data': get_observed_data(disruption_level),    # ✅ 添加observed_data
        'N_monte_carlo': 1000
    }

    # 显示配置信息
    print("\n🔧 参数配置:")
    print(f"  供应商数量: {network_params['num_suppliers']}")
    print(f"  制造商数量: {network_params['num_manufacturers']}")
    print(f"  时间周期: {prediction_params['num_periods']}")
    print(f"  状态数量: {prediction_params['num_states']}")
    print(f"  动作数量: {pomdp_params['action_space_size']}")
    print(f" 折现因子: {pomdp_params['discount_factor']}")
    print(f"  Disruption级别: {disruption_level.upper()}")

    # 显示观测数据状态
    if prediction_params['observed_data']:
        print(f"  观测数据: ✅ 使用预设{disruption_level}级别数据")
        # 显示观测数据详情
        observed_data = prediction_params['observed_data']
        for period, data in observed_data.items():
            print(f"    Period {period}: {len(data)} nodes")
            for node, obs in data.items():
                fulfillment_rate = obs['SD_obs'] / obs['D_obs'] if obs['D_obs'] > 0 else 0
                print(f"      Node {node}: D={obs['D_obs']:.0f}, SD={obs['SD_obs']:.0f}, φ={fulfillment_rate:.3f}")


    print("=" * 60)

    try:
        # 创建优化模型
        print("正在初始化优化模型...")
        optimizer = SupplyChainOptimizationModel(
            network_params=network_params,
            pomdp_params=pomdp_params,
            prediction_params=prediction_params
        )

        # 运行优化
        print("开始优化求解...")
        status = optimizer.run_optimization(
            time_limit=1800,  # 30分钟
            # save_results=True,
            export_excel=True
        )

        # 显示结果
        print("\n优化结果:")
        optimizer.print_model_summary()

        # 🔧 显示观测状态使用情况
        print("\n观测状态验证:")
        print("  从预测器获取的观测状态分布:")
        for t in range(optimizer.num_periods):
            states = [optimizer.o_hat[(k, t)] for k in range(optimizer.num_nodes)]
            print(f"    周期 {t}: {states}")

        # 根据状态显示结论
        if status == GRB.OPTIMAL:
            print(f"\n  {disruption_level.upper()}场景优化成功！找到最优解")
        elif status == GRB.TIME_LIMIT:
            print(f"\n {disruption_level.upper()}场景达到时间限制，返回当前最优解")
        elif status == GRB.INFEASIBLE:
            print(f"\n {disruption_level.upper()}场景模型不可行")
        else:
            print(f"\n {disruption_level.upper()}场景结束，状态: {status}")

        return optimizer

    except Exception as e:
        print(f"\n 运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    # 运行主程序
    optimizer = main()

    if optimizer:
        print("\n 程序执行完成！")
        print(" 结果文件已生成:")
        # print("  - .pkl 文件: 完整求解结果")
        print("  - .xlsx 文件: Excel报告")

    else:
        print("\n 程序执行失败！")

