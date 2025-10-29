"""
R1_solver18.py - 修改版

基于POMDP和动态贝叶斯网络的数学规划模型（支持随机实验）

🔧 修改版本特性：
- 移除所有硬编码的成本参数
- 移除所有硬编码的初始信念状态
- 移除所有硬编码的CPT矩阵
- 改为动态生成（与GA1.py保持一致）
- 调用 R1_network_generate4.py
- 支持任意节点数、层数、状态数

Current Date and Time (UTC): 2025-10-28 13:12:42
Current User's Login: dyy21zyy
"""

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

        print("🔧 供应链优化模型初始化 (支持随机实验)")

    def initialize_components(self):
        """初始化各个组件并生成基础数据"""
        print("\n🔧 初始化组件...")

        try:
            # 🔧 修改点1：导入修改后的网络生成器
            from R1_network_generate4 import generate_supply_chain_network

            # 🔧 修改点2：根据参数类型选择调用方式
            if 'total_nodes' in self.network_params and 'num_layers' in self.network_params:
                # 方式1：使用总节点数和层数
                network_results = generate_supply_chain_network(
                    total_nodes=self.network_params['total_nodes'],
                    num_layers=self.network_params['num_layers'],
                    num_periods=self.prediction_params['num_periods'],
                    num_states=self.prediction_params['num_states'],
                    connection_density=self.network_params['connection_density'],
                    seed=self.network_params['seed'],
                    network_type=self.network_params.get('network_type', 'random'),
                    verbose=False
                )
            elif 'nodes_per_layer' in self.network_params:
                # 方式2：手动指定每层节点数
                network_results = generate_supply_chain_network(
                    nodes_per_layer=self.network_params['nodes_per_layer'],
                    num_periods=self.prediction_params['num_periods'],
                    num_states=self.prediction_params['num_states'],
                    connection_density=self.network_params['connection_density'],
                    seed=self.network_params['seed'],
                    network_type='random',
                    verbose=False
                )
            else:
                # 方式3：传统方式（3层网络）
                network_results = generate_supply_chain_network(
                    num_suppliers=self.network_params['num_suppliers'],
                    num_manufacturers=self.network_params['num_manufacturers'],
                    num_periods=self.prediction_params['num_periods'],
                    num_states=self.prediction_params['num_states'],
                    connection_density=self.network_params['connection_density'],
                    seed=self.network_params['seed'],
                    network_type=self.network_params.get('network_type', 'random'),
                    verbose=False
                )

            (self.network, self.layer_info, self.temporal_network,
             self.temporal_node_info, self.parent_dict, self.independent_nodes,
             self.other_nodes, self.parent_node_dic, self.C_dic, self.G_dic) = network_results

            print("    ✓ 网络生成成功")

        except ImportError as e:
            print(f"❌ 网络生成模块不可用: {e}")
            raise

        # 设置基本参数
        if hasattr(self, 'layer_info'):
            self.num_nodes = self.layer_info['num_nodes']
        else:
            self.num_nodes = self.network_params.get('num_suppliers', 2) + self.network_params.get('num_manufacturers',
                                                                                                   2) + 1

        self.num_periods = self.prediction_params.get('num_periods', 3)
        self.num_states = self.prediction_params.get('num_states', 2)
        self.num_actions = self.pomdp_params.get('action_space_size', 3)
        self.num_obs = self.num_states
        self.gamma = self.pomdp_params.get('discount_factor', 0.9)

        # 初始化其他组件
        self.initialize_other_components()
        self.create_sets()
        self.initialize_parameters()

        print("✓ 组件初始化完成")

    def initialize_other_components(self):
        """ 初始化POMDP和预测组件，传递observed_data"""
        try:
            # 🔧 修改点3：导入修改后的POMDP生成器
            from R1_para_POMDP4 import POMDPParametersGenerator

            self.pomdp_generator = POMDPParametersGenerator(
                network_data=(self.network, self.layer_info, self.temporal_network,
                              self.temporal_node_info, self.parent_dict, self.independent_nodes,
                              self.other_nodes, self.parent_node_dic, self.C_dic, self.G_dic),
                num_states=self.num_states,
                num_actions=self.num_actions,
                seed=self.network_params.get('seed', 42)
            )

            self.pomdp_data, _ = self.pomdp_generator.generate_complete_pomdp_parameters(export_excel=False)
            print("    ✓ POMDP参数生成成功")

        except Exception as e:
            print(f"    ⚠️  POMDP参数生成失败: {e}")
            self.pomdp_data = {}

        try:
            # 🔧 修改点4：导入修改后的预测模块
            from R1_prediction_inputDBN13 import ImprovedBalancedBayesianPredictor

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
            print("    ✓ 预测模型初始化成功")

        except Exception as e:
            print(f"    ⚠️  预测模型初始化失败: {e}")
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

    def _get_node_type(self, node):
        """
        获取节点类型（兼容多层网络）

        🔧 修改点：支持任意层数
        """
        for layer_idx in range(1, self.layer_info.get('num_layers', 3) + 1):
            layer_key = f'layer{layer_idx}'
            if layer_key in self.layer_info:
                start, end, name = self.layer_info[layer_key]
                if start <= node < end:
                    return name
        return "Unknown"

    def initialize_parameters(self):
        """
        🔧 核心修改：动态初始化模型参数（移除所有硬编码）

        与 GA1.py 保持完全一致的参数生成逻辑
        """
        print("🔧 动态初始化模型参数（与GA保持一致）...")

        # 设置随机种子保证可重复性
        np.random.seed(self.network_params.get('seed', 42))

        # ============================================
        # 1. 动态生成成本参数（替代硬编码）
        # ============================================
        self.cost = {}

        # 基础动作成本（加入随机性）
        base_action_costs = {
            0: 0,  # 无动作
            1: np.random.uniform(50, 100),  # mild intervention 基础成本
            2: np.random.uniform(150, 250)  # intense intervention 基础成本
        }

        print(f"   基础动作成本: action_1={base_action_costs[1]:.1f}, "
              f"action_2={base_action_costs[2]:.1f}")

        # 根据节点类型动态生成成本乘子
        for k in self.sets['K']:
            node_type = self._get_node_type(k)

            # 不同节点类型的成本乘子范围
            if node_type == "Suppliers":
                multiplier = np.random.uniform(0.8, 1.2)
            elif node_type in ["Manufacturers", "Intermediate_1", "Intermediate_2"]:
                multiplier = np.random.uniform(1.0, 1.5)
            else:  # Retailer or other
                multiplier = np.random.uniform(1.2, 1.8)

            # 为每个时期和动作分配成本
            for t in self.sets['T'][:-1]:
                for a in self.sets['A_kt'][(k, t)]:
                    self.cost[(k, t, a)] = base_action_costs[a] * multiplier

        print(f"   ✓ 成本参数动态生成完成（{len(self.cost)} 个参数）")

        # ============================================
        # 2. 从预测数据提取观测状态
        # ============================================
        self._extract_observations_from_prediction()

        # ============================================
        # 3. 动态生成初始动作 a_hat_0
        # ============================================
        self.a_hat_0 = {}
        last_node = max(self.sets['K'])

        for k in self.sets['K']:
            if k == last_node:
                self.a_hat_0[k] = 0  # 最后节点固定为无动作
            else:
                # 其他节点随机初始动作（倾向于无动作）
                self.a_hat_0[k] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])

        print(f"   ✓ 初始动作 a_hat_0 动态生成完成")

        # ============================================
        # 4. 动态生成初始信念状态 u_hat_0
        # ============================================
        self.u_hat_0 = {}

        disruption_level = self.prediction_params.get('disruption_level', 'moderate')

        for k in self.sets['K']:
            if self.num_states == 2:
                # 2状态：根据disruption级别调整概率分布
                if disruption_level == 'light':
                    # 轻微disruption：倾向良好状态
                    probs = np.random.dirichlet([2, 4])  # [0.3-0.4, 0.6-0.7]
                elif disruption_level == 'moderate':
                    # 中等disruption：均衡分布
                    probs = np.random.dirichlet([3, 3])  # [0.4-0.6, 0.4-0.6]
                else:  # severe
                    # 严重disruption：倾向差状态
                    probs = np.random.dirichlet([5, 2])  # [0.6-0.8, 0.2-0.4]

                self.u_hat_0[(k, 0)] = probs[0]
                self.u_hat_0[(k, 1)] = probs[1]

            elif self.num_states == 3:
                # 3状态：根据disruption级别调整
                if disruption_level == 'light':
                    probs = np.random.dirichlet([2, 3, 4])
                elif disruption_level == 'moderate':
                    probs = np.random.dirichlet([3, 3, 2])
                else:  # severe
                    probs = np.random.dirichlet([5, 3, 1])

                for r in range(3):
                    self.u_hat_0[(k, r)] = probs[r]

            else:
                # 更多状态：均匀分布加随机扰动
                probs = np.random.dirichlet([2] * self.num_states)
                for r in range(self.num_states):
                    self.u_hat_0[(k, r)] = probs[r]

        print(f"   ✓ 初始信念状态 u_hat_0 动态生成完成")

        # ============================================
        # 5. 动态生成初始CPT矩阵 g_hat_0
        # ============================================
        self.g_hat_0 = {}

        for k in self.sets['K']:
            if (k, 0) not in self.sets['delta_kt']:
                continue

            # 获取父节点数量
            parent_nodes = self.parent_node_dic.get(k, [])
            num_parents = len(parent_nodes)

            # 父节点状态组合数
            num_combinations = len(self.sets['delta_kt'][(k, 0)])

            if num_combinations == 0:
                continue

            # 为每个父节点状态组合生成CPT
            for j in self.sets['delta_kt'][(k, 0)]:
                # 使用Dirichlet分布生成概率分布
                # concentration参数控制分布的集中程度
                concentration = np.random.uniform(1.5, 3.0, self.num_states)
                probs = np.random.dirichlet(concentration)

                # 确保概率和为1
                probs = probs / probs.sum()

                # 存储CPT
                for r in range(self.num_states):
                    self.g_hat_0[(k, j, r)] = float(probs[r])

        print(f"   ✓ 初始CPT g_hat_0 动态生成完成（{len(self.g_hat_0)} 个参数）")

        # ============================================
        # 6. 提取POMDP概率
        # ============================================
        self._extract_pomdp_probabilities()

        print("   ✓ 模型参数初始化完成")

    def _extract_observations_from_prediction(self):
        """从预测数据中提取观测状态"""
        print("   从预测数据中提取观测状态...")

        # 初始化所有观测数据为默认值
        self.o_hat = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.o_hat[(k, t)] = 0

        # 从预测数据中提取观测状态
        if hasattr(self, 'prediction_data') and isinstance(self.prediction_data, dict):
            extraction_successful = 0
            total_extractions = 0

            for k in self.sets['K']:
                for t in self.sets['T']:
                    total_extractions += 1
                    period_key = f'period_{t}'

                    if period_key in self.prediction_data:
                        period_data = self.prediction_data[period_key]

                        # 关键：直接使用预测器确定的observed_state
                        if 'observed_state' in period_data and k in period_data['observed_state']:
                            observed_state = int(period_data['observed_state'][k])
                            self.o_hat[(k, t)] = observed_state
                            extraction_successful += 1

            print(f"      观测状态提取统计: {extraction_successful}/{total_extractions} 成功")

        else:
            print("      未找到预测数据，使用默认观测状态")

        print("   观测数据提取完成")

    def _extract_pomdp_probabilities(self):
        """提取POMDP概率"""
        print("   从POMDP数据中提取概率...")

        # 初始化概率字典
        self.P_transition = {}  # P_transition[(k, t, r_next, r_curr, a)]
        self.P_observation = {}  # P_observation[(k, t, o, r, a_prev)]

        if not hasattr(self, 'pomdp_data') or not self.pomdp_data:
            print("      POMDP数据未找到，使用默认概率")
            return

        # 获取POMDP数据
        transition_probs = self.pomdp_data.get('transition_probabilities', {})
        observation_probs = self.pomdp_data.get('observation_probabilities', {})

        if not transition_probs or not observation_probs:
            print("      POMDP概率矩阵为空，使用默认概率")
            return

        # 初始化转移概率 P(r^{t+1} | r^t, a^t)
        for k in range(self.num_nodes):
            if k not in transition_probs:
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
        for k in range(self.num_nodes):
            if k not in observation_probs:
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

        print(f"      POMDP概率矩阵初始化完成")
        print(f"      转移概率条目数: {len(self.P_transition)}")
        print(f"      观测概率条目数: {len(self.P_observation)}")

    def model_building(self):
        """构建Gurobi模型 - 严格按照数学公式"""
        print('-----------------------------------------------------')
        print('🔧 Model building (Random Experiment Version)')
        self.model = Model("SupplyChainResilience_Random")

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
        print("   添加约束 \\ref{cons1}: 动作选择约束...")
        for k in self.sets['K']:
            for t in self.sets['T'][1:-1]:  # t ∈ T\{|T|}
                o_hat = self.o_hat[(k, t)]
                expr = quicksum(self.x[(k, t, a, o_hat)] for a in self.sets['A_kt'][(k, t)])
                self.model.addConstr(expr == 1, name=f"cons1_{k}_{t}")
                constraint_count += 1

        # 约束 (1.2): 最后节点固定选择动作0 - x_{|K|^t0\hat{o}^t} = 1
        print("   添加约束 \\ref{cons1.2}: 最后节点固定约束...")
        last_node = max(self.sets['K'])  # |K|
        for t in self.sets['T'][1:-1]:
            o_hat = self.o_hat[(last_node, t)]
            self.model.addConstr(
                self.x[(last_node, t, 0, o_hat)] == 1,
                name=f"cons1_2_{t}"
            )
            constraint_count += 1

        # 约束 (1.3): 预算约束 - ∑∑∑ x_{k^ta^t\hat{o}^t} · c_{k^ta^t} ≤ B
        print("   添加约束 \\ref{cons1.3}: 预算约束...")
        budget_expr = quicksum(
            self.x[(k, t, a, self.o_hat[(k, t)])] * self.cost[(k, t, a)]
            for k in self.sets['K']
            for t in self.sets['T'][1:-1]
            for a in self.sets['A_kt'][(k, t)]
        )
        self.model.addConstr(budget_expr <= self.budget, name="cons1_3_budget")
        constraint_count += 1

        # ===========================================
        # 初始条件设置 (t=0) - 使用动态生成的值
        # ===========================================
        print("    设置初始条件 (t=0) - 使用动态生成的参数...")

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
                            if initial_g_prob is not None:
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
                            p_obs = self.P_observation.get((k, 1, o_hat_1, r, a_hat_0), 1e-8)
                            numerator_sum = sum(
                                self.P_transition.get((k, 0, r, r0, a_hat_0), 1e-8) * self.g_hat_0.get((k, j, r0), 1e-8)
                                for r0 in self.sets['R_kt'][(k, 0)]
                            )
                            numerator = p_obs * numerator_sum

                            # 分母: Σ P(o^1|r̃^1,a^0) * Σ P(r̃^1|r^0,a^0) * ĝ_{k^0}^j(r^0)
                            denominator = 0.0
                            for r_tilde in self.sets['R_kt'][(k, 1)]:
                                p_obs_tilde = self.P_observation.get((k, 1, o_hat_1, r_tilde, a_hat_0), 1e-8)
                                denominator_sum = sum(
                                    self.P_transition.get((k, 0, r_tilde, r0, a_hat_0), 1e-8) * self.g_hat_0.get(
                                        (k, j, r0), 1e-8)
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

                            p_obs = self.P_observation.get((k, t, o_hat_t, r, a), 1e-8)
                            inner_sum = quicksum(
                                self.x[(k, t - 1, a, o_hat_prev)] *
                                self.P_transition.get((k, t - 1, r, r_prev, a), 1e-8) *
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
                                p_obs_tilde = self.P_observation.get((k, t, o_hat_t, r_tilde, a), 1e-8)
                                inner_sum_den = quicksum(
                                    self.x[(k, t - 1, a, o_hat_prev)] *
                                    self.P_transition.get((k, t - 1, r_tilde, r_prev, a), 1e-8) *
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

        # min \sum_{t=1}^{|\mathcal{T}|} \gamma^{t} u_{|\mathcal{K}|^t}(0)
        last_node = max(self.sets['K'])  # |\mathcal{K}|
        worst_state = 0  # 最差状态是状态0

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
                print('❌ 模型不可行')
                # print('计算不可行子系统 (IIS)...')
                # self.model.computeIIS()
                # self.model.write("infeasible_model.ilp")
                # print('IIS已保存到 infeasible_model.ilp')

            elif self.model.status == GRB.Status.UNBOUNDED:
                print('❌ 模型无界')

            elif self.model.status == GRB.Status.INF_OR_UNBD:
                print('❌ 模型不可行或无界')

            elif self.model.status == GRB.Status.TIME_LIMIT:
                print('⏱️  达到时间限制')
                if self.model.SolCount > 0:
                    print('提取当前最优解...')
                    self.extract_solution()

            else:
                print(f'求解状态: {self.model.status}')

        except Exception as e:
            print(f'❌ 求解过程中出现错误: {str(e)}')
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
                node_type = self._get_node_type(k)
                print(
                    f'  节点 {k} ({node_type}), 时期 {t}, 动作 {a}({action_name.get(a, "未知")}), 观测 {o_hat}: {var.x:.0f}')

        print('*' * 50)

        # 提取信念状态解（只显示部分）
        print('信念状态 u_{k^t}(r^t) (前5个节点):')
        u_solution = {}
        for key, var in self.u.items():
            if hasattr(var, 'x'):
                k, t, r = key
                u_solution[key] = var.x
                if var.x > 0.001 and k < 5:  # 只显示前5个节点的非零概率
                    node_type = self._get_node_type(k)
                    print(f'  节点 {k} ({node_type}), 时期 {t}, 状态 {r}: {var.x:.4f}')

        print('*' * 50)

        # 分析解的质量
        self.analyze_solution_quality(x_solution, u_solution)

        return x_solution, u_solution

    def analyze_solution_quality(self, x_solution, u_solution):
        """分析解的质量"""
        print('解质量分析:')

        # 最后节点的风险状态概率
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
            disruption = self.prediction_params.get('disruption_level', 'unknown')
            filename = f"Gurobi_B{self.budget}_{self.num_nodes}nodes_{disruption}_{timestamp}.xlsx"

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 决策变量表
                if hasattr(self, 'x') and self.x:
                    x_data = []
                    for (k, t, a, o_hat), var in self.x.items():
                        if hasattr(var, 'x'):
                            action_name = {0: "无动作", 1: "mild intervention", 2: "intense intervention"}
                            node_type = self._get_node_type(k)
                            x_data.append({
                                '节点': k,
                                '节点类型': node_type,
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
                            node_type = self._get_node_type(k)
                            u_data.append({
                                '节点': k,
                                '节点类型': node_type,
                                '时期': t,
                                '状态': r,
                                '概率': var.x
                            })

                    df_u = pd.DataFrame(u_data)
                    df_u.to_excel(writer, sheet_name='信念状态', index=False)

                # 参数汇总表
                params_data = {
                    '参数名': ['节点数', '层数', '时期数', '状态数', '动作数', '预算', '折现因子', '求解时间(秒)',
                               '目标函数值', 'Disruption级别', 'MIP Gap', '网络类型'],
                    '值': [
                        self.num_nodes,
                        self.layer_info.get('num_layers', 'N/A'),
                        self.num_periods,
                        self.num_states,
                        self.num_actions,
                        self.budget,
                        self.gamma,
                        round(self.time_used, 2),
                        self.model.objVal if hasattr(self.model, 'objVal') else 'N/A',
                        self.prediction_params.get('disruption_level', 'N/A'),
                        self.model.MIPGap if hasattr(self.model, 'MIPGap') else 'N/A',
                        self.layer_info.get('network_type', 'UNKNOWN')
                    ]
                }
                df_params = pd.DataFrame(params_data)
                df_params.to_excel(writer, sheet_name='参数汇总', index=False)

            print(f'✅ 结果已导出到Excel: {filename}')
            return filename

        except Exception as e:
            print(f'❌ Excel导出失败: {str(e)}')
            return None

    def run_optimization(self, time_limit=3600, save_results=True, export_excel=True):
        """运行完整优化流程"""
        print("🚀 开始供应链韧性优化 (Random Experiment Version)...")
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
                if export_excel:
                    self.export_to_excel()

            print("\n✅ 优化流程完成！")
            print("=" * 60)

            return self.model.Status if self.model else None

        except Exception as e:
            print(f"❌ 优化过程中出现错误: {str(e)}")
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
            if hasattr(self.model, 'MIPGap'):
                print(f"  MIP Gap: {self.model.MIPGap:.4f}")


def get_observed_data(disruption_level):
    """
    根据disruption级别返回观测数据 - 体现供应链层级抗风险能力差异

    🔧 修改点：这是示例数据，实际使用时应该从实验配置动态生成
    """

    if disruption_level.lower() == 'light':
        return {
            1: {  # Period 1 - 只有suppliers受到显著影响
                # Suppliers (最脆弱) - 履约率0.42-0.52，容易进入状态0
                0: {'D_obs': 90, 'SD_obs': 38},  # φ = 0.422
                1: {'D_obs': 88, 'SD_obs': 42},  # φ = 0.477
                2: {'D_obs': 92, 'SD_obs': 48},  # φ = 0.522
                # Manufacturers (中等抗风险) - 履约率0.73-0.81，主要保持状态1
                3: {'D_obs': 45, 'SD_obs': 33},  # φ = 0.733
                4: {'D_obs': 47, 'SD_obs': 38},  # φ = 0.809
                # Retailer (最强抗风险) - 履约率0.87，稳定在状态1
                5: {'D_obs': 45, 'SD_obs': 39}  # φ = 0.867
            }
        }

    elif disruption_level.lower() == 'moderate':
        return {
            1: {  # Period 1 - suppliers和manufacturers都受到影响
                # Suppliers (最脆弱) - 履约率0.24-0.32，强烈倾向状态0
                0: {'D_obs': 100, 'SD_obs': 24},  # φ = 0.240
                1: {'D_obs': 98, 'SD_obs': 27},  # φ = 0.276
                2: {'D_obs': 102, 'SD_obs': 33},  # φ = 0.324
                # Manufacturers (中等抗风险) - 履约率0.38-0.46，边缘状态
                3: {'D_obs': 55, 'SD_obs': 21},  # φ = 0.382
                4: {'D_obs': 52, 'SD_obs': 24},  # φ = 0.462
                # Retailer (最强抗风险) - 履约率0.62，勉强维持状态1
                5: {'D_obs': 50, 'SD_obs': 31}  # φ = 0.620
            }
        }

    elif disruption_level.lower() == 'severe':
        return {
            1: {  # Period 1 - 所有层级都受到严重影响
                # Suppliers (最脆弱) - 履约率0.14-0.21，完全进入状态0
                0: {'D_obs': 110, 'SD_obs': 15},  # φ = 0.136
                1: {'D_obs': 105, 'SD_obs': 19},  # φ = 0.181
                2: {'D_obs': 108, 'SD_obs': 23},  # φ = 0.213
                # Manufacturers (中等抗风险) - 履约率0.11-0.35，严重受损
                3: {'D_obs': 60, 'SD_obs': 6},  # φ = 0.100
                4: {'D_obs': 58, 'SD_obs': 18},  # φ = 0.310
                # Retailer (最强抗风险) - 履约率0.44，勉强高于0.5的边界
                5: {'D_obs': 55, 'SD_obs': 22}  # φ = 0.400
            }
        }

    else:
        return None


def main():
    """
    主函数 - 演示如何使用修改后的求解器

    Current Date and Time (UTC): 2025-10-28 13:18:25
    Current User's Login: dyy21zyy
    """
    print("=" * 80)
    print("🔧 供应链韧性优化 - Gurobi求解器 (Random Experiment Version)")
    print("   Compatible with R1_network_generate4.py")
    print("=" * 80)

    # 选择disruption级别
    print("\n请选择Disruption级别:")
    print("  1 - Light")
    print("  2 - Moderate")
    print("  3 - Severe")

    while True:
        choice = input("请输入选项 (1/2/3): ").strip()
        if choice == '1':
            disruption_level = 'light'
            break
        elif choice == '2':
            disruption_level = 'moderate'
            break
        elif choice == '3':
            disruption_level = 'severe'
            break
        else:
            print("❌ 无效输入！")

    print(f"\n✅ 选择: {disruption_level.upper()}")

    # 参数配置（示例：使用传统3层网络）
    network_params = {
        'num_suppliers': 3,
        'num_manufacturers': 2,
        'connection_density': 0.7,
        'seed': 21,
        'network_type': 'random'  # 🔧 使用随机网络
    }

    pomdp_params = {
        'discount_factor': 0.9,
        'action_space_size': 3
    }

    prediction_params = {
        'num_periods': 4,
        'num_states': 2,
        'mcmc_samples': 1000,
        'mc_samples': 1000,
        'disruption_level': disruption_level,
        'observed_data': get_observed_data(disruption_level)
    }

    # 创建优化模型
    optimizer = SupplyChainOptimizationModel(
        network_params=network_params,
        pomdp_params=pomdp_params,
        prediction_params=prediction_params
    )

    # 设置预算
    optimizer.budget = 100

    # 运行优化
    status = optimizer.run_optimization(
        time_limit=3600,  # 60分钟
        save_results=True,
        export_excel=True
    )

    # 打印模型摘要
    if optimizer.model:
        optimizer.print_model_summary()

    print("\n✅ Gurobi求解完成！")

    return optimizer


def test_random_network():
    """
    测试函数 - 使用多层随机网络

    Current Date and Time (UTC): 2025-10-28 13:18:25
    Current User's Login: dyy21zyy
    """
    print("=" * 80)
    print("🧪 测试 Gurobi 求解器 - 多层随机网络")
    print("=" * 80)

    # 配置参数（使用多层随机网络）
    network_params = {
        'total_nodes': 10,  # 🔧 10个节点
        'num_layers': 3,  # 🔧 3层网络
        'connection_density': 0.7,
        'seed': 42,
        'network_type': 'random'
    }

    pomdp_params = {
        'discount_factor': 0.9,
        'action_space_size': 3
    }

    # 生成随机观测数据（用于测试）
    print("\n🔧 生成随机观测数据...")
    np.random.seed(42)

    observed_data = {1: {}}
    for node in range(10):
        D_obs = np.random.uniform(50, 150)
        phi = np.random.uniform(0.3, 0.6)  # Moderate disruption
        SD_obs = D_obs * phi

        observed_data[1][node] = {
            'D_obs': D_obs,
            'SD_obs': SD_obs
        }

    print(f"   ✅ 生成了 {len(observed_data[1])} 个节点的观测数据")

    prediction_params = {
        'num_periods': 4,
        'num_states': 2,
        'mcmc_samples': 500,  # 减少采样以加快测试
        'mc_samples': 500,
        'disruption_level': 'moderate',
        'observed_data': observed_data
    }

    # 创建优化模型
    optimizer = SupplyChainOptimizationModel(
        network_params=network_params,
        pomdp_params=pomdp_params,
        prediction_params=prediction_params
    )

    # 设置预算
    optimizer.budget = 200  # 更多节点需要更高预算

    # 运行优化
    print("\n🚀 开始优化...")
    status = optimizer.run_optimization(
        time_limit=600,  # 10分钟测试
        save_results=True,
        export_excel=True
    )

    if status == GRB.Status.OPTIMAL:
        print("\n✅ 测试成功！找到最优解")
    elif status == GRB.Status.TIME_LIMIT:
        print("\n⏱️  测试达到时间限制，但找到了可行解")
    else:
        print(f"\n⚠️  测试完成，状态: {status}")

    return optimizer


if __name__ == "__main__":
    print("🔧 R1_solver18.py - Random Experiment Version")
    print("Current Date and Time (UTC): 2025-10-28 13:18:25")
    print("Current User's Login: dyy21zyy")
    print()

    # 选择运行模式
    print("请选择运行模式:")
    print("  1 - 标准模式（3层固定网络）")
    print("  2 - 测试模式（多层随机网络）")

    mode = input("请输入选项 (1/2): ").strip()

    if mode == '2':
        optimizer = test_random_network()
    else:
        optimizer = main()