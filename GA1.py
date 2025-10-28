import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt


class GeneticAlgorithmSolver:
    """
    遗传算法求解供应链韧性优化问题
    替代Gurobi求解器，处理POMDP和DBN约束
    """

    def __init__(self, network_params, pomdp_params, prediction_params,
                 population_size=100, max_generations=500, crossover_rate=0.8,
                 mutation_rate=0.1, elitism_rate=0.1, tournament_size=5):
        """
        初始化遗传算法求解器

        参数:
            population_size: 种群大小
            max_generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elitism_rate: 精英保留比例
            tournament_size: 锦标赛选择大小
        """
        self.network_params = network_params
        self.pomdp_params = pomdp_params
        self.prediction_params = prediction_params

        # GA参数
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size

        # 问题参数
        self.num_nodes = None
        self.num_periods = None
        self.num_states = None
        self.num_actions = None
        self.num_obs = None
        self.budget = 100
        self.gamma = 0.9

        # 网络数据
        self.network_data = None
        self.parent_node_dic = {}
        self.G_dic = {}
        self.C_dic = {}

        # POMDP参数
        self.P_transition = {}
        self.P_observation = {}
        self.cost = {}
        self.o_hat = {}
        self.a_hat_0 = {}
        self.u_hat_0 = {}
        self.g_hat_0 = {}

        # 集合
        self.sets = {}

        # 进化记录
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')

        # 时间记录
        self.start_time = None
        self.time_used = 0

        print("🧬 遗传算法求解器初始化")
        print(f"   种群大小: {population_size}")
        print(f"   最大迭代代数: {max_generations}")
        print(f"   交叉率: {crossover_rate}, 变异率: {mutation_rate}")
        print(f"   精英保留率: {elitism_rate}")

    def initialize_components(self):
        """初始化所有组件"""
        print("\n🔧 初始化组件...")

        try:
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

        except Exception as e:
            print(f"❌ 网络生成失败: {e}")
            raise

        # 设置基本参数
        self.num_nodes = self.layer_info['num_nodes']
        self.num_periods = self.prediction_params.get('num_periods', 4)
        self.num_states = self.prediction_params.get('num_states', 2)
        self.num_actions = self.pomdp_params.get('action_space_size', 3)
        self.num_obs = self.num_states
        self.gamma = self.pomdp_params.get('discount_factor', 0.9)

        # 初始化其他组件
        self._initialize_pomdp_components()
        self._initialize_prediction_components()
        self._create_sets()
        self._initialize_parameters()

        print("✓ 组件初始化完成")

    def _initialize_pomdp_components(self):
        """初始化POMDP组件"""
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
            print("    ✓ POMDP参数生成成功")

        except Exception as e:
            print(f"    ⚠️  POMDP参数生成失败: {e}")
            self.pomdp_data = {}

    def _initialize_prediction_components(self):
        """初始化预测组件"""
        try:
            from R1_prediction_inputDBN12 import ImprovedBalancedBayesianPredictor

            self.predictor = ImprovedBalancedBayesianPredictor(
                network_data=(self.network, self.layer_info, self.temporal_network,
                              self.temporal_node_info, self.parent_dict, self.independent_nodes,
                              self.other_nodes, self.parent_node_dic, self.C_dic, self.G_dic),
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

    def _create_sets(self):
        """创建集合参数"""
        self.sets['K'] = list(range(self.num_nodes))
        self.sets['T'] = list(range(self.num_periods))

        self.sets['R_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.sets['R_kt'][(k, t)] = list(range(self.num_states))

        self.sets['A_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T'][:-1]:
                self.sets['A_kt'][(k, t)] = list(range(self.num_actions))

        self.sets['O_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.sets['O_kt'][(k, t)] = list(range(self.num_obs))

        # Theta_kt: 父节点集合
        self.sets['Theta_kt'] = {}
        for t in self.sets['T']:
            for k in self.sets['K']:
                parents = []
                if t == 0:
                    parents.append((k, -1))
                else:
                    parents.append((k, t - 1))

                if k in self.parent_node_dic:
                    for parent_k in self.parent_node_dic[k]:
                        parents.append((parent_k, t))

                self.sets['Theta_kt'][(k, t)] = parents

        # delta_kt: 父节点状态组合索引
        self.sets['delta_kt'] = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                if k in self.G_dic:
                    self.sets['delta_kt'][(k, t)] = list(range(len(self.G_dic[k])))
                else:
                    self.sets['delta_kt'][(k, t)] = [0]

    def _initialize_parameters(self):
        """初始化模型参数"""
        # 成本参数
        base_action_costs = {0: 0, 1: 80, 2: 200}
        resource_cost_multipliers = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}


        self.cost = {}
        for k in self.sets['K']:
            multiplier = resource_cost_multipliers.get(k, 1.0)
            for t in self.sets['T'][:-1]:
                for a in self.sets['A_kt'][(k, t)]:
                    self.cost[(k, t, a)] = base_action_costs[a] * multiplier

        # # Example成本设置
        # for t in self.sets['T'][:-1]:
        #     self.cost[(0, t, 1)] = 12
        #     self.cost[(0, t, 2)] = 24
        #     self.cost[(1, t, 1)] = 5
        #     self.cost[(1, t, 2)] = 10
        #     self.cost[(2, t, 1)] = 9
        #     self.cost[(2, t, 2)] = 18
        #     self.cost[(3, t, 1)] = 39
        #     self.cost[(3, t, 2)] = 78
        #     self.cost[(4, t, 1)] = 36
        #     self.cost[(4, t, 2)] = 72

        # 提取观测状态
        self._extract_observations_from_prediction()

        # 初始动作
        last_node = max(self.sets['K'])
        self.a_hat_0 = {}
        for k in self.sets['K']:
            if k == last_node:
                self.a_hat_0[k] = 0
            else:
                self.a_hat_0[k] = np.random.choice([0, 1, 2])

        # 初始信念状态
        self.u_hat_0 = {
            (0, 0): 0.3, (0, 1): 0.7,
            (1, 0): 0.4, (1, 1): 0.6,
            (2, 0): 0.2, (2, 1): 0.8,
            (3, 0): 0.4, (3, 1): 0.6,
            (4, 0): 0.5, (4, 1): 0.5,
            (5, 0): 0.3, (5, 1): 0.7
        }

        # 初始CPT
        self.g_hat_0 = {}
        cpt_matrices = {
            0: np.array([[0.7, 0.3], [0.3, 0.7]]),
            1: np.array([[0.7, 0.3], [0.3, 0.7]]),
            2: np.array([[0.7, 0.3], [0.3, 0.7]]),
            3: np.array([[0.6, 0.5, 0.5, 0.3, 0.5, 0.3, 0.2, 0.1],
                         [0.4, 0.5, 0.5, 0.7, 0.5, 0.7, 0.8, 0.9]]),
            4: np.array([[0.8, 0.6, 0.6, 0.2], [0.2, 0.4, 0.4, 0.8]]),
            5: np.array([[0.9, 0.7, 0.6, 0.3, 0.6, 0.3, 0.5, 0.2],
                         [0.1, 0.3, 0.4, 0.7, 0.4, 0.7, 0.5, 0.8]])
        }

        for k in self.sets['K']:
            if (k, 0) in self.sets['delta_kt']:
                for j in self.sets['delta_kt'][(k, 0)]:
                    if k in cpt_matrices:
                        cpt_matrix = cpt_matrices[k]
                        probs = cpt_matrix[:, j]
                        for r in range(len(probs)):
                            if r < self.num_states:
                                self.g_hat_0[(k, j, r)] = float(probs[r])
                    else:
                        uniform_prob = 1.0 / self.num_states
                        for r in self.sets['R_kt'][(k, 0)]:
                            self.g_hat_0[(k, j, r)] = uniform_prob

        # POMDP概率
        self._extract_pomdp_probabilities()

    def _extract_observations_from_prediction(self):
        """从预测数据提取观测"""
        self.o_hat = {}
        for k in self.sets['K']:
            for t in self.sets['T']:
                self.o_hat[(k, t)] = 0

        if hasattr(self, 'prediction_data') and isinstance(self.prediction_data, dict):
            for k in self.sets['K']:
                for t in self.sets['T']:
                    period_key = f'period_{t}'
                    if period_key in self.prediction_data:
                        period_data = self.prediction_data[period_key]
                        if 'observed_state' in period_data and k in period_data['observed_state']:
                            self.o_hat[(k, t)] = int(period_data['observed_state'][k])

    def _extract_pomdp_probabilities(self):
        """提取POMDP概率"""
        if not hasattr(self, 'pomdp_data') or not self.pomdp_data:
            return

        transition_probs = self.pomdp_data.get('transition_probabilities', {})
        observation_probs = self.pomdp_data.get('observation_probabilities', {})

        # 转移概率
        for k in range(self.num_nodes):
            if k in transition_probs:
                trans_matrix = transition_probs[k]
                for t in range(self.num_periods):
                    for r_curr in range(self.num_states):
                        for a in range(self.num_actions):
                            for r_next in range(self.num_states):
                                prob = trans_matrix[r_curr, a, r_next]
                                self.P_transition[(k, t, r_next, r_curr, a)] = float(prob)

        # 观测概率
        for k in range(self.num_nodes):
            if k in observation_probs:
                obs_matrix = observation_probs[k]
                for t in range(self.num_periods):
                    for r in range(self.num_states):
                        for a_prev in range(self.num_actions):
                            for o in range(self.num_obs):
                                prob = obs_matrix[r, a_prev, o]
                                self.P_observation[(k, t, o, r, a_prev)] = float(prob)

    # ==================== 遗传算法核心方法 ====================

    def encode_solution(self, actions):
        """
        编码解：将动作决策编码为染色体
        染色体结构: [node0_t1_action, node0_t2_action, ..., node1_t1_action, ...]
        """
        chromosome = []
        for k in self.sets['K']:
            for t in self.sets['T'][1:-1]:  # t=1 to T-1
                chromosome.append(actions.get((k, t), 0))
        return np.array(chromosome)

    def decode_solution(self, chromosome):
        """
        解码染色体：将染色体解码为动作决策
        """
        actions = {}
        idx = 0
        for k in self.sets['K']:
            for t in self.sets['T'][1:-1]:
                actions[(k, t)] = int(chromosome[idx])
                idx += 1
        return actions

    def initialize_population(self):
        """初始化种群"""
        print("\n🧬 初始化种群...")

        # 染色体长度
        chromosome_length = len(self.sets['K']) * (len(self.sets['T']) - 2)

        population = []

        # 生成随机个体
        for i in range(self.population_size):
            chromosome = np.random.randint(0, self.num_actions, size=chromosome_length)

            # 强制最后节点选择动作0
            last_node = max(self.sets['K'])
            for t_idx, t in enumerate(self.sets['T'][1:-1]):
                node_offset = last_node * len(self.sets['T'][1:-1])
                chromosome[node_offset + t_idx] = 0

            population.append(chromosome)

        print(f"    ✓ 生成 {len(population)} 个初始个体")
        print(f"    染色体长度: {chromosome_length}")

        return np.array(population)

    def repair_solution(self, chromosome):
        """
        修复解：确保满足约束
        1. 预算约束
        2. 最后节点固定动作0
        """
        actions = self.decode_solution(chromosome)

        # 修复最后节点
        last_node = max(self.sets['K'])
        for t in self.sets['T'][1:-1]:
            actions[(last_node, t)] = 0

        # 修复预算约束
        total_cost = sum(self.cost.get((k, t, actions[(k, t)]), 0)
                         for k in self.sets['K']
                         for t in self.sets['T'][1:-1])

        if total_cost > self.budget:
            # 贪心修复：从成本最高的动作开始降级
            action_costs = []
            for k in self.sets['K']:
                for t in self.sets['T'][1:-1]:
                    if k != last_node:  # 排除最后节点
                        a = actions[(k, t)]
                        cost = self.cost.get((k, t, a), 0)
                        action_costs.append(((k, t), a, cost))

            # 按成本降序排序
            action_costs.sort(key=lambda x: x[2], reverse=True)

            # 逐个降级动作直到满足预算
            for (k, t), a, cost in action_costs:
                if total_cost <= self.budget:
                    break

                if a > 0:
                    new_action = a - 1
                    old_cost = self.cost.get((k, t, a), 0)
                    new_cost = self.cost.get((k, t, new_action), 0)
                    actions[(k, t)] = new_action
                    total_cost = total_cost - old_cost + new_cost

        return self.encode_solution(actions)

    def evaluate_fitness(self, chromosome):
        """
        评估适应度
        目标: min Σ γ^t * u_{|K|,t}(0)
        """
        try:
            # 修复解
            chromosome = self.repair_solution(chromosome)
            actions = self.decode_solution(chromosome)

            # 计算信念状态
            u, G = self._compute_belief_states(actions)

            # 计算目标函数
            last_node = max(self.sets['K'])
            worst_state = 0
            objective = 0.0

            for t in range(1, len(self.sets['T'])):
                if (last_node, t, worst_state) in u:
                    objective += (self.gamma ** t) * u[(last_node, t, worst_state)]

            # 惩罚预算超支
            total_cost = sum(self.cost.get((k, t, actions[(k, t)]), 0)
                             for k in self.sets['K']
                             for t in self.sets['T'][1:-1])

            if total_cost > self.budget:
                penalty = 1000 * (total_cost - self.budget)
                objective += penalty

            return objective

        except Exception as e:
            print(f"⚠️  适应度评估出错: {e}")
            return float('inf')

    def _compute_belief_states(self, actions):
        """计算信念状态和条件信念概率"""
        u = {}
        G = {}

        # t=0: 初始化
        for k in self.sets['K']:
            for r in self.sets['R_kt'][(k, 0)]:
                u[(k, 0, r)] = self.u_hat_0.get((k, r), 1.0 / self.num_states)

        for k in self.sets['K']:
            if self.sets['Theta_kt'][(k, 0)]:
                for j in self.sets['delta_kt'][(k, 0)]:
                    for r in self.sets['R_kt'][(k, 0)]:
                        G[(k, 0, j, r)] = self.g_hat_0.get((k, j, r), 1.0 / self.num_states)

        # t=1: 使用初始动作更新
        for k in self.sets['K']:
            if self.sets['Theta_kt'][(k, 1)]:
                for j in self.sets['delta_kt'][(k, 1)]:
                    for r in self.sets['R_kt'][(k, 1)]:
                        G[(k, 1, j, r)] = self._compute_G_t1(k, j, r)

        # t=1的信念状态
        for k in self.sets['K']:
            for r in self.sets['R_kt'][(k, 1)]:
                u[(k, 1, r)] = self._compute_u_from_G(k, 1, r, G, u)

        # t>=2: 使用决策动作更新
        for t in range(2, len(self.sets['T'])):
            for k in self.sets['K']:
                if self.sets['Theta_kt'][(k, t)]:
                    for j in self.sets['delta_kt'][(k, t)]:
                        for r in self.sets['R_kt'][(k, t)]:
                            G[(k, t, j, r)] = self._compute_G_t_general(k, t, j, r, actions, G)

            for k in self.sets['K']:
                for r in self.sets['R_kt'][(k, t)]:
                    u[(k, t, r)] = self._compute_u_from_G(k, t, r, G, u)

        return u, G

    def _compute_G_t1(self, k, j, r):
        """计算t=1的条件信念概率"""
        o_hat_1 = self.o_hat[(k, 1)]
        a_hat_0 = self.a_hat_0[k]

        numerator = 0.0
        denominator = 0.0

        # 分子
        p_obs = self.P_observation.get((k, 1, o_hat_1, r, a_hat_0), 1e-8)
        for r0 in self.sets['R_kt'][(k, 0)]:
            p_trans = self.P_transition.get((k, 0, r, r0, a_hat_0), 1e-8)
            g_hat = self.g_hat_0.get((k, j, r0), 1e-8)
            numerator += p_trans * g_hat
        numerator *= p_obs

        # 分母
        for r_tilde in self.sets['R_kt'][(k, 1)]:
            p_obs_tilde = self.P_observation.get((k, 1, o_hat_1, r_tilde, a_hat_0), 1e-8)
            inner_sum = 0.0
            for r0 in self.sets['R_kt'][(k, 0)]:
                p_trans = self.P_transition.get((k, 0, r_tilde, r0, a_hat_0), 1e-8)
                g_hat = self.g_hat_0.get((k, j, r0), 1e-8)
                inner_sum += p_trans * g_hat
            denominator += p_obs_tilde * inner_sum

        if denominator < 1e-10:
            return 1.0 / self.num_states

        return numerator / denominator

    def _compute_G_t_general(self, k, t, j, r, actions, G):
        """计算t>=2的条件信念概率"""
        o_hat_t = self.o_hat[(k, t)]
        o_hat_prev = self.o_hat[(k, t - 1)]

        numerator = 0.0
        denominator = 0.0

        for a in self.sets['A_kt'][(k, t - 1)]:
            # 检查动作是否被选择（简化：假设所有动作有可能性）
            action_selected = 1.0 if actions.get((k, t - 1)) == a else 0.0

            if action_selected < 0.5:
                continue

            # 分子
            p_obs = self.P_observation.get((k, t, o_hat_t, r, a), 1e-8)
            inner_sum_num = 0.0
            for r_prev in self.sets['R_kt'][(k, t - 1)]:
                p_trans = self.P_transition.get((k, t - 1, r, r_prev, a), 1e-8)
                g_prev = G.get((k, t - 1, j, r_prev), 1e-8)
                inner_sum_num += p_trans * g_prev

            numerator += action_selected * p_obs * inner_sum_num

            # 分母
            for r_tilde in self.sets['R_kt'][(k, t)]:
                p_obs_tilde = self.P_observation.get((k, t, o_hat_t, r_tilde, a), 1e-8)
                inner_sum_den = 0.0
                for r_prev in self.sets['R_kt'][(k, t - 1)]:
                    p_trans = self.P_transition.get((k, t - 1, r_tilde, r_prev, a), 1e-8)
                    g_prev = G.get((k, t - 1, j, r_prev), 1e-8)
                    inner_sum_den += p_trans * g_prev

                denominator += action_selected * p_obs_tilde * inner_sum_den

        if denominator < 1e-10:
            return 1.0 / self.num_states

        return numerator / denominator

    def _compute_u_from_G(self, k, t, r, G, u):
        """从G计算u（信念状态）"""
        if not self.sets['Theta_kt'][(k, t)]:
            return 1.0 / self.num_states

        belief_sum = 0.0

        for j in self.sets['delta_kt'][(k, t)]:
            g_val = G.get((k, t, j, r), 0.0)

            # 计算父节点乘积
            parent_product = 1.0
            parent_set = self.sets['Theta_kt'][(k, t)]

            for parent_idx, (parent_k, parent_t) in enumerate(parent_set):
                parent_state = self._get_parent_state(k, j, parent_idx)

                if parent_t == -1:
                    # 虚拟时间层
                    parent_prob = self.u_hat_0.get((parent_k, parent_state), 1e-8)
                else:
                    parent_prob = u.get((parent_k, parent_t, parent_state), 1e-8)

                parent_product *= parent_prob

            belief_sum += g_val * parent_product

        return belief_sum

    def _get_parent_state(self, k, j, parent_idx):
        """从组合索引j中提取父节点状态"""
        if k in self.G_dic and j < len(self.G_dic[k]):
            combination = self.G_dic[k][j]
            if parent_idx < len(combination):
                if parent_idx < len(combination) - 1:
                    return combination[parent_idx]
                else:
                    return combination[-1]
        return 0

    def tournament_selection(self, population, fitness_values, tournament_size):
        """锦标赛选择"""
        selected_idx = np.random.choice(len(population), size=tournament_size, replace=False)
        selected_fitness = fitness_values[selected_idx]
        winner_idx = selected_idx[np.argmin(selected_fitness)]
        return population[winner_idx].copy()

    def crossover(self, parent1, parent2):
        """单点交叉"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1, child2

    def mutate(self, chromosome):
        """变异操作"""
        mutated = chromosome.copy()

        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                mutated[i] = np.random.randint(0, self.num_actions)

        return mutated

    def evolve(self):
        """主进化循环"""
        print("\n🚀 开始遗传算法优化...")
        self.start_time = time.time()

        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])

        # 记录最优解
        best_idx = np.argmin(fitness_values)
        self.best_fitness = fitness_values[best_idx]
        self.best_solution = population[best_idx].copy()

        print(f"    初始最优适应度: {self.best_fitness:.6f}")

        # 进化循环
        for generation in range(self.max_generations):
            # 选择、交叉、变异
            new_population = []

            # 精英保留
            elite_size = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness_values)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self.tournament_selection(population, fitness_values, self.tournament_size)
                parent2 = self.tournament_selection(population, fitness_values, self.tournament_size)

                # 交叉
                child1, child2 = self.crossover(parent1, parent2)

                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # 截断到种群大小
            new_population = new_population[:self.population_size]
            population = np.array(new_population)

            # 评估新种群
            fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])

            # 更新最优解
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_solution = population[best_idx].copy()

            # 记录历史
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))

            # 输出进度
            if (generation + 1) % 50 == 0 or generation == 0:
                print(f"    代 {generation + 1}/{self.max_generations}: "
                      f"最优={self.best_fitness:.6f}, 平均={np.mean(fitness_values):.6f}")

        self.time_used = time.time() - self.start_time
        print(f"\n✅ 优化完成！用时: {self.time_used:.2f} 秒")
        print(f"    最优适应度: {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness

    def extract_solution(self):
        """提取解"""
        if self.best_solution is None:
            print("❌ 没有可用的解")
            return None, None

        actions = self.decode_solution(self.best_solution)

        # 计算信念状态
        u, G = self._compute_belief_states(actions)

        # 计算总成本
        total_cost = sum(self.cost.get((k, t, actions[(k, t)]), 0)
                         for k in self.sets['K']
                         for t in self.sets['T'][1:-1])

        print("\n📋 最优解详情:")
        print(f"    目标函数值: {self.best_fitness:.6f}")
        print(f"    总成本: {total_cost:.2f} / {self.budget}")

        print("\n    决策动作:")
        action_names = {0: "无动作", 1: "mild", 2: "intense"}
        for k in self.sets['K']:
            for t in self.sets['T'][1:-1]:
                a = actions[(k, t)]
                cost = self.cost.get((k, t, a), 0)
                print(f"      节点 {k}, 时期 {t}: 动作 {a} ({action_names[a]}), 成本 {cost}")

        print("\n    最后节点风险状态概率:")
        last_node = max(self.sets['K'])
        worst_state = 0
        for t in self.sets['T']:
            prob = u.get((last_node, t, worst_state), 0.0)
            print(f"      时期 {t}: {prob:.4f}")

        return actions, u

    def export_results(self, filename=None):
        """导出结果到Excel"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            disruption = self.prediction_params.get('disruption_level', 'unknown')
            filename = f"GA_B{self.budget}_Gamma{self.gamma:.2f}_{disruption}_{timestamp}.xlsx"

        try:
            actions, u = self.extract_solution()

            if actions is None:
                return None

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 决策变量
                action_names = {0: "无动作", 1: "mild", 2: "intense"}
                x_data = []
                for k in self.sets['K']:
                    for t in self.sets['T'][1:-1]:
                        a = actions[(k, t)]
                        x_data.append({
                            '节点': k,
                            '时期': t,
                            '动作': a,
                            '动作名称': action_names[a],
                            '成本': self.cost.get((k, t, a), 0)
                        })

                df_x = pd.DataFrame(x_data)
                df_x.to_excel(writer, sheet_name='决策变量', index=False)

                # 信念状态
                u_data = []
                for (k, t, r), prob in u.items():
                    u_data.append({
                        '节点': k,
                        '时期': t,
                        '状态': r,
                        '概率': prob
                    })

                df_u = pd.DataFrame(u_data)
                df_u.to_excel(writer, sheet_name='信念状态', index=False)

                # 风险分析
                last_node = max(self.sets['K'])
                worst_state = 0
                risk_data = []
                for t in self.sets['T']:
                    prob = u.get((last_node, t, worst_state), 0.0)
                    risk_data.append({
                        '时期': t,
                        '风险状态概率': prob
                    })

                df_risk = pd.DataFrame(risk_data)
                df_risk.to_excel(writer, sheet_name='风险分析', index=False)

                # 参数汇总
                params_data = {
                    '参数名': ['节点数', '时期数', '状态数', '动作数', '预算', '折现因子',
                               '求解时间(秒)', '目标函数值', 'Disruption级别',
                               '种群大小', '最大迭代代数', '交叉率', '变异率'],
                    '值': [
                        self.num_nodes, self.num_periods, self.num_states,
                        self.num_actions, self.budget, self.gamma,
                        round(self.time_used, 2), self.best_fitness,
                        self.prediction_params.get('disruption_level', 'N/A'),
                        self.population_size, self.max_generations,
                        self.crossover_rate, self.mutation_rate
                    ]
                }
                df_params = pd.DataFrame(params_data)
                df_params.to_excel(writer, sheet_name='参数汇总', index=False)

                # 进化历史
                history_data = {
                    '代数': list(range(1, len(self.best_fitness_history) + 1)),
                    '最优适应度': self.best_fitness_history,
                    '平均适应度': self.avg_fitness_history
                }
                df_history = pd.DataFrame(history_data)
                df_history.to_excel(writer, sheet_name='进化历史', index=False)

            print(f"✅ 结果已导出: {filename}")
            return filename

        except Exception as e:
            print(f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_convergence(self, save_path=None):
        """绘制收敛曲线"""
        if not self.best_fitness_history:
            print("❌ 没有进化历史数据")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Best Fitness', linewidth=2)
        plt.plot(self.avg_fitness_history, label='Average Fitness', linewidth=2, alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 收敛曲线已保存: {save_path}")
        else:
            plt.show()


# ==================== 辅助函数 ====================

def get_observed_data(disruption_level):
    """获取观测数据（复用原代码）"""
    if disruption_level.lower() == 'light':
        return {
            1: {
                0: {'D_obs': 90, 'SD_obs': 38},
                1: {'D_obs': 88, 'SD_obs': 42},
                2: {'D_obs': 92, 'SD_obs': 48},
                3: {'D_obs': 45, 'SD_obs': 33},
                4: {'D_obs': 47, 'SD_obs': 38},
                5: {'D_obs': 45, 'SD_obs': 39}
            }
        }
    elif disruption_level.lower() == 'moderate':
        return {
            1: {
                0: {'D_obs': 100, 'SD_obs': 24},
                1: {'D_obs': 98, 'SD_obs': 27},
                2: {'D_obs': 102, 'SD_obs': 33},
                3: {'D_obs': 55, 'SD_obs': 21},
                4: {'D_obs': 52, 'SD_obs': 24},
                5: {'D_obs': 50, 'SD_obs': 31}
            }
        }
    elif disruption_level.lower() == 'severe':
        return {
            1: {
                0: {'D_obs': 110, 'SD_obs': 15},
                1: {'D_obs': 105, 'SD_obs': 19},
                2: {'D_obs': 108, 'SD_obs': 23},
                3: {'D_obs': 60, 'SD_obs': 6},
                4: {'D_obs': 58, 'SD_obs': 18},
                5: {'D_obs': 55, 'SD_obs': 22}
            }
        }
    else:
        return None


def main():
    """主函数"""
    print("=" * 80)
    print("🧬 遗传算法求解供应链韧性优化问题")
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

    # 参数配置
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

    prediction_params = {
        'num_periods': 4,
        'num_states': 2,
        'mcmc_samples': 1000,
        'mc_samples': 1000,
        'disruption_level': disruption_level,
        'observed_data': get_observed_data(disruption_level)
    }

    # 创建求解器
    solver = GeneticAlgorithmSolver(
        network_params=network_params,
        pomdp_params=pomdp_params,
        prediction_params=prediction_params,
        population_size=100,
        max_generations=300,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_rate=0.1,
        tournament_size=5
    )

    # 设置预算
    solver.budget = 100

    # 初始化
    solver.initialize_components()

    # 运行优化
    best_solution, best_fitness = solver.evolve()

    # 提取解
    solver.extract_solution()

    # 导出结果
    excel_file = solver.export_results()

    # 绘制收敛曲线
    if excel_file:
        plot_file = excel_file.replace('.xlsx', '_convergence.png')
        solver.plot_convergence(save_path=plot_file)

    print("\n✅ 遗传算法求解完成！")

    return solver


if __name__ == "__main__":
    solver = main()