"""
Enhanced POMDP Parameters Generator for Supply Chain Resilience

🔧 修改版本：支持随机实验
- 移除所有硬编码的节点特定概率
- 改为参数化公式动态生成
- 支持任意节点数、状态数、动作数
- 调用 R1_network_generate4.py

Current Date and Time (UTC): 2025-10-28 12:48:41
Current User's Login: dyy21zyy
"""

import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import warnings

warnings.filterwarnings('ignore')

# 🔧 修改点1：调用修改后的网络生成器
from R1_network_generate4 import generate_supply_chain_network


class POMDPParametersGenerator:
    """
    POMDP参数生成器
    生成状态转移概率矩阵和观察概率矩阵
    所有关键参数都从主函数config传入，无默认值

    🔧 修改版本特性：
    1. 移除所有硬编码节点特定概率
    2. 使用参数化公式动态生成
    3. 支持任意节点数、层数
    4. 保证节点间有差异但不硬编码
    """

    def __init__(self, network_data, num_states, num_actions, seed):
        """
        初始化POMDP参数生成器
        所有参数都必须从主函数传入，无默认值

        Args:
            network_data: 网络数据
            num_states: 状态数量（从主函数config传入，必须）
            num_actions: 行动数量（从主函数config传入，必须）
            seed: 随机种子（从主函数config传入，必须）
        """
        # 验证所有必须参数
        required_params = {
            'num_states': num_states,
            'num_actions': num_actions,
            'seed': seed
        }

        for param_name, param_value in required_params.items():
            if param_value is None:
                raise ValueError(
                    f"Required parameter '{param_name}' cannot be None - must be provided by main function config")

        print(f"🔧 Initializing Enhanced POMDP Parameters Generator (Random Experiment Version)")
        print(f"   All parameters from main function config:")
        print(f"   - num_states: {num_states}")
        print(f"   - num_actions: {num_actions}")
        print(f"   - seed: {seed}")

        self.seed = seed
        np.random.seed(seed)

        # 从主函数config传入的参数（无默认值）
        self.num_states = num_states  # |R_k^t|
        self.num_actions = num_actions  # |A_k^t|

        # 解包网络数据（来自R1_network_generate4）
        (self.network, self.layer_info, self.temporal_network,
         self.temporal_node_info, self.parent_dict,
         self.independent_nodes, self.other_nodes,
         self.parent_node_dic, self.C_dic, self.G_dic) = network_data

        self.num_nodes = self.layer_info['num_nodes']
        self.num_observations = self.num_states  # |O_k^t| = |R_k^t| (观察空间等同于状态空间)

        # 验证网络数据一致性
        self._validate_network_data()

        # 定义状态、行动和观察的含义
        self._define_spaces()

        # 存储生成的概率矩阵
        self.transition_probabilities = {}  # P(r^{t+1} | r^t, a^t)
        self.observation_probabilities = {}  # P(o^t | r^t, a^{t-1})

        print("POMDP Parameters Generator Initialized")
        print(f"Configuration: {self.num_nodes} nodes, {self.num_states} states, {self.num_actions} actions")

    def _validate_network_data(self):
        """验证网络数据的完整性"""
        print("🔍 Validating network data consistency...")

        required_components = [
            'network', 'layer_info', 'temporal_network', 'temporal_node_info',
            'parent_dict', 'independent_nodes', 'other_nodes', 'parent_node_dic',
            'C_dic', 'G_dic'
        ]

        for component in required_components:
            if not hasattr(self, component):
                raise ValueError(f"Missing network component: {component}")

        # 验证节点数量一致性
        expected_nodes = self.layer_info['num_nodes']
        if self.network.shape[0] != expected_nodes:
            raise ValueError(f"Network dimension mismatch: expected {expected_nodes}, got {self.network.shape[0]}")

        print(f"   ✅ Network validation passed")
        print(f"   - Spatial network: {self.network.shape}")
        print(f"   - Temporal network: {self.temporal_network.shape}")
        print(f"   - Independent nodes: {self.independent_nodes}")
        print(f"   - Other nodes: {self.other_nodes}")

    def _define_spaces(self):
        """
        定义状态空间、行动空间和观察空间的含义
        基于主函数传入的num_states和num_actions参数
        """

        # 状态空间 R_k^t (基于履行率的离散化)
        if self.num_states == 2:
            self.state_definitions = {
                0: "Good (High Fulfillment Rate ≥ 0.5)",
                1: "Poor (Low Fulfillment Rate < 0.5)"
            }
        elif self.num_states == 3:
            self.state_definitions = {
                0: "Excellent (Fulfillment Rate ≥ 0.67)",
                1: "Moderate (0.33 ≤ Fulfillment Rate < 0.67)",
                2: "Poor (Fulfillment Rate < 0.33)"
            }
        elif self.num_states == 4:
            self.state_definitions = {
                0: "Excellent (Fulfillment Rate ≥ 0.75)",
                1: "Good (0.5 ≤ Fulfillment Rate < 0.75)",
                2: "Fair (0.25 ≤ Fulfillment Rate < 0.5)",
                3: "Poor (Fulfillment Rate < 0.25)"
            }
        else:
            self.state_definitions = {i: f"State_{i}" for i in range(self.num_states)}

        # 行动空间 A_k^t (基于主函数参数)
        if self.num_actions == 2:
            self.action_definitions = {
                0: "Maintain Current Operations (No Action)",
                1: "Increase Safety Stock"
            }
        elif self.num_actions == 3:
            self.action_definitions = {
                0: "Maintain Current Operations (No Action)",
                1: "mild intervention",
                2: "intense intervention"
            }
        else:
            self.action_definitions = {i: f"Action_{i}" for i in range(self.num_actions)}

        # 观察空间 O_k^t (与状态空间相同)
        self.observation_definitions = self.state_definitions.copy()

        print("\n🔧 Space Definitions (from main config):")
        print("States (R_k^t):")
        for i, desc in self.state_definitions.items():
            print(f"  {i}: {desc}")
        print("Actions (A_k^t):")
        for i, desc in self.action_definitions.items():
            print(f"  {i}: {desc}")
        print("Observations (O_k^t):")
        for i, desc in self.observation_definitions.items():
            print(f"  {i}: {desc}")

    def _get_node_characteristics(self, node):
        """
        获取节点特征（兼容多层网络）

        🔧 修改点：支持任意层数
        """
        # 遍历所有层，找到节点所属层
        for layer_idx in range(1, self.layer_info['num_layers'] + 1):
            layer_key = f'layer{layer_idx}'
            if layer_key in self.layer_info:
                start, end, name = self.layer_info[layer_key]
                if start <= node < end:
                    return name, node - start  # 返回层名和层内索引

        return 'Unknown', 0

    def generate_transition_probabilities(self):
        """
        生成状态转移概率矩阵 P(r^{t+1} | r^t, a^t)
        基于主函数传入的num_states和num_actions参数

        🔧 修改版本：
        - 移除硬编码的特殊节点处理
        - 使用参数化公式动态生成
        """
        print(f"\n🔧 Generating ENHANCED State Transition Probabilities P(r^{{t+1}} | r^t, a^t)")
        print(f"   Matrix dimensions from main config: ({self.num_states}, {self.num_actions}, {self.num_states})")
        print(f"   Using parameterized formulas (no hardcoded node-specific values)")
        print("=" * 70)

        self.transition_probabilities = {}

        for node in range(self.num_nodes):
            node_type, node_idx = self._get_node_characteristics(node)

            print(f"\n📍 Node {node} ({node_type} {node_idx}) - Parameterized Probabilities")

            # 为每个节点创建转移概率矩阵
            # 维度: [current_state, action, next_state] - 基于主函数参数
            transition_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))

            for current_state in range(self.num_states):
                for action in range(self.num_actions):
                    # 根据当前状态和行动生成下一状态的概率分布
                    next_state_probs = self._compute_transition_probabilities(
                        node, node_type, current_state, action
                    )

                    transition_matrix[current_state, action, :] = next_state_probs

                    print(
                        f"  P(r^{{t+1}} | r^t={current_state}, a^t={action}): {[f'{p:.3f}' for p in next_state_probs]}")

            self.transition_probabilities[node] = transition_matrix

            # 验证概率矩阵的有效性
            self._validate_transition_matrix(transition_matrix, node)

        print(f"\n✅ State transition probabilities generated successfully!")
        print(f"   Generated for {len(self.transition_probabilities)} nodes")
        print(f"   Each matrix shape: ({self.num_states}, {self.num_actions}, {self.num_states})")
        return self.transition_probabilities

    def _compute_transition_probabilities(self, node, node_type, current_state, action):
        """
        🔧 核心修改：参数化计算转移概率（移除所有硬编码）

        计算策略：
        1. 基于节点哈希生成节点特异性因子
        2. 基于节点类型调整概率倾向
        3. 基于动作计算改善/恶化效果
        4. 不再有任何节点的硬编码特殊处理

        状态定义：0=最差状态，数字越大状态越好
        动作效果（必须严格区分）：
        - 转移到最好状态概率：Action 2 > Action 1 > Action 0
        - 转移到最差状态概率：Action 0 > Action 1 > Action 2
        """

        # 🔧 移除节点5的特殊处理，改为统一公式

        # 设置概率边界
        MIN_PROB = 0.1
        MAX_PROB = 0.8

        # ============================================
        # 步骤1：生成节点特异性因子（基于哈希）
        # ============================================
        # 使用多个质数生成节点哈希，确保节点间有差异
        node_hash_1 = (node * 19 + 11) % 31
        node_hash_2 = (node * 23 + 13) % 37
        node_hash_3 = (node * 29 + 17) % 41

        # 节点基础因子 [0.0, 0.3]
        node_base_factor = (node_hash_1 / 31.0) * 0.3

        # 节点波动因子 [-0.1, +0.1]
        node_volatility = (node_hash_2 / 37.0) * 0.2 - 0.1

        # 节点偏好因子（倾向好状态或差状态）[-0.15, +0.15]
        node_bias = (node_hash_3 / 41.0) * 0.3 - 0.15

        # ============================================
        # 步骤2：节点类型调整
        # ============================================
        # 不同类型的节点有不同的稳定性和响应能力
        type_factors = {
            'Suppliers': {
                'stability': 0.7,  # 稳定性较低
                'action_response': 1.2  # 对投资响应较好
            },
            'Manufacturers': {
                'stability': 0.75,
                'action_response': 1.3
            },
            'Intermediate_1': {  # 中间层1
                'stability': 0.8,
                'action_response': 1.15
            },
            'Intermediate_2': {  # 中间层2
                'stability': 0.82,
                'action_response': 1.1
            },
            'Retailer': {
                'stability': 0.85,  # 稳定性最高
                'action_response': 1.4  # 对投资响应最好（效果最直接）
            },
            'Unknown': {
                'stability': 0.75,
                'action_response': 1.0
            }
        }

        # 获取节点类型因子
        type_factor = type_factors.get(node_type, type_factors['Unknown'])
        stability = type_factor['stability']
        action_response = type_factor['action_response']

        # ============================================
        # 步骤3：动作效果计算
        # ============================================
        # 不同动作对状态转移的影响（基础效果）
        action_base_effects = {
            0: {  # 无动作 - 最差效果
                'improvement': 0.05,  # 改善倾向很低
                'degradation': 0.40,  # 恶化倾向高
                'stability_penalty': 0.2  # 降低稳定性
            },
            1: {  # Mild intervention - 中等效果
                'improvement': 0.25,
                'degradation': 0.15,
                'stability_penalty': 0.0
            },
            2: {  # Intense intervention - 最好效果
                'improvement': 0.45,
                'degradation': 0.05,
                'stability_penalty': 0.0
            }
        }

        # 高级动作（如果存在）
        if action >= 3:
            # 递增改善效果
            extra_improvement = (action - 2) * 0.08
            action_effects = {
                'improvement': min(0.55, 0.45 + extra_improvement),
                'degradation': max(0.02, 0.05 - extra_improvement * 0.3),
                'stability_penalty': 0.0
            }
        else:
            action_effects = action_base_effects.get(action, action_base_effects[0])

        # 应用节点类型的动作响应系数
        action_effects['improvement'] *= action_response
        action_effects['degradation'] /= action_response

        # ============================================
        # 步骤4：计算转移概率
        # ============================================

        if self.num_states == 2:
            # 2状态：0(最差), 1(最好)

            if current_state == 0:  # 当前最差状态
                # 基础保持最差状态的概率
                base_worst_prob = 0.5 + node_base_factor

                # 根据动作调整
                if action == 0:  # 无动作
                    base_worst_prob += action_effects['degradation']
                    base_worst_prob -= action_effects['improvement']
                    base_worst_prob += action_effects['stability_penalty']
                else:  # 有投资
                    base_worst_prob -= action_effects['improvement']
                    base_worst_prob += action_effects['degradation']

                # 应用节点偏好
                base_worst_prob += node_bias

                # 应用稳定性
                base_worst_prob = base_worst_prob * (1 - stability) + 0.5 * stability

                # 应用波动
                base_worst_prob += node_volatility

                # 限制范围
                worst_prob = np.clip(base_worst_prob, MIN_PROB, MAX_PROB)
                best_prob = 1.0 - worst_prob
                next_state_probs = np.array([worst_prob, best_prob])

            else:  # current_state == 1，当前最好状态
                # 在好状态下，不同action对保持好状态的能力不同
                base_worst_prob = 0.3 + node_base_factor

                if action == 0:  # 无动作 - 容易退化
                    base_worst_prob += action_effects['degradation']
                    base_worst_prob += action_effects['stability_penalty']
                else:  # 有投资 - 容易保持
                    base_worst_prob -= action_effects['improvement']
                    base_worst_prob += action_effects['degradation'] * 0.5

                # 应用节点偏好（反向）
                base_worst_prob -= node_bias

                # 应用稳定性
                base_worst_prob = base_worst_prob * (1 - stability) + 0.2 * stability

                # 应用波动
                base_worst_prob += node_volatility * 0.5

                worst_prob = np.clip(base_worst_prob, MIN_PROB, MAX_PROB)
                best_prob = 1.0 - worst_prob
                next_state_probs = np.array([worst_prob, best_prob])

        elif self.num_states == 3:
            # 3状态：0(最差), 1(中等), 2(最好)
            next_state_probs = np.zeros(3)

            # 基础概率分布
            if current_state == 0:  # 最差状态
                if action == 0:  # 无动作
                    base_probs = [0.55 + node_base_factor, 0.30, 0.15 - node_base_factor]
                elif action == 1:  # Mild
                    base_probs = [0.30 + node_base_factor, 0.40, 0.30 - node_base_factor]
                elif action == 2:  # Intense
                    base_probs = [0.12 + node_base_factor, 0.28, 0.60 - node_base_factor]
                else:  # Advanced
                    improvement = min((action - 2) * 0.08, 0.20)
                    base_probs = [
                        max(0.08, 0.12 - improvement) + node_base_factor,
                        0.25,
                        min(0.67, 0.60 + improvement) - node_base_factor
                    ]

            elif current_state == 1:  # 中等状态
                if action == 0:
                    base_probs = [0.40 + node_base_factor, 0.40, 0.20 - node_base_factor]
                elif action == 1:
                    base_probs = [0.20 + node_base_factor, 0.40, 0.40 - node_base_factor]
                elif action == 2:
                    base_probs = [0.08 + node_base_factor, 0.22, 0.70 - node_base_factor]
                else:
                    improvement = min((action - 2) * 0.06, 0.15)
                    base_probs = [
                        max(0.05, 0.08 - improvement) + node_base_factor,
                        max(0.15, 0.22 - improvement),
                        min(0.80, 0.70 + improvement) - node_base_factor
                    ]

            else:  # current_state == 2，最好状态
                if action == 0:
                    base_probs = [0.30 + node_base_factor, 0.40, 0.30 - node_base_factor]
                elif action == 1:
                    base_probs = [0.15 + node_base_factor, 0.35, 0.50 - node_base_factor]
                elif action == 2:
                    base_probs = [0.05 + node_base_factor, 0.20, 0.75 - node_base_factor]
                else:
                    maintenance = min((action - 2) * 0.05, 0.12)
                    base_probs = [
                        max(0.03, 0.05 - maintenance) + node_base_factor,
                        max(0.12, 0.20 - maintenance),
                        min(0.85, 0.75 + maintenance) - node_base_factor
                    ]

            # 应用节点偏好和波动
            adjustments = np.array([node_bias, 0, -node_bias]) + np.array([node_volatility, 0, -node_volatility])
            adjustments *= (1 - stability)  # 稳定性越高，调整越小

            next_state_probs = np.array(base_probs) + adjustments

        else:  # num_states >= 4
            # 多状态情况：基于当前状态和动作的通用公式
            next_state_probs = np.zeros(self.num_states)
            best_state = self.num_states - 1

            # 计算改善和恶化倾向
            if action == 0:  # 无动作
                improvement_tendency = 0.05
                degradation_tendency = 0.40
            elif action == 1:
                improvement_tendency = 0.25
                degradation_tendency = 0.15
            elif action == 2:
                improvement_tendency = 0.45
                degradation_tendency = 0.05
            else:
                improvement_tendency = min(0.60, 0.45 + (action - 2) * 0.08)
                degradation_tendency = max(0.02, 0.05 - (action - 2) * 0.02)

            # 应用节点响应
            improvement_tendency *= action_response
            degradation_tendency /= action_response

            # 分配概率
            # 最差状态
            next_state_probs[0] = degradation_tendency + node_base_factor
            # 最好状态
            next_state_probs[best_state] = improvement_tendency - node_base_factor
            # 中间状态
            remaining = 1.0 - next_state_probs[0] - next_state_probs[best_state]
            if self.num_states > 2:
                middle_states = self.num_states - 2
                for i in range(1, best_state):
                    # 根据距离当前状态的远近分配概率
                    distance = abs(i - current_state)
                    weight = 1.0 / (1.0 + distance)
                    next_state_probs[i] = remaining * weight / middle_states

        # ============================================
        # 步骤5：归一化和验证
        # ============================================
        next_state_probs = np.clip(next_state_probs, MIN_PROB, MAX_PROB)

        # 归一化
        prob_sum = np.sum(next_state_probs)
        if prob_sum > 0:
            next_state_probs = next_state_probs / prob_sum
        else:
            next_state_probs = np.ones(self.num_states) / self.num_states

        # 精确到小数点后一位
        next_state_probs = np.round(next_state_probs, 1)

        # 最终归一化检查
        final_sum = np.sum(next_state_probs)
        if not np.isclose(final_sum, 1.0, atol=0.05):
            diff = 1.0 - final_sum
            max_idx = np.argmax(next_state_probs)
            next_state_probs[max_idx] += diff
            next_state_probs = np.clip(next_state_probs, MIN_PROB, MAX_PROB)
            next_state_probs = next_state_probs / np.sum(next_state_probs)
            next_state_probs = np.round(next_state_probs, 1)

        return next_state_probs

    def _validate_transition_matrix(self, transition_matrix, node):
        """验证转移矩阵的有效性"""
        print(f"📊 验证节点 {node} 的转移概率矩阵...")

        for s in range(self.num_states):
            print(f"  当前状态 {s}:")

            # 收集不同动作的转移概率用于比较
            action_effects = {}

            for a in range(self.num_actions):
                probs = transition_matrix[s, a, :]
                prob_sum = np.sum(probs)

                # 检查概率和
                if not np.isclose(prob_sum, 1.0, rtol=1e-2):
                    print(f"    ⚠️  动作 {a} - 概率和: {prob_sum:.3f}")

                # 检查负概率
                if np.any(probs < 0):
                    print(f"    ⚠️  动作 {a} - 存在负概率")

                # 记录关键概率
                worst_state_prob = probs[0]  # 转移到最差状态(0)的概率
                best_state_prob = probs[-1]  # 转移到最好状态的概率
                action_effects[a] = {
                    'worst': worst_state_prob,
                    'best': best_state_prob
                }

                print(f"    动作{a}: 最差状态概率={worst_state_prob:.1f}, 最好状态概率={best_state_prob:.1f}")

            # 验证动作效果排序
            if len(action_effects) >= 3:
                # 验证转移到最差状态概率排序: Action 0 > Action 1 > Action 2
                worst_0 = action_effects[0]['worst']
                worst_1 = action_effects[1]['worst'] if 1 in action_effects else 0
                worst_2 = action_effects[2]['worst'] if 2 in action_effects else 0

                print(f"    🔍 转移到最差状态概率排序验证:")
                print(f"       Action 0: {worst_0:.1f} (应该最高)")
                print(f"       Action 1: {worst_1:.1f} (应该中等)")
                print(f"       Action 2: {worst_2:.1f} (应该最低)")

                if worst_0 >= worst_1 >= worst_2:
                    print(f"    ✅ 最差状态概率排序正确")
                else:
                    print(f"    ⚠️  最差状态概率排序可能需要调整")

                # 验证转移到最好状态概率排序: Action 2 > Action 1 > Action 0
                best_0 = action_effects[0]['best']
                best_1 = action_effects[1]['best'] if 1 in action_effects else 0
                best_2 = action_effects[2]['best'] if 2 in action_effects else 0

                print(f"    🔍 转移到最好状态概率排序验证:")
                print(f"       Action 2: {best_2:.1f} (应该最高)")
                print(f"       Action 1: {best_1:.1f} (应该中等)")
                print(f"       Action 0: {best_0:.1f} (应该最低)")

                if best_2 >= best_1 >= best_0:
                    print(f"    ✅ 最好状态概率排序正确")
                else:
                    print(f"    ⚠️  最好状态概率排序可能需要调整")

    def generate_observation_probabilities(self):
        """
        生成观察概率矩阵 P(o^t | r^t, a^{t-1})

        🔧 修改版本：移除所有硬编码，使用参数化公式
        """
        print(f"\n👁️  Generating ENHANCED Observation Probabilities P(o^t | r^t, a^{{t-1}})")
        print(
            f"   Matrix dimensions from main config: ({self.num_states}, {self.num_actions}, {self.num_observations})")
        print(f"   Using parameterized formulas (no hardcoded node-specific values)")
        print("=" * 70)

        self.observation_probabilities = {}

        for node in range(self.num_nodes):
            node_type, node_idx = self._get_node_characteristics(node)

            print(f"\n📍 Node {node} ({node_type} {node_idx}) - Parameterized Observations")

            # 为每个节点创建观察概率矩阵
            # 维度: [current_state, previous_action, observation]
            observation_matrix = np.zeros((self.num_states, self.num_actions, self.num_observations))

            for current_state in range(self.num_states):
                for prev_action in range(self.num_actions):
                    # 生成观察概率分布
                    obs_probs = self._compute_observation_probabilities(
                        node, node_type, current_state, prev_action
                    )

                    observation_matrix[current_state, prev_action, :] = obs_probs

                    print(f"  P(o^t | r^t={current_state}, a^{{t-1}}={prev_action}): {[f'{p:.3f}' for p in obs_probs]}")

            self.observation_probabilities[node] = observation_matrix

            # 验证观察矩阵的有效性
            self._validate_observation_matrix(observation_matrix, node)

        print(f"\n✅ Enhanced observation probabilities generated successfully!")
        print(f"   Generated for {len(self.observation_probabilities)} nodes")
        print(f"   Each matrix shape: ({self.num_states}, {self.num_actions}, {self.num_observations})")
        return self.observation_probabilities

    def _compute_observation_probabilities(self, node, node_type, current_state, prev_action):
        """
        🔧 核心修改：参数化计算观测概率（移除所有硬编码）

        计算策略：
        1. 基于节点哈希生成观测特性
        2. 基于前一期动作调整观测准确性
        3. 不再有任何节点的硬编码特殊处理

        Args:
            node: 节点索引
            node_type: 节点类型
            current_state: 当前真实状态
            prev_action: 前一期采取的动作

        Returns:
            obs_probs: 观测概率分布（完全参数化生成）
        """

        MIN_PROB = 0.1
        MAX_PROB = 0.8

        # 🔧 移除节点5的特殊处理，改为统一公式

        # ============================================
        # 步骤1：生成节点观测特性（基于哈希）
        # ============================================
        # 使用不同的质数生成观测相关的节点特性
        obs_hash_1 = (node * 31 + 7) % 43
        obs_hash_2 = (node * 37 + 11) % 47
        obs_hash_3 = (node * 41 + 13) % 53

        # 节点基础观测准确度 [0.65, 0.85]
        base_accuracy = 0.65 + (obs_hash_1 / 43.0) * 0.20

        # 节点观测噪声水平 [0.05, 0.20]
        noise_level = 0.05 + (obs_hash_2 / 47.0) * 0.15

        # 节点观测偏差（倾向乐观或悲观）[-0.10, +0.10]
        observation_bias = (obs_hash_3 / 53.0) * 0.20 - 0.10

        # ============================================
        # 步骤2：节点类型调整观测能力
        # ============================================
        type_obs_factors = {
            'Suppliers': {
                'accuracy_bonus': 0.00,
                'noise_reduction': 0.90
            },
            'Manufacturers': {
                'accuracy_bonus': 0.05,
                'noise_reduction': 0.85
            },
            'Intermediate_1': {
                'accuracy_bonus': 0.03,
                'noise_reduction': 0.88
            },
            'Intermediate_2': {
                'accuracy_bonus': 0.04,
                'noise_reduction': 0.86
            },
            'Retailer': {
                'accuracy_bonus': 0.08,
                'noise_reduction': 0.80
            },
            'Unknown': {
                'accuracy_bonus': 0.00,
                'noise_reduction': 1.00
            }
        }

        type_factor = type_obs_factors.get(node_type, type_obs_factors['Unknown'])
        base_accuracy += type_factor['accuracy_bonus']
        noise_level *= type_factor['noise_reduction']

        # ============================================
        # 步骤3：前一期动作对观测的影响
        # ============================================
        # 投资会提高观测准确性
        action_obs_improvement = {
            0: -0.05,  # 无动作：观测准确性下降
            1: 0.05,  # Mild：观测准确性提升
            2: 0.12,  # Intense：观测准确性显著提升
        }

        if prev_action >= 3:
            # 高级动作：递增观测改善
            improvement = 0.12 + (prev_action - 2) * 0.03
            action_improvement = min(0.20, improvement)
        else:
            action_improvement = action_obs_improvement.get(prev_action, 0.0)

        # 应用动作改善
        adjusted_accuracy = base_accuracy + action_improvement
        adjusted_noise = noise_level * (1.0 - abs(action_improvement))

        # ============================================
        # 步骤4：计算观测概率
        # ============================================

        if self.num_observations == 2:
            # 2状态观测

            # 正确观测的概率
            correct_obs_prob = adjusted_accuracy

            # 应用噪声
            correct_obs_prob = correct_obs_prob * (1 - adjusted_noise) + 0.5 * adjusted_noise

            # 应用偏差
            if current_state == 0:  # 真实最差状态
                # 悲观偏差增加正确识别最差状态的概率
                correct_obs_prob += observation_bias
            else:  # 真实最好状态
                # 乐观偏差增加正确识别最好状态的概率
                correct_obs_prob -= observation_bias

            # 限制范围
            correct_obs_prob = np.clip(correct_obs_prob, MIN_PROB, MAX_PROB)

            if current_state == 0:
                obs_probs = np.array([correct_obs_prob, 1.0 - correct_obs_prob])
            else:
                obs_probs = np.array([1.0 - correct_obs_prob, correct_obs_prob])

        elif self.num_observations == 3:
            # 3状态观测
            obs_probs = np.zeros(3)

            # 基础正确观测概率
            correct_obs_prob = adjusted_accuracy

            # 应用噪声（分散到相邻状态）
            error_prob = adjusted_noise

            if current_state == 0:  # 真实最差状态
                obs_probs[0] = correct_obs_prob
                obs_probs[1] = error_prob * 0.7
                obs_probs[2] = error_prob * 0.3

                # 应用偏差
                obs_probs[0] += observation_bias
                obs_probs[2] -= observation_bias

            elif current_state == 1:  # 真实中等状态
                obs_probs[0] = error_prob * 0.4
                obs_probs[1] = correct_obs_prob
                obs_probs[2] = error_prob * 0.4

            else:  # current_state == 2，真实最好状态
                obs_probs[0] = error_prob * 0.3
                obs_probs[1] = error_prob * 0.7
                obs_probs[2] = correct_obs_prob

                # 应用偏差
                obs_probs[2] -= observation_bias
                obs_probs[0] += observation_bias

        else:  # num_observations >= 4
            # 多状态观测：基于距离的混淆矩阵
            obs_probs = np.zeros(self.num_observations)

            for obs in range(self.num_observations):
                distance = abs(obs - current_state)

                if distance == 0:
                    # 正确观测
                    obs_probs[obs] = adjusted_accuracy
                elif distance == 1:
                    # 相邻状态
                    obs_probs[obs] = adjusted_noise * 0.4
                elif distance == 2:
                    # 距离2
                    obs_probs[obs] = adjusted_noise * 0.2
                else:
                    # 更远
                    obs_probs[obs] = adjusted_noise * 0.1 / (distance - 1)

        # ============================================
        # 步骤5：归一化和验证
        # ============================================
        obs_probs = np.clip(obs_probs, MIN_PROB, MAX_PROB)

        # 归一化
        prob_sum = np.sum(obs_probs)
        if prob_sum > 0:
            obs_probs = obs_probs / prob_sum
        else:
            obs_probs = np.ones(self.num_observations) / self.num_observations

        # 精确到小数点后一位
        obs_probs = np.round(obs_probs, 1)

        # 最终归一化检查
        final_sum = np.sum(obs_probs)
        if not np.isclose(final_sum, 1.0, atol=0.05):
            diff = 1.0 - final_sum
            max_idx = np.argmax(obs_probs)
            obs_probs[max_idx] += diff
            obs_probs = np.clip(obs_probs, MIN_PROB, MAX_PROB)
            obs_probs = obs_probs / np.sum(obs_probs)
            obs_probs = np.round(obs_probs, 1)

        return obs_probs

    def _validate_observation_matrix(self, observation_matrix, node):
        """验证观察矩阵的有效性"""
        print(f"👁️ 验证节点 {node} 的观测概率矩阵...")

        for s in range(self.num_states):
            for a in range(min(3, self.num_actions)):  # 只验证前3个动作
                probs = observation_matrix[s, a, :]
                prob_sum = np.sum(probs)

                if not np.isclose(prob_sum, 1.0, rtol=1e-2):
                    print(f"    ⚠️  状态 {s}, 前一动作 {a} - 观测概率和: {prob_sum:.3f}")

                if np.any(probs < 0):
                    print(f"    ⚠️  状态 {s}, 前一动作 {a} - 存在负观测概率")

    def export_pomdp_parameters_to_excel(self, filename=None):
        """
        将POMDP参数导出到Excel文件
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Enhanced_POMDP_Parameters_{timestamp}.xlsx"

        print(f"\n📊 Exporting Enhanced POMDP Parameters to Excel: {filename}")
        print("=" * 70)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: 概述和定义
            self._create_overview_sheet(writer)

            # Sheet 2: 状态转移概率
            self._create_transition_probabilities_sheet(writer)

            # Sheet 3: 观察概率
            self._create_observation_probabilities_sheet(writer)

            # Sheet 4: 矩阵汇总
            self._create_matrix_summary_sheet(writer)

        print(f"✅ Enhanced POMDP parameters exported to: {filename}")
        return filename

    def _create_overview_sheet(self, writer):
        """创建概述和定义sheet"""
        overview_data = [
            ['Enhanced POMDP Parameters for Supply Chain Resilience', ''],
            ['Version', 'Random Experiment - Parameterized Generation'],
            ['Parameters Source', 'All from main function config (no defaults)'],
            ['Generation Method', 'Parameterized formulas (no hardcoded values)'],
            ['', ''],
            ['Model Configuration', ''],
            ['Number of Nodes', self.num_nodes],
            ['Number of Layers', self.layer_info.get('num_layers', 'N/A')],
            ['Number of States |R_k^t|', self.num_states],
            ['Number of Actions |A_k^t|', self.num_actions],
            ['Number of Observations |O_k^t|', self.num_observations],
            ['', ''],
            ['State Space Definitions (R_k^t)', ''],
        ]

        # 添加状态定义
        for state_id, description in self.state_definitions.items():
            overview_data.append([f'State {state_id}', description])

        overview_data.extend([
            ['', ''],
            ['Action Space Definitions (A_k^t)', ''],
        ])

        # 添加行动定义
        for action_id, description in self.action_definitions.items():
            overview_data.append([f'Action {action_id}', description])

        overview_data.extend([
            ['', ''],
            ['Observation Space Definitions (O_k^t)', ''],
        ])

        # 添加观察定义
        for obs_id, description in self.observation_definitions.items():
            overview_data.append([f'Observation {obs_id}', description])

        overview_data.extend([
            ['', ''],
            ['Network Structure', ''],
            ['Network Type', self.layer_info.get('network_type', 'UNKNOWN')],
            ['Total Nodes', self.num_nodes],
            ['Nodes per Layer', str(self.layer_info.get('nodes_per_layer', 'N/A'))],
            ['', ''],
            ['Properties', ''],
            ['Time-Invariant Transitions', 'Yes - P(r^{t+1}|r^t,a^t) same for all periods'],
            ['Time-Invariant Observations', 'Yes - P(o^t|r^t,a^{t-1}) same for all periods'],
            ['Parameter Generation', 'Fully parameterized - no hardcoded values'],
            ['Node Differentiation', 'Hash-based node-specific factors'],
            ['Random Seed', self.seed],
        ])

        overview_df = pd.DataFrame(overview_data, columns=['Parameter', 'Description'])
        overview_df.to_excel(writer, sheet_name='Enhanced_Overview', index=False, header=False)

    def _create_transition_probabilities_sheet(self, writer):
        """创建状态转移概率sheet"""
        transition_data = []

        # 动态生成表头
        header = ['Node_ID', 'Node_Type', 'Current_State', 'Action']
        for i in range(self.num_states):
            header.append(f'Next_State_{i}')
        transition_data.append(header)

        # 数据行
        for node in range(self.num_nodes):
            node_type, _ = self._get_node_characteristics(node)
            transition_matrix = self.transition_probabilities[node]

            for current_state in range(self.num_states):
                for action in range(self.num_actions):
                    row = [
                        node,
                        node_type,
                        current_state,
                        action
                    ]

                    # 添加转移概率
                    for next_state in range(self.num_states):
                        prob = transition_matrix[current_state, action, next_state]
                        row.append(f"{prob:.6f}")

                    transition_data.append(row)

        transition_df = pd.DataFrame(transition_data[1:], columns=transition_data[0])
        transition_df.to_excel(writer, sheet_name='Enhanced_Transitions', index=False)

    def _create_observation_probabilities_sheet(self, writer):
        """创建观察概率sheet"""
        observation_data = []

        # 动态生成表头
        header = ['Node_ID', 'Node_Type', 'Current_State', 'Prev_Action']
        for i in range(self.num_observations):
            header.append(f'Obs_{i}')
        observation_data.append(header)

        # 数据行
        for node in range(self.num_nodes):
            node_type, _ = self._get_node_characteristics(node)
            observation_matrix = self.observation_probabilities[node]

            for current_state in range(self.num_states):
                for prev_action in range(self.num_actions):
                    row = [
                        node,
                        node_type,
                        current_state,
                        prev_action
                    ]

                    # 添加观察概率
                    for obs in range(self.num_observations):
                        prob = observation_matrix[current_state, prev_action, obs]
                        row.append(f"{prob:.6f}")

                    observation_data.append(row)

        observation_df = pd.DataFrame(observation_data[1:], columns=observation_data[0])
        observation_df.to_excel(writer, sheet_name='Enhanced_Observations', index=False)

    def _create_matrix_summary_sheet(self, writer):
        """创建矩阵汇总sheet"""
        summary_data = [
            ['Enhanced POMDP Parameters Summary', ''],
            ['Version', 'Random Experiment - Parameterized'],
            ['', ''],
            ['Matrix Dimensions', ''],
            ['Transition Matrix per Node', f'{self.num_states} × {self.num_actions} × {self.num_states}'],
            ['Observation Matrix per Node', f'{self.num_states} × {self.num_actions} × {self.num_observations}'],
            ['Total Transition Parameters', self.num_nodes * self.num_states * self.num_actions * self.num_states],
            ['Total Observation Parameters',
             self.num_nodes * self.num_states * self.num_actions * self.num_observations],
            ['', ''],
            ['Configuration', ''],
            ['num_nodes', self.num_nodes],
            ['num_states from main config', self.num_states],
            ['num_actions from main config', self.num_actions],
            ['num_observations', self.num_observations],
            ['Random Seed', self.seed],
            ['', ''],
            ['Validation Results', ''],
        ]

        # 添加验证统计
        total_transition_errors = 0
        total_observation_errors = 0

        for node in range(self.num_nodes):
            trans_matrix = self.transition_probabilities[node]
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    if not np.isclose(np.sum(trans_matrix[s, a, :]), 1.0, rtol=1e-5):
                        total_transition_errors += 1

            obs_matrix = self.observation_probabilities[node]
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    if not np.isclose(np.sum(obs_matrix[s, a, :]), 1.0, rtol=1e-5):
                        total_observation_errors += 1

        summary_data.extend([
            ['Transition Probability Errors', total_transition_errors],
            ['Observation Probability Errors', total_observation_errors],
            ['Overall Validation Status',
             'PASSED' if (total_transition_errors + total_observation_errors) == 0 else 'WARNING'],
        ])

        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Enhanced_Summary', index=False, header=False)

    def generate_complete_pomdp_parameters(self, export_excel=True, filename=None):
        """
        生成完整的POMDP参数
        所有关键参数都来自主函数config

        Returns:
            tuple: (pomdp_parameters字典, excel文件路径)
        """
        print("🎯 GENERATING COMPLETE ENHANCED POMDP PARAMETERS")
        print("   🔧 Random Experiment Version - Parameterized Generation")
        print("   All critical parameters from main function config (no defaults)")
        print("=" * 80)

        # 生成转移概率
        transition_probs = self.generate_transition_probabilities()

        # 生成观察概率
        observation_probs = self.generate_observation_probabilities()

        # 组织结果
        pomdp_parameters = {
            'configuration': {
                'num_nodes': self.num_nodes,
                'num_states': self.num_states,
                'num_actions': self.num_actions,
                'num_observations': self.num_observations,
                'state_definitions': self.state_definitions,
                'action_definitions': self.action_definitions,
                'observation_definitions': self.observation_definitions,
                'parameter_source': 'All from main function config',
                'version': 'Random Experiment - Parameterized',
                'generation_method': 'Hash-based parameterized formulas',
                'seed': self.seed
            },
            'transition_probabilities': transition_probs,
            'observation_probabilities': observation_probs,
            'network_info': {
                'num_nodes': self.num_nodes,
                'num_layers': self.layer_info.get('num_layers', 'N/A'),
                'nodes_per_layer': self.layer_info.get('nodes_per_layer', 'N/A'),
                'network_type': self.layer_info.get('network_type', 'UNKNOWN'),
                'network_structure': self.network.tolist()
            }
        }

        # 导出到Excel
        excel_file = None
        if export_excel:
            excel_file = self.export_pomdp_parameters_to_excel(filename)

        print(f"\n🎉 Enhanced POMDP parameters generation completed!")
        print(f"📋 Generated parameters for {self.num_nodes} nodes")
        print(f"🔄 Transition matrices: {self.num_states}×{self.num_actions}×{self.num_states} each")
        print(f"👁️  Observation matrices: {self.num_states}×{self.num_actions}×{self.num_observations} each")
        if excel_file:
            print(f"📊 Excel export: {excel_file}")

        return pomdp_parameters, excel_file


def main():
    """
    主函数演示 - 使用R1_network_generate4（支持多层随机网络）
    所有参数都必须显式传入

    Current Date and Time (UTC): 2025-10-28 12:48:41
    Current User's Login: dyy21zyy
    """
    print("🚀 ENHANCED POMDP PARAMETERS GENERATOR FOR SUPPLY CHAIN RESILIENCE")
    print("   Compatible with R1_network_generate4.py (multi-layer random networks)")
    print("   🔧 Random Experiment Version - No Hardcoded Values")
    print("=" * 80)

    # 模拟主函数的配置参数（所有参数必须显式指定）
    main_config = {
        'total_nodes': 15,  # 🔧 使用总节点数
        'num_layers': 4,  # 🔧 4层网络
        'num_periods': 5,
        'num_states': 3,  # 🔧 测试3状态
        'num_actions': 3,
        'connection_density': 0.7,
        'seed': 42
    }

    print(f"\n📋 Main Function Configuration (all explicit, no defaults):")
    for key, value in main_config.items():
        print(f"   {key}: {value}")

    # 生成供应链网络（使用R1_network_generate4）
    print(f"\n🏭 Generating Supply Chain Network using R1_network_generate4...")
    network_data = generate_supply_chain_network(
        total_nodes=main_config['total_nodes'],  # 🔧 修改点
        num_layers=main_config['num_layers'],  # 🔧 修改点
        num_periods=main_config['num_periods'],
        num_states=main_config['num_states'],
        connection_density=main_config['connection_density'],
        seed=main_config['seed'],
        network_type='random',  # 🔧 修改点
        verbose=False
    )

    # 创建POMDP参数生成器
    print(f"\n🔧 Initializing Enhanced POMDP Parameters Generator...")
    pomdp_generator = POMDPParametersGenerator(
        network_data=network_data,
        num_states=main_config['num_states'],
        num_actions=main_config['num_actions'],
        seed=main_config['seed']
    )

    # 生成完整的POMDP参数
    print(f"\n🔧 Generating Enhanced POMDP Parameters...")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"Enhanced_SupplyChain_POMDP_Random_{timestamp}.xlsx"

    pomdp_params, excel_file = pomdp_generator.generate_complete_pomdp_parameters(
        export_excel=True,
        filename=excel_filename
    )

    # 显示一个节点的转移概率示例
    sample_node = 0
    sample_transition = pomdp_params['transition_probabilities'][sample_node]

    print(f"\n🔍 Transition Probabilities for Node {sample_node}:")
    print("P(next_state | current_state, action):")

    for current_state in range(min(2, pomdp_generator.num_states)):
        for action in range(min(3, pomdp_generator.num_actions)):
            probs = sample_transition[current_state, action, :]
            state_desc = pomdp_generator.state_definitions[current_state]
            action_desc = pomdp_generator.action_definitions[action]

            print(f"  Current: {current_state}({state_desc[:10]}...), Action: {action}({action_desc[:15]}...)")
            print(f"    → {[f'{p:.3f}' for p in probs]}")

    return pomdp_generator, pomdp_params, excel_file


if __name__ == "__main__":
    try:
        generator, parameters, excel_file = main()
        print(f"\n✅ Success!")
        print(f"📁 Excel file: {excel_file}")
        print(f"🔧 Fully compatible with random experiment framework!")
    except Exception as e:
        import traceback

        print(f"\n❌ Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()