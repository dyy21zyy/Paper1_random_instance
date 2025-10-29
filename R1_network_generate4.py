"""
R1_network_generate3.py - 修改版
支持 n≥3 层随机供应链网络生成

Current Date and Time (UTC): 2025-10-28 12:43:39
Current User's Login: dyy21zyy

主要修改：
1. 支持 n≥3 层网络结构
2. 随机连接生成（替代固定连接）
3. connection_density 参数实际生效
4. 保证网络连通性
5. 最后一层固定为单个零售商
6. 保留 Case Study 版本（向后兼容）
"""

import numpy as np
import random
from copy import deepcopy


def comb_generator(state_list, res, num_nodes, comb):
    """
    递归生成状态组合
    """
    if len(res) == num_nodes:
        comb.append(deepcopy(res))
        return

    for i in range(len(state_list)):
        for state in state_list[i]:
            res.append(state)
            next_state_list = state_list[i + 1:] if i < len(state_list) - 1 else []
            comb_generator(next_state_list, res, num_nodes, comb)
            del res[-1]


def combinations(state_list):
    """
    生成所有可能的状态组合
    """
    num_nodes = len(state_list)
    comb = []
    res = []
    comb_generator(state_list, res, num_nodes, comb)
    return comb


# ============================================================================
# ✅ 新增：多层随机网络生成
# ============================================================================

def allocate_nodes_to_layers(total_nodes, num_layers, min_nodes_per_layer=2):
    """
    将总节点数分配到各层（除最后一层外）

    策略：
    1. 最后一层固定为1（零售商）
    2. 其余层节点数递减（上游多，下游少）
    3. 每层至少 min_nodes_per_layer 个节点

    参数：
        total_nodes: 总节点数（包括零售商）
        num_layers: 层数（≥3）
        min_nodes_per_layer: 每层最小节点数（除零售商外）

    返回：
        nodes_per_layer: [layer1_nodes, layer2_nodes, ..., 1]

    示例：
        total_nodes=15, num_layers=4
        → [6, 4, 3, 2, 1] = 16? 不对
        → 正确：[5, 4, 3, 2, 1] = 15
    """
    if num_layers < 3:
        raise ValueError(f"num_layers must be ≥3, got {num_layers}")

    if total_nodes < num_layers:
        raise ValueError(f"total_nodes ({total_nodes}) must be ≥ num_layers ({num_layers})")

    # 零售商占1个节点
    retailer_nodes = 1
    remaining_nodes = total_nodes - retailer_nodes

    # 剩余层数
    remaining_layers = num_layers - 1

    # 检查是否满足最小节点数要求
    min_required = remaining_layers * min_nodes_per_layer
    if remaining_nodes < min_required:
        raise ValueError(
            f"total_nodes ({total_nodes}) too small for {num_layers} layers "
            f"with min {min_nodes_per_layer} nodes/layer. "
            f"Need at least {min_required + 1} nodes."
        )

    # 分配策略：递减分配
    nodes_per_layer = []

    # 计算基础节点数（均匀分配）
    base_nodes = remaining_nodes // remaining_layers
    extra_nodes = remaining_nodes % remaining_layers

    # 前面的层多分配一些（上游节点更多）
    for i in range(remaining_layers):
        if i < extra_nodes:
            nodes_per_layer.append(base_nodes + 1)
        else:
            nodes_per_layer.append(base_nodes)

    # 排序：递减（可选，让上游节点更多）
    nodes_per_layer.sort(reverse=True)

    # 添加零售商层
    nodes_per_layer.append(retailer_nodes)

    # 验证
    assert sum(nodes_per_layer) == total_nodes, f"Node allocation error: {nodes_per_layer} != {total_nodes}"
    assert all(n >= min_nodes_per_layer for n in nodes_per_layer[:-1]), f"Layer too small: {nodes_per_layer}"

    return nodes_per_layer


def create_multi_layer_random_network(nodes_per_layer, connection_density, seed):
    """
    ✅ 核心函数：创建多层随机供应链网络

    参数：
        nodes_per_layer: 每层节点数列表，例如 [5, 4, 3, 2, 1]
        connection_density: 连接密度 [0.0, 1.0]
        seed: 随机种子

    返回：
        network: 邻接矩阵
        layer_info: 层信息字典

    算法：
        1. 初始化邻接矩阵
        2. 逐层建立随机连接（Layer i → Layer i+1）
        3. 保证每个节点至少有一个入边（除第一层外）
        4. 返回网络和层信息
    """
    np.random.seed(seed)
    random.seed(seed)

    num_layers = len(nodes_per_layer)
    total_nodes = sum(nodes_per_layer)

    print(f"    Creating {num_layers}-layer random network:")
    print(f"      Nodes per layer: {nodes_per_layer}")
    print(f"      Total nodes: {total_nodes}")
    print(f"      Connection density: {connection_density}")

    # 初始化网络
    network = np.zeros([total_nodes, total_nodes], dtype=int)

    # 计算每层的节点索引范围
    layer_ranges = []
    start_idx = 0
    for num_nodes in nodes_per_layer:
        end_idx = start_idx + num_nodes
        layer_ranges.append((start_idx, end_idx))
        start_idx = end_idx

    # 逐层建立连接
    for layer_idx in range(num_layers - 1):
        source_layer = layer_ranges[layer_idx]
        target_layer = layer_ranges[layer_idx + 1]

        source_nodes = list(range(source_layer[0], source_layer[1]))
        target_nodes = list(range(target_layer[0], target_layer[1]))

        print(f"      Connecting Layer {layer_idx + 1} {source_nodes} → Layer {layer_idx + 2} {target_nodes}")

        # 随机连接
        for source in source_nodes:
            for target in target_nodes:
                if np.random.rand() < connection_density:
                    network[source, target] = 1

        # 保证连通性：每个目标节点至少有一个入边
        for target in target_nodes:
            if network[:, target].sum() == 0:
                # 随机选择一个源节点连接
                random_source = np.random.choice(source_nodes)
                network[random_source, target] = 1
                print(f"        ⚠️  Forced connection: {random_source} → {target} (connectivity)")

    # 构建 layer_info
    layer_info = {
        'num_nodes': total_nodes,
        'num_layers': num_layers,
        'nodes_per_layer': nodes_per_layer,
        'network_type': 'MULTI_LAYER_RANDOM'
    }

    # 添加每层的详细信息
    for i, (start, end) in enumerate(layer_ranges):
        if i == num_layers - 1:
            layer_name = 'Retailer'
        elif i == 0:
            layer_name = 'Suppliers'
        else:
            layer_name = f'Intermediate_{i}'

        layer_info[f'layer{i + 1}'] = (start, end, layer_name)

    # 为了兼容性，添加旧的键名
    layer_info['layer1'] = layer_info['layer1']  # Suppliers
    if num_layers >= 3:
        layer_info['layer2'] = layer_info['layer2']  # First intermediate
        layer_info['layer3'] = layer_info[f'layer{num_layers}']  # Retailer

    # 添加供应商和制造商数量（兼容性）
    layer_info['num_suppliers'] = nodes_per_layer[0]
    layer_info['num_manufacturers'] = sum(nodes_per_layer[1:-1])  # 所有中间层

    print(f"    ✓ Random network created: {total_nodes} nodes, {network.sum()} edges")

    return network, layer_info


# ============================================================================
# ⚪ 保留：Case Study 固定网络（向后兼容）
# ============================================================================

def create_case_study_fixed_network(num_suppliers, num_manufacturers,
                                    connection_density, seed):
    """
    为 Case Study 创建固定三层供应链网络

    固定连接模式：
    - 供应商1,2 → 制造商1
    - 供应商3 → 制造商2
    - 制造商1,2 → 零售商

    参数：
        num_suppliers: 第一层供应商数量（必须为3）
        num_manufacturers: 第二层制造商数量（必须为2）
        connection_density: 连接密度(此参数在case study中被忽略)
        seed: 随机种子（保持兼容性）

    返回：
        network: 邻接矩阵
        layer_info: 层信息字典
    """
    # 验证 case study 的参数要求
    if num_suppliers != 3:
        raise ValueError(f"Case study requires exactly 3 suppliers, got {num_suppliers}")
    if num_manufacturers != 2:
        raise ValueError(f"Case study requires exactly 2 manufacturers, got {num_manufacturers}")

    # 设置随机种子（保持兼容性，虽然这里是固定连接）
    np.random.seed(seed)
    random.seed(seed)

    # 总节点数 = 3供应商 + 2制造商 + 1零售商
    num_nodes = 6

    # 初始化网络
    network = np.zeros([num_nodes, num_nodes], dtype=int)

    # 层的索引定义（固定结构）
    # 供应商: 节点 0, 1, 2
    # 制造商: 节点 3, 4
    # 零售商: 节点 5
    layer1_start, layer1_end = 0, 3  # 供应商 [0,1,2]
    layer2_start, layer2_end = 3, 5  # 制造商 [3,4]
    layer3_start, layer3_end = 5, 6  # 零售商 [5]

    print("    Creating fixed connections for case study:")

    # 固定的供应商到制造商连接
    # 供应商1 (节点0) → 制造商1 (节点3)
    network[0, 3] = 1
    print("      Supplier 0 → Manufacturer 3")

    # 供应商2 (节点1) → 制造商1 (节点3)
    network[1, 3] = 1
    print("      Supplier 1 → Manufacturer 3")

    # 供应商3 (节点2) → 制造商2 (节点4)
    network[2, 4] = 1
    print("      Supplier 2 → Manufacturer 4")

    # 固定的制造商到零售商连接
    # 制造商1 (节点3) → 零售商 (节点5)
    network[3, 5] = 1
    print("      Manufacturer 3 → Retailer 5")

    # 制造商2 (节点4) → 零售商 (节点5)
    network[4, 5] = 1
    print("      Manufacturer 4 → Retailer 5")

    # 更新 layer_info 以匹配固定结构
    layer_info = {
        'num_nodes': num_nodes,
        'num_layers': 3,
        'nodes_per_layer': [3, 2, 1],
        'layer1': (layer1_start, layer1_end, 'Suppliers'),
        'layer2': (layer2_start, layer2_end, 'Manufacturers'),
        'layer3': (layer3_start, layer3_end, 'Retailer'),
        'num_suppliers': num_suppliers,
        'num_manufacturers': num_manufacturers,
        'network_type': 'CASE_STUDY_FIXED'
    }

    return network, layer_info


# ============================================================================
# ⚪ 保持不变：时空网络展开
# ============================================================================

def create_temporal_network(network, layer_info, num_periods):
    """
    创建时空网络结构

    重要逻辑修正：设立一个虚拟的时间层-1，
    - 不存在 num_periods = 1
    - 当 num_periods > 1 时：period -1 只有空间连接，period 0及以后有空间连接+时间连接

    Args:
        network: 空间网络邻接矩阵
        layer_info: 层信息（从主函数传入）
        num_periods: 时间周期数（从主函数传入，必须，无默认值）

    Returns:
        temporal_network: 时空网络邻接矩阵
        temporal_node_info: 时空节点信息
    """
    # 验证参数有效性，不允许 num_periods = 1
    if num_periods < 2:
        raise ValueError(f"num_periods must be at least 2 (no single period allowed), got {num_periods}")

    num_nodes = layer_info['num_nodes']
    # 总时空节点数包含虚拟时间层-1，即 (num_periods + 1) 个时间层
    total_temporal_nodes = num_nodes * (num_periods + 1)

    # 初始化时空网络
    temporal_network = np.zeros([total_temporal_nodes, total_temporal_nodes], dtype=int)

    # 创建节点映射信息
    temporal_node_info = {}

    # 时间层从-1开始，到num_periods-1结束
    for t in range(-1, num_periods):
        for k in range(num_nodes):
            # 时空节点ID的计算方式，t=-1对应索引0
            temporal_node_id = (t + 1) * num_nodes + k
            temporal_node_info[temporal_node_id] = {
                'period': t,
                'original_node': k,
                'layer': None
            }

            # 确定节点所属层（兼容多层网络）
            for layer_idx in range(1, layer_info['num_layers'] + 1):
                layer_key = f'layer{layer_idx}'
                if layer_key in layer_info:
                    start, end, name = layer_info[layer_key]
                    if start <= k < end:
                        temporal_node_info[temporal_node_id]['layer'] = layer_idx
                        temporal_node_info[temporal_node_id]['type'] = name
                        break

    # 构建时空网络连接逻辑
    print(f"   Creating temporal network with virtual period -1 (num_periods = {num_periods})")

    # 时间层从-1到num_periods-1
    for t in range(-1, num_periods):
        current_period_offset = (t + 1) * num_nodes

        if t == -1:  # Period -1：只有空间连接
            print("     Period -1 (Virtual): Adding spatial connections only")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if network[i, j] == 1:
                        temporal_network[current_period_offset + i, current_period_offset + j] = 1

        else:  # Period 0及以后：空间连接 + 时间连接
            print(f"     Period {t}: Adding spatial + temporal connections")

            # 当前period的空间连接
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if network[i, j] == 1:
                        temporal_network[current_period_offset + i, current_period_offset + j] = 1

            # 时间连接：从上一时间层到当前时间层
            previous_period_offset = t * num_nodes

            for k in range(num_nodes):
                current_node = current_period_offset + k
                previous_self = previous_period_offset + k

                # 所有节点都连接到上一周期的自身
                temporal_network[previous_self, current_node] = 1

    return temporal_network, temporal_node_info


# ============================================================================
# ⚪ 保持不变：状态组合生成
# ============================================================================

def generate_state_combinations(network, num_states):
    """
    生成C_dic和G_dic状态组合字典

    Args:
        network: 网络邻接矩阵
        num_states: 状态数量（从主函数传入，无默认值）

    Returns:
        C_dic: 空间父节点状态组合字典
        G_dic: 时空父节点状态组合字典（包含时间维度）
    """
    if num_states < 1:
        raise ValueError(f"num_states must be at least 1, got {num_states}")

    num_nodes = network.shape[0]

    # 生成C_dic (空间父节点组合)
    C_dic = {}
    for i in range(num_nodes):
        num_parent_nodes = network[:, i].sum()
        if num_parent_nodes > 0:
            state_list = [list(range(num_states)) for j in range(int(num_parent_nodes))]
            c = combinations(state_list)
            C_dic[i] = c

    # 生成G_dic (时空父节点组合，包含时间维度)
    G_dic = {}
    for i in range(num_nodes):
        num_parent_nodes = network[:, i].sum() + 1  # +1 for temporal dimension
        if num_parent_nodes >= 1:
            state_list = [list(range(num_states)) for j in range(int(num_parent_nodes))]
            g = combinations(state_list)
            G_dic[i] = g

    return C_dic, G_dic


def get_parent_child_relationships(temporal_network, temporal_node_info, num_periods):
    """
    获取父子节点关系

    Args:
        temporal_network: 时空网络（基于主函数参数生成）
        temporal_node_info: 时空节点信息
        num_periods: 周期数（从主函数传入，无默认值）

    Returns:
        parent_dict: {child_node: [parent_nodes]} 的字典
    """
    if num_periods < 2:
        raise ValueError(f"num_periods must be at least 2, got {num_periods}")

    parent_dict = {}

    for child_node in temporal_node_info.keys():
        parents = []
        for parent_node in temporal_node_info.keys():
            if temporal_network[parent_node, child_node] == 1:
                parents.append(parent_node)
        parent_dict[child_node] = parents

    return parent_dict


def node_partition(network):
    """
    节点分区函数
    """
    independent_nodes = []
    other_nodes = []
    num_nodes = network.shape[0]
    parent_node_dic = {}

    for i in range(num_nodes):
        if network[:, i].sum() == 0:
            independent_nodes.append(i)
        else:
            other_nodes.append(i)
            parent_node = []
            for j in range(num_nodes):
                if network[j, i] == 1:
                    parent_node.append(j)
            parent_node_dic[i] = parent_node

    return independent_nodes, other_nodes, parent_node_dic


# ============================================================================
# ⚪ 保持不变：打印函数
# ============================================================================

def print_network_summary(network, layer_info, temporal_network=None, temporal_node_info=None,
                          C_dic=None, G_dic=None, num_periods=None):
    """
    打印网络摘要信息，包含时空网络结构的详细说明
    """
    print("\n" + "=" * 80)
    print("🏭 SUPPLY CHAIN NETWORK GENERATION SUMMARY")
    print("=" * 80)

    print(f"📊 Network Configuration (All from Main Function):")
    print(f"   Spatial network shape: {network.shape}")
    print(f"   Total nodes: {layer_info['num_nodes']}")
    print(f"   Number of layers: {layer_info['num_layers']}")
    print(f"   Nodes per layer: {layer_info['nodes_per_layer']}")
    print(f"   Network type: {layer_info.get('network_type', 'UNKNOWN')}")

    if num_periods is not None:
        print(f"   Periods: {num_periods} (from main config)")

    if temporal_network is not None and num_periods is not None:
        print(f"\n🕐 Temporal Network Structure:")
        print(f"   Temporal network shape: {temporal_network.shape}")
        print(f"   Total temporal nodes: {len(temporal_node_info)}")

        print(f"   Structure: Multi-period temporal network with virtual period -1")
        print(f"     - Period -1 (Virtual): Spatial connections only")
        print(f"     - Periods 0 to {num_periods - 1}: Spatial + temporal connections")

    if C_dic is not None:
        print(f"\n🔗 Spatial Parent Combinations (C_dic):")
        print(f"   Nodes with spatial parents: {list(C_dic.keys())}")
        for key, value in C_dic.items():
            print(f"     Node {key}: {len(value)} spatial parent combinations")

    if G_dic is not None:
        print(f"\n⏰ Temporal Parent Combinations (G_dic):")
        print(f"   Nodes with temporal parents: {list(G_dic.keys())}")
        for key, value in G_dic.items():
            print(f"     Node {key}: {len(value)} temporal parent combinations")


def print_combination_details(C_dic, G_dic, max_display=5):
    """
    打印状态组合的详细信息
    """
    print("\n" + "=" * 60)
    print("📋 STATE COMBINATION DETAILS")
    print("=" * 60)

    if C_dic:
        print("\n🔗 C_dic (Spatial Parent Combinations):")
        for node, combinations in C_dic.items():
            print(f"\nNode {node}: {len(combinations)} combinations")
            if combinations:
                print(f"  First {min(max_display, len(combinations))} combinations:")
                for i, combo in enumerate(combinations[:max_display]):
                    print(f"    {i + 1}: {combo}")
                if len(combinations) > max_display:
                    print(f"    ... and {len(combinations) - max_display} more")

    if G_dic:
        print("\n⏰ G_dic (Temporal Parent Combinations):")
        for node, combinations in G_dic.items():
            print(f"\nNode {node}: {len(combinations)} combinations")
            if combinations:
                print(f"  First {min(max_display, len(combinations))} combinations:")
                for i, combo in enumerate(combinations[:max_display]):
                    print(f"    {i + 1}: {combo}")
                if len(combinations) > max_display:
                    print(f"    ... and {len(combinations) - max_display} more")


def validate_network_structure(network, temporal_network, num_periods, layer_info):
    """
    验证网络结构的正确性
    """
    print("\n🔍 Validating network structure...")

    num_nodes = layer_info['num_nodes']
    expected_temporal_nodes = num_nodes * (num_periods + 1)

    # 验证基本维度
    assert network.shape[0] == network.shape[1] == num_nodes, f"Spatial network dimension mismatch"
    assert temporal_network.shape[0] == temporal_network.shape[1] == expected_temporal_nodes, \
        f"Temporal network dimension mismatch"

    # 验证各周期结构（包含虚拟层-1）
    for t in range(-1, num_periods):
        period_offset = (t + 1) * num_nodes
        period_spatial = temporal_network[period_offset:period_offset + num_nodes,
                         period_offset:period_offset + num_nodes]

        # 每个周期的空间结构都应该相同
        assert np.array_equal(period_spatial, network), f"Period {t} spatial structure mismatch"

        if t >= 0:  # 从period 0开始检查时间连接
            # 检查时间连接
            prev_offset = t * num_nodes
            for k in range(num_nodes):
                prev_node = prev_offset + k
                curr_node = period_offset + k
                assert temporal_network[prev_node, curr_node] == 1, \
                    f"Missing temporal connection from period {t - 1} to {t}, node {k}"

    print(f"   ✅ Multi-period structure with virtual layer validated ({num_periods} periods + virtual layer -1)")
    print("   ✅ Network structure validation passed")


# ============================================================================
# 🔧 修改：主接口函数
# ============================================================================

def generate_supply_chain_network(num_suppliers=None, num_manufacturers=None,
                                  num_periods=None, num_states=None,
                                  connection_density=None, seed=None,
                                  network_type='random',
                                  total_nodes=None, num_layers=None,
                                  nodes_per_layer=None,
                                  verbose=False):
    """
    🔧 主接口：生成供应链网络（支持多种配置方式）

    Current Date and Time (UTC): 2025-10-28 12:43:39
    Current User's Login: dyy21zyy

    配置方式：

    方式1：Case Study（固定网络）
        network_type='case_study_fixed'
        num_suppliers=3
        num_manufacturers=2

    方式2：三层随机网络（兼容原接口）
        network_type='random'
        num_suppliers=5
        num_manufacturers=4
        → 自动生成 [5, 4, 1] 三层网络

    方式3：多层随机网络（指定总节点数）
        network_type='random'
        total_nodes=15
        num_layers=4
        → 自动分配：[5, 4, 3, 2, 1]

    方式4：多层随机网络（手动指定）
        network_type='random'
        nodes_per_layer=[6, 5, 4, 3, 1]
        → 直接使用指定配置

    参数：
        num_suppliers: 供应商数量（方式1,2）
        num_manufacturers: 制造商数量（方式1,2）
        num_periods: 时间周期数（必须）
        num_states: 状态数量（必须）
        connection_density: 连接密度（必须）
        seed: 随机种子（必须）
        network_type: 'random' | 'case_study_fixed'
        total_nodes: 总节点数（方式3）
        num_layers: 层数（方式3）
        nodes_per_layer: 每层节点数列表（方式4）
        verbose: 是否打印详细信息

    返回：
        tuple: 包含所有网络数据的元组
    """
    # 参数验证
    if num_periods is None or num_states is None or connection_density is None or seed is None:
        raise ValueError("num_periods, num_states, connection_density, seed are required")

    if num_periods < 2:
        raise ValueError(f"num_periods must be at least 2, got {num_periods}")

    print(f"🚀 Generating supply chain network:")
    print(f"   network_type: {network_type}")
    print(f"   num_periods: {num_periods}")
    print(f"   num_states: {num_states}")
    print(f"   connection_density: {connection_density}")
    print(f"   seed: {seed}")

    # ========================================
    # 步骤1：生成空间网络
    # ========================================

    if network_type == 'case_study_fixed':
        # 方式1：Case Study 固定网络
        if num_suppliers is None or num_manufacturers is None:
            raise ValueError("num_suppliers and num_manufacturers required for case_study_fixed")

        print(f"   Using Case Study fixed network: {num_suppliers} suppliers + {num_manufacturers} manufacturers")
        network, layer_info = create_case_study_fixed_network(
            num_suppliers, num_manufacturers, connection_density, seed
        )

    elif nodes_per_layer is not None:
        # 方式4：手动指定每层节点数
        print(f"   Using manual layer configuration: {nodes_per_layer}")

        if nodes_per_layer[-1] != 1:
            raise ValueError(f"Last layer must be 1 (retailer), got {nodes_per_layer[-1]}")

        network, layer_info = create_multi_layer_random_network(
            nodes_per_layer, connection_density, seed
        )

    elif total_nodes is not None and num_layers is not None:
        # 方式3：指定总节点数和层数
        print(f"   Auto-allocating {total_nodes} nodes to {num_layers} layers")

        nodes_per_layer = allocate_nodes_to_layers(total_nodes, num_layers, min_nodes_per_layer=2)
        print(f"   Allocated: {nodes_per_layer}")

        network, layer_info = create_multi_layer_random_network(
            nodes_per_layer, connection_density, seed
        )

    elif num_suppliers is not None and num_manufacturers is not None:
        # 方式2：三层随机网络（兼容原接口）
        nodes_per_layer = [num_suppliers, num_manufacturers, 1]
        print(f"   Using 3-layer random network: {nodes_per_layer}")

        network, layer_info = create_multi_layer_random_network(
            nodes_per_layer, connection_density, seed
        )

    else:
        raise ValueError(
            "Invalid configuration. Please provide one of:\n"
            "  1. num_suppliers + num_manufacturers (3-layer)\n"
            "  2. total_nodes + num_layers (auto-allocate)\n"
            "  3. nodes_per_layer (manual)\n"
            "  4. network_type='case_study_fixed' + num_suppliers + num_manufacturers"
        )

    # ========================================
    # 步骤2：生成时空网络
    # ========================================

    temporal_network, temporal_node_info = create_temporal_network(
        network, layer_info, num_periods
    )

    # 验证网络结构的正确性
    validate_network_structure(network, temporal_network, num_periods, layer_info)

    # ========================================
    # 步骤3：获取父子关系
    # ========================================

    parent_dict = get_parent_child_relationships(temporal_network, temporal_node_info, num_periods)

    # ========================================
    # 步骤4：节点分区
    # ========================================

    independent_nodes, other_nodes, parent_node_dic = node_partition(network)

    # ========================================
    # 步骤5：生成状态组合字典
    # ========================================

    C_dic, G_dic = generate_state_combinations(network, num_states)

    # ========================================
    # 步骤6：打印详细信息
    # ========================================

    if verbose:
        print_network_summary(network, layer_info, temporal_network, temporal_node_info, C_dic, G_dic, num_periods)
        print_combination_details(C_dic, G_dic)

    print("✅ Supply chain network generation completed successfully")

    return (network, layer_info, temporal_network, temporal_node_info,
            parent_dict, independent_nodes, other_nodes, parent_node_dic, C_dic, G_dic)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing R1 Network Generator - MULTI-LAYER RANDOM VERSION")
    print("Current Date and Time (UTC): 2025-10-28 12:43:39")
    print("Current User's Login: dyy21zyy")
    print("=" * 80)

    # ========================================
    # 测试1：Case Study（固定网络）
    # ========================================
    print("\n" + "=" * 80)
    print("📌 TEST 1: Case Study Fixed Network")
    print("=" * 80)

    results1 = generate_supply_chain_network(
        num_suppliers=3,
        num_manufacturers=2,
        num_periods=4,
        num_states=2,
        connection_density=0.7,
        seed=42,
        network_type='case_study_fixed',
        verbose=True
    )

    # ========================================
    # 测试2：三层随机网络（兼容原接口）
    # ========================================
    print("\n" + "=" * 80)
    print("📌 TEST 2: 3-Layer Random Network (Compatible)")
    print("=" * 80)

    results2 = generate_supply_chain_network(
        num_suppliers=5,
        num_manufacturers=4,
        num_periods=4,
        num_states=2,
        connection_density=0.7,
        seed=123,
        network_type='random',
        verbose=True
    )

    # ========================================
    # 测试3：多层随机网络（自动分配）
    # ========================================
    print("\n" + "=" * 80)
    print("📌 TEST 3: Multi-Layer Random Network (Auto-Allocate)")
    print("=" * 80)

    results3 = generate_supply_chain_network(
        total_nodes=15,
        num_layers=4,
        num_periods=5,
        num_states=3,
        connection_density=0.65,
        seed=456,
        network_type='random',
        verbose=True
    )

    # ========================================
    # 测试4：多层随机网络（手动指定）
    # ========================================
    print("\n" + "=" * 80)
    print("📌 TEST 4: Multi-Layer Random Network (Manual)")
    print("=" * 80)

    results4 = generate_supply_chain_network(
        nodes_per_layer=[6, 5, 4, 3, 1],
        num_periods=4,
        num_states=2,
        connection_density=0.75,
        seed=789,
        network_type='random',
        verbose=True
    )

    print(f"\n🎉 All tests completed!")
    print("=" * 80)