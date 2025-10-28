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


def create_three_layer_network(num_suppliers, num_manufacturers, connection_density, seed):
    """
    为case study创建固定三层供应链网络

    固定连接模式：
    - 供应商1 → 制造商1
    - 供应商2 → 制造商1
    - 供应商3 → 制造商2
    - 制造商1 → 零售商
    - 制造商2 → 零售商

    Args:
        num_suppliers: 第一层供应商数量（必须为3）
        num_manufacturers: 第二层制造商数量（必须为2）
        connection_density: 连接密度(此参数在case study中被忽略)
        seed: 随机种子（保持兼容性）

    Returns:
        network: 邻接矩阵
        layer_info: 层信息字典
    """
    # 🔴 修改点1：验证case study的参数要求
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

    # 🔴 修改点2：层的索引定义（固定结构）
    # 供应商: 节点 0, 1, 2
    # 制造商: 节点 3, 4
    # 零售商: 节点 5
    layer1_start, layer1_end = 0, 3  # 供应商 [0,1,2]
    layer2_start, layer2_end = 3, 5  # 制造商 [3,4]
    layer3_start, layer3_end = 5, 6  # 零售商 [5]

    # 🔴 修改点3：固定的供应商到制造商连接
    print("    Creating fixed connections for case study:")

    # 供应商1 (节点0) → 制造商1 (节点3)
    network[0, 3] = 1
    print("      Supplier 0 → Manufacturer 3")

    # 供应商2 (节点1) → 制造商1 (节点3)
    network[1, 3] = 1
    print("      Supplier 1 → Manufacturer 3")

    # 供应商3 (节点2) → 制造商2 (节点4)
    network[2, 4] = 1
    print("      Supplier 2 → Manufacturer 4")

    # 🔴 修改点4：固定的制造商到零售商连接
    # 制造商1 (节点3) → 零售商 (节点5)
    network[3, 5] = 1
    print("      Manufacturer 3 → Retailer 5")

    # 制造商2 (节点4) → 零售商 (节点5)
    network[4, 5] = 1
    print("      Manufacturer 4 → Retailer 5")

    # 🔴 修改点5：更新layer_info以匹配固定结构
    layer_info = {
        'num_nodes': num_nodes,
        'layer1': (layer1_start, layer1_end, 'Suppliers'),
        'layer2': (layer2_start, layer2_end, 'Manufacturers'),
        'layer3': (layer3_start, layer3_end, 'Retailer'),
        'num_suppliers': num_suppliers,
        'num_manufacturers': num_manufacturers,
        'network_type': 'CASE_STUDY_FIXED'  # 🔴 新增：标识这是case study版本
    }

    return network, layer_info


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
    # 🔴 修改点1: 验证参数有效性，不允许 num_periods = 1
    if num_periods < 2:
        raise ValueError(f"num_periods must be at least 2 (no single period allowed), got {num_periods}")

    num_nodes = layer_info['num_nodes']
    # 🔴 修改点2: 总时空节点数包含虚拟时间层-1，即 (num_periods + 1) 个时间层
    total_temporal_nodes = num_nodes * (num_periods + 1)

    # 初始化时空网络
    temporal_network = np.zeros([total_temporal_nodes, total_temporal_nodes], dtype=int)

    # 创建节点映射信息
    temporal_node_info = {}

    # 🔴 修改点3: 时间层从-1开始，到num_periods-1结束
    for t in range(-1, num_periods):
        for k in range(num_nodes):
            # 🔴 修改点4: 时空节点ID的计算方式，t=-1对应索引0
            temporal_node_id = (t + 1) * num_nodes + k
            temporal_node_info[temporal_node_id] = {
                'period': t,  # 🔴 修改点5: period可以为-1
                'original_node': k,
                'layer': None
            }

            # 确定节点所属层
            if layer_info['layer1'][0] <= k < layer_info['layer1'][1]:
                temporal_node_info[temporal_node_id]['layer'] = 1
                temporal_node_info[temporal_node_id]['type'] = 'Supplier'
            elif layer_info['layer2'][0] <= k < layer_info['layer2'][1]:
                temporal_node_info[temporal_node_id]['layer'] = 2
                temporal_node_info[temporal_node_id]['type'] = 'Manufacturer'
            else:
                temporal_node_info[temporal_node_id]['layer'] = 3
                temporal_node_info[temporal_node_id]['type'] = 'Retailer'

    # 🔴 修改点6: 构建时空网络连接逻辑
    print(f"   Creating temporal network with virtual period -1 (num_periods = {num_periods})")

    # 时间层从-1到num_periods-1
    for t in range(-1, num_periods):
        current_period_offset = (t + 1) * num_nodes

        if t == -1:  # 🔴 修改点7: Period -1：只有空间连接
            print("     Period -1 (Virtual): Adding spatial connections only")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if network[i, j] == 1:
                        temporal_network[current_period_offset + i, current_period_offset + j] = 1

        else:  # 🔴 修改点8: Period 0及以后：空间连接 + 时间连接
            print(f"     Period {t}: Adding spatial + temporal connections")

            # 当前period的空间连接
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if network[i, j] == 1:
                        temporal_network[current_period_offset + i, current_period_offset + j] = 1

            # 🔴 修改点9: 时间连接：从上一时间层到当前时间层
            previous_period_offset = t * num_nodes  # 上一层的offset

            for k in range(num_nodes):
                current_node = current_period_offset + k
                previous_self = previous_period_offset + k

                # 所有节点都连接到上一周期的自身
                temporal_network[previous_self, current_node] = 1

    return temporal_network, temporal_node_info


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
    # 🔴 修改点10: 修改参数验证，不允许单周期
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
    print(f"   Suppliers: {layer_info['num_suppliers']}")
    print(f"   Manufacturers: {layer_info['num_manufacturers']}")
    print(f"   Retailers: 1")
    if num_periods is not None:
        print(f"   Periods: {num_periods} (from main config)")

    if temporal_network is not None and num_periods is not None:
        print(f"\n🕐 Temporal Network Structure:")
        print(f"   Temporal network shape: {temporal_network.shape}")
        print(f"   Total temporal nodes: {len(temporal_node_info)}")

        # 🔴 修改点11: 更新打印逻辑
        print(f"   Structure: Multi-period temporal network with virtual period -1")
        print(f"     - Period -1 (Virtual): Spatial connections only")
        print(f"     - Periods 0 to {num_periods-1}: Spatial + temporal connections")

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
    # 🔴 修改点12: 期望的时空节点数包含虚拟层-1
    expected_temporal_nodes = num_nodes * (num_periods + 1)

    # 验证基本维度
    assert network.shape[0] == network.shape[1] == num_nodes, f"Spatial network dimension mismatch"
    assert temporal_network.shape[0] == temporal_network.shape[
        1] == expected_temporal_nodes, f"Temporal network dimension mismatch"

    # 🔴 修改点13: 验证时空结构逻辑，不再有单周期情况
    # 验证各周期结构（包含虚拟层-1）
    for t in range(-1, num_periods):
        period_offset = (t + 1) * num_nodes
        period_spatial = temporal_network[period_offset:period_offset + num_nodes,
                         period_offset:period_offset + num_nodes]

        # 每个周期的空间结构都应该相同
        assert np.array_equal(period_spatial, network), f"Period {t} spatial structure mismatch"

        if t >= 0:  # 🔴 修改点14: 从period 0开始检查时间连接
            # 检查时间连接
            prev_offset = t * num_nodes
            for k in range(num_nodes):
                prev_node = prev_offset + k
                curr_node = period_offset + k
                assert temporal_network[
                           prev_node, curr_node] == 1, f"Missing temporal connection from period {t-1} to {t}, node {k}"

    print(f"   ✅ Multi-period structure with virtual layer validated ({num_periods} periods + virtual layer -1)")
    print("   ✅ Network structure validation passed")


def generate_supply_chain_network(num_suppliers, num_manufacturers, num_periods,
                                             num_states, connection_density, seed, verbose=False):
    """
    🔴 修改版本：为case study生成固定结构的供应链网络

    固定网络结构：
    - 供应商1,2 → 制造商1
    - 供应商3 → 制造商2
    - 制造商1,2 → 零售商

    Args:
        num_suppliers: 供应商数量（必须为3）
        num_manufacturers: 制造商数量（必须为2）
        num_periods: 时间周期数（从主函数传入，必须）
        num_states: 状态数量（从主函数传入，必须）
        connection_density: 连接密度（在case study中被忽略）
        seed: 随机种子（从主函数传入，必须）
        verbose: 是否打印详细信息（可选，默认False）

    Returns:
        tuple: 包含所有网络数据的元组
    """
    # 验证case study参数
    if num_suppliers != 3 or num_manufacturers != 2:
        raise ValueError(f"Case study requires exactly 3 suppliers and 2 manufacturers, "
                         f"got {num_suppliers} suppliers and {num_manufacturers} manufacturers")

    if num_periods < 2:
        raise ValueError(f"num_periods must be at least 2 (no single period allowed), got {num_periods}")

    print(f"🚀 Generating CASE STUDY supply chain network with fixed structure:")
    print(f"   num_suppliers: {num_suppliers} (fixed)")
    print(f"   num_manufacturers: {num_manufacturers} (fixed)")
    print(f"   num_periods: {num_periods} (includes virtual period -1)")
    print(f"   num_states: {num_states}")
    print(f"   connection_density: {connection_density} (ignored in case study)")
    print(f"   seed: {seed}")
    print(f"   🎯 Fixed connection pattern:")
    print(f"      Suppliers [0,1] → Manufacturer 3")
    print(f"      Supplier 2 → Manufacturer 4")
    print(f"      Manufacturers [3,4] → Retailer 5")

    # 🔴 修改点6：使用case study版本的网络生成函数
    network, layer_info = create_three_layer_network(
        num_suppliers, num_manufacturers, connection_density, seed
    )

    # 其余部分保持不变，使用原有的时空网络生成逻辑
    temporal_network, temporal_node_info = create_temporal_network(
        network, layer_info, num_periods
    )

    # 验证网络结构的正确性
    validate_network_structure(network, temporal_network, num_periods, layer_info)

    # 获取父子关系
    parent_dict = get_parent_child_relationships(temporal_network, temporal_node_info, num_periods)

    # 节点分区
    independent_nodes, other_nodes, parent_node_dic = node_partition(network)

    # 生成状态组合字典
    C_dic, G_dic = generate_state_combinations(network, num_states)

    if verbose:
        print_network_summary(network, layer_info, temporal_network, temporal_node_info, C_dic, G_dic, num_periods)
        print_combination_details(C_dic, G_dic)



    print("✅ Case study supply chain network generation completed successfully")

    return (network, layer_info, temporal_network, temporal_node_info,
            parent_dict, independent_nodes, other_nodes, parent_node_dic, C_dic, G_dic)


if __name__ == "__main__":
    print("🧪 Testing R1 Network Generator - CASE STUDY VERSION")
    print("Current Date and Time (UTC): 2025-09-14 13:28:02")
    print("Current User's Login: dyy21zyy")
    print("=" * 80)

    # 🔴 修改点8：case study测试配置
    case_study_config = {
        'num_suppliers': 3,        # 固定为3
        'num_manufacturers': 2,    # 固定为2
        'num_periods': 4,          # 可调整
        'num_states': 2,          # 可调整
        'connection_density': 0.7, # 被忽略
        'seed': 42
    }

    print(f"\n🎯 CASE STUDY Configuration:")
    print("-" * 60)
    print("📋 Fixed network structure parameters:")
    for param, value in case_study_config.items():
        print(f"   {param}: {value}")

    # 🔴 修改点9：使用case study版本
    results = generate_supply_chain_network(
        num_suppliers=case_study_config['num_suppliers'],
        num_manufacturers=case_study_config['num_manufacturers'],
        num_periods=case_study_config['num_periods'],
        num_states=case_study_config['num_states'],
        connection_density=case_study_config['connection_density'],
        seed=case_study_config['seed'],
        verbose=True
    )

    print(f"\n🎉 Case study network generation completed!")
    print("=" * 80)