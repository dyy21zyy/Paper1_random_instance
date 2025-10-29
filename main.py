"""
main_systematic_experiments.py - 系统性实验设计（完善版）

Current Date and Time (UTC): 2025-10-28 18:26:54
Current User's Login: dyy21zyy

适配模块:
- GA3.py (遗传算法求解器)
- R1_solver19.py (Gurobi求解器)
- R1_network_generate4.py (多层随机网络生成)
- R1_para_POMDP4.py (POMDP参数生成)
- R1_prediction_inputDBN13.py (贝叶斯状态预测)

功能特性:
1. 多种实验类型（单因素、双因素、全因子）
2. 统一管理实验参数，确保GA和Gurobi求解相同问题
3. 自动生成观测数据（基于disruption级别）
4. 支持多次重复实验降低随机性
5. 生成详细对比报告（Excel格式）
6. 中间结果自动保存
7. 实验进度实时显示
8. 异常处理和恢复机制
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import json
import os
import sys
import traceback
from pathlib import Path


class SystematicExperimentManager:
    """
    系统性实验管理器

    负责管理整个实验流程，确保GA和Gurobi在相同条件下进行对比
    """

    def __init__(self, experiment_type='full_factorial', output_dir='./experiment_results'):
        """
        初始化实验管理器

        Args:
            experiment_type: 实验类型
                - 'single_node': 只变节点数
                - 'single_period': 只变周期数
                - 'single_state': 只变状态数
                - 'node_period': 节点数×周期数
                - 'node_state': 节点数×状态数
                - 'period_state': 周期数×状态数
                - 'full_factorial': 全因子设计
            output_dir: 输出目录
        """
        self.experiment_type = experiment_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 实验因素水平定义
        self.factor_levels = {
            'num_nodes': [6, 8, 10, 12, 14, 16, 18, 20],
            'num_periods': [3, 4, 5, 6],
            'num_states': [2, 3]
        }

        # 固定参数
        self.fixed_params = {
            'connection_density': 0.7,
            'discount_factor': 0.9,
            'action_space_size': 3,
            'budget_base': 200,  # 基础预算
            'mcmc_samples': 500,  # MCMC采样数（可根据需要调整）
            'mc_samples': 500,  # MC采样数（可根据需要调整）
            'time_limit_gurobi': 3600,  # Gurobi时间限制（60分钟）
            'ga_population_size': 100,  # GA种群大小
            'ga_max_generations': 300,  # GA最大迭代代数
            'disruption_level': 'moderate',  # 默认disruption级别
            'base_seed': 42
        }

        # 每个配置重复次数（降低随机性）
        self.replications = 3

        # 结果存储
        self.results = []

        # 实验统计
        self.total_experiments = 0
        self.completed_experiments = 0
        self.failed_experiments = 0

        # 时间记录
        self.start_time = None
        self.experiment_start_time = None

        print(f"🔬 系统性实验管理器初始化")
        print(f"   实验类型: {self.experiment_type}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   重复次数: {self.replications}")

    def allocate_nodes(self, total_nodes: int) -> Tuple[int, int, int]:
        """
        智能分配节点数到各层

        策略：供应商 ≈ 45%, 制造商 ≈ 45%, 零售商 = 1
        """
        retailer = 1
        remaining = total_nodes - retailer

        # 供应商占40-50%
        num_suppliers = max(2, int(remaining * 0.45))
        num_manufacturers = remaining - num_suppliers

        # 确保至少有2个制造商
        if num_manufacturers < 2:
            num_manufacturers = 2
            num_suppliers = remaining - num_manufacturers

        return (num_suppliers, num_manufacturers, retailer)

    def adjust_budget(self, num_nodes: int, num_periods: int,
                      num_states: int) -> int:
        """
        根据问题规模动态调整预算

        规则：
        - 节点数每增加1：预算+20
        - 周期数每增加1：预算+15
        - 状态数=3：预算+30
        """
        base = self.fixed_params['budget_base']

        # 节点数影响
        node_factor = (num_nodes - 6) * 20

        # 周期数影响
        period_factor = (num_periods - 3) * 15

        # 状态数影响
        state_factor = 30 if num_states == 3 else 0

        budget = base + node_factor + period_factor + state_factor

        return int(budget)

    def generate_experiment_configs(self) -> List[Dict]:
        """
        根据实验类型生成配置列表

        Returns:
            配置列表，每个配置包含完整的实验参数
        """
        configs = []

        if self.experiment_type == 'single_node':
            # 固定周期=4, 状态=2, 只变节点数
            for num_nodes in self.factor_levels['num_nodes']:
                config = self._create_config(
                    num_nodes=num_nodes,
                    num_periods=4,
                    num_states=2
                )
                configs.append(config)

        elif self.experiment_type == 'single_period':
            # 固定节点=10, 状态=2, 只变周期数
            for num_periods in self.factor_levels['num_periods']:
                config = self._create_config(
                    num_nodes=10,
                    num_periods=num_periods,
                    num_states=2
                )
                configs.append(config)

        elif self.experiment_type == 'single_state':
            # 固定节点=10, 周期=4, 只变状态数
            for num_states in self.factor_levels['num_states']:
                config = self._create_config(
                    num_nodes=10,
                    num_periods=4,
                    num_states=num_states
                )
                configs.append(config)

        elif self.experiment_type == 'node_period':
            # 固定状态=2, 变节点数×周期数
            for num_nodes, num_periods in itertools.product(
                    self.factor_levels['num_nodes'],
                    self.factor_levels['num_periods']
            ):
                config = self._create_config(
                    num_nodes=num_nodes,
                    num_periods=num_periods,
                    num_states=2
                )
                configs.append(config)

        elif self.experiment_type == 'node_state':
            # 固定周期=4, 变节点数×状态数
            for num_nodes, num_states in itertools.product(
                    self.factor_levels['num_nodes'],
                    self.factor_levels['num_states']
            ):
                config = self._create_config(
                    num_nodes=num_nodes,
                    num_periods=4,
                    num_states=num_states
                )
                configs.append(config)

        elif self.experiment_type == 'period_state':
            # 固定节点=10, 变周期数×状态数
            for num_periods, num_states in itertools.product(
                    self.factor_levels['num_periods'],
                    self.factor_levels['num_states']
            ):
                config = self._create_config(
                    num_nodes=10,
                    num_periods=num_periods,
                    num_states=num_states
                )
                configs.append(config)

        elif self.experiment_type == 'full_factorial':
            # 全因子设计：所有组合
            for num_nodes, num_periods, num_states in itertools.product(
                    self.factor_levels['num_nodes'],
                    self.factor_levels['num_periods'],
                    self.factor_levels['num_states']
            ):
                config = self._create_config(
                    num_nodes=num_nodes,
                    num_periods=num_periods,
                    num_states=num_states
                )
                configs.append(config)

        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

        # 为每个配置添加重复实验
        configs_with_reps = []
        for i, config in enumerate(configs):
            for rep in range(self.replications):
                config_copy = config.copy()
                config_copy['config_id'] = i
                config_copy['replication'] = rep
                config_copy['seed'] = self.fixed_params['base_seed'] + i * 100 + rep
                config_copy['exp_id'] = len(configs_with_reps)  # 全局实验ID
                configs_with_reps.append(config_copy)

        return configs_with_reps

    def _create_config(self, num_nodes: int, num_periods: int,
                       num_states: int) -> Dict:
        """创建单个实验配置"""
        num_suppliers, num_manufacturers, _ = self.allocate_nodes(num_nodes)
        budget = self.adjust_budget(num_nodes, num_periods, num_states)

        return {
            'num_nodes': num_nodes,
            'num_suppliers': num_suppliers,
            'num_manufacturers': num_manufacturers,
            'num_periods': num_periods,
            'num_states': num_states,
            'budget': budget,
            'connection_density': self.fixed_params['connection_density'],
            'discount_factor': self.fixed_params['discount_factor'],
            'action_space_size': self.fixed_params['action_space_size'],
            'mcmc_samples': self.fixed_params['mcmc_samples'],
            'mc_samples': self.fixed_params['mc_samples'],
            'time_limit_gurobi': self.fixed_params['time_limit_gurobi'],
            'ga_population_size': self.fixed_params['ga_population_size'],
            'ga_max_generations': self.fixed_params['ga_max_generations'],
            'disruption_level': self.fixed_params['disruption_level']
        }

    def run_single_experiment(self, config: Dict) -> Dict:
        """运行单个实验（GA + Gurobi）"""
        print(f"\n{'=' * 80}")
        print(f"🧪 实验 {config['exp_id'] + 1}/{self.total_experiments}")
        print(f"   配置ID={config['config_id']}, 重复={config['replication']}")
        print(f"   节点={config['num_nodes']}, 周期={config['num_periods']}, "
              f"状态={config['num_states']}, 预算={config['budget']}")
        print(f"   Disruption={config['disruption_level']}, 种子={config['seed']}")
        print(f"{'=' * 80}")

        result = {
            'exp_id': config['exp_id'],
            'config_id': config['config_id'],
            'replication': config['replication'],
            'config': config,
            'ga_result': None,
            'gurobi_result': None,
            'comparison': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            # 统一生成观测数据
            print(f"\n🔧 生成统一观测数据...")
            observed_data = self._generate_random_observations(config)
            print(f"   ✓ 生成了 {len(observed_data[1])} 个节点的观测数据")

            # 运行GA
            print("\n🧬 运行 GA 求解器...")
            ga_result = self._run_ga(config, observed_data)
            result['ga_result'] = ga_result

            if ga_result['objective'] is not None:
                print(f"   ✓ GA完成: 目标值={ga_result['objective']:.6f}, 时间={ga_result['time']:.2f}s")
            else:
                print(f"   ❌ GA失败")

            # 运行Gurobi
            print("\n🔧 运行 Gurobi 求解器...")
            gurobi_result = self._run_gurobi(config, observed_data)
            result['gurobi_result'] = gurobi_result

            # 🔧 修复点：详细打印 Gurobi 状态
            if gurobi_result['objective'] is not None:
                print(f"   ✓ Gurobi完成: 目标值={gurobi_result['objective']:.6f}, "
                      f"时间={gurobi_result['time']:.2f}s, Gap={gurobi_result['gap']:.4f}")
            else:
                print(f"   ⚠️  Gurobi未找到解")
                print(f"      状态: {gurobi_result.get('status', 'UNKNOWN')}")
                print(f"      时间: {gurobi_result['time']:.2f}s")
                if gurobi_result.get('status') == 2:  # GRB.Status.OPTIMAL
                    print(f"      可能原因: 模型构建问题")
                elif gurobi_result.get('status') == 3:  # GRB.Status.INFEASIBLE
                    print(f"      可能原因: 模型不可行")
                elif gurobi_result.get('status') == 9:  # GRB.Status.TIME_LIMIT
                    print(f"      可能原因: 达到时间限制")

            # 对比分析
            result['comparison'] = self._compare_results(ga_result, gurobi_result)

            # 🔧 修复点：根据对比有效性打印结果
            if result['comparison']['comparison_valid']:
                print(f"\n📊 对比结果:")
                print(f"   目标值Gap: {result['comparison']['objective_gap']:.2%}")
                print(f"   时间比率: {result['comparison']['time_ratio']:.2f}x")
                print(f"   GA {'胜出' if result['comparison']['ga_better'] else '落后'}")
            else:
                print(f"\n⚠️  无法进行对比（Gurobi未找到解）")
                if result['comparison']['time_ratio'] is not None:
                    print(f"   时间比率: {result['comparison']['time_ratio']:.2f}x")

            self.completed_experiments += 1

        except Exception as e:
            print(f"\n❌ 实验失败: {e}")
            traceback.print_exc()
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            self.failed_experiments += 1

        return result
    def _run_ga(self, config: Dict, observed_data: Dict) -> Dict:
        """
        运行 GA 求解器

        Args:
            config: 实验配置
            observed_data: 观测数据（与Gurobi共享）
        """
        # 🔧 修改点：导入 GA3.py
        from GA3 import GeneticAlgorithmSolver

        # 准备参数
        network_params = {
            'num_suppliers': config['num_suppliers'],
            'num_manufacturers': config['num_manufacturers'],
            'connection_density': config['connection_density'],
            'seed': config['seed'],
            'network_type': 'random'
        }

        pomdp_params = {
            'discount_factor': config['discount_factor'],
            'action_space_size': config['action_space_size']
        }

        prediction_params = {
            'num_periods': config['num_periods'],
            'num_states': config['num_states'],
            'mcmc_samples': config['mcmc_samples'],
            'mc_samples': config['mc_samples'],
            'disruption_level': config['disruption_level'],
            'observed_data': observed_data  # ← 使用统一生成的观测数据
        }

        # 创建求解器
        solver = GeneticAlgorithmSolver(
            network_params=network_params,
            pomdp_params=pomdp_params,
            prediction_params=prediction_params,
            population_size=config['ga_population_size'],
            max_generations=config['ga_max_generations']
        )

        solver.budget = config['budget']

        # 求解
        start_time = time.time()
        solver.initialize_components()
        best_solution, best_fitness = solver.evolve()
        solve_time = time.time() - start_time

        return {
            'objective': best_fitness,
            'time': solve_time,
            'solution': best_solution.tolist() if hasattr(best_solution, 'tolist') else None,
            'converged': True,
            'method': 'GA'
        }

    def _run_gurobi(self, config: Dict, observed_data: Dict) -> Dict:
        """
        运行 Gurobi 求解器

        Args:
            config: 实验配置
            observed_data: 观测数据（与GA共享）
        """
        # 🔧 修改点：导入 R1_solver19.py
        from R1_solver19 import SupplyChainOptimizationModel
        from gurobipy import GRB

        # 准备参数（与GA完全相同）
        network_params = {
            'num_suppliers': config['num_suppliers'],
            'num_manufacturers': config['num_manufacturers'],
            'connection_density': config['connection_density'],
            'seed': config['seed'],
            'network_type': 'random'
        }

        pomdp_params = {
            'discount_factor': config['discount_factor'],
            'action_space_size': config['action_space_size']
        }

        prediction_params = {
            'num_periods': config['num_periods'],
            'num_states': config['num_states'],
            'mcmc_samples': config['mcmc_samples'],
            'mc_samples': config['mc_samples'],
            'disruption_level': config['disruption_level'],
            'observed_data': observed_data  # ← 使用统一生成的观测数据
        }

        # 创建优化模型
        optimizer = SupplyChainOptimizationModel(
            network_params=network_params,
            pomdp_params=pomdp_params,
            prediction_params=prediction_params
        )

        optimizer.budget = config['budget']

        # 求解
        start_time = time.time()
        status = optimizer.run_optimization(
            time_limit=config['time_limit_gurobi'],
            save_results=False,
            export_excel=False
        )
        solve_time = time.time() - start_time

        # 提取结果
        objective = None
        gap = None

        if hasattr(optimizer, 'model') and optimizer.model is not None:
            if hasattr(optimizer.model, 'objVal'):
                objective = optimizer.model.objVal
            if hasattr(optimizer.model, 'MIPGap'):
                gap = optimizer.model.MIPGap

        return {
            'objective': objective,
            'time': solve_time,
            'gap': gap if gap is not None else 0.0,
            'status': status,
            'method': 'Gurobi'
        }

    def _generate_random_observations(self, config: Dict) -> Dict:
        """
        生成随机观测数据

        关键：基于相同的seed生成，保证可重复性
        """
        np.random.seed(config['seed'])

        # 根据disruption级别设置履约率范围
        if config['disruption_level'] == 'light':
            phi_range = (0.6, 0.9)  # 轻微disruption
        elif config['disruption_level'] == 'moderate':
            phi_range = (0.4, 0.7)  # 中等disruption
        else:  # severe
            phi_range = (0.2, 0.5)  # 严重disruption

        observations = {1: {}}  # Period 1的观测数据

        for node in range(config['num_nodes']):
            # 随机需求
            D_obs = np.random.uniform(50, 150)

            # 根据disruption级别随机履约率
            phi = np.random.uniform(*phi_range)
            SD_obs = D_obs * phi

            observations[1][node] = {
                'D_obs': D_obs,
                'SD_obs': SD_obs
            }

        return observations

    def _compare_results(self, ga_result: Dict, gurobi_result: Dict) -> Dict:
        """
        对比GA和Gurobi的结果

        🔧 修复：处理 Gurobi 未找到解的情况
        """
        comparison = {
            'objective_gap': None,
            'ga_better': None,
            'time_ratio': None,
            'quality_score': None,
            'comparison_valid': False  # 🔧 新增：标记对比是否有效
        }

        # 目标值对比（只有两个都成功时才有效）
        if (ga_result and ga_result['objective'] is not None and
                gurobi_result and gurobi_result['objective'] is not None):

            ga_obj = ga_result['objective']
            gurobi_obj = gurobi_result['objective']

            # 避免除零
            if abs(gurobi_obj) > 1e-10:
                gap = abs(ga_obj - gurobi_obj) / abs(gurobi_obj)
                comparison['objective_gap'] = gap
                comparison['ga_better'] = ga_obj < gurobi_obj
                comparison['quality_score'] = (gurobi_obj - ga_obj) / abs(gurobi_obj)
                comparison['comparison_valid'] = True

        # 时间对比（总是可以比较）
        if (ga_result and ga_result['time'] > 0 and
                gurobi_result and gurobi_result['time'] > 0):
            comparison['time_ratio'] = ga_result['time'] / gurobi_result['time']

        return comparison

    def run_all_experiments(self):
        """运行所有实验"""
        self.start_time = time.time()

        # 生成实验配置
        configs = self.generate_experiment_configs()
        self.total_experiments = len(configs)

        print(f"\n🚀 开始系统性实验")
        print(f"{'=' * 80}")
        print(f"   实验类型: {self.experiment_type}")
        print(f"   总配置数: {len(configs) // self.replications}")
        print(f"   每配置重复: {self.replications} 次")
        print(f"   总实验数: {self.total_experiments}")
        print(f"   输出目录: {self.output_dir}")
        print(f"{'=' * 80}")

        # 运行实验
        for i, config in enumerate(configs):
            self.experiment_start_time = time.time()

            result = self.run_single_experiment(config)
            self.results.append(result)

            # 显示进度
            self._print_progress()

            # 定期保存中间结果
            if (i + 1) % 5 == 0:
                self._save_intermediate_results()

        # 生成最终报告
        self.generate_report()

        # 打印总结
        self._print_summary()

    def _print_progress(self):
        """打印实验进度"""
        progress = (self.completed_experiments + self.failed_experiments) / self.total_experiments * 100
        elapsed = time.time() - self.start_time

        if self.completed_experiments > 0:
            avg_time = elapsed / (self.completed_experiments + self.failed_experiments)
            remaining = (self.total_experiments - self.completed_experiments - self.failed_experiments) * avg_time

            print(
                f"\n📊 进度: {progress:.1f}% ({self.completed_experiments + self.failed_experiments}/{self.total_experiments})")
            print(f"   已完成: {self.completed_experiments}, 失败: {self.failed_experiments}")
            print(f"   已用时: {elapsed / 60:.1f}分钟, 预计剩余: {remaining / 60:.1f}分钟")

    def _save_intermediate_results(self):
        """保存中间结果（JSON格式）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f'intermediate_{self.experiment_type}_{timestamp}.json'

        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            print(f"   💾 中间结果已保存: {filename}")
        except Exception as e:
            print(f"   ⚠️  保存中间结果失败: {e}")

    def generate_report(self):
        """生成系统性分析报告（Excel格式）"""
        print(f"\n{'=' * 80}")
        print("📊 生成实验报告...")
        print(f"{'=' * 80}")

        # 🔧 修复点1：分类统计
        total_results = len(self.results)
        valid_results = 0
        ga_failed = 0
        gurobi_failed = 0
        both_success = 0

        # 提取数据
        data = []
        for result in self.results:
            # 统计各种情况
            if 'error' in result:
                continue

            ga_ok = result['ga_result'] and result['ga_result']['objective'] is not None
            gurobi_ok = result['gurobi_result'] and result['gurobi_result']['objective'] is not None

            if ga_ok and gurobi_ok:
                both_success += 1
            elif not ga_ok:
                ga_failed += 1
            elif not gurobi_ok:
                gurobi_failed += 1

            # 🔧 修复点2：即使 Gurobi 失败，也保存数据
            if result['ga_result'] and result['gurobi_result']:
                valid_results += 1

                row = {
                    'exp_id': result['exp_id'],
                    'config_id': result['config_id'],
                    'replication': result['replication'],
                    'num_nodes': result['config']['num_nodes'],
                    'num_periods': result['config']['num_periods'],
                    'num_states': result['config']['num_states'],
                    'budget': result['config']['budget'],
                    'disruption': result['config']['disruption_level'],
                    'seed': result['config']['seed'],

                    # GA结果
                    'ga_objective': result['ga_result']['objective'],
                    'ga_time': result['ga_result']['time'],
                    'ga_success': result['ga_result']['objective'] is not None,

                    # Gurobi结果（可能为 None）
                    'gurobi_objective': result['gurobi_result']['objective'],
                    'gurobi_time': result['gurobi_result']['time'],
                    'gurobi_gap': result['gurobi_result'].get('gap', None),
                    'gurobi_status': str(result['gurobi_result'].get('status', 'UNKNOWN')),
                    'gurobi_success': result['gurobi_result']['objective'] is not None,

                    # 对比结果（可能为 None）
                    'objective_gap': result['comparison'].get('objective_gap'),
                    'time_ratio': result['comparison'].get('time_ratio'),
                    'ga_better': result['comparison'].get('ga_better'),

                    'timestamp': result['timestamp']
                }
                data.append(row)

        # 🔧 修复点3：打印统计信息
        print(f"\n📊 数据统计:")
        print(f"   总实验数: {total_results}")
        print(f"   有效结果: {valid_results}")
        print(f"   GA+Gurobi都成功: {both_success}")
        print(f"   仅GA失败: {ga_failed}")
        print(f"   仅Gurobi失败: {gurobi_failed}")

        if not data:
            print("   ⚠️  没有有效数据可供生成报告")
            return

        df = pd.DataFrame(data)

        # 🔧 修复点4：分别处理成功和失败的案例
        # 只用成功的案例计算统计指标
        df_success = df[df['gurobi_success'] == True].copy()

        print(f"\n   可用于对比的实验数: {len(df_success)}/{len(df)}")

        # 聚合重复实验（只聚合成功的案例）
        if len(df_success) > 0:
            df_agg = df_success.groupby(['config_id', 'num_nodes', 'num_periods', 'num_states', 'budget']).agg({
                'ga_objective': ['mean', 'std', 'count'],
                'ga_time': ['mean', 'std'],
                'gurobi_objective': ['mean', 'std'],
                'gurobi_time': ['mean', 'std'],
                'gurobi_gap': 'mean',
                'objective_gap': ['mean', 'std'],
                'time_ratio': ['mean', 'std'],
                'ga_better': 'sum'
            })

            # 扁平化列名
            df_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                              for col in df_agg.columns.values]

            # 重置索引
            df_agg = df_agg.reset_index()
        else:
            df_agg = None
            print("   ⚠️  没有成功的对比案例，跳过聚合统计")

        # 保存Excel报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = self.output_dir / f'report_{self.experiment_type}_{timestamp}.xlsx'

        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 🔧 修复点5：所有数据（包括失败案例）
                df.to_excel(writer, sheet_name='所有数据', index=False)

                # 🔧 修复点6：只有成功的案例
                if len(df_success) > 0:
                    df_success.to_excel(writer, sheet_name='成功案例', index=False)

                # 🔧 修复点7：聚合结果（如果有）
                if df_agg is not None:
                    df_agg.to_excel(writer, sheet_name='聚合结果', index=False)

                # 🔧 修复点8：失败案例分析
                df_failed = df[df['gurobi_success'] == False].copy()
                if len(df_failed) > 0:
                    df_failed.to_excel(writer, sheet_name='Gurobi失败案例', index=False)
                    print(f"   ⚠️  发现 {len(df_failed)} 个 Gurobi 失败案例")

                # 按节点数分组（只用成功案例）
                if len(df_success) > 0 and 'num_nodes' in df_success.columns and len(
                        df_success['num_nodes'].unique()) > 1:
                    grouped_nodes = df_success.groupby('num_nodes').agg({
                        'ga_objective': 'mean',
                        'gurobi_objective': 'mean',
                        'ga_time': 'mean',
                        'gurobi_time': 'mean',
                        'objective_gap': 'mean',
                        'time_ratio': 'mean'
                    })
                    grouped_nodes = grouped_nodes.reset_index()
                    grouped_nodes.to_excel(writer, sheet_name='按节点数统计', index=False)

                # 按周期数分组（只用成功案例）
                if len(df_success) > 0 and 'num_periods' in df_success.columns and len(
                        df_success['num_periods'].unique()) > 1:
                    grouped_periods = df_success.groupby('num_periods').agg({
                        'ga_objective': 'mean',
                        'gurobi_objective': 'mean',
                        'ga_time': 'mean',
                        'gurobi_time': 'mean',
                        'objective_gap': 'mean',
                        'time_ratio': 'mean'
                    })
                    grouped_periods = grouped_periods.reset_index()
                    grouped_periods.to_excel(writer, sheet_name='按周期数统计', index=False)

                # 按状态数分组（只用成功案例）
                if len(df_success) > 0 and 'num_states' in df_success.columns and len(
                        df_success['num_states'].unique()) > 1:
                    grouped_states = df_success.groupby('num_states').agg({
                        'ga_objective': 'mean',
                        'gurobi_objective': 'mean',
                        'ga_time': 'mean',
                        'gurobi_time': 'mean',
                        'objective_gap': 'mean',
                        'time_ratio': 'mean'
                    })
                    grouped_states = grouped_states.reset_index()
                    grouped_states.to_excel(writer, sheet_name='按状态数统计', index=False)

                # 🔧 修复点9：更完整的统计摘要
                summary = {
                    '指标': [
                        '总实验数',
                        '有效实验数',
                        '成功实验数',
                        '失败实验数',
                        'GA失败数',
                        'Gurobi失败数',
                        '',
                        'GA平均目标值（成功案例）',
                        'Gurobi平均目标值（成功案例）',
                        'GA平均时间(秒)',
                        'Gurobi平均时间(秒)',
                        '',
                        '平均目标值gap（成功案例）',
                        '平均时间比率',
                        'GA获胜次数',
                        'Gurobi获胜次数'
                    ],
                    '值': [
                        self.total_experiments,
                        valid_results,
                        both_success,
                        self.failed_experiments + ga_failed + gurobi_failed,
                        ga_failed,
                        gurobi_failed,
                        '',
                        df_success['ga_objective'].mean() if len(df_success) > 0 else 'N/A',
                        df_success['gurobi_objective'].mean() if len(df_success) > 0 else 'N/A',
                        df['ga_time'].mean(),
                        df['gurobi_time'].mean(),
                        '',
                        df_success['objective_gap'].mean() if len(df_success) > 0 else 'N/A',
                        df['time_ratio'].mean() if df['time_ratio'].notna().any() else 'N/A',
                        (df_success['ga_objective'] < df_success['gurobi_objective']).sum() if len(
                            df_success) > 0 else 0,
                        (df_success['gurobi_objective'] < df_success['ga_objective']).sum() if len(
                            df_success) > 0 else 0
                    ]
                }
                pd.DataFrame(summary).to_excel(writer, sheet_name='统计摘要', index=False)

                # 实验配置
                config_summary = {
                    '参数': list(self.fixed_params.keys()),
                    '值': list(self.fixed_params.values())
                }
                pd.DataFrame(config_summary).to_excel(writer, sheet_name='实验配置', index=False)

            print(f"   ✅ 报告已生成: {excel_file}")

        except Exception as e:
            print(f"   ❌ 报告生成失败: {e}")
            traceback.print_exc()

    def _print_summary(self):
        """打印实验总结"""
        total_time = time.time() - self.start_time

        print(f"\n{'=' * 80}")
        print("📈 实验总结")
        print(f"{'=' * 80}")
        print(f"   实验类型: {self.experiment_type}")
        print(f"   总实验数: {self.total_experiments}")
        print(f"   成功: {self.completed_experiments}")
        print(f"   失败: {self.failed_experiments}")
        print(f"   总用时: {total_time / 60:.1f} 分钟 ({total_time / 3600:.2f} 小时)")

        if self.completed_experiments > 0:
            # 计算统计数据
            data = []
            for result in self.results:
                if 'error' not in result and result['ga_result'] and result['gurobi_result']:
                    data.append(result)

            if data:
                ga_objs = [r['ga_result']['objective'] for r in data]
                gurobi_objs = [r['gurobi_result']['objective'] for r in data if
                               r['gurobi_result']['objective'] is not None]
                gaps = [r['comparison']['objective_gap'] for r in data if r['comparison']['objective_gap'] is not None]
                time_ratios = [r['comparison']['time_ratio'] for r in data if r['comparison']['time_ratio'] is not None]

                print(f"\n   GA平均目标值: {np.mean(ga_objs):.6f} ± {np.std(ga_objs):.6f}")
                if gurobi_objs:
                    print(f"   Gurobi平均目标值: {np.mean(gurobi_objs):.6f} ± {np.std(gurobi_objs):.6f}")
                if gaps:
                    print(f"   平均目标值Gap: {np.mean(gaps):.2%} ± {np.std(gaps):.2%}")
                if time_ratios:
                    print(f"   平均时间比率: {np.mean(time_ratios):.2f}x ± {np.std(time_ratios):.2f}x")

                ga_wins = sum(1 for r in data if r['comparison'].get('ga_better'))
                gurobi_wins = len(data) - ga_wins
                print(f"\n   GA获胜: {ga_wins} 次 ({ga_wins / len(data) * 100:.1f}%)")
                print(f"   Gurobi获胜: {gurobi_wins} 次 ({gurobi_wins / len(data) * 100:.1f}%)")

        print(f"{'=' * 80}")


def main():
    """主函数"""
    print("=" * 80)
    print("🎯 供应链韧性优化 - 系统性实验框架")
    print(f"   Current Date and Time (UTC): 2025-10-28 18:26:54")
    print(f"   Current User's Login: dyy21zyy")
    print("=" * 80)
    print()
    print("📌 适配模块:")
    print("   - GA3.py (遗传算法求解器)")
    print("   - R1_solver19.py (Gurobi求解器)")
    print("   - R1_network_generate4.py (多层随机网络)")
    print("   - R1_para_POMDP4.py (POMDP参数)")
    print("   - R1_prediction_inputDBN13.py (状态预测)")
    print("=" * 80)

    # 选择实验类型
    print("\n请选择实验类型:")
    print("  1 - single_node     (只变节点数, 8个水平, ~24个实验)")
    print("  2 - single_period   (只变周期数, 4个水平, ~12个实验)")
    print("  3 - single_state    (只变状态数, 2个水平, ~6个实验)")
    print("  4 - node_period     (节点数×周期数, 32组合, ~96个实验)")
    print("  5 - node_state      (节点数×状态数, 16组合, ~48个实验)")
    print("  6 - period_state    (周期数×状态数, 8组合, ~24个实验)")
    print("  7 - full_factorial  (全因子, 64组合, ~192个实验)")
    print()
    print("💡 提示: 数字越大，实验数量越多，耗时越长")
    print("   建议先选择 1-3 进行快速测试")

    choice = input("\n请输入选项 (1-7): ").strip()

    experiment_types = {
        '1': 'single_node',
        '2': 'single_period',
        '3': 'single_state',
        '4': 'node_period',
        '5': 'node_state',
        '6': 'period_state',
        '7': 'full_factorial'
    }

    if choice not in experiment_types:
        print("❌ 无效选项！")
        return

    experiment_type = experiment_types[choice]
    print(f"\n✅ 已选择: {experiment_type}")

    # 确认开始
    confirm = input("\n是否开始实验? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 实验已取消")
        return

    # 创建实验管理器
    output_dir = './experiment_results'
    manager = SystematicExperimentManager(
        experiment_type=experiment_type,
        output_dir=output_dir
    )

    # 运行实验
    try:
        manager.run_all_experiments()
        print("\n✅ 所有实验完成！")
        print(f"   结果保存在: {manager.output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  实验被用户中断")
        print("   部分结果已保存")
        manager._save_intermediate_results()

    except Exception as e:
        print(f"\n\n❌ 实验过程中发生错误: {e}")
        traceback.print_exc()
        manager._save_intermediate_results()


if __name__ == "__main__":
    main()