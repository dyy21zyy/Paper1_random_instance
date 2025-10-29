"""
R1_prediction_inputDBN12.py - 修改版

贝叶斯状态预测器 - 支持随机实验
- 移除硬编码的初始信念状态
- 移除硬编码的CPT矩阵
- 改为动态生成（基于Dirichlet分布）
- 支持任意节点数、层数
- 调用 R1_network_generate4.py

Current Date and Time (UTC): 2025-10-28 13:05:56
Current User's Login: dyy21zyy

观测: Demand (D) 和 Satisfied Demand (SD)
预测: State

t=0: 先验分布 + 初始状态概率
t=1: 实际观测 → 真实φ → 直接确定状态 → 后验更新
t≥2: 预测观测 → 后验更新 → MC采样确定状态
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, invgamma, gamma
from datetime import datetime


class ImprovedBalancedBayesianPredictor:
    """
    贝叶斯状态预测器 - t=1使用真实履约率确定状态

    🔧 修改版本特性：
    1. 移除所有硬编码的初始信念状态
    2. 移除所有硬编码的CPT矩阵
    3. 使用Dirichlet分布动态生成
    4. 支持任意节点数、层数
    """

    def __init__(
            self,
            network_data,
            num_states,
            num_periods,
            disruption_level=None,
            observed_data=None,
            mcmc_samples=1000,
            mc_samples=1000,
            seed=21
    ):
        (self.network, self.layer_info, self.temporal_network,
         self.temporal_node_info, self.parent_dict,
         self.independent_nodes, self.other_nodes,
         self.parent_node_dic, self.C_dic, self.G_dic) = network_data

        self.num_nodes = self.layer_info['num_nodes']
        self.num_states = num_states
        self.num_periods = num_periods
        self.disruption_level = disruption_level

        if observed_data is None or 1 not in observed_data:
            raise ValueError("❌ ERROR: Must provide actual observation data for period t=1")

        self.observed_data = observed_data
        self.mcmc_samples = mcmc_samples
        self.mc_samples = mc_samples
        self.seed = seed
        np.random.seed(seed)

        self._set_prior_parameters()
        self.t0_state_probs = {}
        self.t0_cpts = {}
        self._initialize_t0_parameters()
        self.posterior_history = {}

        print(f"🔧 Initialized Bayesian Predictor (Random Experiment Version)")
        print(f"   Disruption: {self.disruption_level.upper() if self.disruption_level else 'N/A'}")
        print(f"   Nodes: {self.num_nodes}, States: {self.num_states}, Periods: {self.num_periods}")
        print(f"   t=1: Actual observations → Direct state from real φ → Posterior update")
        print(f"   t≥2: Predicted observations → Posterior update → MC sampled states")

    def _set_prior_parameters(self):
        """
        设置先验参数

        🔧 修改点：适应任意节点数和层数
        """
        print(f"\n🔧 Setting prior parameters for {self.num_nodes} nodes...")

        self.prior_mu_D = []
        self.prior_mu_SD = []
        self.prior_tau_D = []
        self.prior_tau_SD = []

        for node in range(self.num_nodes):
            node_type = self._get_node_type(node)

            # 根据节点类型设置不同的先验
            if node_type == "Suppliers":
                mu_D, mu_SD = 90.0, 90.0
                tau_D, tau_SD = 2.0, 2.0
            elif node_type in ["Manufacturers", "Intermediate_1", "Intermediate_2"]:
                mu_D, mu_SD = 45.0, 45.0
                tau_D, tau_SD = 1.5, 2.0
            else:  # Retailer or other
                mu_D, mu_SD = 45.0, 45.0
                tau_D, tau_SD = 1.0, 1.0

            self.prior_mu_D.append(mu_D)
            self.prior_mu_SD.append(mu_SD)
            self.prior_tau_D.append(tau_D)
            self.prior_tau_SD.append(tau_SD)

        self.prior_alpha_D = [3.0] * self.num_nodes
        self.prior_beta_D = []
        self.prior_alpha_SD = [3.0] * self.num_nodes
        self.prior_beta_SD = []

        for node in range(self.num_nodes):
            expected_var_D = (self.prior_tau_D[node] * 0.3) ** 2
            expected_var_SD = (self.prior_tau_SD[node] * 0.3) ** 2
            self.prior_beta_D.append(expected_var_D * (self.prior_alpha_D[node] - 1))
            self.prior_beta_SD.append(expected_var_SD * (self.prior_alpha_SD[node] - 1))

        self._set_external_factor_priors()
        print(f"   ✅ Prior parameters configured for {self.num_nodes} nodes")

    def _set_external_factor_priors(self):
        """设置外部因子χ的先验"""
        configs = {
            'light': {'chi_d': (278, 267.31), 'chi_sd': (123, 141.38)},
            'moderate': {'chi_d': (123, 110.81), 'chi_sd': (51, 66.23)},
            'severe': {'chi_d': (69, 58.97), 'chi_sd': (31, 46.27)}
        }

        config = configs.get(self.disruption_level, configs['moderate'])
        alpha_d, beta_d = config['chi_d']
        alpha_sd, beta_sd = config['chi_sd']

        self.prior_alpha_chi_d = [alpha_d] * self.num_nodes
        self.prior_beta_chi_d = [beta_d] * self.num_nodes
        self.prior_alpha_chi_sd = [alpha_sd] * self.num_nodes
        self.prior_beta_chi_sd = [beta_sd] * self.num_nodes

        print(f"   χ factors: E[χ_d]={alpha_d / beta_d:.3f}, E[χ_sd]={alpha_sd / beta_sd:.3f}")

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

    def _initialize_t0_parameters(self):
        """
        🔧 核心修改：动态初始化t=0的状态概率和CPT（移除硬编码）

        修改策略：
        1. 使用Dirichlet分布生成初始信念状态
        2. 根据disruption级别调整concentration参数
        3. 使用Dirichlet分布生成CPT矩阵
        4. 保证每个节点有不同的随机生成结果
        """
        print(f"\n🔧 Dynamically initializing t=0 parameters (no hardcoded values)...")

        np.random.seed(self.seed)

        # ============================================
        # 步骤1：动态生成初始状态概率 u_hat_0
        # ============================================
        print(f"   Generating initial belief states for {self.num_nodes} nodes...")

        for node in range(self.num_nodes):
            # 根据disruption级别设置concentration参数
            if self.disruption_level == 'light':
                # 轻微disruption：倾向良好状态
                if self.num_states == 2:
                    concentration = [2.0, 5.0]  # 倾向状态1（好状态）
                elif self.num_states == 3:
                    concentration = [1.5, 3.0, 5.0]  # 倾向状态2（最好状态）
                else:
                    # 多状态：后面的状态concentration更高
                    concentration = [1.0 + i * 0.8 for i in range(self.num_states)]

            elif self.disruption_level == 'moderate':
                # 中等disruption：相对均衡但略倾向差状态
                if self.num_states == 2:
                    concentration = [4.0, 3.0]  # 略倾向状态0（差状态）
                elif self.num_states == 3:
                    concentration = [3.0, 3.0, 2.0]  # 相对均衡
                else:
                    concentration = [3.0] * self.num_states  # 均匀

            else:  # severe
                # 严重disruption：倾向差状态
                if self.num_states == 2:
                    concentration = [6.0, 2.0]  # 强烈倾向状态0（差状态）
                elif self.num_states == 3:
                    concentration = [5.0, 3.0, 1.5]  # 倾向状态0（最差状态）
                else:
                    # 多状态：前面的状态concentration更高
                    concentration = [5.0 - i * 0.6 for i in range(self.num_states)]

            # 添加节点特异性（基于节点索引的扰动）
            node_perturbation = (node % 3) * 0.3  # [0, 0.3, 0.6]
            concentration = [c + node_perturbation for c in concentration]

            # 使用Dirichlet分布生成概率
            probs = np.random.dirichlet(concentration)

            # 存储到t0_state_probs
            self.t0_state_probs[node] = probs

            # 打印前几个节点的示例
            if node < 3:
                print(f"      Node {node}: u_hat_0 = {[f'{p:.3f}' for p in probs]}")

        print(f"   ✅ Initial belief states generated for {self.num_nodes} nodes")

        # ============================================
        # 步骤2：动态生成初始CPT矩阵 g_hat_0
        # ============================================
        print(f"   Generating initial CPT matrices...")

        cpt_count = 0
        for node in range(self.num_nodes):
            # 获取父节点
            parent_nodes = np.where(self.network[:, node] == 1)[0]

            if len(parent_nodes) > 0:
                # 计算父节点状态组合数
                num_parent_combinations = self.num_states ** len(parent_nodes)

                # 为每个父节点状态组合生成CPT
                # 使用Dirichlet分布生成概率分布

                # concentration参数：略微倾向对角线（父节点好→子节点好）
                base_concentration = np.ones(self.num_states) * 2.0

                # 添加节点特异性
                node_hash = (node * 17 + 13) % 23
                node_factor = 0.5 + (node_hash / 23.0) * 1.5  # [0.5, 2.0]

                concentration = base_concentration * node_factor

                # 为每个父节点状态组合生成CPT
                cpt_matrix = np.zeros((self.num_states, num_parent_combinations))

                for j in range(num_parent_combinations):
                    # 为不同的父节点状态组合使用不同的concentration
                    # 添加组合索引的扰动
                    combination_perturbation = (j % self.num_states) * 0.4
                    adjusted_concentration = concentration + combination_perturbation

                    # 生成该组合对应的概率分布
                    probs = np.random.dirichlet(adjusted_concentration)
                    cpt_matrix[:, j] = probs

                # 存储CPT矩阵
                self.t0_cpts[node] = cpt_matrix.T  # 转置为 [combinations, states]
                cpt_count += 1

                # 打印前几个节点的示例
                if node < 3 and num_parent_combinations <= 4:
                    print(f"      Node {node}: CPT shape = {cpt_matrix.T.shape}")
                    for j in range(min(2, num_parent_combinations)):
                        print(f"         Combination {j}: {[f'{p:.3f}' for p in cpt_matrix[:, j]]}")

        print(f"   ✅ CPT matrices generated for {cpt_count} nodes with parents")

    def _compute_boundary_points(self, k, t):
        """计算边界点 ω(r) = r / |R|"""
        R_kt_size = self.num_states
        return [r / R_kt_size for r in range(R_kt_size + 1)]

    def _map_single_phi_to_state(self, phi):
        """
        对于 num_states=2:
        - State 0: φ ∈ [0.0, 0.5)
        - State 1: φ ∈ [0.5, 1.0]
        """
        phi = max(0.0, min(1.0, phi))

        boundary_points = [r / self.num_states for r in range(self.num_states + 1)]

        for r_kt in range(1, self.num_states + 1):
            omega_prev = boundary_points[r_kt - 1]
            omega_curr = boundary_points[r_kt]

            if r_kt == self.num_states:
                if omega_prev <= phi <= omega_curr:
                    return r_kt - 1  # 返回状态索引
            else:
                if omega_prev <= phi < omega_curr:
                    return r_kt - 1

        return 0  # 默认返回状态0

    def _map_fulfillment_to_states(self, fulfillment_rates, k, t):
        """
        将履约率样本映射到离散状态（用于 t≥2 的MC采样）
        """
        boundary_points = self._compute_boundary_points(k, t)
        R_kt_size = self.num_states
        state_counts = [0] * R_kt_size
        N = len(fulfillment_rates)

        for phi in fulfillment_rates:
            phi = max(0.0, min(1.0, phi))

            for r_kt in range(1, R_kt_size + 1):
                omega_prev = boundary_points[r_kt - 1]
                omega_curr = boundary_points[r_kt]

                if r_kt == R_kt_size:
                    if omega_prev <= phi <= omega_curr:
                        state_counts[r_kt - 1] += 1
                        break
                else:
                    if omega_prev <= phi < omega_curr:
                        state_counts[r_kt - 1] += 1
                        break

        state_probabilities = [count / N for count in state_counts]
        r_max_index = np.argmax(state_probabilities)

        return state_counts, state_probabilities, r_max_index

    def _get_actual_observations(self, period):
        """获取实际观测数据"""
        if period not in self.observed_data:
            raise ValueError(f"No actual observations for period {period}")

        observations = {}
        provided_data = self.observed_data[period]

        for node in range(self.num_nodes):
            if node not in provided_data:
                raise ValueError(f"Missing data for node {node} at period {period}")

            node_data = provided_data[node]
            D_obs = node_data['D_obs']
            SD_obs = node_data['SD_obs']

            observations[node] = {
                'D_obs': D_obs,
                'SD_obs': SD_obs,
                'phi_actual': min(1.0, SD_obs / D_obs) if D_obs > 0 else 0.0
            }

        return observations

    def _generate_predicted_observations(self, posterior_samples, period):
        """从后验生成预测观测"""
        observations = {}

        for node in range(self.num_nodes):
            s = posterior_samples[node]
            idx = np.random.randint(self.mcmc_samples)

            mu_D = s['mu_D'][idx]
            sigma_D = s['sigma_D'][idx]
            chi_d = s['chi_d'][idx]
            mu_SD = s['mu_SD'][idx]
            sigma_SD = s['sigma_SD'][idx]
            chi_sd = s['chi_sd'][idx]

            mu_D_eff = mu_D * chi_d
            mu_SD_eff = mu_SD * chi_sd
            D_pred = norm.rvs(loc=mu_D_eff, scale=sigma_D)
            SD_pred = norm.rvs(loc=mu_SD_eff, scale=sigma_SD)

            observations[node] = {
                'D_obs': D_pred,
                'SD_obs': SD_pred,
                'phi_actual': min(1.0, SD_pred / D_pred) if D_pred > 0 else 0.0
            }

        return observations

    def _bayesian_update(self, observations, prior_params):
        """贝叶斯更新"""
        posterior_samples = {}

        for node in range(self.num_nodes):
            obs = observations[node]
            prior = prior_params[node]

            samples = self._mcmc_posterior_samples(
                node=node,
                D_obs=obs['D_obs'],
                SD_obs=obs['SD_obs'],
                mu_D_prior=prior['mu_D'],
                tau_D_prior=prior['tau_D'],
                mu_SD_prior=prior['mu_SD'],
                tau_SD_prior=prior['tau_SD'],
                alpha_D=prior['alpha_D'],
                beta_D=prior['beta_D'],
                alpha_SD=prior['alpha_SD'],
                beta_SD=prior['beta_SD'],
                alpha_chi_d=prior['alpha_chi_d'],
                beta_chi_d=prior['beta_chi_d'],
                alpha_chi_sd=prior['alpha_chi_sd'],
                beta_chi_sd=prior['beta_chi_sd'],
                chi_d_init=prior['chi_d'],
                chi_sd_init=prior['chi_sd']
            )

            posterior_samples[node] = samples

        return posterior_samples

    def _determine_states_from_actual_phi(self, observations):
        """
        🔧 新方法：直接从真实履约率确定状态（用于 t=1）
        """
        print(f"  🎯 Step 3: Determining states directly from actual φ")

        predicted_states = {}
        state_probs = {}

        for node in range(self.num_nodes):
            obs = observations[node]
            phi_actual = obs['phi_actual']

            # 直接映射
            state = self._map_single_phi_to_state(phi_actual)
            predicted_states[node] = state

            # 构造伪概率（100%确定性）
            probs = [0.0] * self.num_states
            probs[state] = 1.0
            state_probs[node] = probs

            node_type = self._get_node_type(node)

            # 状态名称（适应多状态）
            if self.num_states == 2:
                state_name = "Good" if state == 1 else "Poor"
            elif self.num_states == 3:
                state_names = ["Poor", "Moderate", "Excellent"]
                state_name = state_names[state]
            else:
                state_name = f"State_{state}"

            if node < 3:  # 只打印前3个节点
                print(f"   Node {node} ({node_type}): φ={phi_actual:.3f} → State {state} ({state_name})")

        return predicted_states, state_probs

    def _predict_states_from_posterior(self, posterior_samples, period):
        """
        从后验采样预测状态（用于 t≥2）
        """
        print(f"  🎯 Step 3: Predicting states via MC sampling (N={self.mc_samples})")

        fulfillment_rates = {n: [] for n in range(self.num_nodes)}

        for i in range(self.mc_samples):
            for n in range(self.num_nodes):
                s = posterior_samples[n]
                idx = np.random.randint(self.mcmc_samples)

                mu_D = s['mu_D'][idx]
                sigma_D = s['sigma_D'][idx]
                chi_d = s['chi_d'][idx]
                mu_SD = s['mu_SD'][idx]
                sigma_SD = s['sigma_SD'][idx]
                chi_sd = s['chi_sd'][idx]

                mu_D_eff = mu_D * chi_d
                mu_SD_eff = mu_SD * chi_sd
                D_pred = norm.rvs(loc=mu_D_eff, scale=sigma_D)
                SD_pred = norm.rvs(loc=mu_SD_eff, scale=sigma_SD)

                phi = min(1.0, SD_pred / D_pred) if D_pred > 0 else 0.0
                fulfillment_rates[n].append(phi)

        predicted_states = {}
        state_probs = {}

        for n in range(self.num_nodes):
            _, state_probabilities, obs_state = self._map_fulfillment_to_states(
                fulfillment_rates[n], n, period
            )
            state_probs[n] = state_probabilities
            predicted_states[n] = obs_state

        print(f"   Predicted states (first 5): {[predicted_states[i] for i in range(min(5, self.num_nodes))]}")

        return predicted_states, state_probs, fulfillment_rates

    def _mcmc_posterior_samples(
            self, node, D_obs, SD_obs,
            mu_D_prior, tau_D_prior, mu_SD_prior, tau_SD_prior,
            alpha_D, beta_D, alpha_SD, beta_SD,
            alpha_chi_d, beta_chi_d, alpha_chi_sd, beta_chi_sd,
            chi_d_init, chi_sd_init
    ):
        """MCMC采样"""
        samples = {
            'mu_D': [], 'sigma_D': [], 'chi_d': [],
            'mu_SD': [], 'sigma_SD': [], 'chi_sd': []
        }

        mu_D = mu_D_prior
        sigma_D = np.sqrt(beta_D / alpha_D)
        chi_d = chi_d_init
        mu_SD = mu_SD_prior
        sigma_SD = np.sqrt(beta_SD / alpha_SD)
        chi_sd = chi_sd_init

        for i in range(self.mcmc_samples):
            mu_D_prop = mu_D + np.random.normal(0, tau_D_prior * 0.1)
            sigma_D_prop = abs(sigma_D + np.random.normal(0, 10.0))
            chi_d_prop = abs(chi_d + np.random.normal(0, 0.1))
            mu_SD_prop = mu_SD + np.random.normal(0, tau_SD_prior * 0.1)
            sigma_SD_prop = abs(sigma_SD + np.random.normal(0, 10.0))
            chi_sd_prop = abs(chi_sd + np.random.normal(0, 0.1))

            log_post_cur = (
                    norm.logpdf(D_obs, mu_D * chi_d, sigma_D) +
                    norm.logpdf(SD_obs, mu_SD * chi_sd, sigma_SD) +
                    norm.logpdf(mu_D, mu_D_prior, tau_D_prior) +
                    norm.logpdf(mu_SD, mu_SD_prior, tau_SD_prior) +
                    invgamma.logpdf(sigma_D ** 2, a=alpha_D, scale=beta_D) +
                    invgamma.logpdf(sigma_SD ** 2, a=alpha_SD, scale=beta_SD) +
                    gamma.logpdf(chi_d, a=alpha_chi_d, scale=1 / beta_chi_d) +
                    gamma.logpdf(chi_sd, a=alpha_chi_sd, scale=1 / beta_chi_sd)
            )

            log_post_prop = (
                    norm.logpdf(D_obs, mu_D_prop * chi_d_prop, sigma_D_prop) +
                    norm.logpdf(SD_obs, mu_SD_prop * chi_sd_prop, sigma_SD_prop) +
                    norm.logpdf(mu_D_prop, mu_D_prior, tau_D_prior) +
                    norm.logpdf(mu_SD_prop, mu_SD_prior, tau_SD_prior) +
                    invgamma.logpdf(sigma_D_prop ** 2, a=alpha_D, scale=beta_D) +
                    invgamma.logpdf(sigma_SD_prop ** 2, a=alpha_SD, scale=beta_SD) +
                    gamma.logpdf(chi_d_prop, a=alpha_chi_d, scale=1 / beta_chi_d) +
                    gamma.logpdf(chi_sd_prop, a=alpha_chi_sd, scale=1 / beta_chi_sd)
            )

            alpha_accept = min(1, np.exp(log_post_prop - log_post_cur))
            if np.random.rand() < alpha_accept:
                mu_D, sigma_D, chi_d = mu_D_prop, sigma_D_prop, chi_d_prop
                mu_SD, sigma_SD, chi_sd = mu_SD_prop, sigma_SD_prop, chi_sd_prop

            samples['mu_D'].append(mu_D)
            samples['sigma_D'].append(sigma_D)
            samples['chi_d'].append(chi_d)
            samples['mu_SD'].append(mu_SD)
            samples['sigma_SD'].append(sigma_SD)
            samples['chi_sd'].append(chi_sd)

        for k in samples:
            samples[k] = np.array(samples[k])

        return samples

    def _posterior_to_prior(self, posterior_samples):
        """后验转先验"""
        prior_params = {}

        for node in range(self.num_nodes):
            s = posterior_samples[node]

            chi_d_mean = np.mean(s['chi_d'])
            chi_d_var = np.var(s['chi_d'])
            chi_sd_mean = np.mean(s['chi_sd'])
            chi_sd_var = np.var(s['chi_sd'])

            prior_params[node] = {
                'mu_D': np.mean(s['mu_D']),
                'tau_D': max(10.0, np.std(s['mu_D'])),
                'mu_SD': np.mean(s['mu_SD']),
                'tau_SD': max(10.0, np.std(s['mu_SD'])),
                'alpha_D': 3.0,
                'beta_D': max(10.0, np.var(s['sigma_D']) * 2.0),
                'alpha_SD': 3.0,
                'beta_SD': max(10.0, np.var(s['sigma_SD']) * 2.0),
                'alpha_chi_d': chi_d_mean ** 2 / max(1e-6, chi_d_var),
                'beta_chi_d': chi_d_mean / max(1e-6, chi_d_var),
                'alpha_chi_sd': chi_sd_mean ** 2 / max(1e-6, chi_sd_var),
                'beta_chi_sd': chi_sd_mean / max(1e-6, chi_sd_var),
                'chi_d': chi_d_mean,
                'chi_sd': chi_sd_mean
            }

        return prior_params

    def run(self):
        """
        运行贝叶斯状态预测

        🔧 t=1: 真实φ直接确定状态
        🔧 t≥2: MC采样确定状态
        """
        results = {}

        # t=0
        print(f"\n{'=' * 80}")
        print(f"📌 PERIOD 0: Prior + Initial State Probabilities (Dynamically Generated)")
        print(f"{'=' * 80}")

        prior_params = {}
        for node in range(self.num_nodes):
            prior_params[node] = {
                'mu_D': self.prior_mu_D[node],
                'tau_D': self.prior_tau_D[node],
                'mu_SD': self.prior_mu_SD[node],
                'tau_SD': self.prior_tau_SD[node],
                'alpha_D': self.prior_alpha_D[node],
                'beta_D': self.prior_beta_D[node],
                'alpha_SD': self.prior_alpha_SD[node],
                'beta_SD': self.prior_beta_SD[node],
                'alpha_chi_d': self.prior_alpha_chi_d[node],
                'beta_chi_d': self.prior_beta_chi_d[node],
                'alpha_chi_sd': self.prior_alpha_chi_sd[node],
                'beta_chi_sd': self.prior_beta_chi_sd[node],
                'chi_d': self.prior_alpha_chi_d[node] / self.prior_beta_chi_d[node],
                'chi_sd': self.prior_alpha_chi_sd[node] / self.prior_beta_chi_sd[node]
            }

        results['period_0'] = {
            'state_probabilities': self.t0_state_probs,
            'conditional_probability_tables': self.t0_cpts,
            'observed_state': {},
            'disruption_level': self.disruption_level,
            'type': 'INITIAL_PROBABILITIES_ONLY'
        }

        print(f"✅ Period 0 initialized with dynamic parameters")

        current_prior = prior_params

        for t in range(1, self.num_periods):
            print(f"\n{'=' * 80}")
            print(f"🎯 PERIOD {t}")
            print(f"{'=' * 80}")

            # Step 1: 获取观测
            if t == 1:
                print(f"  📊 Step 1: Using ACTUAL observations")
                observations_t = self._get_actual_observations(t)
                observation_type = "ACTUAL"

                for node in range(min(3, self.num_nodes)):  # 只打印前3个节点
                    obs = observations_t[node]
                    node_type = self._get_node_type(node)
                    print(
                        f"       Node {node} ({node_type}): D={obs['D_obs']:.0f}, SD={obs['SD_obs']:.0f}, φ={obs['phi_actual']:.3f}")
            else:
                print(f"  🔮 Step 1: Generating PREDICTED observations")
                prev_posterior = results[f'period_{t - 1}']['posterior_samples']
                observations_t = self._generate_predicted_observations(prev_posterior, t)
                observation_type = "PREDICTED"

                for node in range(min(3, self.num_nodes)):
                    obs = observations_t[node]
                    node_type = self._get_node_type(node)
                    print(
                        f"       Node {node} ({node_type}): D̂={obs['D_obs']:.0f}, ŜD={obs['SD_obs']:.0f}, φ̂={obs['phi_actual']:.3f}")

            # Step 2: 贝叶斯更新
            print(f"\n  🔄 Step 2: Bayesian update (MCMC N={self.mcmc_samples})")
            posterior_t = self._bayesian_update(observations_t, current_prior)
            print(f"       ✅ Posterior computed for {self.num_nodes} nodes")

            # Step 3: 确定状态
            if t == 1:
                # 🔧 t=1: 直接从真实φ确定状态
                predicted_states, state_probs = self._determine_states_from_actual_phi(observations_t)
                fulfillment_rates = None
            else:
                # t≥2: MC采样确定状态
                predicted_states, state_probs, fulfillment_rates = self._predict_states_from_posterior(posterior_t, t)

            # 存储结果
            results[f'period_{t}'] = {
                'observations': observations_t,
                'state_probs': state_probs,
                'observed_state': predicted_states,
                'fulfillment_rates': fulfillment_rates,
                'posterior_samples': posterior_t,
                'disruption_level': self.disruption_level,
                'type': 'DISRUPTION_OBSERVATIONS' if t == 1 else 'PREDICTED_OBSERVATIONS'
            }

            # Step 4: 滚动更新
            current_prior = self._posterior_to_prior(posterior_t)

        self.posterior_history = results
        return results

    def print_summary(self, results):
        """打印摘要"""
        print("\n" + "=" * 80)
        print("BAYESIAN STATE PREDICTION SUMMARY (Random Experiment Version)")
        print("=" * 80)
        print(f"Disruption: {self.disruption_level.upper() if self.disruption_level else 'N/A'}")
        print(f"Nodes: {self.num_nodes}, States: {self.num_states}, Periods: {self.num_periods}")
        print(f"t=1: States from actual φ (deterministic)")
        print(f"t≥2: States from MC sampling (probabilistic)")

        # 状态名称（适应多状态）
        if self.num_states == 2:
            state_names = {0: "Poor", 1: "Good"}
        elif self.num_states == 3:
            state_names = {0: "Poor", 1: "Moderate", 2: "Excellent"}
        else:
            state_names = {i: f"State_{i}" for i in range(self.num_states)}

        for period_key, period_data in results.items():
            period_num = period_key.split('_')[1]

            if period_data['type'] == 'INITIAL_PROBABILITIES_ONLY':
                print(f"\n📌 PERIOD {period_num}: Initial State Probabilities (Dynamically Generated)")

            else:
                print(f"\n📌 PERIOD {period_num}: {period_data['type'].replace('_', ' ').title()}")

                for n in range(min(5, self.num_nodes)):  # 只打印前5个节点
                    node_type = self._get_node_type(n)
                    obs = period_data['observations'][n]
                    observed_state = period_data['observed_state'][n]
                    state_name = state_names.get(observed_state, f"State_{observed_state}")

                    if period_num == '1':
                        # t=1: 显示真实φ
                        print(f"   Node {n} ({node_type}): D={obs['D_obs']:.0f}, SD={obs['SD_obs']:.0f}, "
                              f"φ_actual={obs['phi_actual']:.3f} → State {observed_state} ({state_name})")
                    else:
                        # t≥2: 显示MC统计
                        phi_mean = np.mean(period_data['fulfillment_rates'][n])
                        probs = period_data['state_probs'][n]
                        print(f"   Node {n} ({node_type}): D̂={obs['D_obs']:.0f}, ŜD={obs['SD_obs']:.0f}, "
                              f"E[φ]={phi_mean:.3f}, P={[f'{p:.2f}' for p in probs]} → State {observed_state} ({state_name})")

        print("=" * 80)

    def export_to_excel(self, results, filename=None):
        """导出Excel"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bayesian_{self.disruption_level}_{self.num_nodes}nodes_{timestamp}.xlsx"

        rows = []

        # 状态名称（适应多状态）
        if self.num_states == 2:
            state_names = {0: "Poor", 1: "Good"}
        elif self.num_states == 3:
            state_names = {0: "Poor", 1: "Moderate", 2: "Excellent"}
        else:
            state_names = {i: f"State_{i}" for i in range(self.num_states)}

        for period_key, period_data in results.items():
            period_num = period_key.split('_')[1]

            if period_data['type'] == 'INITIAL_PROBABILITIES_ONLY':
                for node, probs in period_data['state_probabilities'].items():
                    rows.append({
                        "周期": period_num,
                        "节点": node,
                        "节点类型": self._get_node_type(node),
                        "数据类型": "初始概率（动态生成）",
                        "状态确定方法": "N/A",
                        "需求(D)": "N/A",
                        "满足需求(SD)": "N/A",
                        "实际φ": "N/A",
                        "预测φ均值": "N/A",
                        "状态概率": ", ".join([f"{p:.3f}" for p in probs]),
                        "状态": "N/A"
                    })
            else:
                method = "真实φ直接映射" if period_num == '1' else "MC采样"

                for node in range(self.num_nodes):
                    obs = period_data['observations'][node]
                    state = period_data['observed_state'][node]
                    probs = period_data['state_probs'][node]

                    if period_num == '1':
                        phi_pred = "N/A"
                    else:
                        phi_pred = f"{np.mean(period_data['fulfillment_rates'][node]):.3f}"

                    rows.append({
                        "周期": period_num,
                        "节点": node,
                        "节点类型": self._get_node_type(node),
                        "数据类型": period_data['type'],
                        "状态确定方法": method,
                        "需求(D)": f"{obs['D_obs']:.1f}",
                        "满足需求(SD)": f"{obs['SD_obs']:.1f}",
                        "实际φ": f"{obs['phi_actual']:.3f}",
                        "预测φ均值": phi_pred,
                        "状态概率": ", ".join([f"{p:.3f}" for p in probs]),
                        "状态": f"{state}({state_names.get(state, 'UNKNOWN')})"
                    })

        df = pd.DataFrame(rows)
        df.to_excel(filename, index=False)
        print(f"\n✅ Results exported to {filename}")


# 测试代码
if __name__ == "__main__":
    from R1_network_generate4 import generate_supply_chain_network

    print("🧪 Testing Bayesian Predictor (Random Experiment Version)")
    print("Current Date and Time (UTC): 2025-10-28 13:05:56")
    print("Current User's Login: dyy21zyy")
    print("=" * 80)

    config = {
        'total_nodes': 10,  # 🔧 10个节点
        'num_layers': 3,  # 🔧 3层网络
        'num_periods': 4,
        'num_states': 2,
        'connection_density': 0.7,
        'mcmc_samples': 500,  # 减少采样以加快测试
        'mc_samples': 500,
        'seed': 21
    }

    disruption_level = 'moderate'

    # 生成网络
    network_data = generate_supply_chain_network(
        total_nodes=config['total_nodes'],
        num_layers=config['num_layers'],
        num_periods=config['num_periods'],
        num_states=config['num_states'],
        connection_density=config['connection_density'],
        seed=config['seed'],
        network_type='random',
        verbose=False
    )

    # 生成随机观测数据（用于测试）
    print(f"\n🔧 Generating random observation data for {config['total_nodes']} nodes...")
    np.random.seed(config['seed'])

    # Moderate disruption 观测数据（动态生成）
    observed_data = {1: {}}
    for node in range(config['total_nodes']):
        D_obs = np.random.uniform(50, 150)
        # Moderate disruption: 履约率 0.3-0.6
        phi = np.random.uniform(0.3, 0.6)
        SD_obs = D_obs * phi

        observed_data[1][node] = {
            'D_obs': D_obs,
            'SD_obs': SD_obs
        }

    print(f"   ✅ Generated observation data for {config['total_nodes']} nodes")

    try:
        predictor = ImprovedBalancedBayesianPredictor(
            network_data=network_data,
            num_states=config['num_states'],
            num_periods=config['num_periods'],
            disruption_level=disruption_level,
            observed_data=observed_data,
            mcmc_samples=config['mcmc_samples'],
            mc_samples=config['mc_samples'],
            seed=config['seed']
        )

        results = predictor.run()
        predictor.print_summary(results)
        predictor.export_to_excel(results)

        print(f"\n✅ Test completed successfully!")

    except ValueError as e:
        print(f"\n❌ Error: {e}")