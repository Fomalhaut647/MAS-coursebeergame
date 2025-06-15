import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import copy


def set_seed(seed=42):
    """
    设置所有随机种子以确保结果可复现
    
    :param seed: 随机种子值，默认为42
    """
    # 设置Python内置random模块的种子
    random.seed(seed)
    
    # 设置NumPy的随机种子
    np.random.seed(seed)
    
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    
    # 如果使用CUDA，设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        # 设置CUDA的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {seed}")


# --- 1. 环境定义 ---
class Env:
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100):
        """
        初始化供应链管理仿真环境。
        
        :param num_firms: 企业数量
        :param p: 各企业的价格列表
        :param h: 库存持有成本
        :param c: 损失销售成本
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业需求的泊松分布均值
        :param max_steps: 每个episode的最大步数
        """
        self.num_firms = num_firms
        self.p = p
        self.h = h
        self.c = c
        self.poisson_lambda = poisson_lambda
        self.max_steps = max_steps
        self.initial_inventory = initial_inventory
        self.reset()

    def reset(self):
        """重置环境状态"""
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """获取每个企业的观察信息"""
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """生成每个企业的需求"""
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                demand[i] = self.orders[i - 1]
        return demand

    def step(self, actions):
        """执行一个时间步的仿真"""
        self.orders = actions
        self.demand = self._generate_demand()

        for i in range(self.num_firms):
            self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])

        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]

        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))

        for i in range(self.num_firms):
            purchase_cost = (self.p[i + 1] if i + 1 < len(self.p) - 1 else 0) * self.orders[i]
            rewards[i] += self.p[i] * self.satisfied_demand[i] - purchase_cost - self.h * self.inventory[i]

            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        rewards -= loss_sales

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), rewards, self.done


# --- 2. 网络定义 ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --- 3. 智能体定义 ---
class DQNAgent:
    """标准DQN智能体（基线）"""

    def __init__(self, state_size, action_size, firm_id, max_order=20, buffer_size=10000, batch_size=64, gamma=0.99,
                 learning_rate=1e-3, tau=1e-3, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()) + 1
        else:
            return random.randint(1, self.max_order)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(np.vstack([a - 1 for a in actions])).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.q_network.state_dict(), filename)
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        """加载已保存的模型权重"""
        if os.path.isfile(filename):
            self.q_network.load_state_dict(torch.load(filename))
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"从 {filename} 加载了模型")
            return True
        else:
            print(f"错误：在 {filename} 找不到模型")
            return False


class DoubleDQNAgent(DQNAgent):
    """改进1：Double DQN智能体"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self, experiences):
        """重写学习方法实现Double DQN"""
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(np.vstack([a - 1 for a in actions])).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        # Double DQN: 用主网络选择动作，用目标网络计算Q值
        best_actions_next = self.q_network(next_states).argmax(1).unsqueeze(1)
        Q_targets_next = self.target_network(next_states).gather(1, best_actions_next)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()


# --- 4. 训练和测试函数 ---
def train_single_agent(env, agent, num_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """训练单个智能体（用于DQN和DoubleDQN）"""
    scores = []
    eps = eps_start

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0

        for t in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, eps)
                    actions[firm_id] = action
                else:  # 其他企业随机行动
                    actions[firm_id] = np.random.randint(1, 21)

            next_state, rewards, done = env.step(actions)
            reward = rewards[agent.firm_id][0]

            agent.step(state[agent.firm_id].reshape(1, -1), actions[agent.firm_id], reward,
                       next_state[agent.firm_id].reshape(1, -1), done)

            state = next_state
            score += reward
            if done: break

        eps = max(eps_end, eps_decay * eps)
        scores.append(score)

        if i_episode % 100 == 0:
            print(f'回合 {i_episode}/{num_episodes} | 平均得分: {np.mean(scores[-100:]):.2f}')

    agent.save(f'models/{type(agent).__name__}_firm_{agent.firm_id}.pth')
    return scores


def train_marl(env, agents, num_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """改进2：训练多个智能体（独立学习）"""
    scores = []  # 记录所有智能体的总奖励
    eps = eps_start

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_score = 0

        for t in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for firm_id, agent in enumerate(agents):
                firm_state = state[firm_id].reshape(1, -1)
                action = agent.act(firm_state, eps)
                actions[firm_id] = action

            next_state, rewards, done = env.step(actions)

            for firm_id, agent in enumerate(agents):
                reward = rewards[firm_id][0]
                agent.step(state[firm_id].reshape(1, -1), actions[firm_id], reward, next_state[firm_id].reshape(1, -1),
                           done)

            state = next_state
            episode_score += np.sum(rewards)
            if done: break

        eps = max(eps_end, eps_decay * eps)
        scores.append(episode_score)

        if i_episode % 100 == 0:
            print(f'回合 {i_episode}/{num_episodes} | 平均总得分: {np.mean(scores[-100:]):.2f}')

    for agent in agents:
        agent.save(f'models/IDQN_firm_{agent.firm_id}.pth')
    return scores


def test_policy(env, agents):
    """测试最终策略，返回一个回合中所有企业的订单历史"""
    state = env.reset()
    orders_history = [[] for _ in agents]
    inventory_history = [[] for _ in agents]
    demand_history = [[] for _ in agents]

    for t in range(env.max_steps):
        actions = np.zeros((env.num_firms, 1))
        for i, agent in enumerate(agents):
            if isinstance(agent, (DQNAgent, DoubleDQNAgent)):
                firm_state = state[i].reshape(1, -1)
                actions[i] = agent.act(firm_state, epsilon=0.0)  # 使用确定性策略
            else:  # 非智能体企业的随机策略
                actions[i] = np.random.randint(1, 21)

        next_state, _, done = env.step(actions)
        state = next_state

        for i in range(len(agents)):
            orders_history[i].append(actions[i][0])
            inventory_history[i].append(env.inventory[i][0])
            demand_history[i].append(env.demand[i][0])

        if done: break

    return orders_history, inventory_history, demand_history


# --- 5. 绘图函数（中文） ---
def plot_learning_curves(results, title='学习曲线对比'):
    """绘制学习曲线对比"""
    plt.figure(figsize=(12, 7))
    for label, scores in results.items():
        moving_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')
        plt.plot(moving_avg, label=label)

    plt.title(title, fontsize=16)
    plt.xlabel('回合数', fontsize=12)
    plt.ylabel('移动平均奖励', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(f'figures/{title.replace(" ", "_")}.png')
    plt.show()


def plot_policy_comparison(policy_results, title='最终策略对比'):
    """绘制策略对比"""
    num_agents = len(policy_results[list(policy_results.keys())[0]][0])

    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 4 * num_agents), sharex=True)
    if num_agents == 1: axes = [axes]

    for label, (orders_hist, _, _) in policy_results.items():
        for i in range(num_agents):
            if len(orders_hist[i]) > 0:
                axes[i].plot(orders_hist[i], label=f'{label} - 订单量', marker='o', alpha=0.7)

    for i in range(num_agents):
        # 使用一致的需求历史作为参考
        if "IDQN" in policy_results:
            _, _, demand_hist = policy_results["IDQN"]
            if len(demand_hist[i]) > 0:
                axes[i].plot(demand_hist[i], label=f'企业 {i} 需求', linestyle='--', color='gray')

        axes[i].set_title(f'企业 {i} 订购策略', fontsize=14)
        axes[i].set_ylabel('数量', fontsize=12)
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('时间步', fontsize=12)
    fig.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'figures/{title.replace(" ", "_")}.png')
    plt.show()


def plot_comprehensive_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history, agent_name="DQN"):
    """绘制单个智能体综合测试结果"""
    # 计算平均值
    avg_inventory = np.mean(inventory_history, axis=0)
    avg_orders = np.mean(orders_history, axis=0)
    avg_demand = np.mean(demand_history, axis=0)
    avg_satisfied_demand = np.mean(satisfied_demand_history, axis=0)
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 库存图表
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title(f'{agent_name} - 平均库存变化')
    axs[0, 0].set_xlabel('时间步')
    axs[0, 0].set_ylabel('库存量')
    axs[0, 0].grid(True)
    
    # 订单图表
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title(f'{agent_name} - 平均订单量变化')
    axs[0, 1].set_xlabel('时间步')
    axs[0, 1].set_ylabel('订单量')
    axs[0, 1].grid(True)
    
    # 需求和满足需求图表
    axs[1, 0].plot(avg_demand, label='需求')
    axs[1, 0].plot(avg_satisfied_demand, label='满足的需求')
    axs[1, 0].set_title(f'{agent_name} - 平均需求 vs 满足的需求')
    axs[1, 0].set_xlabel('时间步')
    axs[1, 0].set_ylabel('数量')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 奖励柱状图
    axs[1, 1].bar(range(len(test_scores)), test_scores)
    axs[1, 1].set_title(f'{agent_name} - 测试回合奖励')
    axs[1, 1].set_xlabel('回合')
    axs[1, 1].set_ylabel('总奖励')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/{agent_name}_comprehensive_results.png')
    plt.show()


def plot_marl_comprehensive_results(total_scores, individual_scores, inventory_history, orders_history, demand_history, satisfied_demand_history, system_name="IDQN"):
    """绘制多智能体系统综合测试结果"""
    num_agents = len(individual_scores)
    
    # 创建大图表
    fig = plt.figure(figsize=(16, 12))
    
    # 总奖励图
    ax1 = plt.subplot(3, 3, 1)
    plt.bar(range(len(total_scores)), total_scores)
    plt.title(f'{system_name} - 系统总奖励')
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.grid(True)
    
    # 个体奖励对比
    ax2 = plt.subplot(3, 3, 2)
    for i in range(num_agents):
        plt.plot(individual_scores[i], label=f'企业 {i}', marker='o')
    plt.title(f'{system_name} - 个体奖励对比')
    plt.xlabel('回合')
    plt.ylabel('个体奖励')
    plt.legend()
    plt.grid(True)
    
    # 平均个体奖励柱状图
    ax3 = plt.subplot(3, 3, 3)
    avg_individual_scores = [np.mean(scores) for scores in individual_scores]
    plt.bar(range(num_agents), avg_individual_scores)
    plt.title(f'{system_name} - 平均个体奖励')
    plt.xlabel('企业编号')
    plt.ylabel('平均奖励')
    plt.grid(True)
    
    # 每个智能体的库存变化
    for i in range(num_agents):
        ax = plt.subplot(3, 3, 4 + i)
        avg_inventory = np.mean(inventory_history[i], axis=0)
        plt.plot(avg_inventory)
        plt.title(f'企业 {i} - 平均库存变化')
        plt.xlabel('时间步')
        plt.ylabel('库存量')
        plt.grid(True)
    
    # 所有智能体的订单量对比
    ax7 = plt.subplot(3, 3, 7)
    for i in range(num_agents):
        avg_orders = np.mean(orders_history[i], axis=0)
        plt.plot(avg_orders, label=f'企业 {i}', marker='o', alpha=0.7)
    plt.title(f'{system_name} - 订单量对比')
    plt.xlabel('时间步')
    plt.ylabel('订单量')
    plt.legend()
    plt.grid(True)
    
    # 所有智能体的需求vs满足需求
    ax8 = plt.subplot(3, 3, 8)
    for i in range(num_agents):
        avg_demand = np.mean(demand_history[i], axis=0)
        avg_satisfied = np.mean(satisfied_demand_history[i], axis=0)
        plt.plot(avg_demand, label=f'企业 {i} 需求', linestyle='--', alpha=0.7)
        plt.plot(avg_satisfied, label=f'企业 {i} 满足', alpha=0.7)
    plt.title(f'{system_name} - 需求 vs 满足需求')
    plt.xlabel('时间步')
    plt.ylabel('数量')
    plt.legend()
    plt.grid(True)
    
    # 系统效率指标
    ax9 = plt.subplot(3, 3, 9)
    efficiency_rates = []
    for i in range(num_agents):
        total_demand = np.sum([np.sum(episode) for episode in demand_history[i]])
        total_satisfied = np.sum([np.sum(episode) for episode in satisfied_demand_history[i]])
        efficiency = total_satisfied / total_demand if total_demand > 0 else 0
        efficiency_rates.append(efficiency)
    
    plt.bar(range(num_agents), efficiency_rates)
    plt.title(f'{system_name} - 需求满足率')
    plt.xlabel('企业编号')
    plt.ylabel('满足率')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/{system_name}_comprehensive_results.png')
    plt.show()


def test_agent_comprehensive(env, agent, num_episodes=10):
    """综合测试单个智能体性能"""
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []
        
        for t in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, epsilon=0.0)
                    actions[firm_id] = action
                else:
                    actions[firm_id] = np.random.randint(1, 21)
            
            next_state, rewards, done = env.step(actions)
            
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])
            
            reward = rewards[agent.firm_id][0]
            score += reward
            state = next_state
            
            if done:
                break
        
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)
        
        print(f'测试回合 {i_episode}/{num_episodes} | 得分: {score:.2f}')
    
    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history


def test_marl_comprehensive(env, agents, num_episodes=10):
    """综合测试多智能体系统性能"""
    total_scores = []
    individual_scores = [[] for _ in agents]
    inventory_history = [[] for _ in agents]
    orders_history = [[] for _ in agents]
    demand_history = [[] for _ in agents]
    satisfied_demand_history = [[] for _ in agents]
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        episode_total_score = 0
        episode_individual_scores = [0] * len(agents)
        episode_inventory = [[] for _ in agents]
        episode_orders = [[] for _ in agents]
        episode_demand = [[] for _ in agents]
        episode_satisfied_demand = [[] for _ in agents]
        
        for t in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for firm_id, agent in enumerate(agents):
                firm_state = state[firm_id].reshape(1, -1)
                action = agent.act(firm_state, epsilon=0.0)
                actions[firm_id] = action
            
            next_state, rewards, done = env.step(actions)
            
            # 记录每个智能体的数据
            for firm_id in range(len(agents)):
                episode_inventory[firm_id].append(env.inventory[firm_id][0])
                episode_orders[firm_id].append(actions[firm_id][0])
                episode_demand[firm_id].append(env.demand[firm_id][0])
                episode_satisfied_demand[firm_id].append(env.satisfied_demand[firm_id][0])
                
                reward = rewards[firm_id][0]
                episode_individual_scores[firm_id] += reward
                episode_total_score += reward
            
            state = next_state
            if done:
                break
        
        total_scores.append(episode_total_score)
        for firm_id in range(len(agents)):
            individual_scores[firm_id].append(episode_individual_scores[firm_id])
            inventory_history[firm_id].append(episode_inventory[firm_id])
            orders_history[firm_id].append(episode_orders[firm_id])
            demand_history[firm_id].append(episode_demand[firm_id])
            satisfied_demand_history[firm_id].append(episode_satisfied_demand[firm_id])
        
        print(f'测试回合 {i_episode}/{num_episodes} | 总得分: {episode_total_score:.2f} | 个体得分: {[f"{score:.1f}" for score in episode_individual_scores]}')
    
    return total_scores, individual_scores, inventory_history, orders_history, demand_history, satisfied_demand_history


# --- 6. 主执行模块 ---
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    set_seed(42)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['PingFang HK']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # 环境参数
    NUM_FIRMS = 3
    P = [10, 8, 6, 4]
    H = 0.5
    C = 2
    INITIAL_INVENTORY = 100
    POISSON_LAMBDA = 10
    MAX_STEPS = 100

    env = Env(NUM_FIRMS, P, H, C, INITIAL_INVENTORY, POISSON_LAMBDA, MAX_STEPS)
    STATE_SIZE = 3
    ACTION_SIZE = 30
    NUM_EPISODES = 2000

    learning_results = {}
    policy_results = {}

    # --- 实验1：运行基线DQN ---
    print("--- 训练基线DQN ---")
    baseline_agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, firm_id=1, max_order=ACTION_SIZE)
    learning_results['基线DQN'] = train_single_agent(env, baseline_agent, num_episodes=NUM_EPISODES)
    test_agents_baseline = [None] * NUM_FIRMS
    test_agents_baseline[1] = baseline_agent
    policy_results['基线DQN'] = test_policy(env, test_agents_baseline)

    # --- 实验2：运行改进1（Double DQN） ---
    print("\n--- 训练Double DQN ---")
    double_dqn_agent = DoubleDQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, firm_id=1, max_order=ACTION_SIZE)
    learning_results['Double DQN'] = train_single_agent(env, double_dqn_agent, num_episodes=NUM_EPISODES)
    test_agents_double = [None] * NUM_FIRMS
    test_agents_double[1] = double_dqn_agent
    policy_results['Double DQN'] = test_policy(env, test_agents_double)

    # --- 实验3：运行改进2（独立DQN - MARL） ---
    print("\n--- 训练多智能体（IDQN） ---")
    marl_agents = [DoubleDQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, firm_id=i, max_order=ACTION_SIZE) for
                   i in range(NUM_FIRMS)]
    learning_results['IDQN（多智能体）'] = train_marl(env, marl_agents, num_episodes=NUM_EPISODES)
    policy_results['IDQN'] = test_policy(env, marl_agents)

    # --- 可视化结果 ---
    plot_learning_curves(learning_results, title='学习曲线对比')
    
    # 单智能体策略对比
    single_agent_policy_results = {k: v for k, v in policy_results.items() if k != 'IDQN'}
    plot_policy_comparison(single_agent_policy_results, title='单智能体最终策略对比（企业1）')
    
    # 多智能体策略对比
    multi_agent_policy_results = {'IDQN': policy_results['IDQN']}
    plot_policy_comparison(multi_agent_policy_results, title='多智能体最终策略对比（所有企业）')

    # --- 综合测试 ---
    print("\n--- 进行综合测试 ---")
    
    # 测试基线DQN
    baseline_test_results = test_agent_comprehensive(env, baseline_agent, num_episodes=10)
    plot_comprehensive_results(*baseline_test_results, agent_name="基线DQN")
    
    # 测试Double DQN
    double_test_results = test_agent_comprehensive(env, double_dqn_agent, num_episodes=10)
    plot_comprehensive_results(*double_test_results, agent_name="DoubleDQN")
    
    # 测试IDQN（多智能体）
    print("\n--- 测试IDQN多智能体系统 ---")
    idqn_test_results = test_marl_comprehensive(env, marl_agents, num_episodes=10)
    plot_marl_comprehensive_results(*idqn_test_results, system_name="IDQN多智能体")
    
    print("\n--- 训练和测试完成！ ---")
    print(f"基线DQN平均测试得分: {np.mean(baseline_test_results[0]):.2f}")
    print(f"Double DQN平均测试得分: {np.mean(double_test_results[0]):.2f}")
    print(f"IDQN系统平均总得分: {np.mean(idqn_test_results[0]):.2f}")
    print(f"IDQN各企业平均得分: {[f'{np.mean(scores):.2f}' for scores in idqn_test_results[1]]}")
