import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v1")

alpha = 0.1
gamma = 0.99
episodes = 500

n_states = env.observation_space.n
n_actions = env.action_space.n      

def q_learning(strategy):

    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for episode in range(episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0

        if strategy == "high":
            epsilon = 0.5

        elif strategy == "low":
            epsilon = 0.05

        elif strategy == "decay":
            epsilon = max(0.01, 1.0 * (0.995 ** episode))

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return rewards_per_episode     


high_rewards = q_learning("high")
low_rewards = q_learning("low")
decay_rewards = q_learning("decay")


plt.figure(figsize=(12, 6))

plt.plot(high_rewards, label="High Exploration (ε=0.5)")
plt.plot(low_rewards, label="Low Exploration (ε=0.05)")
plt.plot(decay_rewards, label="Decaying Exploration")

plt.xlabel("Episode")
plt.ylabel("Sum of Rewards")
plt.title("Q-Learning on CliffWalking-v1")
plt.legend()

plt.show()