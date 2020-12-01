import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLander-v2')
print("step 1")
# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
print("step 2")
# Train the agent
model.learn(total_timesteps=int(10))
print("step 3")
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading
print("step 4")
# Load the trained agent
model = DQN.load("dqn_lunar")
print("step 5")
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(mean_reward)
print("Test Sucessfull")