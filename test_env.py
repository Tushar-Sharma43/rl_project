import numpy as np
from flood_response_ai.envs.flood_env_2 import FloodEvacuationEnv

def test_environment():
    print("Initializing FloodEvacuationEnv...")
    # Initialize with a short max_steps for testing
    env = FloodEvacuationEnv(graph_path="data/processed/bengaluru_graph.pkl", max_steps=10)

    print("\n--- Testing Reset ---")
    obs, info = env.reset()
    print(f"Observation space shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial Info: {info}")

    print("\n--- Testing Step Loop ---")
    total_reward = 0.0
    
    # Run 5 simulation hours
    for step in range(1, 6):
        # Sample a random routing action for all 50 vehicles
        action = env.action_space.sample()
        
        # Advance the environment by one step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step} | Rescued: {info['step_rescued']:>4} | Blocked Nodes: {info['blocked_nodes_count']:>4} | Reward: {reward:7.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}.")
            break

    print(f"\nEnvironment check passed! Cumulative reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_environment()