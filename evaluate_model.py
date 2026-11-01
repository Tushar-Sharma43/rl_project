import numpy as np
from stable_baselines3 import PPO
from envs.flood_env_2 import FloodEvacuationEnv

def evaluate_agent():
    print("Loading environment and trained model...")
    # Load the environment
    env = FloodEvacuationEnv(graph_path="data/processed/bengaluru_graph.pkl", max_steps=72)
    
    # Load the 200,000 step PPO model
    model_path = "./models/best_models/best_model"
    try:
        model = PPO.load(model_path)
        print(f"Model {model_path} loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    obs, info = env.reset()
    total_reward = 0.0
    
    print("\n--- Starting AI Rescue Operation ---")
    
    for step in range(1, 73):
        # deterministic=True forces the agent to use its optimal learned policy instead of guessing
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0 or terminated or truncated:
            print(f"Hour {step:2} | Total Rescued: {info['total_rescued']:>6} | Blocked Roads: {info['blocked_nodes_count']}")
            
        if terminated or truncated:
            print(f"\nStorm simulation ended at hour {step}.")
            break
            
    print("\n" + "="*30)
    print("FINAL MISSION METRICS")
    print("="*30)
    print(f"Total People Rescued : {info['total_rescued']}")
    print(f"Equity Variance      : {info['equity_variance']:.5f}")
    print(f"Total Model Reward   : {total_reward:.2f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_agent()