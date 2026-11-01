import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Ensure this matches your file name (e.g., envs.flood_env_2 if you are still using that)
from envs.flood_env import FloodEvacuationEnv

def deep_training_run():
    print("Initializing environment for deep convergence training...")
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = Monitor(FloodEvacuationEnv(graph_path="data/processed/bengaluru_graph.pkl", max_steps=72))
    eval_env = Monitor(FloodEvacuationEnv(graph_path="data/processed/bengaluru_graph.pkl", max_steps=72))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_models/',
        log_path='./logs/',
        eval_freq=10000, 
        deterministic=True,
        render=False
    )
    
    # Load the fine-tuned model that just achieved the 7% improvement
    model_path = "./models/ppo_flood_finetuned"
    print(f"Loading checkpoint: {model_path}...")
    model = PPO.load(model_path, env=env, device="cpu")
    
    # 200,000 steps to reach the 15% improvement threshold
    TOTAL_TIMESTEPS = 200000
    print(f"Starting deep learning loop for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    final_path = "./models/ppo_flood_15_percent"
    model.save(final_path)
    print(f"Training complete. Saved to {final_path}.zip")
    
if __name__ == "__main__":
    deep_training_run()