import os
import sys

# Ensure Python can resolve modules from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from envs.flood_env_2 import FloodEvacuationEnv
import numpy as np

def run_dijkstra_baseline():
    print("Initializing Dijkstra Greedy Baseline...")
    env = FloodEvacuationEnv(graph_path="data/processed/bengaluru_graph.pkl", max_steps=72)
    
    # Load precomputed travel time matrix
    dist_matrix_path = "data/processed/distance_matrix.pkl"
    with open(dist_matrix_path, "rb") as f:
        distance_matrix = pickle.load(f)
        
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    
    print("\n--- Starting Baseline Dispatch Operation ---")
    
    for step in range(1, 73):
        actions = []
        blocked_nodes = set(np.where(env.water_depths > 0.4)[0])
        
        # Greedy heuristic: route to the nearest accessible node with stranded population
        for v_id in range(env.num_vehicles):
            curr_node = env.idx_to_node[env.vehicle_positions[v_id]]
            best_target_idx = env.vehicle_positions[v_id]
            min_dist = float('inf')
            
            for node_idx in range(env.num_nodes):
                if node_idx in blocked_nodes or env.remaining_population[node_idx] <= 0:
                    continue
                target_node = env.idx_to_node[node_idx]
                dist = distance_matrix.get(curr_node, {}).get(target_node, float('inf'))
                if dist < min_dist:
                    min_dist = dist
                    best_target_idx = node_idx
                    
            actions.append(best_target_idx)
            
        actions = np.array(actions, dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        if step % 10 == 0 or terminated or truncated:
            print(f"Hour {step:2} | Baseline Rescued: {info['total_rescued']:>6} | Blocked Roads: {info['blocked_nodes_count']}")
            
        if terminated or truncated:
            break
            
    print("\n" + "="*35)
    print("BASELINE DIJKSTRA FINAL METRICS")
    print("="*35)
    print(f"Total People Rescued : {info['total_rescued']}")
    print(f"Equity Variance      : {info['equity_variance']:.5f}")
    print(f"Total Reward         : {total_reward:.2f}")
    print("="*35)

if __name__ == "__main__":
    run_dijkstra_baseline()