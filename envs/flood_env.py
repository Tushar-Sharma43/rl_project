import os
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FloodEvacuationEnv(gym.Env):
    """
    Gymnasium environment for urban flood evacuation routing.
    Designed for the Bengaluru NetworkX road graph.
    """
    # 4 FPS max for visual rendering (if implemented later)
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, graph_path="data/processed/bengaluru_graph.pkl", max_steps=72, num_vehicles=50):
        super().__init__()
        
        # 1. Load the preprocessed NetworkX road network
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at {graph_path}. Run Step 2 first.")
            
        with open(graph_path, "rb") as f:
            self.base_graph = pickle.load(f)
            
        self.nodes = list(self.base_graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        
        # Identify designated shelters from node attributes
        self.shelter_indices = [
            self.node_to_idx[n] for n in self.nodes 
            if self.base_graph.nodes[n].get("is_shelter", 0) == 1
        ]
        self.num_shelters = len(self.shelter_indices)
        
        # 2. Simulation parameters
        self.max_steps = max_steps          # Default 72 hours (3-day event)
        self.num_vehicles = num_vehicles    # Default fleet of 50 rescue vehicles
        self.vehicle_capacity = 20          # Persons evacuated per trip
        self.current_step = 0
        
        # 3. Define Action Space
        # MultiDiscrete space: Each of the 50 vehicles selects one of the available target nodes
        self.action_space = spaces.MultiDiscrete([self.num_nodes] * self.num_vehicles)
        
        # 4. Define Observation Space (Box normalized between 0.0 and 1.0)
        # Vector size = (Water depths) + (Remaining pop ratios) + (Shelter occupancies) + (Vehicle normalized positions)
        obs_dim = (self.num_nodes * 2) + self.num_shelters + self.num_vehicles
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Clone the graph so dynamic attributes don't corrupt the base graph
        self.graph = self.base_graph.copy()
        
        # Initialize NumPy arrays for fast vectorized environment updates
        self.water_depths = np.zeros(self.num_nodes, dtype=np.float32)
        self.initial_population = np.zeros(self.num_nodes, dtype=np.float32)
        self.remaining_population = np.zeros(self.num_nodes, dtype=np.float32)
        self.evacuated_per_zone = np.zeros(self.num_nodes, dtype=np.float32)
        
        for idx, node in enumerate(self.nodes):
            pop = float(self.graph.nodes[node].get("population", 500))
            self.initial_population[idx] = pop
            self.remaining_population[idx] = pop
            
        # Shelter capacities and occupancies arrays
        self.shelter_capacities = np.array([
            float(self.graph.nodes[self.idx_to_node[s_idx]].get("shelter_capacity", 1500))
            for s_idx in self.shelter_indices
        ], dtype=np.float32)
        self.shelter_occupancies = np.zeros(self.num_shelters, dtype=np.float32)
        
        # Initialize vehicle positions at random shelter nodes
        self.vehicle_positions = np.random.choice(self.shelter_indices, size=self.num_vehicles)
        
        # Set storm intensity profile for this specific training episode (randomized peak rainfall)
        self.rainfall_rate = float(self.np_random.uniform(0.05, 0.15))
        
        obs = self._get_observation()
        info = {"total_evacuated": 0.0, "active_shelters": self.num_shelters}
        return obs, info

    def _get_observation(self):
        """Constructs the normalized 1D state vector for the PPO agent."""
        # 1. Normalized water depth (clipped at 2.0 meters max)
        norm_depth = np.clip(self.water_depths / 2.0, 0.0, 1.0)
        
        # 2. Remaining population ratio
        norm_pop = np.divide(
            self.remaining_population, 
            np.maximum(self.initial_population, 1.0), 
            dtype=np.float32
        )
        
        # 3. Shelter occupancy ratio
        norm_shelter = np.divide(
            self.shelter_occupancies, 
            np.maximum(self.shelter_capacities, 1.0), 
            dtype=np.float32
        )
        
        # 4. Normalized vehicle positions
        norm_vehicles = (self.vehicle_positions / max(1, self.num_nodes - 1)).astype(np.float32)
        
        return np.concatenate([norm_depth, norm_pop, norm_shelter, norm_vehicles]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # --- PHASE A: Propagate Flood Dynamics ---
        # Water accumulates based on the episode's rainfall profile
        flood_increment = self.np_random.normal(self.rainfall_rate, 0.02, size=self.num_nodes)
        self.water_depths += np.clip(flood_increment, 0.0, None)
        
        # Block nodes where water depth exceeds 0.4 meters (impassable threshold)
        blocked_nodes = set(np.where(self.water_depths > 0.4)[0])

        # --- PHASE B: Execute Vehicle Dispatch and Rescue Operations ---
        step_rescued = 0
        transit_penalties = 0.0
        
        # Loop through each vehicle and process its chosen target node action
        for v_id, target_node_idx in enumerate(action):
            
            # Agent penalized for attempting to route to submerged/impassable nodes
            if target_node_idx in blocked_nodes:
                transit_penalties += 0.5 
                continue
                
            # If the vehicle reaches a node with stranded citizens, evacuate up to capacity
            if self.remaining_population[target_node_idx] > 0:
                people_to_pick = min(self.vehicle_capacity, self.remaining_population[target_node_idx])
                
                self.remaining_population[target_node_idx] -= people_to_pick
                self.evacuated_per_zone[target_node_idx] += people_to_pick
                step_rescued += people_to_pick
                
                # Automatically route evacuees to the first shelter with available capacity
                for s_local_idx, s_node_idx in enumerate(self.shelter_indices):
                    available_cap = self.shelter_capacities[s_local_idx] - self.shelter_occupancies[s_local_idx]
                    if available_cap > 0:
                        placed = min(people_to_pick, available_cap)
                        self.shelter_occupancies[s_local_idx] += placed
                        break
            
            # Update vehicle location to the target node
            self.vehicle_positions[v_id] = target_node_idx

        # --- PHASE C: Compute Equity-Aware Reward ---
        # R = (w1 * Evacuated) - (w2 * Inequity_Penalty) - (w3 * Transit_Penalty)
        evac_ratios = np.divide(
            self.evacuated_per_zone, 
            np.maximum(self.initial_population, 1.0)
        )
        # The core innovation: Variance minimization across zones acts as the equity penalty
        equity_variance = float(np.var(evac_ratios))
        
        reward = (float(step_rescued) * 1.0) - (equity_variance * 50.0) - (transit_penalties * 0.1)
        
        # --- PHASE D: Check Termination Conditions ---
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Terminate successfully if all citizens are evacuated
        if np.sum(self.remaining_population) <= 0:
            terminated = True
            reward += 100.0  # Completion bonus
            
        obs = self._get_observation()
        info = {
            "step_rescued": step_rescued,
            "total_rescued": float(np.sum(self.evacuated_per_zone)),
            "equity_variance": equity_variance,
            "blocked_nodes_count": len(blocked_nodes)
        }
        
        return obs, reward, terminated, truncated, info