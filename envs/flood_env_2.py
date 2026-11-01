import os
import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FloodEvacuationEnv(gym.Env):
    """Gymnasium environment for urban flood evacuation routing and resource allocation.

    Designed for the Bengaluru NetworkX road graph with dynamic hydrograph
    accumulation and localized drainage.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        graph_path="data/processed/bengaluru_graph.pkl",
        max_steps=72,
        num_vehicles=50,
    ):
        super().__init__()

        # 1. Load the preprocessed NetworkX road network
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"Graph file not found at {graph_path}. Run Step 2 first."
            )

        with open(graph_path, "rb") as f:
            self.base_graph = pickle.load(f)

        self.nodes = list(self.base_graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        # Precompute susceptibility array to avoid slow graph queries during step()
        self.node_susceptibility = np.array(
            [
                float(
                    self.base_graph.nodes[n].get("flood_susceptibility", 0.5)
                )
                for n in self.nodes
            ],
            dtype=np.float32,
        )

        # Identify shelters
        self.shelter_indices = [
            self.node_to_idx[n]
            for n in self.nodes
            if self.base_graph.nodes[n].get("is_shelter", 0) == 1
        ]
        self.num_shelters = len(self.shelter_indices)

        # 2. Simulation parameters
        self.max_steps = max_steps  # 72 hours (3-day event)
        self.num_vehicles = num_vehicles  # Fleet of 50 rescue vehicles
        self.vehicle_capacity = 20  # Persons per trip
        self.current_step = 0

        # 3. Action Space: Each vehicle selects one target node index [0, num_nodes - 1]
        self.action_space = spaces.MultiDiscrete(
            [self.num_nodes] * self.num_vehicles
        )

        # 4. Observation Space: Vector normalized between 0.0 and 1.0
        # Dimensions: Water depths (N) + Pop ratios (N) + Shelter occupancies (S) + Vehicle locations (V)
        obs_dim = (self.num_nodes * 2) + self.num_shelters + self.num_vehicles
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Maintain isolated runtime state arrays
        self.water_depths = np.zeros(self.num_nodes, dtype=np.float32)
        self.initial_population = np.zeros(self.num_nodes, dtype=np.float32)
        self.remaining_population = np.zeros(self.num_nodes, dtype=np.float32)
        self.evacuated_per_zone = np.zeros(self.num_nodes, dtype=np.float32)

        for idx, node in enumerate(self.nodes):
            pop = float(self.base_graph.nodes[node].get("population", 500))
            self.initial_population[idx] = pop
            self.remaining_population[idx] = pop

        # Shelter capacities and occupancies
        self.shelter_capacities = np.array(
            [
                float(
                    self.base_graph.nodes[self.idx_to_node[s_idx]].get(
                        "shelter_capacity", 1500
                    )
                )
                for s_idx in self.shelter_indices
            ],
            dtype=np.float32,
        )
        self.shelter_occupancies = np.zeros(self.num_shelters, dtype=np.float32)

        # Initialize vehicles at shelter nodes
        self.vehicle_positions = np.random.choice(
            self.shelter_indices, size=self.num_vehicles
        )

        # Storm intensity profile for this episode
        self.peak_rainfall = float(self.np_random.uniform(0.06, 0.12))

        obs = self._get_observation()
        info = {
            "total_evacuated": 0.0,
            "active_shelters": self.num_shelters,
            "blocked_nodes_count": 0,
        }
        return obs, info

    def _get_observation(self):
        # 1. Normalized water depth (clipped at 2.0 meters max)
        norm_depth = np.clip(self.water_depths / 2.0, 0.0, 1.0)

        # 2. Remaining population ratio
        norm_pop = np.divide(
            self.remaining_population,
            np.maximum(self.initial_population, 1.0),
            dtype=np.float32,
        )

        # 3. Shelter occupancy ratio
        norm_shelter = np.divide(
            self.shelter_occupancies,
            np.maximum(self.shelter_capacities, 1.0),
            dtype=np.float32,
        )

        # 4. Normalized vehicle positions
        norm_vehicles = (
            self.vehicle_positions / max(1, self.num_nodes - 1)
        ).astype(np.float32)

        return np.concatenate(
            [norm_depth, norm_pop, norm_shelter, norm_vehicles]
        ).astype(np.float32)

    def step(self, action):
        self.current_step += 1

        # --- 1. Realistic Storm Hydrograph & Localized Elevation Dynamics ---
        # Storm bell curve: peak rainfall between hours 6 and 18, tapering off afterwards
        storm_intensity = np.exp(
            -((self.current_step - 12) ** 2) / (2 * (6.0**2))
        )
        base_rain = self.peak_rainfall * storm_intensity

        # Low-lying nodes accumulate water faster; natural drainage drains 0.015 m/hour
        flood_increment = self.node_susceptibility * base_rain
        drainage_rate = 0.010
        self.water_depths = np.clip(
            self.water_depths + flood_increment - drainage_rate, 0.0, 3.0
        )

        # Threshold: Roads with >0.4m depth are impassable for standard vans
        blocked_nodes = set(np.where(self.water_depths > 0.4)[0])

        # --- 2. Execute Vehicle Dispatch and Rescue Operations ---
        step_rescued = 0
        transit_penalties = 0.0

        for v_id, target_node_idx in enumerate(action):
            # Penalize vehicle attempts to traverse impassable/submerged nodes
            if target_node_idx in blocked_nodes:
                transit_penalties += 0.5
                continue

            # Evacuate stranded population if present at target node
            if self.remaining_population[target_node_idx] > 0:
                people_to_pick = min(
                    self.vehicle_capacity,
                    self.remaining_population[target_node_idx],
                )

                self.remaining_population[target_node_idx] -= people_to_pick
                self.evacuated_per_zone[target_node_idx] += people_to_pick
                step_rescued += people_to_pick

                # Deposit evacuees in first available shelter with capacity
                remaining_to_house = people_to_pick
                for s_local_idx in range(self.num_shelters):
                    available_cap = (
                        self.shelter_capacities[s_local_idx]
                        - self.shelter_occupancies[s_local_idx]
                    )
                    if available_cap > 0:
                        placed = min(remaining_to_house, available_cap)
                        self.shelter_occupancies[s_local_idx] += placed
                        remaining_to_house -= placed
                        if remaining_to_house <= 0:
                            break

            # Update vehicle location
            self.vehicle_positions[v_id] = target_node_idx

        # --- 3. Compute Equity-Aware Reward ---
        # Variance minimization across zones acts as the core equity penalty
        evac_ratios = np.divide(
            self.evacuated_per_zone, np.maximum(self.initial_population, 1.0)
        )
        equity_variance = float(np.var(evac_ratios))

        reward = (
            (float(step_rescued) * 1.0)
            - (equity_variance * 50.0)
            - (transit_penalties * 0.1)
        )

        # --- 4. Check Termination Conditions ---
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Early completion if entire vulnerable population is evacuated
        if np.sum(self.remaining_population) <= 0:
            terminated = True
            reward += 100.0

        obs = self._get_observation()
        info = {
            "step_rescued": step_rescued,
            "total_rescued": float(np.sum(self.evacuated_per_zone)),
            "equity_variance": equity_variance,
            "blocked_nodes_count": len(blocked_nodes),
        }

        return obs, reward, terminated, truncated, info