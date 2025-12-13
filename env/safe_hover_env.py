# env/safe_hover_env.py
import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class SafeHoverAviary(BaseRLAviary):
    """
    SafeHoverAviary: single-drone environment with spherical obstacles and a cost signal.
    Returns Gymnasium-style (obs, reward, terminated, truncated, info) from step().
    info["cost"] is 1 when drone within safe_distance of any obstacle, else 0.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 obstacle_positions=None,
                 obstacle_radii=None,
                 safe_radius: float = 0.4):
        # config
        self.target_pos = np.array([0, 0, 1])
        self.episode_len_sec = 8
        self.MAX_OBSTACLES = 3  # Define this as class attribute

        self.obstacle_positions = [] if obstacle_positions is None else [np.array(x) for x in obstacle_positions]
        if obstacle_radii is None:
            self.obstacle_radii = [0.2 for _ in self.obstacle_positions]
        else:
            self.obstacle_radii = list(obstacle_radii)

        self.safe_radius = float(safe_radius)
        self._obstacle_body_ids = []

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )
        
        # CRITICAL: Update observation space to include obstacle features
        # This must happen AFTER super().__init__() so base observation_space exists
        base_obs_dim = self.observation_space.shape[0] 
        obstacle_features_per_obs = 6  # [rel_x, rel_y, rel_z, distance, radius, exists]
        total_obs_dim = base_obs_dim + (self.MAX_OBSTACLES * obstacle_features_per_obs)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

    def _post_init(self):
        super()._post_init()
        self._spawn_obstacles()

    def _get_pybullet_client(self):
        # try to get the pybullet client used by the aviary, else fallback to global p
        if hasattr(self, "_p"):
            return self._p, getattr(self, "_client", -1)
        return p, -1

    def _spawn_obstacles(self):
        for pos, r in zip(self.obstacle_positions, self.obstacle_radii):
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1,0,0,0.5])
            body_id = p.createMultiBody(
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos
            )
            self._obstacle_body_ids.append(body_id)

    def _remove_obstacles(self):
        pyb, client = self._get_pybullet_client()
        for bid in getattr(self, "_obstacle_body_ids", []):
            try:
                pyb.removeBody(bid, physicsClientId=client)
            except Exception:
                pass
        self._obstacle_body_ids = []

    def _compute_cost(self):
        drone_state = self._getDroneStateVector(0)
        drone_pos = np.array(drone_state[0:3])
        for obs_pos, obs_r in zip(self.obstacle_positions, self.obstacle_radii):
            dist = np.linalg.norm(drone_pos - obs_pos)
            if dist < (self.safe_radius + obs_r):
                return 1.0
        return 0.0

    def step(self, action):
        # BaseRLAviary.step returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = super().step(action)

        cost = self._compute_cost()
        info = dict(info)
        info["cost"] = float(cost)
        info["obstacle_positions"] = [pos.tolist() for pos in self.obstacle_positions]
        info["drone_pos"] = list(self._getDroneStateVector(0)[0:3])
        return obs, reward, terminated, truncated, info

    def reset(self):
        self._remove_obstacles()
        ret = super().reset()
        self._spawn_obstacles()
        return ret

    def set_obstacles(self, obstacle_positions, obstacle_radii=None):
        self._remove_obstacles()
        self.obstacle_positions = [np.array(x) for x in obstacle_positions]
        if obstacle_radii is None:
            self.obstacle_radii = [0.2 for _ in self.obstacle_positions]
        else:
            self.obstacle_radii = list(obstacle_radii)
        self._spawn_obstacles()

    def close(self):
        self._remove_obstacles()
        try:
            super().close()
        except Exception:
            pass

    def _computeInfo(self):
        """
        Compute additional environment info.
        """
        drone_pos = self._getDroneStateVector(0)[0:3]
        
        # compute cost: 1 if near obstacle, else 0
        cost = 0
        for pos, r in zip(self.obstacle_positions, self.obstacle_radii):
            if np.linalg.norm(drone_pos - pos) < self.safe_radius + r:
                cost = 1
                break

        return {
            "cost": cost,
            "drone_position": drone_pos,
            "target_position": self.target_pos
        }
    
    def _computeReward(self):
        pos = self._getDroneStateVector(0)[0:3]
        vel = self._getDroneStateVector(0)[10:13]  # Get velocity
        target = self.target_pos
        dist = np.linalg.norm(pos - target)
        
        # Dense reward shaping
        reward = 0.0
        
        # Distance reward
        reward += np.exp(-2.0 * dist)
        
        # for being far
        reward -= 0.1 * dist
        
        # reward for low velocity near target
        if dist < 0.5:
            speed = np.linalg.norm(vel)
            reward += 0.5 * np.exp(-speed)
        
        # penalty for crash
        if pos[2] < 0.1:
            reward -= 5.0
        
        # penalty for going too far
        if dist > 3.0:
            reward -= 2.0
        
        return reward


    def _computeTerminated(self):
        """
        Episode termination logic.
        """
        pos = self._getDroneStateVector(0)[0:3]
        target = self.target_pos
        dist = np.linalg.norm(pos - target)

        # Done if the drone is extremely far or too low
        if dist > 5:            # too far from target
            return True
        if pos[2] < 0.05:       # crashed / fell
            return True

        return False

    def _computeTruncated(self):
        """
        Episode truncation logic (typically for time limits).
        Returns True if episode should end due to time limit.
        """
        # Check if we've exceeded the episode time limit
        if self.step_counter / self.CTRL_FREQ > self.episode_len_sec:
            return True
        
        return False
    
    def _computeObs(self):
        """
        Returns observation with fixed-size obstacle information.
        """
        # Get base observation
        base_obs = super()._computeObs()
        
        # Get drone position
        drone_pos = self._getDroneStateVector(0)[0:3]
        
        # Fixed number of obstacle slots
        obstacle_features = []
        
        for i in range(self.MAX_OBSTACLES):
            if i < len(self.obstacle_positions):
                obs_pos = self.obstacle_positions[i]
                obs_r = self.obstacle_radii[i]
                
                rel_pos = obs_pos - drone_pos
                distance = np.linalg.norm(rel_pos)
                
                # Features: [rel_x, rel_y, rel_z, distance, radius, exists]
                obstacle_features.extend([
                    rel_pos[0], rel_pos[1], rel_pos[2],
                    distance,
                    obs_r,
                    1.0  # Obstacle exists
                ])
            else:
                # Pad with zeros if obstacle doesn't exist
                obstacle_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        full_obs = np.concatenate([base_obs.flatten(), np.array(obstacle_features)])
        return full_obs
