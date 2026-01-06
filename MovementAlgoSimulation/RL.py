import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math

class SnakePathEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The Goal: Learn to actuate joints to move the 'Head' toward a 'Target Waypoint'.
    """
    def __init__(self, render_mode=False):
        super(SnakePathEnv, self).__init__()
        
        # --- 1. CONFIGURATION ---
        self.NUM_JOINTS = 4  # Matches your RRT config
        self.render_mode = render_mode
        self.dt = 1./240.
        
        # Connect to PyBullet
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 2. ACTION & OBSERVATION SPACES ---
        # Action: Target angle for each joint (in radians)
        # Range: -70 to +70 degrees (approx -1.2 to 1.2 rad)
        self.action_space = spaces.Box(low=-1.2, high=1.2, shape=(self.NUM_JOINTS,), dtype=np.float32)
        
        # Observation: 
        # [0-2] Target Vector (x, y, z) relative to Head
        # [3-6] Current Joint Angles
        # [7-9] Head Orientation (Roll, Pitch, Yaw)
        obs_dim = 3 + self.NUM_JOINTS + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- LOAD ROBOT ---
        # NOTE: You need a snake.urdf file. 
        # For now, this loads a placeholder or you can construct one procedurally.
        start_pos = [0, 0, 0.1]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # IMPORTANT: Replace "snake_robot.urdf" with your actual file path
        # If you don't have one, I can provide a script to generate a simple box-snake.
        try:
            self.robot_id = p.loadURDF("snake_robot.urdf", start_pos, start_orn)
        except:
            print("URDF not found. Please provide a valid 'snake_robot.urdf'")
            # Create a dummy box for code validation if file missing
            self.robot_id = p.loadURDF("r2d2.urdf", start_pos, start_orn) 

        # --- SET TARGET (Simulating the next RRT waypoint) ---
        # Randomize target slightly in front of the robot
        target_x = np.random.uniform(1.0, 2.0)
        target_y = np.random.uniform(-1.0, 1.0)
        self.target_pos = np.array([target_x, target_y, 0])
        
        # Visual marker for the target
        self.target_visual = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1]),
            basePosition=self.target_pos
        )
        
        # Initialize friction (CRITICAL FOR SNAKES)
        # You must set lateral friction high and spinning friction low
        p.changeDynamics(self.robot_id, -1, lateralFriction=2.0, spinningFriction=0.1)

        self.prev_dist = np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.robot_id)[0]) - self.target_pos)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Get Head Position & Orientation
        head_pos, head_orn = p.getBasePositionAndOrientation(self.robot_id)
        head_euler = p.getEulerFromQuaternion(head_orn) # Roll, Pitch, Yaw
        
        # 2. Get Joint States
        joint_states = p.getJointStates(self.robot_id, range(self.NUM_JOINTS))
        joint_angles = [state[0] for state in joint_states]
        
        # 3. Calculate Relative Vector to Target
        rel_target = self.target_pos - np.array(head_pos)
        
        # Combine into one array
        obs = np.concatenate((rel_target, joint_angles, head_euler)).astype(np.float32)
        return obs

    def step(self, action):
        # --- 1. APPLY ACTION ---
        # Set motor control for all joints
        p.setJointMotorControlArray(
            self.robot_id,
            range(self.NUM_JOINTS),
            p.POSITION_CONTROL,
            targetPositions=action,
            forces=[5.0]*self.NUM_JOINTS # Limited torque
        )
        
        p.stepSimulation()
        if self.render_mode:
            import time
            time.sleep(self.dt)
            
        # --- 2. CALCULATE REWARD ---
        head_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        curr_dist = np.linalg.norm(np.array(head_pos) - self.target_pos)
        
        # A. Progress Reward (Positive if getting closer)
        reward_progress = (self.prev_dist - curr_dist) * 100 
        
        # B. Energy Penalty (Negative for wild movements)
        reward_energy = -np.mean(np.abs(action)) * 0.1
        
        # C. Alignment Reward (Bonus for facing the target)
        # (This helps the snake steer correctly)
        
        reward = reward_progress + reward_energy
        self.prev_dist = curr_dist
        
        # --- 3. CHECK DONE ---
        terminated = False
        truncated = False
        
        # Success: Reached target
        if curr_dist < 0.2:
            reward += 100.0 # Big bonus
            terminated = True
            print("Target Reached!")
            
        # Failure: Moved too far away or flipped over
        if curr_dist > 5.0 or head_pos[2] > 0.5: # If flying/flipping
            reward -= 50.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        p.disconnect()

# --- OPTIONAL: Training Loop Example ---
if __name__ == "__main__":
    from stable_baselines3 import PPO
    
    # 1. Create Env
    env = SnakePathEnv(render_mode=False)
    
    # 2. Define Model (PPO is great for continuous control)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 3. Train
    print("Starting Training...")
    model.learn(total_timesteps=10000)
    print("Training Finished.")
    
    # 4. Test Visualization
    env = SnakePathEnv(render_mode=True)
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()