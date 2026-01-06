import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

# --- 1. CONFIGURATION ---
VISUALIZE_SEARCH = False 
SKIP_FRAMES = 3 

NUM_JOINTS = 4          
SEGMENT_LENGTH = 3.0    
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# 6D Freedom
JOINT_LIMIT = 70.0      
MAX_TURN_ANGLE = np.deg2rad(70) 

# --- NEW PARAMETER: JOINT AGILITY ---
# This makes joints move 10x faster than the head position.
# It allows the snake to coil/uncoil instantly while moving slowly.
JOINT_AGILITY = 10.0  

# Weights (Joints are now "cheap" to move, so we weight them low to encourage exploring them)
W_POS = 1.0     
W_JOINT = 0.05
JOINT_SCALE = math.sqrt(W_JOINT / W_POS) 

RRT_STEP_SIZE = 2.0     
MAX_ITER = 100000       # High iteration count for complex paths

# --- 2. MATH HELPERS ---
def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def get_snake_body(state, yaw_override=None):
    x, y = state[0], state[1]
    joint_angles = state[2:]
    
    current_angle = yaw_override
    body_points = [(x, y)] 
    
    cx, cy = x, y
    for i in range(len(joint_angles)):
        current_angle -= math.radians(joint_angles[i])
        bx = cx - SEGMENT_LENGTH * math.cos(current_angle)
        by = cy - SEGMENT_LENGTH * math.sin(current_angle)
        body_points.append((bx, by))
        cx, cy = bx, by

    return body_points

# --- 3. ENVIRONMENT ---
class DebrisMap:
    def __init__(self, width=70, height=70):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width))
        self.planning_grid = np.zeros((height, width))
        self.create_chaos_field()
        self.inflate_obstacles(radius=INFLATION_RADIUS) 

    def create_chaos_field(self):
        self.raw_grid[0, :] = 1; self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1; self.raw_grid[:, -1] = 1
        
        # Static Obstacles
        self.raw_grid[30:33, 0:40] = 1   
        self.raw_grid[50:53, 30:70] = 1  
        self.raw_grid[15:25, 55:60] = 1  
        
        # Safety Clearing
        self.raw_grid[15:25, 30:40] = 0 
        self.raw_grid[55:65, 55:65] = 0 
        
    def inflate_obstacles(self, radius):
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((radius*2, radius*2))).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height): return True
        return self.planning_grid[int(y), int(x)] == 1
        
    def check_line_collision(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        if dist < 1.0: return False
        
        steps = int(dist * 0.8)
        if steps == 0: return self.is_collision(x1, y1)
        
        t = np.linspace(0, 1, steps+1)
        x_points = x1 + t * (x2 - x1)
        y_points = y1 + t * (y2 - y1)
        
        x_idx = x_points.astype(int)
        y_idx = y_points.astype(int)
        
        valid_mask = (x_idx >= 0) & (x_idx < self.width) & \
                     (y_idx >= 0) & (y_idx < self.height)
        
        if not np.all(valid_mask): return True 
        if np.any(self.planning_grid[y_idx, x_idx] == 1): return True
        return False

# --- 4. 6D PLANNER (AGILE JOINTS) ---
class CSpaceRRT:
    class Node:
        def __init__(self, state, parent=None, yaw=0.0):
            self.state = np.array(state) 
            self.parent = parent
            self.yaw = yaw 

    def __init__(self, env, start_conf, goal_conf, start_yaw=0.0):
        self.env = env
        self.start = self.Node(start_conf, yaw=start_yaw)
        self.goal_conf = np.array(goal_conf)
        self.nodes = [self.start]
        
        self.scaled_states = [self.scale_state(start_conf)]
        self.tree_needs_rebuild = False
        self.kdtree = cKDTree(self.scaled_states)
        
        self.finished = False
        self.path = None
        self.joint_limit = JOINT_LIMIT
        self.goal_bias = 0.10 

        if not self.is_valid_configuration(self.start):
            print("CRITICAL: Start Configuration Collides!")

    def scale_state(self, state):
        s = state.copy()
        s[2:] *= JOINT_SCALE
        return s

    def get_random_sample(self):
        if random.random() < self.goal_bias: 
            return self.goal_conf

        margin = 3
        rx = random.uniform(margin, self.env.width-margin)
        ry = random.uniform(margin, self.env.height-margin)
        
        r_joints = []
        for _ in range(NUM_JOINTS):
            # FULL UNIFORM SAMPLING
            angle = random.uniform(-self.joint_limit, self.joint_limit)
            r_joints.append(angle)
        
        return np.array([rx, ry] + r_joints)

    def step(self):
        if self.finished: return False
        
        rnd = self.get_random_sample()
        
        if self.tree_needs_rebuild and len(self.nodes) % 100 == 0:
            self.kdtree = cKDTree(self.scaled_states)
            self.tree_needs_rebuild = False
        
        dist, idx = self.kdtree.query(self.scale_state(rnd))
        current_node = self.nodes[idx]

        # --- AGILE STEERING LOGIC ---
        diff_raw = rnd - current_node.state
        
        # Split diff into Position and Joints
        diff_pos = diff_raw[:2]
        diff_joints = diff_raw[2:]
        
        # Calculate Magnitude primarily based on Position to control step size
        dist_pos = np.linalg.norm(diff_pos)
        
        # If we are close in position, just rotate!
        if dist_pos < 0.1:
            scale_pos = 0
            # Allow pure rotation (up to step size equivalent)
            scale_joint = RRT_STEP_SIZE / (np.linalg.norm(diff_joints) + 0.001)
            scale_joint = min(scale_joint, 1.0)
        else:
            # Normal movement
            if dist_pos > RRT_STEP_SIZE:
                scale_pos = RRT_STEP_SIZE / dist_pos
            else:
                scale_pos = 1.0
            
            # Apply SUPER AGILITY to joints
            # We allow joints to move much further percent-wise than position
            scale_joint = scale_pos * JOINT_AGILITY
            scale_joint = min(scale_joint, 1.0) # Cap at 100% of the way to target

        # Apply separate scales
        step_pos = diff_pos * scale_pos
        step_joints = diff_joints * scale_joint
        
        # Recombine
        new_state = current_node.state.copy()
        new_state[:2] += step_pos
        new_state[2:] += step_joints
        
        # Yaw logic (Head follows movement)
        dx, dy = step_pos[0], step_pos[1]
        if math.hypot(dx, dy) < 0.01:
            new_yaw = current_node.yaw
        else:
            target_yaw = math.atan2(dy, dx)
            diff_yaw = normalize_angle(target_yaw - current_node.yaw)
            if abs(diff_yaw) > MAX_TURN_ANGLE:
                diff_yaw = MAX_TURN_ANGLE * np.sign(diff_yaw)
            new_yaw = normalize_angle(current_node.yaw + diff_yaw)

        joints = new_state[2:]
        if np.any(np.abs(joints) > self.joint_limit):
            return False

        new_node = self.Node(new_state, current_node, yaw=new_yaw)
        
        if self.is_valid_configuration(new_node):
            self.nodes.append(new_node)
            self.scaled_states.append(self.scale_state(new_state))
            self.tree_needs_rebuild = True 
            
            dist_to_goal = math.hypot(new_state[0]-self.goal_conf[0], new_state[1]-self.goal_conf[1])
            if dist_to_goal <= 4.0: 
                print(f"Goal Reached! Final Dist: {dist_to_goal:.2f}")
                self.finished = True
                self.path = self.extract_path(new_node)
                return True
        
        return False

    def is_valid_configuration(self, node):
        body = get_snake_body(node.state, yaw_override=node.yaw)
        for (bx, by) in body:
            if not (0 <= bx < self.env.width and 0 <= by < self.env.height): return False
        for i in range(len(body)-1):
            if self.env.check_line_collision(body[i], body[i+1]): return False 
        return True

    def extract_path(self, node):
        path = []
        while node is not None:
            path.append((node.state, node.yaw))
            node = node.parent
        return path[::-1]

# --- 5. ADAPTIVE SAFE SMOOTHING ---
def check_path_safety(env, states, yaws):
    for i in range(0, len(states), 2): 
        body = get_snake_body(states[i], yaw_override=yaws[i])
        for (bx, by) in body:
            if not (0 <= bx < env.width and 0 <= by < env.height): return False
        for j in range(len(body)-1):
            if env.check_line_collision(body[j], body[j+1]): return False 
    return True

def interpolate_path_linear(path_data, steps_per_segment=10):
    full_animation_states = []
    for i in range(len(path_data) - 1):
        start_state = path_data[i][0]
        start_yaw   = path_data[i][1]
        end_state   = path_data[i+1][0]
        end_yaw     = path_data[i+1][1]
        diff_yaw = normalize_angle(end_yaw - start_yaw)
        for t in np.linspace(0, 1, steps_per_segment):
            interp_state = start_state + (end_state - start_state) * t
            interp_yaw = start_yaw + diff_yaw * t
            full_animation_states.append((interp_state, interp_yaw))
    return full_animation_states

def get_safe_smoothed_path(env, raw_path, num_points=200):
    states = [p[0] for p in raw_path]
    data = np.array(states).T 
    if data.shape[1] < 3: return interpolate_path_linear(raw_path)

    for smoothing_factor in [5.0, 3.0, 1.0, 0.5, 0.0]:
        try:
            tck, u = splprep(data, s=smoothing_factor) 
            u_new = np.linspace(0, 1, num_points)
            new_data = splev(u_new, tck)
            new_states = np.array(new_data).T 
            
            anim_data = [] 
            yaws = []
            yaw = raw_path[0][1] 

            for i in range(len(new_states)):
                s = new_states[i]
                if i < len(new_states) - 1:
                    dx = new_states[i+1][0] - s[0]
                    dy = new_states[i+1][1] - s[1]
                    yaw = math.atan2(dy, dx)
                yaws.append(yaw)
                anim_data.append((s, yaw))
            
            if check_path_safety(env, new_states, yaws):
                print(f"Accepted Safe Smoothing with s={smoothing_factor}")
                return anim_data
        except Exception:
            continue
    print("Fallback to Linear.")
    return interpolate_path_linear(raw_path)

# --- 6. VISUALIZATION ---
def draw_snake_line_explicit(ax, body_points, color='blue', alpha=1.0, lw=3):
    bx, by = zip(*body_points)
    ax.plot(bx, by, color=color, linestyle='-', linewidth=lw, alpha=alpha, zorder=15)
    ax.scatter(bx[:-1], by[:-1], color='white', edgecolor='black', s=30, zorder=16)
    ax.scatter(bx[0], by[0], color='gold', edgecolor='black', marker='D', s=45, zorder=17)

def main():
    WIDTH, HEIGHT = 70, 70
    START_CONF = np.array([35.0, 20.0, 0.0, 0.0, 0.0, 0.0]) 
    START_YAW = 0
    GOAL_CONF  = np.array([60.0, 60.0, 0.0, 0.0, 0.0, 0.0])
    GOAL_YAW = 0.0
    
    env = DebrisMap(WIDTH, HEIGHT)
    planner = CSpaceRRT(env, START_CONF, GOAL_CONF, start_yaw=START_YAW)
    
    fig, ax = plt.subplots(figsize=(9,9))
    plt.ion() 
    
    print("Searching... (Optimization: Agile Joints + KDTree)")
    frame_count = 0
    
    # Initial Draw
    ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
    ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
    draw_snake_line_explicit(ax, get_snake_body(START_CONF, START_YAW), color='green', alpha=0.5)
    draw_snake_line_explicit(ax, get_snake_body(GOAL_CONF, GOAL_YAW), color='red', alpha=0.5)
    
    while not planner.finished:
        if frame_count > MAX_ITER: 
            print("Max iterations reached.")
            break

        # BATCHING
        for _ in range(1000): 
            if planner.step(): break
            frame_count += 1
            
        if VISUALIZE_SEARCH:
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
            ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
            for node in planner.nodes:
                if node.parent:
                    ax.plot([node.parent.state[0], node.state[0]], 
                            [node.parent.state[1], node.state[1]], 
                            color='blue', linewidth=1.5, alpha=0.6)
            plt.pause(0.01)
        else:
            print(f"Iterations: {frame_count} | Nodes: {len(planner.nodes)}")

    if planner.path:
        print("Path Found! Calculating Best Safe Trajectory...")
        raw_path = planner.path
        
        anim_data = get_safe_smoothed_path(env, raw_path)
        trail_indices = list(range(0, len(anim_data), 20)) 
        total_frames = len(anim_data)
        
        for i in range(0, total_frames, SKIP_FRAMES): 
            state, yaw = anim_data[i]
            body = get_snake_body(state, yaw_override=yaw)
            
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
            for idx in trail_indices:
                if idx > i: break 
                t_state, t_yaw = anim_data[idx]
                t_body = get_snake_body(t_state, yaw_override=t_yaw)
                draw_snake_line_explicit(ax, t_body, color='lime', alpha=0.15, lw=4)
            
            draw_snake_line_explicit(ax, body, color='blue', alpha=1.0)
            
            joints = state[2:]
            j_str = f"J1:{joints[0]:.0f} J2:{joints[1]:.0f} J3:{joints[2]:.0f} J4:{joints[3]:.0f}"
            ax.text(2, 65, j_str, color='blue', fontsize=10, fontweight='bold', backgroundcolor='white')
            
            progress = (i / total_frames) * 100
            ax.set_title(f"Replay: {progress:.1f}% | {j_str}")
            
            plt.pause(0.001) 
            
        plt.ioff(); plt.show() 
    else:
        print("No path found.")
        plt.ioff(); plt.show()

if __name__ == "__main__":
    main()