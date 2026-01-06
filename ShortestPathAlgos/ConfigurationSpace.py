import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev

# --- 1. CONFIGURATION ---
NUM_JOINTS = 4          
SEGMENT_LENGTH = 3.0    
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# Constraints
JOINT_LIMIT = 15.0      # Degrees
MAX_TURN_ANGLE = np.deg2rad(15) 

RRT_STEP_SIZE = 2.0     
MAX_ITER = 8000         
W_POS = 1.0             
W_JOINT = 0.5           

# --- 2. KINEMATICS ---
def get_snake_body(state, yaw_override=None):
    """
    Returns the body points given state [x, y, j1, j2, j3, j4] and explicit yaw.
    """
    x, y = state[0], state[1]
    joint_angles = state[2:]
    
    current_angle = yaw_override
    body_points = [(x, y)] # Head
    
    cx, cy = x, y
    
    for i in range(len(joint_angles)):
        bx = cx - SEGMENT_LENGTH * math.cos(current_angle)
        by = cy - SEGMENT_LENGTH * math.sin(current_angle)
        body_points.append((bx, by))
        cx, cy = bx, by
        current_angle -= math.radians(joint_angles[i]) 

    bx = cx - SEGMENT_LENGTH * math.cos(current_angle)
    by = cy - SEGMENT_LENGTH * math.sin(current_angle)
    body_points.append((bx, by))
    
    return body_points

# --- 3. ENVIRONMENT ---
class DebrisMap:
    def __init__(self, width=70, height=70):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width))
        self.planning_grid = np.zeros((height, width))
        self.create_hardcore_maze()
        self.inflate_obstacles(radius=INFLATION_RADIUS) 

    def create_hardcore_maze(self):
        self.raw_grid[0, :] = 1; self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1; self.raw_grid[:, -1] = 1
        mid_x, mid_y = self.width // 2, self.height // 2
        
        self.raw_grid[mid_y-2:mid_y+2, 10:15] = 0 
        self.raw_grid[10:20, 15:20] = 1
        self.raw_grid[25:30, 10:25] = 1 
        self.raw_grid[40:45, 10:25] = 1 
        self.raw_grid[45:50, 40:55] = 1 
        self.raw_grid[10:25, 45:50] = 1
        
    def inflate_obstacles(self, radius):
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((radius*2, radius*2))).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height): return True
        return self.planning_grid[int(y), int(x)] == 1
        
    def check_line_collision(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        steps = int(dist * 3) 
        if steps == 0: return self.is_collision(x1, y1)
        for i in range(steps+1):
            t = i / steps
            px = x1 + t*(x2-x1)
            py = y1 + t*(y2-y1)
            if self.is_collision(px, py): return True
        return False

# --- 4. PLANNER ---
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
        self.finished = False
        self.path = None
        self.joint_limit = JOINT_LIMIT

        if not self.is_valid_configuration(self.start):
            print("CRITICAL: Start Configuration Collides!")
            self.start.state[0] = 35.0
            self.start.state[1] = 20.0

    def get_weighted_dist(self, s1, s2):
        d_pos = (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2
        d_joints = np.sum((s1[2:] - s2[2:])**2)
        return math.sqrt(d_pos * W_POS + d_joints * W_JOINT)

    def get_random_sample(self):
        if random.random() < 0.10: 
            return self.goal_conf
        margin = 3
        rx = random.uniform(margin, self.env.width-margin)
        ry = random.uniform(margin, self.env.height-margin)
        r_joints = [random.uniform(-self.joint_limit, self.joint_limit) for _ in range(NUM_JOINTS)]
        return np.array([rx, ry] + r_joints)
    
    def step(self):
        if self.finished: return False
        
        rnd = self.get_random_sample()
        dlist = [self.get_weighted_dist(node.state, rnd) for node in self.nodes]
        nearest_node = self.nodes[np.argmin(dlist)]
        
        dx = rnd[0] - nearest_node.state[0]
        dy = rnd[1] - nearest_node.state[1]
        target_yaw = math.atan2(dy, dx)
        
        current_yaw = nearest_node.yaw
        diff_yaw = target_yaw - current_yaw
        
        while diff_yaw > math.pi: diff_yaw -= 2*math.pi
        while diff_yaw < -math.pi: diff_yaw += 2*math.pi
        
        if abs(diff_yaw) > MAX_TURN_ANGLE:
            diff_yaw = MAX_TURN_ANGLE * np.sign(diff_yaw)
            
        new_yaw = current_yaw + diff_yaw
        
        new_x = nearest_node.state[0] + RRT_STEP_SIZE * math.cos(new_yaw)
        new_y = nearest_node.state[1] + RRT_STEP_SIZE * math.sin(new_yaw)
        
        current_joints = nearest_node.state[2:]
        target_joints = rnd[2:]
        new_joints = current_joints + (target_joints - current_joints) * 0.5 
        
        new_state = np.array([new_x, new_y, *new_joints])
        new_node = self.Node(new_state, nearest_node, yaw=new_yaw)
        
        if self.is_valid_configuration(new_node):
            self.nodes.append(new_node)
            
            dist_to_goal = self.get_weighted_dist(new_node.state, self.goal_conf)
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

# --- 5. SMOOTHING ---
def smooth_path_bspline(path_data, num_points=100):
    states = [p[0] for p in path_data]
    x = [s[0] for s in states]
    y = [s[1] for s in states]
    
    if len(x) < 3: return path_data 

    tck, u = splprep([x, y], s=3.0)
    u_new = np.linspace(0, 1, num_points)
    new_points = splev(u_new, tck)
    new_x, new_y = new_points[0], new_points[1]
    
    smoothed_path = []
    old_joints = np.array([s[2:] for s in states])
    new_joints_array = np.array([np.interp(u_new, np.linspace(0, 1, len(x)), old_joints[:, j]) for j in range(NUM_JOINTS)]).T

    for i in range(len(new_x)):
        if i < len(new_x) - 1:
            dx = new_x[i+1] - new_x[i]
            dy = new_y[i+1] - new_y[i]
            new_yaw = math.atan2(dy, dx)
        else:
            new_yaw = smoothed_path[-1][1]
            
        state = np.array([new_x[i], new_y[i], *new_joints_array[i]])
        smoothed_path.append((state, new_yaw))
        
    return smoothed_path

# --- 6. VISUALIZATION ---
def draw_snake_line(ax, state, yaw, color='blue', alpha=1.0, lw=3):
    body = get_snake_body(state, yaw_override=yaw)
    bx, by = zip(*body)
    
    ax.plot(bx, by, color=color, linestyle='-', linewidth=lw, alpha=alpha, zorder=15)
    ax.scatter(bx[:-1], by[:-1], color='white', edgecolor='black', s=30, zorder=16)
    ax.scatter(bx[0], by[0], color='gold', edgecolor='black', marker='D', s=45, zorder=17)

def main():
    WIDTH, HEIGHT = 70, 70
    
    START_CONF = np.array([35.0, 20.0, 0.0, 0.0, 0.0, 0.0]) # 6D
    START_YAW = 1.57
    
    GOAL_CONF  = np.array([60.0, 60.0, 0.0, 0.0, 0.0, 0.0])
    GOAL_YAW = 0.0
    
    env = DebrisMap(WIDTH, HEIGHT)
    planner = CSpaceRRT(env, START_CONF, GOAL_CONF, start_yaw=START_YAW)
    
    fig, ax = plt.subplots(figsize=(9,9))
    plt.ion() 
    
    print("Searching 6D C-Space...")
    frame_count = 0
    while not planner.finished:
        if frame_count > MAX_ITER: break

        for _ in range(50): 
            planner.step()
            frame_count += 1
            if planner.finished: break
            
        ax.clear()
        ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
        
        # Draw Branches
        for node in planner.nodes:
            if node.parent:
                ax.plot([node.parent.state[0], node.state[0]], 
                        [node.parent.state[1], node.state[1]], 
                        color='blue', linewidth=1, alpha=1.0)

        # Draw Active Snake
        if planner.nodes:
            curr = planner.nodes[-1]
            draw_snake_line(ax, curr.state, curr.yaw, color='magenta', lw=2)
            
        draw_snake_line(ax, START_CONF, START_YAW, color='green', alpha=0.5)
        draw_snake_line(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.5)
        
        ax.set_title(f"Searching... Nodes: {len(planner.nodes)}")
        plt.pause(0.001)

    # --- REPLAY WITH SWEPT VOLUME ---
    if planner.path:
        print("Path Found! Smoothing...")
        raw_path = planner.path
        smooth_path = smooth_path_bspline(raw_path, num_points=100)
        
        # Extract trail for visualizing the "Full Configuration Path"
        # We will draw "ghost" bodies at intervals
        trail_indices = list(range(0, len(smooth_path), 5))

        for i, (state, yaw) in enumerate(smooth_path):
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
            
            # 1. Draw the Swept Volume (The "Ghost" Trail)
            # This visualizes the PATH OF THE BODY, not just the head
            for idx in trail_indices:
                if idx > i: break # Only draw trail up to current point
                t_state, t_yaw = smooth_path[idx]
                # Draw faint green body
                draw_snake_line(ax, t_state, t_yaw, color='lime', alpha=0.15, lw=4)
            
            # 2. Draw Start/Goal
            draw_snake_line(ax, START_CONF, START_YAW, color='green', alpha=0.3)
            draw_snake_line(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.3)
            
            # 3. Draw Active Snake
            draw_snake_line(ax, state, yaw, color='blue', alpha=1.0)
            
            ax.set_title(f"Replaying Full Body Path: {int(i/len(smooth_path)*100)}%")
            plt.pause(0.02)
        
        ax.set_title("Target Reached. Close window to exit.")
        plt.ioff()
        plt.show() 
    else:
        print("No path found.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()