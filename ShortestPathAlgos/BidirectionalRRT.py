import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time
from scipy.ndimage import binary_dilation

# --- 1. CONFIGURATION ---
NUM_LINKS = 5           
SEGMENT_LENGTH = 3.0    # Fixed length (Constraint)
MAX_JOINT_ANGLE = 70    # Degrees (Constraint)
ANIMATION_SPEED = 1    # Steps per frame

# --- 2. ENVIRONMENT ---
class DebrisMap:
    def __init__(self, width=70, height=70, safe_zones=None):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width)) 
        self.planning_grid = np.zeros((height, width))
        self.safe_zones = safe_zones if safe_zones else []
        
        self.create_hardcore_maze()
        self.carve_guaranteed_path(thickness=6)
        self.inflate_obstacles(radius=3)
        self.enforce_safe_zones()

    def create_hardcore_maze(self):
        self.raw_grid[0, :] = 1; self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1; self.raw_grid[:, -1] = 1
        mid_x, mid_y = self.width // 2, self.height // 2
        self.raw_grid[mid_y-2:mid_y+2, :] = 1 
        self.raw_grid[:, mid_x-2:mid_x+2] = 1 
        self.raw_grid[mid_y-2:mid_y+2, 10:18] = 0 
        self.raw_grid[50:60, mid_x-2:mid_x+2] = 0
        self.raw_grid[mid_y-2:mid_y+2, 50:58] = 0
        self.raw_grid[10:20, 10:20] = 1
        self.raw_grid[25:30, 5:25] = 1 
        self.raw_grid[45:55, 10:20] = 1 
        self.raw_grid[40:42, 5:30] = 1 
        self.raw_grid[45:65, 45:47] = 1 
        self.raw_grid[40:42, 40:60] = 1 
        self.raw_grid[5:25, 45:47] = 1
        self.raw_grid[15:17, 45:60] = 1

    def carve_guaranteed_path(self, thickness=6):
        waypoints = [(5, 5), (14, 15), (14, 40), (14, 55), (35, 55), (54, 55), (54, 35), (54, 10), (65, 5)]
        for i in range(len(waypoints)-1):
            self.clear_line(waypoints[i], waypoints[i+1], thickness=thickness)

    def clear_line(self, p1, p2, thickness):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        if dist == 0: return
        
        # Stability Fix: Prevent ZeroDivisionError
        steps = int(dist * 2) 
        if steps == 0: steps = 1
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            for ty in range(-thickness, thickness+1):
                for tx in range(-thickness, thickness+1):
                    if 0 <= x+tx < self.width and 0 <= y+ty < self.height:
                        self.raw_grid[y+ty, x+tx] = 0
                        self.planning_grid[y+ty, x+tx] = 0

    def inflate_obstacles(self, radius):
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((radius*2, radius*2))).astype(int)

    def enforce_safe_zones(self):
        for (sx, sy) in self.safe_zones:
            sy, sx = int(sy), int(sx)
            safe_area = slice(max(0,sy-4), min(self.height,sy+5)), slice(max(0,sx-4), min(self.width,sx+5))
            self.raw_grid[safe_area] = 0
            self.planning_grid[safe_area] = 0

    def is_collision(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        return self.planning_grid[int(y)][int(x)] == 1

# --- 3. BIDIRECTIONAL KINEMATIC RRT ---
class SnakeBiRRT:
    class Node:
        def __init__(self, x, y, yaw=0.0):
            self.x, self.y, self.yaw = x, y, yaw
            self.parent = None

    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.step_size = SEGMENT_LENGTH
        self.max_angle = math.radians(MAX_JOINT_ANGLE)
        
        self.start_node = self.Node(start[0], start[1], 0.0)
        self.goal_node = self.Node(goal[0], goal[1], 0.0)
        
        self.tree_start = [self.start_node]
        self.tree_goal = [self.goal_node]
        
        self.path = None
        self.finished = False

    def get_angle_diff(self, a1, a2):
        diff = a2 - a1
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    def check_collision(self, n1, n2):
        # Bresenham Line Check
        x0, y0 = int(n1.x), int(n1.y)
        x1, y1 = int(n2.x), int(n2.y)
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.map.is_collision(x, y): return True
                err -= dy
                if err < 0: y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.map.is_collision(x, y): return True
                err -= dx
                if err < 0: x += sx; err += dy
                y += sy
        return self.map.is_collision(x, y)

    def validate_bridge(self, node_a, node_b):
        """
        Strict check for the 'jump' connection between trees.
        A -> Bridge -> B
        """
        # 1. Collision Check
        if self.check_collision(node_a, node_b): return False
        
        # 2. Check Turn: Tree A -> Bridge
        # node_a.yaw is the heading arriving at A.
        # We need to turn to face B.
        if node_a.parent:
            heading_bridge = math.atan2(node_b.y - node_a.y, node_b.x - node_a.x)
            diff_a = self.get_angle_diff(node_a.yaw, heading_bridge)
            if abs(diff_a) > self.max_angle: return False
            
        # 3. Check Turn: Bridge -> Tree B
        # Tree B grows backwards from Goal. node_b.yaw is (b.parent -> b).
        # We are traveling A -> B -> B.parent.
        # So we are LEAVING B opposite to how it grew.
        if node_b.parent:
            heading_bridge = math.atan2(node_b.y - node_a.y, node_b.x - node_a.x)
            
            # Vector B->Parent
            heading_b_out = math.atan2(node_b.parent.y - node_b.y, node_b.parent.x - node_b.x)
            
            # Turn from Bridge vector to Outgoing vector
            diff_b = self.get_angle_diff(heading_bridge, heading_b_out)
            if abs(diff_b) > self.max_angle: return False
            
        return True

    def extend(self, tree, target_x, target_y):
        # Nearest
        dists = [(n.x - target_x)**2 + (n.y - target_y)**2 for n in tree]
        nearest = tree[np.argmin(dists)]

        # Steer
        theta = math.atan2(target_y - nearest.y, target_x - nearest.x)
        
        # Kinematic Clamp
        if nearest.parent is not None:
            diff = self.get_angle_diff(nearest.yaw, theta)
            if abs(diff) > self.max_angle:
                sign = 1 if diff > 0 else -1
                theta = nearest.yaw + (sign * self.max_angle)

        new_x = nearest.x + self.step_size * math.cos(theta)
        new_y = nearest.y + self.step_size * math.sin(theta)
        new_node = self.Node(new_x, new_y, theta)

        if self.check_collision(nearest, new_node): return None
        
        new_node.parent = nearest
        tree.append(new_node)
        return new_node

    def connect(self, tree, target_node):
        """Greedy connect attempt"""
        new_node = self.extend(tree, target_node.x, target_node.y)
        while new_node:
            dist = math.hypot(new_node.x - target_node.x, new_node.y - target_node.y)
            if dist <= self.step_size:
                return new_node, "CONNECTED"
            
            prev_node = new_node
            new_node = self.extend(tree, target_node.x, target_node.y)
            if not new_node:
                return prev_node, "ADVANCED"
        return None, "TRAPPED"

    def step(self):
        if self.finished: return False

        trees = [(self.tree_start, self.tree_goal), (self.tree_goal, self.tree_start)]
        
        for tree_a, tree_b in trees:
            if random.randint(0, 100) > 5:
                rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
            else:
                rnd = [tree_b[-1].x, tree_b[-1].y]

            new_node_a = self.extend(tree_a, rnd[0], rnd[1])

            if new_node_a:
                new_node_b, status = self.connect(tree_b, new_node_a)
                
                if status == "CONNECTED":
                    # Use Strict Validator
                    if self.validate_bridge(new_node_a, new_node_b):
                        self.generate_path(new_node_a, new_node_b, tree_a == self.tree_start)
                        self.finished = True
                        return True
        return False

    def generate_path(self, node_a, node_b, a_is_start):
        if a_is_start:
            start_piece, goal_piece = node_a, node_b
        else:
            start_piece, goal_piece = node_b, node_a
            
        path = []
        curr = start_piece
        while curr:
            path.append((curr.x, curr.y))
            curr = curr.parent
        path = path[::-1] # Reverse to get Start -> Middle

        curr = goal_piece
        temp_path = []
        while curr:
            temp_path.append((curr.x, curr.y))
            curr = curr.parent
        
        path.extend(temp_path)
        self.path = path

# --- 4. VISUALIZATION UTILS ---
def draw_snake_body(ax, path, num_links):
    if len(path) < num_links + 1: return
    body_points = path[-(num_links+1):] 
    bx, by = zip(*body_points)
    ax.plot(bx, by, color='blue', linestyle='-', linewidth=4, alpha=0.8, zorder=5)
    ax.plot(bx, by, 'o', markerfacecolor='white', markeredgecolor='black', markersize=6, zorder=6)
    ax.plot(bx[-1], by[-1], 'D', markerfacecolor='gold', markeredgecolor='black', markersize=8, zorder=7)

def draw_trees(ax, tree_start, tree_goal):
    for node in tree_start:
        if node.parent:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                    color='lime', linestyle='-', linewidth=0.8, alpha=0.4)
    for node in tree_goal:
        if node.parent:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                    color='cyan', linestyle='-', linewidth=0.8, alpha=0.4)

if __name__ == "__main__":
    WIDTH, HEIGHT = 70, 70
    START = (5, 5)
    GOAL = (65, 5)
    
    env = DebrisMap(WIDTH, HEIGHT, safe_zones=[START, GOAL])
    bi_rrt = SnakeBiRRT(env, START, GOAL)
    
    print("Initialize Map...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    
    start_time = time.time()
    
    try:
        while not bi_rrt.finished:
            for _ in range(ANIMATION_SPEED): 
                bi_rrt.step()
                if bi_rrt.finished: break
            
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
            ax.plot(START[0], START[1], 'go', markersize=8)
            ax.plot(GOAL[0], GOAL[1], 'ro', markersize=8)
            
            draw_trees(ax, bi_rrt.tree_start, bi_rrt.tree_goal)
            
            ax.set_title(f"Nodes: {len(bi_rrt.tree_start) + len(bi_rrt.tree_goal)} | Time: {time.time()-start_time:.1f}s")
            plt.pause(0.01)

        print(f"Path Found! Total Time: {time.time() - start_time:.2f}s")
        
        ax.clear()
        ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
        ax.plot(START[0], START[1], 'go', label='Start')
        ax.plot(GOAL[0], GOAL[1], 'ro', label='Goal')
        
        draw_trees(ax, bi_rrt.tree_start, bi_rrt.tree_goal)
        
        if bi_rrt.path:
            px, py = zip(*bi_rrt.path)
            ax.plot(px, py, 'magenta', linewidth=2, label="Bi-Directional Path")
            draw_snake_body(ax, bi_rrt.path, NUM_LINKS)
        
        ax.legend(loc='upper right')
        ax.set_title("Final Bidirectional Kinematic RRT Path")
        plt.ioff()
        plt.show()

    except Exception as e:
        print(f"Simulation stopped: {e}")
        plt.close()