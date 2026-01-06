import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time
import heapq
from scipy.ndimage import binary_dilation

# --- 1. CONFIGURATION ---
NUM_LINKS = 5           
SEGMENT_LENGTH = 3.0    # Fixed length (Constraint)
MAX_JOINT_ANGLE = 70    # Degrees (Constraint)

# Animation Settings
RRT_STEPS_PER_FRAME = 1000  # RRT creates 1 node per step
ASTAR_STEPS_PER_FRAME = 1000 # Reduced A* steps per frame because branching is now HUGE (31 children)

# --- 2. ENVIRONMENT (Shared) ---
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
        steps = int(dist * 2) 
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

# --- 3. ALGORITHM 1: KINEMATIC RRT ---
class SnakeKinematicRRT:
    class Node:
        def __init__(self, x, y, yaw=0.0):
            self.x, self.y, self.yaw = x, y, yaw
            self.parent = None
            self.cost = 0.0

    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.step_size = SEGMENT_LENGTH 
        self.max_angle = math.radians(MAX_JOINT_ANGLE) 
        self.start = self.Node(start[0], start[1], 0.0) 
        self.goal = self.Node(goal[0], goal[1], 0.0)
        
        self.nodes = [self.start]
        self.path = None
        self.finished = False
        self.nodes_expanded = 0

    def get_angle_diff(self, a1, a2):
        diff = a2 - a1
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    def check_collision(self, n1, n2):
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

    def step(self):
        if self.finished: return False
        
        self.nodes_expanded += 1
        # Sample
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
        else: 
            rnd = [self.goal.x, self.goal.y]

        # Nearest
        dists = [(n.x-rnd[0])**2 + (n.y-rnd[1])**2 for n in self.nodes]
        nearest = self.nodes[np.argmin(dists)]

        # Steer
        target_theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)
        if nearest.parent is not None:
            diff = self.get_angle_diff(nearest.yaw, target_theta)
            if abs(diff) > self.max_angle:
                sign = 1 if diff > 0 else -1
                target_theta = nearest.yaw + (sign * self.max_angle)

        new_x = nearest.x + self.step_size * math.cos(target_theta)
        new_y = nearest.y + self.step_size * math.sin(target_theta)
        new_node = self.Node(new_x, new_y, target_theta)

        if self.check_collision(nearest, new_node): return False

        new_node.parent = nearest
        new_node.cost = nearest.cost + self.step_size
        self.nodes.append(new_node)

        # Check Goal
        dist_to_goal = math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y)
        if dist_to_goal <= self.step_size:
            final = self.Node(self.goal.x, self.goal.y)
            final_heading = math.atan2(final.y - new_node.y, final.x - new_node.x)
            if abs(self.get_angle_diff(new_node.yaw, final_heading)) <= self.max_angle:
                if not self.check_collision(new_node, final):
                    final.parent = new_node
                    self.path = self.extract_path(final)
                    self.finished = True
                    return True
            
            if dist_to_goal <= 2.0: # Threshold
                self.path = self.extract_path(new_node)
                self.finished = True
                return True
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

# --- 4. ALGORITHM 2: KINEMATIC A* ---
class SnakeKinematicAStar:
    class Node:
        def __init__(self, x, y, yaw, g=0.0, h=0.0, parent=None):
            self.x, self.y, self.yaw = x, y, yaw
            self.g = g  # Cost from start
            self.h = h  # Heuristic to goal
            self.f = g + h
            self.parent = parent
        
        def __lt__(self, other): return self.f < other.f

    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.step_size = SEGMENT_LENGTH
        self.max_angle = math.radians(MAX_JOINT_ANGLE)
        
        self.start = self.Node(start[0], start[1], 0.0)
        self.goal = self.Node(goal[0], goal[1], 0.0)
        
        self.path = None
        self.finished = False
        self.nodes_expanded = 0
        
        self.open_set = []
        heapq.heappush(self.open_set, self.start)
        self.closed_set = {} 
        self.recently_visited = []

        # --- IMPROVEMENT: HIGH RESOLUTION STEERING ---
        # 31 discrete actions = ~4.5 degree increments
        # This allows A* to "thread the needle" in narrow gaps
        self.steering_actions = np.linspace(-self.max_angle, self.max_angle, 31)

    def check_collision(self, n1, n2):
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

    def get_discrete_state(self, node):
        x_d = int(round(node.x))
        y_d = int(round(node.y))
        # Finer Yaw Resolution (5 degrees) for State Hashing
        yaw_d = int(round(math.degrees(node.yaw) / 5.0))
        return (x_d, y_d, yaw_d)

    def step(self):
        if self.finished: return False
        if not self.open_set: 
            self.finished = True # Failed
            return False

        current = heapq.heappop(self.open_set)
        self.nodes_expanded += 1
        self.recently_visited.append(current)

        # Check Goal
        dist_to_goal = math.hypot(current.x - self.goal.x, current.y - self.goal.y)
        if dist_to_goal <= 2.0:
            self.path = self.extract_path(current)
            self.finished = True
            return True

        # Expand Children
        for steering in self.steering_actions:
            new_yaw = current.yaw + steering
            new_yaw = (new_yaw + math.pi) % (2 * math.pi) - math.pi
            
            new_x = current.x + self.step_size * math.cos(new_yaw)
            new_y = current.y + self.step_size * math.sin(new_yaw)
            
            if new_x < 0 or new_x >= self.map.width or new_y < 0 or new_y >= self.map.height:
                continue

            new_node = self.Node(new_x, new_y, new_yaw, current.g + self.step_size, 0, current)
            
            if self.check_collision(current, new_node):
                continue

            # Weighted Heuristic (1.05) to encourage exploration towards goal
            new_node.h = math.hypot(new_x - self.goal.x, new_y - self.goal.y)
            new_node.f = new_node.g + (new_node.h * 1.05)

            state = self.get_discrete_state(new_node)
            if state in self.closed_set and self.closed_set[state] <= new_node.g:
                continue 
            
            self.closed_set[state] = new_node.g
            heapq.heappush(self.open_set, new_node)
            
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

# --- 5. ANIMATED COMPARISON ---
def draw_snake_body(ax, path, num_links):
    if len(path) < num_links + 1: return
    body_points = path[-(num_links+1):] 
    bx, by = zip(*body_points)
    ax.plot(bx, by, color='lime', linestyle='-', linewidth=4, alpha=0.9, zorder=5)
    ax.plot(bx, by, 'o', markerfacecolor='white', markeredgecolor='black', markersize=4, zorder=6)
    ax.plot(bx[-1], by[-1], 'D', markerfacecolor='gold', markeredgecolor='black', markersize=6, zorder=7)

def draw_current_tree_rrt(ax, nodes):
    # RRT Visualization: Magenta Lines
    for node in nodes:
        if node.parent:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                    color='magenta', linestyle='-', linewidth=0.8, alpha=0.4)

def draw_current_tree_astar(ax, nodes):
    # A* Visualization: Cyan Lines (Closed Set)
    if len(nodes) > 1000:
        nodes_to_draw = nodes[-500:] 
    else:
        nodes_to_draw = nodes
        
    for node in nodes_to_draw:
        if node.parent:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                    color='cyan', linestyle='-', linewidth=0.8, alpha=0.4)

if __name__ == "__main__":
    WIDTH, HEIGHT = 70, 70
    START = (5, 5)
    GOAL = (65, 5)
    
    print("Generating Environment...")
    env = DebrisMap(WIDTH, HEIGHT, safe_zones=[START, GOAL])
    
    rrt = SnakeKinematicRRT(env, START, GOAL)
    astar = SnakeKinematicAStar(env, START, GOAL)

    # Setup Plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    start_time = time.time()
    
    while not (rrt.finished and astar.finished):
        # Step RRT
        if not rrt.finished:
            for _ in range(RRT_STEPS_PER_FRAME):
                rrt.step()
                if rrt.finished: break
        
        # Step A*
        if not astar.finished:
            for _ in range(ASTAR_STEPS_PER_FRAME):
                astar.step()
                if astar.finished: break
        
        # --- DRAW RRT ---
        ax1.clear()
        ax1.set_title(f"RRT: {rrt.nodes_expanded} nodes")
        ax1.imshow(env.raw_grid, cmap='Greys', origin='lower')
        ax1.plot(START[0], START[1], 'go')
        ax1.plot(GOAL[0], GOAL[1], 'ro')
        draw_current_tree_rrt(ax1, rrt.nodes)
        
        if rrt.path:
            px, py = zip(*rrt.path)
            ax1.plot(px, py, 'lime', linewidth=2)

        # --- DRAW A* ---
        ax2.clear()
        ax2.set_title(f"A*: {astar.nodes_expanded} nodes")
        ax2.imshow(env.raw_grid, cmap='Greys', origin='lower')
        ax2.plot(START[0], START[1], 'go')
        ax2.plot(GOAL[0], GOAL[1], 'ro')
        draw_current_tree_astar(ax2, astar.recently_visited)
        
        if astar.path:
            px, py = zip(*astar.path)
            ax2.plot(px, py, 'blue', linewidth=2)

        plt.pause(0.001)
        
        # Failsafe timeout
        if time.time() - start_time > 120: # Extended time out to 120s
            print("Timeout!")
            break

    print("Both algorithms finished (or timed out).")
    
    # Final Draw
    if rrt.path: draw_snake_body(ax1, rrt.path, NUM_LINKS)
    if astar.path: draw_snake_body(ax2, astar.path, NUM_LINKS)
    
    plt.ioff()
    plt.show()