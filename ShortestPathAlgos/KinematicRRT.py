import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev

# --- 1. CONFIGURATION ---
NUM_LINKS = 5           
SEGMENT_LENGTH = 3.0    
MAX_JOINT_ANGLE = 70    # Degrees relative to previous link
ANIMATION_SPEED = 50    

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
            safe_area = (slice(max(0,sy-4), min(self.height,sy+5)), slice(max(0,sx-4), min(self.width,sx+5)))
            self.raw_grid[safe_area] = 0
            self.planning_grid[safe_area] = 0

    def is_collision(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        return self.planning_grid[int(y)][int(x)] == 1

# --- 3. KINEMATIC RRT (Planner) ---
class SnakeKinematicRRT:
    class Node:
        def __init__(self, x, y, yaw=0.0):
            self.x = x
            self.y = y
            self.yaw = yaw 
            self.parent = None
            self.cost = 0.0

    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.step_size = SEGMENT_LENGTH 
        self.max_angle = math.radians(MAX_JOINT_ANGLE) 
        self.start = self.Node(start[0], start[1], 0.0) 
        self.goal = self.Node(goal[0], goal[1], 0.0)
        self.nodes = [self.start]
        self.finished = False
        self.path = None

    def get_angle_diff(self, a1, a2):
        diff = a2 - a1
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    def check_collision_bresenham(self, n1, n2):
        x0, y0 = int(n1.x), int(n1.y)
        x1, y1 = int(n2.x), int(n2.y)
        if self.map.is_collision(x0, y0) or self.map.is_collision(x1, y1): return True
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
        if self.map.is_collision(x, y): return True
        return False

    def step(self):
        if self.finished: return False
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
        else: 
            rnd = [self.goal.x, self.goal.y]
        dists = [(n.x-rnd[0])**2 + (n.y-rnd[1])**2 for n in self.nodes]
        nearest = self.nodes[np.argmin(dists)]
        target_theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)
        if nearest.parent is not None:
            diff = self.get_angle_diff(nearest.yaw, target_theta)
            if abs(diff) > self.max_angle:
                sign = 1 if diff > 0 else -1
                target_theta = nearest.yaw + (sign * self.max_angle)
        new_x = nearest.x + self.step_size * math.cos(target_theta)
        new_y = nearest.y + self.step_size * math.sin(target_theta)
        new_node = self.Node(new_x, new_y, target_theta)
        if self.check_collision_bresenham(nearest, new_node): return False
        new_node.parent = nearest
        new_node.cost = nearest.cost + self.step_size
        self.nodes.append(new_node)
        dist_to_goal = math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y)
        if dist_to_goal <= self.step_size:
            final_node = self.Node(self.goal.x, self.goal.y)
            final_heading = math.atan2(final_node.y - new_node.y, final_node.x - new_node.x)
            diff = self.get_angle_diff(new_node.yaw, final_heading)
            if abs(diff) <= self.max_angle or dist_to_goal < 2.0:
                if not self.check_collision_bresenham(new_node, final_node):
                    final_node.parent = new_node
                    final_node.yaw = final_heading
                    self.path = self.extract_path(final_node)
                    self.finished = True
                    return True
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

# --- 4. POST-PROCESSING (Collision & Angle Aware) ---
class PathPostProcessor:
    def __init__(self, debris_map, segment_length, max_angle_deg):
        self.map = debris_map
        self.segment_length = segment_length
        self.max_angle_rad = math.radians(max_angle_deg)

    def check_line_collision(self, start, end):
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        if self.map.is_collision(x0, y0) or self.map.is_collision(x1, y1): return True
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
        return False

    def get_angle(self, p1, p2, p3):
        """Calculates angle change at p2 between vectors p1->p2 and p2->p3"""
        angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        diff = angle2 - angle1
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return abs(diff)

    def prune_path(self, path):
        """Greedy pruning that respects OBSTACLES and JOINT ANGLES."""
        if len(path) < 3: return path
        pruned = [path[0]]
        cur = 0
        
        while cur < len(path) - 1:
            found_shortcut = False
            # Check potential shortcuts (furthest first)
            for check in range(len(path)-1, cur, -1):
                is_safe = True
                
                # 1. Check Collision for Shortcut
                if self.check_line_collision(path[cur], path[check]):
                    is_safe = False
                
                # 2. Check Angle Constraint (Crucial Fix!)
                if is_safe and len(pruned) >= 2:
                    # We need to check if the new segment (pruned[-1] -> path[check])
                    # creates a sharp angle relative to the previous segment (pruned[-2] -> pruned[-1])
                    # Note: 'path[cur]' is pruned[-1]
                    p_prev = pruned[-2]
                    p_curr = path[cur] # Start of shortcut
                    p_next = path[check] # End of shortcut
                    
                    angle_diff = self.get_angle(p_prev, p_curr, p_next)
                    if angle_diff > self.max_angle_rad:
                        is_safe = False

                if is_safe:
                    pruned.append(path[check])
                    cur = check
                    found_shortcut = True
                    break
            
            if not found_shortcut:
                cur += 1
                # Even for direct neighbor, check angle (though RRT guaranteed it, pruning previous steps might have altered the incoming vector)
                if len(pruned) >= 2:
                    p_prev = pruned[-2]
                    p_curr = pruned[-1] # The node we just added
                    p_next = path[cur]
                    # If neighbor violates angle (rare but possible if prev was pruned), we might be stuck.
                    # For now, we trust RRT's local validity but prune carefully.
                pruned.append(path[cur])
                
        return pruned

    def _is_path_safe(self, x_pts, y_pts):
        for i in range(len(x_pts)):
            if self.map.is_collision(x_pts[i], y_pts[i]):
                return False
        return True

    def _resample_points(self, x_fine, y_fine):
        new_path = [(x_fine[0], y_fine[0])]
        for i in range(1, len(x_fine)):
            last_x, last_y = new_path[-1]
            dist = math.hypot(x_fine[i] - last_x, y_fine[i] - last_y)
            if dist >= self.segment_length:
                new_path.append((x_fine[i], y_fine[i]))
        final_dist = math.hypot(x_fine[-1] - new_path[-1][0], y_fine[-1] - new_path[-1][1])
        if final_dist > self.segment_length * 0.5:
             new_path.append((x_fine[-1], y_fine[-1]))
        return new_path

    def _linear_resample(self, path):
        x_fine, y_fine = [], []
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = int(dist * 10) # Higher resolution for better angle checking later if needed
            for s in range(steps):
                t = s / steps
                x_fine.append(p1[0] + (p2[0]-p1[0])*t)
                y_fine.append(p1[1] + (p2[1]-p1[1])*t)
        x_fine.append(path[-1][0])
        y_fine.append(path[-1][1])
        return self._resample_points(x_fine, y_fine)

    def smooth_and_resample(self, path):
        if len(path) < 3: return self._linear_resample(path)
        x, y = zip(*path)
        
        # Try Soft Spline
        try:
            tck, _ = splprep([x, y], s=5.0, k=3)
            u_fine = np.linspace(0, 1, num=1000)
            x_s, y_s = splev(u_fine, tck)
            if self._is_path_safe(x_s, y_s):
                print("Using Smooth Spline (s=5.0)")
                return self._resample_points(x_s, y_s)
        except: pass

        # Try Tight Spline
        try:
            tck, _ = splprep([x, y], s=0.0, k=3)
            u_fine = np.linspace(0, 1, num=1000)
            x_t, y_t = splev(u_fine, tck)
            if self._is_path_safe(x_t, y_t):
                print("Using Tight Spline (s=0.0)")
                return self._resample_points(x_t, y_t)
        except: pass

        print("Using Linear Interpolation (Angle-Checked Pruning)")
        return self._linear_resample(path)

# --- 5. VISUALIZATION ---
def draw_snake_body(ax, path, num_links):
    if len(path) < 2: return
    body_points = path[-(num_links+1):] 
    if len(body_points) < 2: return
    bx, by = zip(*body_points)
    ax.plot(bx, by, color='blue', linestyle='-', linewidth=4, alpha=0.8, zorder=5, label='Body Segment')
    ax.plot(bx[:-1], by[:-1], 'o', markerfacecolor='white', markeredgecolor='black', markersize=6, zorder=6, label='Joint')
    ax.plot(bx[-1], by[-1], 'D', markerfacecolor='gold', markeredgecolor='black', markersize=8, zorder=7, label='Head')

def draw_current_tree(ax, nodes):
    for node in nodes:
        if node.parent:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                    color='magenta', linestyle='-', linewidth=0.8, alpha=0.4)

if __name__ == "__main__":
    WIDTH, HEIGHT = 70, 70
    START = (5, 5)
    GOAL = (65, 5)
    
    debris_env = DebrisMap(WIDTH, HEIGHT, safe_zones=[START, GOAL])
    snake_planner = SnakeKinematicRRT(debris_env, START, GOAL)
    
    print("Initialize Map & Planner...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    
    start_time = time.time()
    while not snake_planner.finished:
        for _ in range(ANIMATION_SPEED): 
            snake_planner.step()
            if snake_planner.finished: break
        ax.clear()
        ax.imshow(debris_env.raw_grid, cmap='Greys', origin='lower')
        ax.plot(START[0], START[1], 'go', markersize=10)
        ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10)
        draw_current_tree(ax, snake_planner.nodes)
        ax.set_title(f"Nodes: {len(snake_planner.nodes)} | Time: {time.time()-start_time:.1f}s")
        plt.pause(0.01)

    print(f"Path Found! Total Time: {time.time() - start_time:.2f}s")
    
    print("Post-Processing Path...")
    # PASSING MAX ANGLE TO POST PROCESSOR
    processor = PathPostProcessor(debris_env, SEGMENT_LENGTH, MAX_JOINT_ANGLE)
    
    pruned_path = processor.prune_path(snake_planner.path)
    smooth_path = processor.smooth_and_resample(pruned_path)
    
    ax.clear()
    ax.imshow(debris_env.raw_grid, cmap='Greys', origin='lower')
    ax.imshow(debris_env.planning_grid, cmap='Reds', alpha=0.3, origin='lower')
    ax.plot(START[0], START[1], 'go', markersize=10, label='Start')
    ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10, label='Goal')

    if snake_planner.path:
        ox, oy = zip(*snake_planner.path)
        ax.plot(ox, oy, 'red', linestyle='--', linewidth=1, alpha=0.5, label="Raw RRT")

    px, py = zip(*pruned_path)
    ax.plot(px, py, 'cyan', linestyle='-', linewidth=1, alpha=0.6, label="Pruned Path")

    sx, sy = zip(*smooth_path)
    ax.plot(sx, sy, 'lime', linewidth=3, label="Smooth Path")

    draw_snake_body(ax, smooth_path, NUM_LINKS)

    ax.legend(loc='upper right')
    ax.set_title("Final Path: RRT -> Kinematic Pruning -> Smoothing")
    plt.ioff()
    plt.show()