import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from scipy.ndimage import binary_dilation

# --- 1. Ortam: ZORLU LABİRENT (Sabit) ---
class DebrisMap:
    def __init__(self, width=70, height=70, safe_zones=None):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width)) 
        self.planning_grid = np.zeros((height, width))
        self.safe_zones = safe_zones if safe_zones else []
        
        self.create_hardcore_maze()
        self.carve_guaranteed_path(thickness=6) 
        self.inflate_obstacles()
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
        self.raw_grid[10:12, 55:65] = 1

    def carve_guaranteed_path(self, thickness=6):
        waypoints = [(5, 5), (14, 15), (14, 40), (14, 55), (35, 55), (54, 55), (54, 35), (54, 10), (65, 5)]
        for i in range(len(waypoints)-1):
            self.clear_line(waypoints[i], waypoints[i+1], thickness=thickness)

    def clear_line(self, p1, p2, thickness):
        x1, y1 = p1; x2, y2 = p2
        length = int(math.hypot(x2-x1, y2-y1))
        for i in range(length):
            t = i / length if length > 0 else 0
            x, y = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
            for ty in range(-thickness, thickness+1):
                for tx in range(-thickness, thickness+1):
                    if 0 <= x+tx < self.width and 0 <= y+ty < self.height:
                        self.raw_grid[y+ty, x+tx] = 0

    def inflate_obstacles(self):
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((5,5))).astype(int)

    def enforce_safe_zones(self):
        for (sx, sy) in self.safe_zones:
            sy, sx = int(sy), int(sx)
            safe_area = slice(max(0,sy-3), min(self.height,sy+4)), slice(max(0,sx-3), min(self.width,sx+4))
            self.raw_grid[safe_area] = 0
            self.planning_grid[safe_area] = 0

    def is_collision(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        return self.planning_grid[int(y)][int(x)] == 1

# --- 2. Algoritma: A* ---
class AStarStepper:
    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.start = start; self.goal = goal
        self.movements = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.open_set = {start: 0}; self.came_from = {}; self.g_score = {start: 0}
        self.finished = False; self.path = None
        self.visited_x = []; self.visited_y = []

    def step(self):
        if self.finished or not self.open_set: return False
        current = min(self.open_set, key=self.open_set.get)
        self.visited_x.append(current[0]); self.visited_y.append(current[1])
        if current == self.goal:
            self.path = self.reconstruct_path(self.came_from, current)
            self.finished = True
            return True
        del self.open_set[current]
        for dx, dy in self.movements:
            neighbor = (current[0] + dx, current[1] + dy)
            if self.map.is_collision(neighbor[0], neighbor[1]): continue
            tentative_g = self.g_score[current] + math.hypot(dx, dy)
            if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                f_score = tentative_g + math.hypot(neighbor[0]-self.goal[0], neighbor[1]-self.goal[1])
                self.open_set[neighbor] = f_score
        return False

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# --- 3. Algoritma: RRT* (HIZLANDIRILMIŞ - BRESENHAM) ---
class RRTStepper:
    class Node:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.parent = None; self.cost = 0.0

    def __init__(self, debris_map, start, goal, step_size=2.0, search_radius=10.0):
        self.map = debris_map
        self.goal = self.Node(goal[0], goal[1])
        self.start = self.Node(start[0], start[1])
        self.step_size = step_size
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.finished = False; self.path = None; self.latest_edge = None

    def step(self):
        if self.finished: return False
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
        else: rnd = [self.goal.x, self.goal.y]

        nearest = self.nodes[np.argmin([(n.x-rnd[0])**2 + (n.y-rnd[1])**2 for n in self.nodes])]
        theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)
        new_node = self.Node(nearest.x + self.step_size * math.cos(theta), nearest.y + self.step_size * math.sin(theta))
        
        # BRESENHAM KONTROLÜ (Çok hızlı)
        if self.check_collision_bresenham(nearest, new_node): return False
        
        new_node.parent = nearest
        new_node.cost = nearest.cost + math.hypot(new_node.x-nearest.x, new_node.y-nearest.y)
        self.nodes.append(new_node)
        self.latest_edge = ([nearest.x, new_node.x], [nearest.y, new_node.y])

        if math.hypot(new_node.x-self.goal.x, new_node.y-self.goal.y) <= self.step_size:
            final = self.Node(self.goal.x, self.goal.y)
            if not self.check_collision_bresenham(new_node, final):
                final.parent = new_node
                self.path = self.extract_path(final)
                self.finished = True
                return True
        return False

    def check_collision_bresenham(self, n1, n2):
        """Bresenham Algoritması ile optimize edilmiş çarpışma kontrolü"""
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
                if err < 0:
                    y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.map.is_collision(x, y): return True
                err -= dx
                if err < 0:
                    x += sx; err += dy
                y += sy
        if self.map.is_collision(x, y): return True
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

class RRTStarStepper:
    class Node:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.parent = None
            self.children = []  # <--- NEW: Track children to propagate costs
            self.cost = 0.0

    def __init__(self, debris_map, start, goal, step_size=2.0, search_radius=10.0):
        self.map = debris_map
        self.goal = self.Node(goal[0], goal[1])
        self.start = self.Node(start[0], start[1])
        self.step_size = step_size
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.finished = False
        self.path = None
        self.latest_edge = None

    def propagate_cost_to_leaves(self, parent_node):
        """
        Recursively updates the cost of all children (and their children)
        when a parent's cost changes.
        """
        for child in parent_node.children:
            # Calculate distance between parent and child
            dist = math.hypot(child.x - parent_node.x, child.y - parent_node.y)
            # Update child's cost based on new parent cost
            child.cost = parent_node.cost + dist
            # Recursively continue down the tree
            self.propagate_cost_to_leaves(child)

    def step(self):
        if self.finished: return False
        
        # 1. Sampling
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
        else: 
            rnd = [self.goal.x, self.goal.y]

        # 2. Nearest Node
        dists = [(n.x-rnd[0])**2 + (n.y-rnd[1])**2 for n in self.nodes]
        nearest_idx = np.argmin(dists)
        nearest = self.nodes[nearest_idx]

        # 3. Steer
        theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)
        new_x = nearest.x + self.step_size * math.cos(theta)
        new_y = nearest.y + self.step_size * math.sin(theta)
        new_node = self.Node(new_x, new_y)

        # Collision Check (Early exit)
        if self.check_collision_bresenham(nearest, new_node): 
            return False

        # --- RRT* KEY IMPROVEMENT 1: CHOOSE BEST PARENT ---
        neighbors = []
        for node in self.nodes:
            if (node.x - new_node.x)**2 + (node.y - new_node.y)**2 <= self.search_radius**2:
                neighbors.append(node)
        
        best_node = nearest
        min_cost = nearest.cost + math.hypot(new_node.x - nearest.x, new_node.y - nearest.y)

        for neighbor in neighbors:
            cost_to_new = neighbor.cost + math.hypot(new_node.x - neighbor.x, new_node.y - neighbor.y)
            if cost_to_new < min_cost:
                if not self.check_collision_bresenham(neighbor, new_node):
                    min_cost = cost_to_new
                    best_node = neighbor
        
        # Link new_node to best_node
        new_node.parent = best_node
        new_node.cost = min_cost
        best_node.children.append(new_node) # <--- NEW: Register as child
        
        self.nodes.append(new_node)
        self.latest_edge = ([best_node.x, new_node.x], [best_node.y, new_node.y])

        # --- RRT* KEY IMPROVEMENT 2: REWIRE (CORRECTED) ---
        for neighbor in neighbors:
            if neighbor == best_node: continue 
            
            dist = math.hypot(new_node.x - neighbor.x, new_node.y - neighbor.y)
            new_cost = new_node.cost + dist
            
            if new_cost < neighbor.cost:
                if not self.check_collision_bresenham(new_node, neighbor):
                    # 1. Remove neighbor from its OLD parent's children list
                    if neighbor.parent and neighbor in neighbor.parent.children:
                        neighbor.parent.children.remove(neighbor)

                    # 2. Update Parent and Cost
                    neighbor.parent = new_node
                    neighbor.cost = new_cost
                    
                    # 3. Add to NEW parent's children list
                    new_node.children.append(neighbor)

                    # 4. PROPAGATE changes down the tree
                    self.propagate_cost_to_leaves(neighbor)

        # Check Goal
        if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.step_size:
            final = self.Node(self.goal.x, self.goal.y)
            if not self.check_collision_bresenham(new_node, final):
                final.parent = new_node
                final.cost = new_node.cost + math.hypot(final.x - new_node.x, final.y - new_node.y)
                self.path = self.extract_path(final)
                self.finished = True
                return True
        return False

    def check_collision_bresenham(self, n1, n2):
        # ... (Your existing Bresenham code remains exactly the same) ...
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
                if err < 0:
                    y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.map.is_collision(x, y): return True
                err -= dx
                if err < 0:
                    x += sx; err += dy
                y += sy
        if self.map.is_collision(x, y): return True
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    
# --- 4. Main (YARIŞ MODU) ---
if __name__ == "__main__":
    WIDTH, HEIGHT = 70, 70
    START = (5, 5)
    GOAL = (65, 5)
    
    debris_env = DebrisMap(WIDTH, HEIGHT, safe_zones=[START, GOAL])
    astar = AStarStepper(debris_env, START, GOAL)
    # RRT*'ın adım boyunu Bresenham sayesinde güvenle artırabiliriz
    rrt = RRTStarStepper(debris_env, START, GOAL, step_size=3.0) 

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(debris_env.raw_grid, cmap='Greys', origin='lower')
    ax.plot(START[0], START[1], 'go', markersize=10, label='Başlangıç')
    ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10, label='Hedef')
    ax.plot([], [], 'c.', label='A* (Tarama)')
    ax.plot([], [], 'm-', label='RRT* (Dallanma)')
    ax.legend(loc='upper right')
    
    start_time = time.time()
    astar_time = 0; rrt_time = 0
    iteration = 0
    
    # Döngü
    while not (astar.finished and rrt.finished):
        iteration += 1
        current_time = time.time()
        
        # A* (20 adım birden atarak hızlandırıyoruz)
        if not astar.finished:
            for _ in range(20): 
                if astar.step():
                    astar_time = time.time() - start_time
                    px, py = zip(*astar.path)
                    ax.plot(px, py, 'b-', linewidth=3, label="A* Final")
                    break
            if iteration % 5 == 0 and len(astar.visited_x) > 0:
                ax.plot(astar.visited_x, astar.visited_y, 'c.', markersize=1, alpha=0.4)
                astar.visited_x, astar.visited_y = [], []

        # RRT* (Bresenham sayesinde iterasyon başına daha fazla deneme yapabiliriz)
        if not rrt.finished:
            for _ in range(20): # Eskiden 5'ti, şimdi 10 yaptık çünkü çok hızlı
                if rrt.step():
                    rrt_time = time.time() - start_time
                    rx, ry = zip(*rrt.path)
                    ax.plot(rx, ry, 'r-', linewidth=3, label="RRT* Final")
                    break
                if rrt.latest_edge:
                    ax.plot(rrt.latest_edge[0], rrt.latest_edge[1], 'm-', linewidth=0.5, alpha=0.6)
                    rrt.latest_edge = None

        # Durum Metni
        status = f"Zaman: {current_time - start_time:.1f}s | "
        status += f"A*: {'Bitti (' + str(round(astar_time, 2)) + 's)' if astar.finished else '...'}"
        status += f" | RRT*: {'Bitti (' + str(round(rrt_time, 2)) + 's)' if rrt.finished else '...'}"
        ax.set_title(status)
        
        plt.pause(0.001)
        if iteration > 10000: break

    print(f"Sonuçlar:\nA*: {astar_time:.4f} sn\nRRT*: {rrt_time:.4f} sn")
    plt.ioff()
    plt.show()