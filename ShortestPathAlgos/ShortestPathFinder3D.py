import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time
from scipy.ndimage import binary_dilation

# --- 1. Ortam: 3D ENKAZ HARİTASI (Voksel) ---
class DebrisMap3D:
    def __init__(self, size=40, safe_zones=None):
        self.size = size 
        self.raw_grid = np.zeros((size, size, size)) 
        self.planning_grid = np.zeros((size, size, size))
        self.safe_zones = safe_zones if safe_zones else []
        
        self.create_3d_debris()
        self.inflate_obstacles()
        self.enforce_safe_zones()

    def create_3d_debris(self):
        # Dış Duvarlar
        self.raw_grid[0, :, :] = 1; self.raw_grid[-1, :, :] = 1
        self.raw_grid[:, 0, :] = 1; self.raw_grid[:, -1, :] = 1
        self.raw_grid[:, :, 0] = 1; self.raw_grid[:, :, -1] = 1

        # Rastgele 3D Bloklar ve Sütunlar
        num_obstacles = 45 # Engel sayısını artırdım
        for _ in range(num_obstacles):
            x = random.randint(2, self.size-3)
            y = random.randint(2, self.size-3)
            z = random.randint(2, self.size-3)
            
            shape_type = random.choice(['block', 'pillar_z', 'pillar_y', 'pillar_x'])
            
            if shape_type == 'block':
                self.raw_grid[z:z+5, y:y+5, x:x+5] = 1 # Blokları büyüttüm
            elif shape_type == 'pillar_z': 
                self.raw_grid[:, y:y+2, x:x+2] = 1
            elif shape_type == 'pillar_y': 
                self.raw_grid[z:z+2, :, x:x+2] = 1
            elif shape_type == 'pillar_x': 
                self.raw_grid[z:z+2, y:y+2, :] = 1

    def inflate_obstacles(self):
        # 3D Şişirme
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((3,3,3))).astype(int)

    def enforce_safe_zones(self):
        for (sx, sy, sz) in self.safe_zones:
            sx, sy, sz = int(sx), int(sy), int(sz)
            sl = slice(max(0, sz-3), min(self.size, sz+4))
            sy_sl = slice(max(0, sy-3), min(self.size, sy+4))
            sx_sl = slice(max(0, sx-3), min(self.size, sx+4))
            self.raw_grid[sl, sy_sl, sx_sl] = 0
            self.planning_grid[sl, sy_sl, sx_sl] = 0

    def is_collision(self, x, y, z):
        if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size): return True
        return self.planning_grid[int(z)][int(y)][int(x)] == 1

# --- 2. Algoritma: 3D A* ---
class AStarStepper3D:
    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.start = start; self.goal = goal
        self.movements = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx==0 and dy==0 and dz==0: continue
                    self.movements.append((dx, dy, dz))

        self.open_set = {start: 0}; self.came_from = {}; self.g_score = {start: 0}
        self.finished = False; self.path = None; self.visited_nodes = []

    def step(self):
        if self.finished or not self.open_set: return False
        current = min(self.open_set, key=self.open_set.get)
        self.visited_nodes.append(current)

        if current == self.goal:
            self.path = self.reconstruct_path(self.came_from, current)
            self.finished = True
            return True

        del self.open_set[current]
        
        for dx, dy, dz in self.movements:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if self.map.is_collision(neighbor[0], neighbor[1], neighbor[2]): continue
            
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            tentative_g = self.g_score[current] + dist
            if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                h_score = math.sqrt((neighbor[0]-self.goal[0])**2 + (neighbor[1]-self.goal[1])**2 + (neighbor[2]-self.goal[2])**2)
                self.open_set[neighbor] = tentative_g + h_score
        return False

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# --- 3. Algoritma: 3D RRT* ---
class RRTStepper3D:
    class Node:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
            self.parent = None
            self.cost = 0.0

    def __init__(self, debris_map, start, goal, step_size=2.0):
        self.map = debris_map
        self.start = self.Node(*start)
        self.goal = self.Node(*goal)
        self.step_size = step_size
        self.nodes = [self.start]
        self.finished = False
        self.path = None
        self.latest_edge = None

    def step(self):
        if self.finished: return False
        
        # 1. Random Sampling
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.size), random.uniform(0, self.map.size), random.uniform(0, self.map.size)]
        else: 
            rnd = [self.goal.x, self.goal.y, self.goal.z]

        # 2. Find Nearest (Standard RRT Logic)
        dists = [(n.x-rnd[0])**2 + (n.y-rnd[1])**2 + (n.z-rnd[2])**2 for n in self.nodes]
        nearest = self.nodes[np.argmin(dists)]
        dist = math.sqrt(dists[np.argmin(dists)])
        
        if dist == 0: return False
        
        # 3. Steer
        new_x = nearest.x + (rnd[0] - nearest.x) / dist * self.step_size
        new_y = nearest.y + (rnd[1] - nearest.y) / dist * self.step_size
        new_z = nearest.z + (rnd[2] - nearest.z) / dist * self.step_size
        new_node = self.Node(new_x, new_y, new_z)
        
        # 4. Collision Check (UPDATED TO VOXEL TRAVERSAL)
        # This is the "Apples to Apples" update
        if self.check_collision_voxel_3d(nearest, new_node): return False
        
        # 5. Add to Tree (Standard RRT Logic - No Rewiring, No Optimization)
        new_node.parent = nearest
        new_node.cost = nearest.cost + self.step_size
        self.nodes.append(new_node)
        self.latest_edge = ([nearest.x, new_node.x], [nearest.y, new_node.y], [nearest.z, new_node.z])

        # 6. Check Goal
        dist_to_goal = math.sqrt((new_node.x-self.goal.x)**2 + (new_node.y-self.goal.y)**2 + (new_node.z-self.goal.z)**2)
        if dist_to_goal <= self.step_size:
            final = self.Node(self.goal.x, self.goal.y, self.goal.z)
            if not self.check_collision_voxel_3d(new_node, final):
                final.parent = new_node
                self.path = self.extract_path(final)
                self.finished = True
                return True
        return False

    def check_collision_voxel_3d(self, n1, n2):
        # EXACTLY THE SAME AS RRT*
        x1, y1, z1 = int(n1.x), int(n1.y), int(n1.z)
        x2, y2, z2 = int(n2.x), int(n2.y), int(n2.z)
        dx = abs(x2 - x1); dy = abs(y2 - y1); dz = abs(z2 - z1)
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1
        
        # Driving axis logic
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx; p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0: y1 += ys; p1 -= 2 * dx
                if p2 >= 0: z1 += zs; p2 -= 2 * dx
                p1 += 2 * dy; p2 += 2 * dz
                if self.map.is_collision(x1, y1, z1): return True
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy; p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0: x1 += xs; p1 -= 2 * dy
                if p2 >= 0: z1 += zs; p2 -= 2 * dy
                p1 += 2 * dx; p2 += 2 * dz
                if self.map.is_collision(x1, y1, z1): return True
        else:
            p1 = 2 * dy - dz; p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0: y1 += ys; p1 -= 2 * dz
                if p2 >= 0: x1 += xs; p2 -= 2 * dz
                p1 += 2 * dy; p2 += 2 * dx
                if self.map.is_collision(x1, y1, z1): return True
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]

class RRTStarStepper3D:
    class Node:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
            self.parent = None
            self.children = []  # <--- RESTORED: Track children for cost updates
            self.cost = 0.0

    def __init__(self, debris_map, start, goal, step_size=2.0, search_radius=6.0):
        self.map = debris_map
        self.start = self.Node(*start)
        self.goal = self.Node(*goal)
        self.step_size = step_size
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.finished = False
        self.path = None
        self.latest_edge = None

    def propagate_cost_to_leaves(self, parent_node):
        """
        RESTORED: Recursively updates the cost of all descendants
        when a parent's cost changes due to rewiring.
        """
        for child in parent_node.children:
            # Calculate distance between parent and child
            dist = math.sqrt((child.x - parent_node.x)**2 + 
                             (child.y - parent_node.y)**2 + 
                             (child.z - parent_node.z)**2)
            # Update child's cost
            child.cost = parent_node.cost + dist
            # Recursively continue down the tree
            self.propagate_cost_to_leaves(child)

    def step(self):
        if self.finished: return False
        
        # 1. Sample
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.size), 
                   random.uniform(0, self.map.size), 
                   random.uniform(0, self.map.size)]
        else: 
            rnd = [self.goal.x, self.goal.y, self.goal.z]

        # 2. Nearest Node
        dists = [(n.x-rnd[0])**2 + (n.y-rnd[1])**2 + (n.z-rnd[2])**2 for n in self.nodes]
        nearest_idx = np.argmin(dists)
        nearest = self.nodes[nearest_idx]
        dist = math.sqrt(dists[nearest_idx])
        
        if dist == 0: return False
        
        # 3. Steer
        new_x = nearest.x + (rnd[0] - nearest.x) / dist * self.step_size
        new_y = nearest.y + (rnd[1] - nearest.y) / dist * self.step_size
        new_z = nearest.z + (rnd[2] - nearest.z) / dist * self.step_size
        new_node = self.Node(new_x, new_y, new_z)
        
        # Collision Check (Voxel)
        if self.check_collision_voxel_3d(nearest, new_node): return False
        
        # 4. Choose Best Parent (RRT* Logic)
        neighbors = []
        for node in self.nodes:
            d = (node.x - new_node.x)**2 + (node.y - new_node.y)**2 + (node.z - new_node.z)**2
            if d <= self.search_radius**2:
                neighbors.append(node)

        best_node = nearest
        min_cost = nearest.cost + math.sqrt((new_node.x-nearest.x)**2 + 
                                            (new_node.y-nearest.y)**2 + 
                                            (new_node.z-nearest.z)**2)

        for neighbor in neighbors:
            dist_to_new = math.sqrt((new_node.x-neighbor.x)**2 + 
                                    (new_node.y-neighbor.y)**2 + 
                                    (new_node.z-neighbor.z)**2)
            if neighbor.cost + dist_to_new < min_cost:
                if not self.check_collision_voxel_3d(neighbor, new_node):
                    min_cost = neighbor.cost + dist_to_new
                    best_node = neighbor
        
        # Link new_node to best_node
        new_node.parent = best_node
        new_node.cost = min_cost
        best_node.children.append(new_node) # <--- Track as child
        
        self.nodes.append(new_node)
        self.latest_edge = ([best_node.x, new_node.x], 
                            [best_node.y, new_node.y], 
                            [best_node.z, new_node.z])

        # 5. Rewire (RRT* Logic)
        for neighbor in neighbors:
            if neighbor == best_node: continue
            
            dist = math.sqrt((new_node.x-neighbor.x)**2 + 
                             (new_node.y-neighbor.y)**2 + 
                             (new_node.z-neighbor.z)**2)
            
            if new_node.cost + dist < neighbor.cost:
                if not self.check_collision_voxel_3d(new_node, neighbor):
                    # 1. Remove neighbor from OLD parent's children list
                    if neighbor.parent and neighbor in neighbor.parent.children:
                        neighbor.parent.children.remove(neighbor)
                    
                    # 2. Update Parent and Cost
                    neighbor.parent = new_node
                    neighbor.cost = new_node.cost + dist
                    
                    # 3. Add to NEW parent's children list
                    new_node.children.append(neighbor)
                    
                    # 4. Propagate changes down the tree
                    self.propagate_cost_to_leaves(neighbor)

        # Check Goal
        dist_to_goal = math.sqrt((new_node.x-self.goal.x)**2 + 
                                 (new_node.y-self.goal.y)**2 + 
                                 (new_node.z-self.goal.z)**2)
        
        if dist_to_goal <= self.step_size:
            final = self.Node(self.goal.x, self.goal.y, self.goal.z)
            if not self.check_collision_voxel_3d(new_node, final):
                final.parent = new_node
                final.cost = new_node.cost + math.sqrt((final.x - new_node.x)**2 + 
                                                       (final.y - new_node.y)**2 + 
                                                       (final.z - new_node.z)**2)
                self.path = self.extract_path(final)
                self.finished = True
                return True
        return False
        
    def check_collision_voxel_3d(self, n1, n2):
        """
        Robust 3D Voxel Traversal (3D Bresenham).
        Guarantees no tunneling through thin obstacles.
        """
        x1, y1, z1 = int(n1.x), int(n1.y), int(n1.z)
        x2, y2, z2 = int(n2.x), int(n2.y), int(n2.z)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1
        
        # Driving axis is the one with the greatest distance
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                if self.map.is_collision(x1, y1, z1): return True
                
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                if self.map.is_collision(x1, y1, z1): return True
                
        else: # dz is max
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                if self.map.is_collision(x1, y1, z1): return True
                
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]
    
# --- 4. Main (YÜKSEK KONTRASTLI 3D YARIŞ) ---
if __name__ == "__main__":
    SIZE = 40
    START = (3, 3, 3)
    GOAL = (36, 36, 36)
    
    print("Harita Oluşturuluyor...")
    env = DebrisMap3D(size=SIZE, safe_zones=[START, GOAL])
    
    astar = AStarStepper3D(env, START, GOAL)
    #rrt = RRTStepper3D(env, START, GOAL, step_size=3.0)
    rrt = RRTStarStepper3D(env, START, GOAL, step_size=3.0, search_radius=6.0)

    plt.ion()
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Arka plan rengini hafif koyulaştır ki neon renkler patlasın
    ax.set_facecolor('#f0f0f0') 
    
    # 1. ENGELLERİ ÇİZ (Daha belirgin)
    # alpha=0.25: Yarı saydam
    # edgecolors='gray': Küpün kenarları belli olsun
    # s=50: Küpler daha büyük
    ox, oy, oz = np.where(env.raw_grid == 1)
    ax.scatter(ox, oy, oz, c='black', marker='s', s=40, alpha=0.10, edgecolors='gray', linewidths=0.5, label='Enkaz (Debris)')

    # Başlangıç ve Bitiş (Çok belirgin)
    ax.scatter(*START, c='lime', s=150, edgecolors='black', label='Başlangıç', zorder=10)
    ax.scatter(*GOAL, c='red', s=150, edgecolors='black', label='Hedef', zorder=10)

    # Eksen Ayarları
    ax.set_xlim(0, SIZE); ax.set_ylim(0, SIZE); ax.set_zlim(0, SIZE)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    
    start_time = time.time()
    astar_done_time = 0
    rrt_done_time = 0
    iteration = 0
    
    print("Yarış Başlıyor...")

    while not (astar.finished and rrt.finished):
        iteration += 1
        current_time = time.time()
        
        # --- A* GÖRSELLEŞTİRME ---
        if not astar.finished:
            for _ in range(40): # Hızlandırma
                if astar.step():
                    astar_done_time = time.time() - start_time
                    path = np.array(astar.path)
                    # Final Yol: Kalın MAVİ
                    ax.plot(path[:,0], path[:,1], path[:,2], c='blue', linewidth=6, label='A* Yolu', zorder=5)
                    print(f"--> A* TAMAMLANDI! Süre: {astar_done_time:.4f} sn")
                    break
            
            # Taranan Alan (Cyan Bulutu): Daha belirgin
            if len(astar.visited_nodes) > 0 and iteration % 3 == 0:
                recent = np.array(astar.visited_nodes[-60:])
                ax.scatter(recent[:,0], recent[:,1], recent[:,2], c='cyan', s=10, alpha=0.6)

        # --- RRT* GÖRSELLEŞTİRME ---
        if not rrt.finished:
            for _ in range(40):
                if rrt.step():
                    rrt_done_time = time.time() - start_time
                    path = np.array(rrt.path)
                    # Final Yol: Kalın KIRMIZI
                    ax.plot(path[:,0], path[:,1], path[:,2], c='red', linewidth=6, label='RRT* Yolu', zorder=5)
                    print(f"--> RRT* TAMAMLANDI! Süre: {rrt_done_time:.4f} sn")
                    break
                
                # Dallar: Magenta ve daha kalın
                if rrt.latest_edge:
                    edge = rrt.latest_edge
                    ax.plot(edge[0], edge[1], edge[2], c='magenta', linewidth=2, alpha=0.8)
                    rrt.latest_edge = None

        # Başlık
        status = f"Zaman: {current_time - start_time:.1f}s\n"
        status += f"A*: {str(round(astar_done_time, 2))  if astar.finished else 'Arıyor...'} | "
        status += f"RRT*: {str(round(rrt_done_time, 2)) if rrt.finished else 'Arıyor...'}"
        ax.set_title(status, fontsize=14, fontweight='bold')
        
        plt.pause(0.001)
        if iteration > 2500: break

    # Döngü bitti, son durumu konsola yaz
    print("\n" + "="*30)
    print(f"YARIŞ SONUCU:")
    print(f"A* Algoritması   : {astar_done_time:.4f} saniye")
    print(f"RRT* Algoritması : {rrt_done_time:.4f} saniye")
    print("="*30 + "\n")
    
    plt.ioff()
    plt.show()