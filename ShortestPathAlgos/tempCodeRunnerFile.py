import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev

# --- 1. Ortam: KOMPLEKS Enkaz Haritası ---
class DebrisMap:
    def __init__(self, width=60, height=60, safe_zones=None):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width)) 
        self.inflated_grid = np.zeros((height, width))
        self.safe_zones = safe_zones if safe_zones else []
        
        self.create_complex_maze()
        self.add_random_rubble(density=6)
        self.carve_guaranteed_path()
        self.inflate_obstacles()
        self.enforce_safe_zones()

    def create_complex_maze(self):
        self.raw_grid[0, :] = 1
        self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1
        self.raw_grid[:, -1] = 1
        self.raw_grid[15, 0:40] = 1 
        self.raw_grid[15:45, 20] = 1
        self.raw_grid[0:30, 40] = 1
        self.raw_grid[40:55, 35:55] = 1 
        self.raw_grid[42:53, 37:53] = 0 
        self.raw_grid[45:50, 35:38] = 0 

    def add_random_rubble(self, density):
        num_obstacles = int(self.width * self.height * (density / 100))
        for _ in range(num_obstacles):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            self.raw_grid[y, x] = 1

    def carve_guaranteed_path(self):
        waypoints = [(5, 5), (5, 12), (50, 12), (50, 35), (10, 35), (10, 50), (36, 50), (45, 45), (55, 55)]
        for i in range(len(waypoints)-1):
            self.clear_line(waypoints[i], waypoints[i+1], thickness=4)

    def clear_line(self, p1, p2, thickness=2):
        x1, y1 = p1
        x2, y2 = p2
        length = int(math.hypot(x2-x1, y2-y1))
        for i in range(length):
            t = i / length if length > 0 else 0
            x, y = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
            for ty in range(-thickness, thickness+1):
                for tx in range(-thickness, thickness+1):
                    if 0 <= x+tx < self.width and 0 <= y+ty < self.height:
                        self.raw_grid[y+ty, x+tx] = 0

    def inflate_obstacles(self):
        print("Inflation (Güvenlik Payı) hesaplanıyor...")
        self.inflated_grid = binary_dilation(self.raw_grid, structure=np.ones((3,3))).astype(int)

    def enforce_safe_zones(self):
        for (sx, sy) in self.safe_zones:
            sy, sx = int(sy), int(sx)
            safe_area_y = slice(max(0,sy-3), min(self.height,sy+4))
            safe_area_x = slice(max(0,sx-3), min(self.width,sx+4))
            self.raw_grid[safe_area_y, safe_area_x] = 0
            self.inflated_grid[safe_area_y, safe_area_x] = 0

    def is_collision(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        return self.inflated_grid[int(y)][int(x)] == 1

# --- 2. Algoritma: A* ---
class AStar:
    def __init__(self, debris_map):
        self.map = debris_map

    def plan(self, start, goal):
        movements = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_set = {start: 0} 
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            current = min(open_set, key=open_set.get)
            if current == goal: return self.reconstruct_path(came_from, current)
            del open_set[current]
            
            for dx, dy in movements:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.map.is_collision(neighbor[0], neighbor[1]): continue
                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.hypot(neighbor[0]-goal[0], neighbor[1]-goal[1])
                    open_set[neighbor] = f_score
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# --- 3. Algoritma: RRT* (Akıllı Yumuşatma ile) ---
class RRTStar:
    class Node:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.parent = None
            self.cost = 0.0

    def __init__(self, debris_map, max_iter=6000, step_size=2.0, search_radius=6.0):
        self.map = debris_map
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.nodes = []

    def plan(self, start, goal):
        self.start = self.Node(start[0], start[1])
        self.goal = self.Node(goal[0], goal[1])
        self.nodes = [self.start]

        for i in range(self.max_iter):
            if random.randint(0, 100) > 10: 
                rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
            else: 
                rnd = [self.goal.x, self.goal.y]

            nearest = self.nodes[np.argmin([(n.x-rnd[0])**2 + (n.y-rnd[1])**2 for n in self.nodes])]
            theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)
            new_node = self.Node(nearest.x + self.step_size * math.cos(theta),
                                 nearest.y + self.step_size * math.sin(theta))
            
            if self.check_collision_line(nearest, new_node): continue
            
            near_nodes = [n for n in self.nodes if math.hypot(n.x-new_node.x, n.y-new_node.y) <= self.search_radius]
            new_node.parent = nearest
            new_node.cost = nearest.cost + math.hypot(new_node.x-nearest.x, new_node.y-nearest.y)
            
            for near_node in near_nodes:
                if near_node.cost + math.hypot(near_node.x-new_node.x, near_node.y-new_node.y) < new_node.cost:
                    if not self.check_collision_line(near_node, new_node):
                        new_node.parent = near_node
                        new_node.cost = near_node.cost + math.hypot(near_node.x-new_node.x, near_node.y-new_node.y)
            self.nodes.append(new_node)
            
            if math.hypot(new_node.x-self.goal.x, new_node.y-self.goal.y) <= self.step_size:
                final = self.Node(self.goal.x, self.goal.y)
                if not self.check_collision_line(new_node, final):
                    final.parent = new_node
                    return self.extract_path(final)
        return None

    def check_collision_line(self, n1, n2):
        dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
        step_check = 0.2 
        num_steps = int(dist / step_check) + 1
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            x = n1.x + (n2.x - n1.x) * t
            y = n1.y + (n2.y - n1.y) * t
            if self.map.is_collision(x, y):
                return True
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def smart_smooth_path(self, path):
        """
        Güvenli Yumuşatma: Yolu yumuşatır ama engele çarpıp çarpmadığını kontrol eder.
        Eğer çarpıyorsa yumuşatma miktarını (s) düşürüp tekrar dener.
        """
        # Tekrarlayan noktaları temizle (Spline hatası olmaması için)
        clean_path = []
        for p in path:
            if not clean_path or (p[0] != clean_path[-1][0] or p[1] != clean_path[-1][1]):
                clean_path.append(p)
        
        if len(clean_path) < 3: return path

        try:
            y_coords, x_coords = zip(*clean_path)
            
            # Farklı yumuşaklık seviyelerini dene (Çok yumuşaktan -> aza doğru)
            # s=5.0 (Çok kavisli), s=0.1 (Neredeyse düz)
            for s_val in [5.0, 3.0, 1.0, 0.5, 0.1]:
                tck, u = splprep([x_coords, y_coords], s=s_val, per=False)
                u_new = np.linspace(u.min(), u.max(), 200)
                x_new, y_new = splev(u_new, tck, der=0)
                
                # --- YENİ: Çarpışma Testi ---
                # Oluşturulan bu yeni kavisli yolun HERHANGİ bir noktası engele değiyor mu?
                collision_detected = False
                for i in range(len(x_new)):
                    if self.map.is_collision(x_new[i], y_new[i]):
                        collision_detected = True
                        break
                
                # Eğer çarpışma yoksa, bu yumuşak yolu kabul et ve döndür
                if not collision_detected:
                    return list(zip(x_new, y_new))
            
            # Eğer tüm yumuşatma denemeleri başarısız olursa orijinal yolu döndür
            print("Uyarı: Yumuşatma duvara çarptığı için iptal edildi, ham yol kullanılıyor.")
            return path 

        except Exception as e:
            print(f"Smoothing hatası: {e}")
            return path

# --- 4. Main ---
if __name__ == "__main__":
    WIDTH, HEIGHT = 60, 60
    START = (5, 5)
    GOAL = (55, 55)
    
    debris_env = DebrisMap(WIDTH, HEIGHT, safe_zones=[START, GOAL])
    
    print("1. A* Hesaplanıyor...")
    path_astar = AStar(debris_env).plan(START, GOAL)
    
    print("2. RRT* Hesaplanıyor...")
    rrt_algo = RRTStar(debris_env, max_iter=7000, step_size=2.0)
    path_rrt = rrt_algo.plan(START, GOAL)

    # --- Görselleştirme ---
    plt.figure(figsize=(12, 12))
    
    display_map = debris_env.inflated_grid * 0.5 + debris_env.raw_grid * 0.5
    plt.imshow(display_map, cmap='Greys', origin='lower')
    
    plt.plot([], [], 's', color='black', label='Gerçek Duvar')
    plt.plot([], [], 's', color='gray', label='Güvenlik Alanı (Inflation)')
    plt.plot(START[0], START[1], 'go', markersize=12, label='Start')
    plt.plot(GOAL[0], GOAL[1], 'ro', markersize=12, label='Goal')

    if path_astar:
        px, py = zip(*path_astar)
        plt.plot(px, py, 'b-', linewidth=3, alpha=0.6, label='A* (Grid)')

    if path_rrt:
        # Ham Yol
        rx, ry = zip(*path_rrt)
        plt.plot(rx, ry, 'm--', linewidth=1, alpha=0.5, label='RRT* (Ham)')
        
        # Akıllı Yumuşatılmış Yol
        # smooth_path yerine YENİ smart_smooth_path fonksiyonunu çağırıyoruz
        path_smooth = rrt_algo.smart_smooth_path(path_rrt)
        sx, sy = zip(*path_smooth)
        plt.plot(sx, sy, 'm-', linewidth=3, label='RRT* (Akıllı/Güvenli Smooth)')
    else:
        print("RRT* başarısız.")

    plt.title("Kompleks Labirent: Akıllı Yumuşatma (Duvara Çarpmaz)")
    plt.legend(loc='upper left')
    plt.show()