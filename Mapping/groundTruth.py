import open3d as o3d
import numpy as np
import cv2 
import copy

# ==========================================
# HELPER 1: Manual Trajectory Parser
# Fixes the Open3D version incompatibility
# ==========================================
def read_trajectory_manual(filepath):
    traj = []
    try:
        with open(filepath, 'r') as f:
            content = f.read().split()
            # The file format is: ID, ID, ID, then 4x4 matrix
            # We skip metadata and read the matrix
            # A 4x4 matrix has 16 numbers. The log usually has metadata lines.
            # Redwood format: "metadata \n matrix \n metadata..."
            
            # Re-reading line by line is safer for this format
            f.seek(0)
            lines = f.readlines()
            
            for i in range(0, len(lines), 5):
                # Lines i+1 to i+5 contain the matrix
                if i + 4 < len(lines):
                    matrix_lines = lines[i+1 : i+5]
                    mat = []
                    for line in matrix_lines:
                        mat.append([float(x) for x in line.split()])
                    traj.append(np.array(mat))
    except Exception as e:
        print(f"Error reading trajectory: {e}")
    return traj

# ==========================================
# HELPER 2: Fake IMU Generator
# Simulates what an IMU would see based on motion
# ==========================================
def calculate_fake_imu(current_pose, prev_pose, dt=0.033):
    # Relative motion = inv(Previous) * Current
    pose_rel = np.linalg.inv(prev_pose) @ current_pose
    
    rotation_matrix = pose_rel[:3, :3] # Gyroscope equivalent
    translation = pose_rel[:3, 3]      # Accelerometer equivalent
    
    velocity = translation / dt
    return velocity, rotation_matrix

# ==========================================
# MAIN SYSTEM
# ==========================================
def main():
    print("---------------------------------------")
    print("Starting Autonomous Snake Robot Simulation")
    print("   [1] Loading Sensor Data...")
    print("   [2] Initializing SLAM Algorithm...")
    print("---------------------------------------")

    # 1. LOAD DATASET
    dataset = o3d.data.RedwoodIndoorLivingRoom1()
    
    # 2. READ TRAJECTORY (The "Ground Truth" Position)
    trajectory_matrices = read_trajectory_manual(dataset.trajectory_path)
    print(f"Loaded {len(trajectory_matrices)} trajectory steps.")

    # 3. SETUP CAMERAS & VISUALIZATION
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D LIDAR MAP (Real-Time)", width=960, height=720, left=0, top=0)
    
    # Intrinsic parameters (Simulating the specific sensor lens)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    global_map = o3d.geometry.PointCloud()
    
    # 4. MAPPING LOOP
    # We step by 5 frames to make the simulation run faster (like a time-lapse)
    limit = min(len(dataset.color_paths), len(trajectory_matrices))
    
    for i in range(0, limit, 2): 
        
        # -------------------------------------------------
        # MODULE A: PERCEPTION (RGB Camera)
        # -------------------------------------------------
        # This simulates the camera used for VICTIM DETECTION
        color_img_path = dataset.color_paths[i]
        color_img = cv2.imread(color_img_path)
        
        # Overlay Simulation Info
        cv2.putText(color_img, f"System Status: MAPPING", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(color_img, f"Frame: {i}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Robot 'Eye' (RGB Feed)", color_img)

        # -------------------------------------------------
        # MODULE B: MAPPING (LIDAR / Depth)
        # -------------------------------------------------
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        
        # Convert Depth Image -> 3D Point Cloud (Simulating LIDAR Scan)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_raw, intrinsic, 
            depth_trunc=3.0, # Blind beyond 3 meters
            stride=2         # Skip pixels for speed
        )
        
        # --- FIX 1: Noise Filtering ---
        # Remove statistical outliers (Flying pixels / dust)
        if i % 10 == 0: # Only run heavy filter occasionally to save FPS
             pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # --- FIX 2: Voxel Downsampling ---
        # Makes the map cleaner and sharper (2cm resolution)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        # --- FIX 3: Transformation ---
        # Move the local scan to the global map position
        current_pose = trajectory_matrices[i]
        
        # Redwood standard: Pose maps Camera -> World. 
        # So we apply it directly.
        pcd.transform(current_pose)
        
        # Add to global map
        global_map += pcd

        # -------------------------------------------------
        # MODULE C: TELEMETRY (Simulated IMU)
        # -------------------------------------------------
        if i > 5:
            prev_pose = trajectory_matrices[i-2] # Check against previous step
            vel, rot = calculate_fake_imu(current_pose, prev_pose)
            
            # Print status every 20 frames
            if i % 20 == 0:
                print(f"[IMU] Speed: {np.linalg.norm(vel):.2f} m/s | Map Points: {len(global_map.points)}")

        # -------------------------------------------------
        # VISUALIZATION UPDATE
        # -------------------------------------------------
        # Only redraw the 3D map every 10 frames to keep smooth video
        if i % 10 == 0:
            # Periodic Map Cleanup (prevent memory overflow)
            global_map = global_map.voxel_down_sample(voxel_size=0.03)
            
            vis.clear_geometries()
            vis.add_geometry(global_map)
            
            # Camera Follow Logic (Look at the new points)
            ctr = vis.get_view_control()
            ctr.set_lookat([0, 0, 0])
            ctr.set_zoom(0.4)
            
            vis.poll_events()
            vis.update_renderer()

        # Exit Command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Emergency Stop Triggered.")
            break

    # Keep window open at end
    print("Mapping Complete. Interactive Review Mode.")
    vis.run()
    vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()