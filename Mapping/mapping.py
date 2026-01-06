import open3d as o3d
import numpy as np
import cv2 
import copy
import matplotlib.pyplot as plt # Needed for the color map

# ==========================================
# CONFIGURATION
# ==========================================
VOXEL_SIZE = 0.03       # 3cm resolution (Performance vs Quality trade-off)
MAX_DEPTH = 3.0         # Ignore data beyond 3 meters (Noise reduction)
VISUALIZE_STEP = 5      # Render map every 5 frames (Higher = Faster simulation)
DATA_STEP = 2           # Skip every 2nd frame from dataset to speed up processing

def main_visual_slam():
    print("---------------------------------------")
    print("INITIALIZING RGB-D SLAM SYSTEM")
    print(" [1] Sensor: RGB-D Camera (Simulated)")
    print(" [2] Algorithm: Hybrid RGB-D Odometry (Front-End)")
    print(" [3] Visualization: Depth Heatmap (Red=Close, Blue=Far)")
    print("---------------------------------------")

    # 1. LOAD DATASET (Simulating Sensor Feed)
    dataset = o3d.data.RedwoodIndoorLivingRoom1()
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    # 2. SETUP VISUALIZATION
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D SLAM (Depth Heatmap Mode)", width=960, height=720, left=0, top=0)
    
    # 3. INITIALIZE SLAM VARIABLES
    global_map = o3d.geometry.PointCloud()
    current_trans = np.identity(4) 
    prev_rgbd = None 
    limit = len(dataset.color_paths)
    
    print(f"Starting SLAM Loop on {limit} frames...")

    # Pre-load colormap for speed
    cmap = plt.get_cmap("jet")

    # 4. MAIN SLAM LOOP
    for i in range(0, limit, DATA_STEP): 
        
        # -------------------------------------------------
        # MODULE A: SENSOR ACQUISITION
        # -------------------------------------------------
        color_raw = o3d.io.read_image(dataset.color_paths[i])
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        
        # Create RGB-D Object
        curr_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, 
            depth_trunc=MAX_DEPTH, 
            convert_rgb_to_intensity=False
        )

        # -------------------------------------------------
        # MODULE B: LOCALIZATION (Visual Odometry)
        # -------------------------------------------------
        is_tracking_success = False
        
        if prev_rgbd is not None:
            # RETURNS: (success, transformation_matrix, info_matrix)
            success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                curr_rgbd, prev_rgbd, 
                intrinsic, 
                np.identity(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                o3d.pipelines.odometry.OdometryOption()
            )
            
            if success:
                current_trans = current_trans @ trans
                is_tracking_success = True
            else:
                print(f"[WARNING] Frame {i}: Tracking Lost")

        # -------------------------------------------------
        # MODULE C: MAPPING & COLORING
        # -------------------------------------------------
        # 1. Generate local cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, intrinsic)

        # --- FIX: Noise Filtering (Added back to match presentation) ---
        # Removes points that are too far from their neighbors (dust/noise)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 2. APPLY HEATMAP COLORING (Override RGB)
        # Get the Z-coordinate (Depth) of every point in the local camera frame
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            dist = pts[:, 2] # Z is depth in camera frame
            
            # Normalize distance: 0m -> 1.0 (Red), 3m -> 0.0 (Blue)
            norm_dist = 1.0 - np.clip(dist / MAX_DEPTH, 0, 1)
            
            # Map normalized distance to RGB colors
            colors = cmap(norm_dist)[:, :3]
            
            # Assign new colors to the point cloud
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 3. Transform to Global Position
        pcd.transform(current_trans)
        
        # 4. Add to global map
        global_map += pcd

        if i % 10 == 0:
            global_map = global_map.voxel_down_sample(voxel_size=VOXEL_SIZE)

        # -------------------------------------------------
        # MODULE D: VISUALIZATION
        # -------------------------------------------------
        
        # OpenCV Window
        cv_img = np.asarray(color_raw)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        status_text = "TRACKING: GOOD" if is_tracking_success or i==0 else "TRACKING: LOST"
        cv2.putText(cv_img, "VISUAL SLAM", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cv_img, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Robot Camera Feed", cv_img)

        # Open3D Window
        if i % VISUALIZE_STEP == 0:
            vis.clear_geometries()
            vis.add_geometry(global_map)
            
            ctr = vis.get_view_control()
            robot_pos = current_trans[:3, 3] 
            ctr.set_lookat(robot_pos) 
            
            vis.poll_events()
            vis.update_renderer()

        prev_rgbd = curr_rgbd
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vis.run()
    vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_visual_slam()