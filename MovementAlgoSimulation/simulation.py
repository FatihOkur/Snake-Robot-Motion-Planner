import pybullet as p
import pybullet_data
import time
import math
import os
import random

# --- AYARLAR ---
URDF_PATH = "MovementAlgoSimulation/snakerobotmodel/yilansimu_description/urdf/yilansimu.urdf"
ALT_URDF_PATH = "snakerobotmodel/yilansimu_description/urdf/yilansimu.urdf"

# Simülasyon
GRAVITY = -9.81
CONTROL_FREQ = 240
DT = 1./CONTROL_FREQ

# --- HAREKET PARAMETRELERİ (ARCADE MODU - YAVAŞLATILDI) ---
ROBOT_SCALE = 2.0      
MOVE_SPEED = 0.5       # Hız düşük (Sakin hareket)
TURN_SPEED = 2.0       # Dönüşler yumuşak
WAVE_AMPLITUDE = 0.5   # Görsel kıvrılma miktarı
WAVE_FREQ = 4.0        # Görsel kıvrılma hızı

# --- HEDEF NOKTASI ---
TARGET_POS = [1.5, 0.5, 0] 

def get_yaw_from_quaternion(quat):
    _, _, yaw = p.getEulerFromQuaternion(quat)
    return yaw

def main():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    
    # Zemin
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=0.1)

    # Robot Yükleme
    startPos = [0, 0, 0.1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    final_path = URDF_PATH
    if not os.path.exists(URDF_PATH):
        if os.path.exists(ALT_URDF_PATH):
            final_path = ALT_URDF_PATH
        else:
            print("HATA: URDF bulunamadı.")
            return

    robotId = p.loadURDF(final_path, startPos, startOrientation, 
                         globalScaling=ROBOT_SCALE, 
                         flags=p.URDF_USE_SELF_COLLISION)

    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=TARGET_POS)

    # Eklemleri Tanımla
    num_joints = p.getNumJoints(robotId)
    actuated_joints = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        joint_type = joint_info[2]
        
        # Robotun sürtünmelerini kapatıyoruz
        p.changeDynamics(robotId, i, 
                         lateralFriction=0.0, 
                         spinningFriction=0.0,
                         rollingFriction=0.0,
                         linearDamping=0.0,
                         angularDamping=0.0)

        if joint_type == p.JOINT_REVOLUTE:
            actuated_joints.append(i)

    # --- KONTROL BUTONLARI ---
    # Ekrana bir buton ekliyoruz. Basıldıkça sayacı artar.
    pause_btn = p.addUserDebugParameter("SIMULASYONU DURDUR/BASLAT", 1, 0, 0)
    last_pause_click = 0
    is_paused = False

    # --- SİMÜLASYON DÖNGÜSÜ ---
    t = 0
    print(f"\nHedef: {TARGET_POS}")
    print("Mod: ARCADE (Yavaş ve Yerde)")
    print("DURDURMAK İÇİN: 'SPACE' tuşuna basın veya ekrandaki butonu kullanın.")
    
    current_target = list(TARGET_POS)

    while True:
        try:
            # --- DURDURMA MANTIĞI ---
            # 1. Buton kontrolü
            btn_clicks = p.readUserDebugParameter(pause_btn)
            if btn_clicks > last_pause_click:
                is_paused = not is_paused
                last_pause_click = btn_clicks
            
            # 2. Klavye kontrolü (Space tuşu)
            keys = p.getKeyboardEvents()
            # 32 = Space bar ASCII kodu
            if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
                is_paused = not is_paused
                print(f"Simülasyon Durumu: {'DURDU' if is_paused else 'DEVAM EDİYOR'}")

            # Eğer durdurulduysa döngünün geri kalanını (fizik hesaplamalarını) atla
            if is_paused:
                time.sleep(0.1) # İşlemciyi yormamak için bekle
                continue

            # --- 1. NAVİGASYON HESABI ---
            head_pos, head_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = get_yaw_from_quaternion(head_orn)

            dx = current_target[0] - head_pos[0]
            dy = current_target[1] - head_pos[1]
            dist_to_target = math.sqrt(dx**2 + dy**2)
            target_angle = math.atan2(dy, dx)
            
            angle_error = target_angle - current_yaw
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

            # --- 2. HAREKET ---
            if dist_to_target > 0.5:
                vx = (dx / dist_to_target) * MOVE_SPEED
                vy = (dy / dist_to_target) * MOVE_SPEED
            else:
                vx, vy = 0, 0

            wz = angle_error * TURN_SPEED

            p.resetBaseVelocity(robotId, linearVelocity=[vx, vy, -0.5], angularVelocity=[0, 0, wz])

            # --- 3. GÖRSEL ANİMASYON ---
            for index, joint_id in enumerate(actuated_joints):
                phase_offset = index * 1.0 
                target_pos = WAVE_AMPLITUDE * math.sin(WAVE_FREQ * t - phase_offset)
                
                p.setJointMotorControl2(
                    bodyUniqueId=robotId,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=100.0
                )

            p.stepSimulation()
            time.sleep(DT)
            t += DT
            '''
            # Kamera Takibi
            p.resetDebugVisualizerCamera(
                cameraDistance=8.0, 
                cameraYaw=math.degrees(current_yaw) - 90, 
                cameraPitch=-45, 
                cameraTargetPosition=head_pos
            )
            '''

            # Hedef Kontrolü
            if dist_to_target < 0.5:
                print("HEDEFE ULAŞILDI!")
                current_target = [head_pos[0] + random.uniform(-3, 3), 
                                  head_pos[1] + random.uniform(-3, 3), 0]
                p.resetBasePositionAndOrientation(visualShapeId, current_target, [0,0,0,1])

        except KeyboardInterrupt:
            break

    p.disconnect()

if __name__ == "__main__":
    main()