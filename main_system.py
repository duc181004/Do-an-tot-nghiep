import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import math
from PIL import Image
from collections import deque
import time

# 1. CẤU HÌNH THÔNG SỐ VÀ ĐƯỜNG DẪN MODEL
WEIGHTS_EYE = 'weights/best_eye_state_mobilenetv2.pt'
WEIGHTS_MOUTH = 'weights/best_yawn_ultimate.pt'
WEIGHTS_EMOTION = 'weights/best_emotion_model.pt'

# Cấu hình Ngáp & Mắt
EMA_ALPHA = 0.15
FPS = 30
WINDOW_SIZE = FPS * 1       # Cửa sổ 1 giây
ALARM_RATIO = 0.6           # 60% frame ngáp -> Cảnh báo
PERCLOS_WINDOW = FPS * 2    # Cửa sổ 2 giây cho mắt
PERCLOS_THRESH = 0.5        # Nếu 50% thời gian nhắm mắt -> Cảnh báo

EMOTION_LABELS = ['Angry', 'Happy', 'Neutral','Surprise']

# 2. KHỞI TẠO MEDIAPIPE & THIẾT BỊ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

MOUTH_IDXS = [61, 291, 0, 17, 39, 40, 37, 267, 269, 270, 409, 287, 375, 321, 405, 314, 84, 181, 91, 146]
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Hệ thống khởi động trên: {device}")

# 3. KHỞI TẠO 3 MÔ HÌNH AI 
def load_mobilenet_v2(weights_path):
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_resnet50(weights_path, num_classes=7):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    
    # Xây lại khối Sequential 
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, num_classes) 
    )
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

print("📦 Đang nạp Model Mắt (MobileNetV2)...")
eye_model = load_mobilenet_v2(WEIGHTS_EYE)
print("📦 Đang nạp Model Miệng (MobileNetV2)...")
mouth_model = load_mobilenet_v2(WEIGHTS_MOUTH)
print("📦 Đang nạp Model Cảm xúc (ResNet50)...")
emotion_model = load_resnet50(WEIGHTS_EMOTION, len(EMOTION_LABELS))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. HÀM HỖ TRỢ 
def calculate_mar(landmarks, w, h):
    p_left = (landmarks.landmark[61].x * w, landmarks.landmark[61].y * h)
    p_right = (landmarks.landmark[291].x * w, landmarks.landmark[291].y * h)
    p_top = (landmarks.landmark[13].x * w, landmarks.landmark[13].y * h)
    p_bottom = (landmarks.landmark[14].x * w, landmarks.landmark[14].y * h)
    dist_h = math.hypot(p_right[0] - p_left[0], p_right[1] - p_left[1])
    dist_v = math.hypot(p_bottom[0] - p_top[0], p_bottom[1] - p_top[1])
    return dist_v / dist_h if dist_h > 0 else 0.0

def calculate_ear(landmarks, idxs, w, h):
    # p1 đến p6 là 6 điểm bao quanh con mắt
    p1 = (landmarks.landmark[idxs[0]].x * w, landmarks.landmark[idxs[0]].y * h)
    p2 = (landmarks.landmark[idxs[1]].x * w, landmarks.landmark[idxs[1]].y * h)
    p3 = (landmarks.landmark[idxs[2]].x * w, landmarks.landmark[idxs[2]].y * h)
    p4 = (landmarks.landmark[idxs[3]].x * w, landmarks.landmark[idxs[3]].y * h)
    p5 = (landmarks.landmark[idxs[4]].x * w, landmarks.landmark[idxs[4]].y * h)
    p6 = (landmarks.landmark[idxs[5]].x * w, landmarks.landmark[idxs[5]].y * h)
    
    # Tính khoảng cách dọc
    dist_v1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
    dist_v2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
    # Tính khoảng cách ngang
    dist_h = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
    
    return (dist_v1 + dist_v2) / (2.0 * dist_h) if dist_h > 0 else 0.0

def calculate_head_ratio(landmarks, w, h):
    # Lấy tọa độ
    nose_x, nose_y = landmarks.landmark[1].x * w, landmarks.landmark[1].y * h
    chin_x, chin_y = landmarks.landmark[152].x * w, landmarks.landmark[152].y * h
    forehead_x, forehead_y = landmarks.landmark[10].x * w, landmarks.landmark[10].y * h

    # Tính khoảng cách bằng math.hypot cho tối ưu hiệu năng
    dist_nose_chin = math.hypot(nose_x - chin_x, nose_y - chin_y)
    dist_nose_forehead = math.hypot(nose_x - forehead_x, nose_y - forehead_y)

    if dist_nose_forehead == 0:
        return 0.0
    return dist_nose_chin / dist_nose_forehead

def get_crop_from_idxs(frame, landmarks, idxs, w, h, pad_ratio=0.2):
    coords = np.array([[int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)] for i in idxs])
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    pad_x, pad_y = int(pad_ratio * (x_max - x_min)), int(pad_ratio * (y_max - y_min))
    x_min, y_min = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x_max, y_max = min(w, x_max + pad_x), min(h, y_max + pad_y)
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

def get_full_face_crop(frame, landmarks, w, h):
    # Lấy min/max của tất cả 468 điểm để tạo Bounding Box 
    coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark])
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    pad = int(0.1 * (x_max - x_min))
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

# 5. CHẠY HỆ THỐNG DMS 
cap = cv2.VideoCapture(0)
baseline_ear = None
baseline_head_ratio = None
ema_head_ratio = None
last_alert_time = 0
active_warning_msg = ""
active_warning_color = (0, 255, 0)
consecutive_closed_frames = 0

# Bộ đệm thời gian
yawn_window = deque(maxlen=WINDOW_SIZE)
eye_window = deque(maxlen=PERCLOS_WINDOW)

# Biến EMA và Cảm xúc bất đồng bộ
ema_mar, ema_yawn_prob = None, None
ema_left_prob, ema_right_prob = None, None  
current_emotion = "Normal"
frame_count = 0

print("🎥 Đã bật Webcam. Nhấn 'q' để thoát.")

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)
        frame_count += 1
        
        is_yawning = False
        is_eyes_closed = False
        is_head_down = False
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # --- LUỒNG 1: XỬ LÝ MIỆNG ---
            raw_mar = calculate_mar(landmarks, w, h)
            mouth_crop, mouth_bbox = get_crop_from_idxs(frame, landmarks, MOUTH_IDXS, w, h, 0.2)
            
            if mouth_crop.size > 0:
                img_t = transform(Image.fromarray(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                yawn_prob = torch.softmax(mouth_model(img_t), dim=1)[0][1].item()
                
                if ema_mar is None:
                    ema_mar, ema_yawn_prob = raw_mar, yawn_prob
                else:
                    ema_mar = EMA_ALPHA * raw_mar + (1 - EMA_ALPHA) * ema_mar
                    ema_yawn_prob = EMA_ALPHA * yawn_prob + (1 - EMA_ALPHA) * ema_yawn_prob
                
                fusion_score = 0.6 * ema_mar + 0.4 * ema_yawn_prob
                is_yawning = (fusion_score > 0.35)
                
                cv2.rectangle(frame, (mouth_bbox[0], mouth_bbox[1]), (mouth_bbox[2], mouth_bbox[3]), (255, 255, 0), 1)

            # --- LUỒNG 2: XỬ LÝ MẮT ---
            left_ear = calculate_ear(landmarks, LEFT_EYE_IDXS, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if baseline_ear is None:
                baseline_ear = avg_ear
            elif avg_ear > baseline_ear * 0.8: # Chỉ update khi mắt không nhắm
                baseline_ear = 0.95 * baseline_ear + 0.05 * avg_ear
            
            EAR_THRESH = 0.75 * baseline_ear
            
            cv2.putText(frame, f"Baseline EAR: {baseline_ear:.2f}", (w - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            left_eye_crop, _ = get_crop_from_idxs(frame, landmarks, LEFT_EYE_IDXS, w, h, 0.3)
            right_eye_crop, _ = get_crop_from_idxs(frame, landmarks, RIGHT_EYE_IDXS, w, h, 0.3)
            
            if left_eye_crop.size > 0 and right_eye_crop.size > 0:
                l_t = transform(Image.fromarray(cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2RGB)))
                r_t = transform(Image.fromarray(cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2RGB)))
                eyes_batch = torch.stack([l_t, r_t]).to(device)
                
                eye_probs = torch.softmax(eye_model(eyes_batch), dim=1)
                raw_left_closed = eye_probs[0][0].item()
                raw_right_closed = eye_probs[1][0].item()
                
                if ema_left_prob is None:
                    ema_left_prob = raw_left_closed
                    ema_right_prob = raw_right_closed
                else:
                    ema_left_prob = EMA_ALPHA * raw_left_closed + (1 - EMA_ALPHA) * ema_left_prob
                    ema_right_prob = EMA_ALPHA * raw_right_closed + (1 - EMA_ALPHA) * ema_right_prob
                
                cv2.putText(frame, f"Left Closed (AI): {ema_left_prob*100:.1f}%", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, f"Right Closed (AI): {ema_right_prob*100:.1f}%", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                avg_ai_prob = (ema_left_prob + ema_right_prob) / 2.0
                
                if avg_ear < EAR_THRESH:
                    is_eyes_closed = True
                elif avg_ai_prob > 0.4:
                    is_eyes_closed = True

            # --- LUỒNG 3: XỬ LÝ CẢM XÚC  ---
            face_crop, face_bbox = get_full_face_crop(frame, landmarks, w, h)
            cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (200, 200, 200), 1)
            
            # Chạy ResNet50 mỗi 30 frames (1 giây)
            if face_crop.size > 0 and frame_count % 30 == 0:
                img_t = transform(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                emotion_preds = torch.softmax(emotion_model(img_t), dim=1)
                emotion_idx = torch.argmax(emotion_preds, dim=1).item()
                current_emotion = EMOTION_LABELS[emotion_idx]
                
                if ema_mar is not None and ema_mar > 0.4:
                    current_emotion = "Yawning..."

            cv2.putText(frame, f"Emotion: {current_emotion}", (face_bbox[0], face_bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # --- LUỒNG 4: XỬ LÝ HEAD POSE ---
            raw_head_ratio = calculate_head_ratio(landmarks, w, h)
            
            if ema_head_ratio is None:
                ema_head_ratio = raw_head_ratio
            else:
                ema_head_ratio = EMA_ALPHA * raw_head_ratio + (1 - EMA_ALPHA) * ema_head_ratio
                
            # Khởi tạo Baseline tự động
            if baseline_head_ratio is None:
                baseline_head_ratio = ema_head_ratio
            # Cập nhật baseline từ từ 
            elif ema_head_ratio > baseline_head_ratio * 0.9:
                baseline_head_ratio = 0.95 * baseline_head_ratio + 0.05 * ema_head_ratio
                
            # Cúi đầu
            HEAD_DROP_THRESH = 0.80 * baseline_head_ratio
            is_head_down = (ema_head_ratio < HEAD_DROP_THRESH)
            
            # Hiển thị Debug
            cv2.putText(frame, f"Head Ratio (EMA): {ema_head_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # XỬ LÝ MẤT MẶT
        else:
            # Face tracking lost -> Reset toàn bộ EMA và biến trạng thái
            ema_mar, ema_yawn_prob = None, None
            ema_left_prob, ema_right_prob = None, None
            yawn_window.clear()
            eye_window.clear()
            is_yawning = False
            is_eyes_closed = False
            consecutive_closed_frames = 0
            current_emotion = "No Face"
            baseline_head_ratio = None
            ema_head_ratio = None
            is_head_down = False
            cv2.putText(frame, "TRACKING LOST", (w//2 - 100, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # --- CẬP NHẬT TRẠNG THÁI & CẢNH BÁO TỔNG HỢP ---
        yawn_window.append(1 if is_yawning else 0)
        eye_window.append(1 if is_eyes_closed else 0)

        if is_eyes_closed:
            consecutive_closed_frames += 1
        else:
            consecutive_closed_frames = 0
        
        yawn_ratio = sum(yawn_window) / WINDOW_SIZE if len(yawn_window) > 0 else 0
        perclos = sum(eye_window) / PERCLOS_WINDOW if len(eye_window) > 0 else 0
        
        # Giao diện thông số
        cv2.putText(frame, f"Yawn Ratio: {yawn_ratio*100:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # HỆ THỐNG CẢNH BÁO ĐA TẦNG
        new_warning = ""
        new_color = (0, 255, 0)
        CRITICAL_PERCLOS_THRESH = 0.75
        
        # 1. ƯU TIÊN CAO NHẤT: Nhắm mắt lâu 
        if consecutive_closed_frames >= 15:
            new_warning = "CRITICAL: MICRO-SLEEP DETECTED!"
            new_color = (0, 0, 255)
        # 2. Cảnh báo Đỏ thứ cấp: Ngủ sâu, Ngủ + Ngáp, Gục đầu
        elif perclos >= CRITICAL_PERCLOS_THRESH:
            new_warning = "CRITICAL: DEEP SLEEP DETECTED!"
            new_color = (0, 0, 255)
        elif perclos >= PERCLOS_THRESH and yawn_ratio >= ALARM_RATIO:
            new_warning = "CRITICAL: SLEEPING & YAWNING!"
            new_color = (0, 0, 255)
        elif is_head_down and perclos >= PERCLOS_THRESH:
            new_warning = "CRITICAL: HEAD DROP!"
            new_color = (0, 0, 255)
        # 3. Cảnh báo Cam 
        elif perclos >= PERCLOS_THRESH:
            new_warning = "WARNING: DROWSINESS!"
            new_color = (0, 165, 255)
        elif yawn_ratio >= ALARM_RATIO:
            new_warning = "WARNING: YAWNING DETECTED!"
            new_color = (0, 165, 255)
        elif is_head_down:
            new_warning = "WARNING: HEAD DROP!"
            new_color = (0, 165, 255)

        # UX COOLDOWN CHO CẢNH BÁO
        current_time = time.time()
        if new_warning != "":
            active_warning_msg = new_warning
            active_warning_color = new_color
            last_alert_time = current_time
        elif current_time - last_alert_time > 2.0:
            active_warning_msg = ""

        if active_warning_msg:
            cv2.putText(frame, active_warning_msg, (w//2 - 280, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, active_warning_color, 2)

        cv2.imshow("HAUI - Advanced Driver Monitoring System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()