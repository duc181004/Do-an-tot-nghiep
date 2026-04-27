import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import time
import pygame

# 1. CẤU HÌNH THÔNG SỐ HỆ THỐNG
WEIGHTS_EYE = 'weights/best_eye_state_mobilenetv2.pt'
WEIGHTS_MOUTH = 'weights/best_yawn_ultimate.pt'
WEIGHTS_EMOTION = 'weights/best_emotion_model.pt'

EMA_ALPHA = 0.15
TARGET_FPS = 30
WINDOW_SIZE = TARGET_FPS * 1
PERCLOS_WINDOW = TARGET_FPS * 2
PERCLOS_THRESH = 0.5
ALARM_RATIO = 0.6

EMOTION_LABELS = ['Angry', 'Happy', 'Neutral', 'Surprise']

CAM_XUC = {
    'Angry': 'Tức giận',
    'Happy': 'Vui vẻ',
    'Neutral': 'Bình thường',
    'Surprise': 'Bất ngờ'
}

try:
    font_text = ImageFont.truetype("arial.ttf", 32)
    font_warning = ImageFont.truetype("arial.ttf", 48)
except IOError:
    print("⚠️ Không tìm thấy arial.ttf, chuyển sang font mặc định.")
    font_text = ImageFont.load_default()
    font_warning = ImageFont.load_default()

# Âm thanh
pygame.mixer.init()
try:
    sound_warning = pygame.mixer.Sound("sound_effects/warning.wav")
    sound_alarm = pygame.mixer.Sound("sound_effects/alarm.wav")
except:
    print("⚠️ Không tìm thấy âm thanh. Vận hành ở chế độ Im lặng.")
    sound_warning = sound_alarm = None

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🚀 Hệ thống DMS khởi động thành công trên phần cứng: {device.type.upper()}")

# 2. KHỞI TẠO MEDIAPIPE & AI MODELS
mp_face_mesh = mp.solutions.face_mesh
# Tắt refine_landmarks (Không theo dõi Iris) 
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)

MOUTH_IDXS = [61, 291, 0, 17, 39, 40, 37, 267, 269, 270, 409, 287, 375, 321, 405, 314, 84, 181, 91, 146]
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

def load_mobilenet_v2(path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_resnet50(path, classes):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, classes))
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

eye_model = load_mobilenet_v2(WEIGHTS_EYE)
mouth_model = load_mobilenet_v2(WEIGHTS_MOUTH)
emotion_model = load_resnet50(WEIGHTS_EMOTION, len(EMOTION_LABELS))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. CÁC HÀM XỬ LÝ TOÁN HỌC & ĐỒ HỌA
def draw_dashboard_vn(img, fps, ear, perclos, mar, yawn_ratio, emotion, score, warning_msg, warning_color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font_sm = ImageFont.truetype("arialbd.ttf", 18)
        font_md = ImageFont.truetype("arialbd.ttf", 22)
        font_lg = ImageFont.truetype("arialbd.ttf", 35)
    except:
        font_sm = font_md = font_lg = ImageFont.load_default()

    draw.text((10, 20), f"FPS: {int(fps)}", font=font_sm, fill=(0, 255, 0))
    draw.text((10, 60), f"EAR: {ear:.2f} | PERCLOS: {perclos*100:.1f}%", font=font_md, fill=(255, 255, 255))
    draw.text((10, 100), f"MAR: {mar:.2f} | Yawn: {yawn_ratio*100:.1f}%", font=font_md, fill=(255, 255, 255))
    
    score_color = (0, 255, 0) if score > 0.65 else ((255, 255, 0) if score > 0.4 else (255, 0, 0))
    draw.text((10, 140), f"Chỉ số tập trung: {score*100:.1f}%", font=font_md, fill=score_color)

    cam_xuc_hien_tai = CAM_XUC.get(current_emotion)

    draw.text((20, 250), f"Cảm xúc: {cam_xuc_hien_tai}", font=font_text, fill=(255, 255, 255))

    if warning_msg:
        draw.text((380, img.shape[0] - 80), warning_msg, font=font_lg, fill=(warning_color[2], warning_color[1], warning_color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_mar(landmarks, w, h):
    p_left, p_right = landmarks.landmark[61], landmarks.landmark[291]
    p_top, p_bottom = landmarks.landmark[13], landmarks.landmark[14]
    dist_h = math.hypot((p_right.x - p_left.x)*w, (p_right.y - p_left.y)*h)
    dist_v = math.hypot((p_bottom.x - p_top.x)*w, (p_bottom.y - p_top.y)*h)
    return dist_v / dist_h if dist_h > 0 else 0.0

def calculate_ear(landmarks, idxs, w, h):
    p = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in idxs]
    v1 = math.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
    v2 = math.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
    h_dist = math.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
    return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0.0

def calculate_head_ratio(landmarks, w, h):
    n, c, f = landmarks.landmark[1], landmarks.landmark[152], landmarks.landmark[10]
    dist_nc = math.hypot((n.x - c.x)*w, (n.y - c.y)*h)
    dist_nf = math.hypot((n.x - f.x)*w, (n.y - f.y)*h)
    return dist_nc / dist_nf if dist_nf > 0 else 0.0

def get_crop(frame, coords, idxs, w, h, pad=0.2):
    pts = coords[idxs]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    pw, ph = int(pad * (x_max - x_min)), int(pad * (y_max - y_min))
    return frame[max(0, int(y_min)-ph):min(h, int(y_max)+ph), max(0, int(x_min)-pw):min(w, int(x_max)+pw)], (x_min, y_min, x_max, y_max)

# 4. VÒNG LẶP HỆ THỐNG DMS
cap = cv2.VideoCapture(0)

prev_time = time.time()
smoothed_fps = 30.0
baseline_ear = 0.3
ema_head_ratio, baseline_head_ratio = 1.0, 1.0
attention_score = 1.0

yawn_window = deque(maxlen=WINDOW_SIZE)
eye_window = deque(maxlen=PERCLOS_WINDOW)
consecutive_closed_frames = 0
head_down_counter = 0 

ema_mar, ema_left_prob, ema_right_prob = None, None, None
last_yawn_prob = 0.0      
current_emotion = "Normal"
frame_count = 0

warning_counter = 0
candidate_msg = ""
active_warning_msg = ""
last_alert_time = 0
last_sound_time = 0 

show_debug_mode = False 

print("🎥 Hệ thống đang quét luồng Camera. Nhấn 'q' để thoát. Nhấn 'd' bật Debug.")

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (1280, 720))

        h, w = frame.shape[:2]
        
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0: smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1 / time_diff)
        prev_time = curr_time
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        
        is_eyes_closed = is_yawning = False
        is_head_down_final = False
        avg_ear = raw_mar = perclos = yawn_ratio = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
            face_w = np.max(coords[:, 0]) - np.min(coords[:, 0])
            
            if face_w < w * 0.12:
                active_warning_msg = "FACE_TOO_FAR"
                yawn_window.clear(); eye_window.clear()
            else:
                # --- XỬ LÝ MIỆNG ---
                raw_mar = calculate_mar(landmarks, w, h)
                mouth_crop, m_bbox = get_crop(frame, coords, MOUTH_IDXS, w, h, 0.2)
                
                if mouth_crop.size > 0:
                    if frame_count % 2 == 0:
                        img_t = transform(Image.fromarray(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                        last_yawn_prob = torch.softmax(mouth_model(img_t), dim=1)[0][1].item()
                    
                    ema_mar = raw_mar if ema_mar is None else 0.15 * raw_mar + 0.85 * ema_mar
                    is_yawning = ((0.6 * ema_mar + 0.4 * last_yawn_prob) > 0.35)
                    
                    if show_debug_mode:
                        cv2.rectangle(frame, (int(m_bbox[0]), int(m_bbox[1])), (int(m_bbox[2]), int(m_bbox[3])), (255, 255, 0), 1)

                # --- XỬ LÝ MẮT ---
                left_ear = calculate_ear(landmarks, LEFT_EYE_IDXS, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear > baseline_ear * 0.8: baseline_ear = 0.99 * baseline_ear + 0.01 * avg_ear
                EAR_THRESH = 0.75 * baseline_ear

                left_eye_crop, l_bbox = get_crop(frame, coords, LEFT_EYE_IDXS, w, h, 0.3)
                right_eye_crop, r_bbox = get_crop(frame, coords, RIGHT_EYE_IDXS, w, h, 0.3)
                
                if show_debug_mode:
                    cv2.rectangle(frame, (int(l_bbox[0]), int(l_bbox[1])), (int(l_bbox[2]), int(l_bbox[3])), (0, 255, 0), 1)
                    cv2.rectangle(frame, (int(r_bbox[0]), int(r_bbox[1])), (int(r_bbox[2]), int(r_bbox[3])), (0, 255, 0), 1)

                if left_eye_crop.size > 0 and right_eye_crop.size > 0:
                    if frame_count % 2 != 0:
                        l_t = transform(Image.fromarray(cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2RGB)))
                        r_t = transform(Image.fromarray(cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2RGB)))
                        eye_probs = torch.softmax(eye_model(torch.stack([l_t, r_t]).to(device)), dim=1)
                        raw_l, raw_r = eye_probs[0][0].item(), eye_probs[1][0].item()
                        
                        ema_left_prob = raw_l if ema_left_prob is None else 0.15 * raw_l + 0.85 * ema_left_prob
                        ema_right_prob = raw_r if ema_right_prob is None else 0.15 * raw_r + 0.85 * ema_right_prob
                    
                    if ema_left_prob is not None:
                        if avg_ear < EAR_THRESH or ((ema_left_prob + ema_right_prob) / 2.0) > 0.4:
                            is_eyes_closed = True

                # --- XỬ LÝ CẢM XÚC & FULL FACE CROP ---
                x_min, y_min = np.max([0, int(np.min(coords[:, 0]))]), np.max([0, int(np.min(coords[:, 1]))])
                x_max, y_max = np.min([w, int(np.max(coords[:, 0]))]), np.min([h, int(np.max(coords[:, 1]))])
                pw, ph = int(0.1 * (x_max - x_min)), int(0.1 * (y_max - y_min))
                
                face_crop = frame[max(0, y_min-ph):min(h, y_max+ph), max(0, x_min-pw):min(w, x_max+pw)]
                
                if show_debug_mode:
                    cv2.rectangle(frame, (max(0, x_min-pw), max(0, y_min-ph)), (min(w, x_max+pw), min(h, y_max+ph)), (255, 0, 255), 1)

                if face_crop.size > 0 and frame_count % 30 == 0:
                    img_t = transform(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                    current_emotion = EMOTION_LABELS[torch.argmax(torch.softmax(emotion_model(img_t), dim=1), dim=1).item()]

                # --- XỬ LÝ ĐẦU ---
                raw_head = calculate_head_ratio(landmarks, w, h)
                ema_head_ratio = 0.15 * raw_head + 0.85 * ema_head_ratio
                if ema_head_ratio > baseline_head_ratio * 0.9: 
                    baseline_head_ratio = 0.99 * baseline_head_ratio + 0.01 * ema_head_ratio
                
                # Bộ lọc độ trễ cho Head Down
                raw_head_down = (ema_head_ratio < 0.80 * baseline_head_ratio and avg_ear < EAR_THRESH * 1.1)
                head_down_counter = head_down_counter + 1 if raw_head_down else 0
                is_head_down_final = (head_down_counter >= 5)

                # --- CẬP NHẬT CỬA SỔ & TÍNH ATTENTION SCORE ---
                eye_window.append(1 if is_eyes_closed else 0)
                yawn_window.append(1 if is_yawning else 0)
                perclos = sum(eye_window) / PERCLOS_WINDOW if len(eye_window) > 0 else 0
                yawn_ratio = sum(yawn_window) / WINDOW_SIZE if len(yawn_window) > 0 else 0
                
                # Attention Score 
                penalty = 0.6 * perclos + 0.25 * yawn_ratio + 0.15 * (1 if is_head_down_final else 0)
                attention_score = np.clip(1.0 - penalty, 0.0, 1.0)
                
                if is_eyes_closed: consecutive_closed_frames += 1
                else: consecutive_closed_frames = 0

                # --- MÁY TRẠNG THÁI CẢNH BÁO ---
                raw_warning = ""
                raw_color = (0, 255, 0)
                
                if consecutive_closed_frames >= 25: 
                    raw_warning, raw_color = "CRITICAL: MICRO-SLEEP!", (0, 0, 255)
                elif attention_score < 0.4:
                    raw_warning, raw_color = "CRITICAL: EXTREME FATIGUE!", (0, 0, 255)
                elif is_head_down_final and perclos > 0.4:
                    raw_warning, raw_color = "CRITICAL: HEAD DROP SLEEP!", (0, 0, 255)
                elif is_head_down_final:
                    raw_warning, raw_color = "WARNING: HEAD DROP!", (0, 165, 255)
                elif perclos > 0.5 or yawn_ratio > 0.6:
                    raw_warning, raw_color = "WARNING: DROWSINESS!", (0, 165, 255)
                elif current_emotion == 'Angry':
                    raw_warning, raw_color = "WARNING: ROAD RAGE!", (0, 165, 255)

                if raw_warning != "" and raw_warning == candidate_msg:
                    warning_counter += 1
                else:
                    candidate_msg = raw_warning
                    warning_counter = 0

                trigger_frames = 5 if candidate_msg.startswith("CRITICAL") else 12
                
                if warning_counter >= trigger_frames:
                    active_warning_msg, active_warning_color = candidate_msg, raw_color
                    last_alert_time = curr_time
                elif curr_time - last_alert_time > 2.0:
                    active_warning_msg = ""
        else:
            active_warning_msg = "TRACKING_LOST"
            yawn_window.clear(); eye_window.clear()
            consecutive_closed_frames = 0
            head_down_counter = 0

        # 5. RENDER ĐỒ HỌA & ÂM THANH
        msg_vn = ""
        color_vn = (0, 255, 0)
        
        if active_warning_msg == "TRACKING_LOST":
            msg_vn, color_vn = "MẤT DẤU KHUÔN MẶT", (0, 0, 255)
        elif active_warning_msg == "FACE_TOO_FAR":
            msg_vn, color_vn = "NGỒI QUÁ XA CAMERA", (0, 165, 255)
        elif active_warning_msg != "":
            color_vn = active_warning_color
            if "MICRO-SLEEP" in active_warning_msg or "FATIGUE" in active_warning_msg: msg_vn = "NGỦ GẬT NGUY HIỂM!"
            elif "HEAD DROP SLEEP" in active_warning_msg: msg_vn = "GỤC ĐẦU NGỦ GẬT!"
            elif "HEAD DROP" in active_warning_msg: msg_vn = "CÚI GẦM ĐẦU!"
            elif "DROWSINESS" in active_warning_msg or "YAWNING" in active_warning_msg: msg_vn = "TÀI XẾ MẤT TẬP TRUNG!"
            elif "ROAD RAGE" in active_warning_msg: msg_vn = "TÀI XẾ CĂNG THẲNG!"
            
            #  Quản lý Cooldown âm thanh
            if curr_time - last_sound_time > 2.0:
                if active_warning_msg.startswith("CRITICAL"):
                    if sound_alarm: sound_alarm.play()
                    last_sound_time = curr_time
                elif active_warning_msg.startswith("WARNING"):
                    if sound_warning: sound_warning.play()
                    last_sound_time = curr_time

            if active_warning_msg.startswith("CRITICAL"):
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (350, h), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)

        frame = draw_dashboard_vn(frame, smoothed_fps, avg_ear, perclos, raw_mar, yawn_ratio, current_emotion, attention_score, msg_vn, color_vn)
        
        cv2.imshow("HAUI - Advanced Driver Monitoring System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): show_debug_mode = not show_debug_mode

cap.release()
cv2.destroyAllWindows()