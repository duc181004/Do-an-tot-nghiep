import cv2
import torch
import mediapipe as mp
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont

from config import *
from core_vision import *
from core_ai import *

print(f"🚀 Hệ thống DMS khởi động thành công trên phần cứng: {device.type.upper()}")

# 1. KHỞI TẠO MEDIAPIPE
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)

# 2. HÀM ĐỒ HỌA
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

    cam_xuc_hien_tai = CAM_XUC.get(emotion, "Bình thường")
    draw.text((20, 250), f"Cảm xúc: {cam_xuc_hien_tai}", font=font_text, fill=(255, 255, 255))

    if warning_msg:
        draw.text((380, img.shape[0] - 80), warning_msg, font=font_lg, fill=(warning_color[2], warning_color[1], warning_color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 3. VÒNG LẶP HỆ THỐNG
print("⏳ Đang khởi động luồng Camera độc lập...")
cap = WebcamVideoStream(src=0).start()

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
    while True:
        ret, frame = cap.read()
        if not ret: 
            continue

        # frame = cv2.resize(frame, (1280, 720))

        h, w = frame.shape[:2]
        
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0: smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1 / time_diff)
        prev_time = curr_time
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

        if frame_count < 45:
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                l_ear = calculate_ear(landmarks, LEFT_EYE_IDXS, w, h)
                r_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS, w, h)
                current_ear = (l_ear + r_ear) / 2.0
                current_head = calculate_head_ratio(landmarks, w, h)
                
                if frame_count == 1: 
                    baseline_ear = current_ear
                    baseline_head_ratio = current_head
                else: 
                    baseline_ear = 0.9 * baseline_ear + 0.1 * current_ear
                    baseline_head_ratio = 0.9 * baseline_head_ratio + 0.1 * current_head

            phan_tram = int((frame_count / 45) * 100)
            cv2.putText(frame, f"DANG HIEU CHINH CA NHAN HOA... {phan_tram}%", 
                        (int(w/2) - 300, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            cv2.imshow("HAUI - Advanced Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
        
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

                # --- XỬ LÝ CẢM XÚC ---
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
                
                raw_head_down = (ema_head_ratio < 0.80 * baseline_head_ratio and avg_ear < EAR_THRESH * 1.1)
                head_down_counter = head_down_counter + 1 if raw_head_down else 0
                is_head_down_final = (head_down_counter >= 5)

                # --- CẬP NHẬT CỬA SỔ & TÍNH SCORE ---
                eye_window.append(1 if is_eyes_closed else 0)
                yawn_window.append(1 if is_yawning else 0)
                perclos = sum(eye_window) / PERCLOS_WINDOW if len(eye_window) > 0 else 0
                yawn_ratio = sum(yawn_window) / WINDOW_SIZE if len(yawn_window) > 0 else 0
                
                penalty = 0.6 * perclos + 0.25 * yawn_ratio + 0.15 * (1 if is_head_down_final else 0)
                attention_score = np.clip(1.0 - penalty, 0.0, 1.0)
                
                if is_eyes_closed: consecutive_closed_frames += 1
                else: consecutive_closed_frames = 0

                # --- LOGIC CẢNH BÁO ---
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

        # --- HIỂN THỊ GIAO DIỆN ---
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
            
            if curr_time - last_sound_time > 2.0:
                if active_warning_msg.startswith("CRITICAL") and sound_alarm:
                    sound_alarm.play()
                    last_sound_time = curr_time
                elif active_warning_msg.startswith("WARNING") and sound_warning:
                    sound_warning.play()
                    last_sound_time = curr_time

            if active_warning_msg.startswith("CRITICAL"):
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        display_frame = cv2.resize(frame, (1280, 720))
        dh, dw = display_frame.shape[:2] # dh=720, dw=1280

        if active_warning_msg.startswith("CRITICAL"):
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (dw, dh), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

        panel = display_frame.copy()
        cv2.rectangle(panel, (0, 0), (350, dh), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.6, display_frame, 0.4, 0, display_frame)

        display_frame = draw_dashboard_vn(display_frame, smoothed_fps, avg_ear, perclos, raw_mar, yawn_ratio, current_emotion, attention_score, msg_vn, color_vn)

        cv2.imshow("HAUI - Advanced Driver Monitoring System", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            cap.stop()
            break
        elif key == ord('d'): 
            show_debug_mode = not show_debug_mode

cv2.destroyAllWindows()