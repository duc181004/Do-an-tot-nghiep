import torch
import pygame
from PIL import ImageFont

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

MOUTH_IDXS = [61, 291, 0, 17, 39, 40, 37, 267, 269, 270, 409, 287, 375, 321, 405, 314, 84, 181, 91, 146]
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

# Khởi tạo Font
try:
    font_text = ImageFont.truetype("arial.ttf", 32)
    font_warning = ImageFont.truetype("arial.ttf", 48)
except IOError:
    print("⚠️ Không tìm thấy arial.ttf, chuyển sang font mặc định.")
    font_text = ImageFont.load_default()
    font_warning = ImageFont.load_default()

# Khởi tạo Âm thanh
pygame.mixer.init()
try:
    sound_warning = pygame.mixer.Sound("sound_effects/warning.wav")
    sound_alarm = pygame.mixer.Sound("sound_effects/alarm.wav")
except:
    print("⚠️ Không tìm thấy âm thanh. Vận hành ở chế độ Im lặng.")
    sound_warning = sound_alarm = None

# Khởi tạo Thiết bị
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))