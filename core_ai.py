import torch
import torch.nn as nn
from torchvision import transforms, models
from config import *

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