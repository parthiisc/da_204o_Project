# ============================================================
#                IMPORTS
# ============================================================
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image, ImageFile
import random
from pathlib import Path

# Fix corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
#                DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
#                CLASS NAMES
# ============================================================
class_names = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Becterial Blight in Rice', 'Brownspot', 'Common_Rust',
    'Cotton Aphid', 'Flag Smut', 'Gray_Leaf_Spot',
    'Healthy Maize', 'Healthy Wheat', 'Healthy cotton',
    'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane',
    'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast',
    'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust',
    'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust',
    'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew',
    'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt',
    'Yellow Rust Sugarcane', 'bacterial_blight in Cotton'
    , 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm',
    'maize stem borer', 'pink bollworm in cotton',
    'red cotton bug', 'thirps on  cotton'
]
num_classes = len(class_names)

# ============================================================
#                TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ============================================================
#                CNN MODEL DEFINITION
# ============================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.4)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        return self.classifier(x)

# ============================================================
#                LOAD MODELS
# ============================================================
cnn_weights_path = "CNN_Plant_best.pth"
resnet_weights_path = "ResNet_Plant_best.pth"

# CNN
cnn_model = ImprovedCNN(num_classes).to(device)
cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
cnn_model.eval()

# ResNet50
resnet_model = models.resnet50(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load(resnet_weights_path, map_location=device))
resnet_model.to(device)
resnet_model.eval()

# ============================================================
#                SHOW SAMPLE IMAGES
# ============================================================
def get_sample_images(dataset_root, num_samples_per_class=3):
    """
    Returns a list of (PIL Image, class_name) tuples for Streamlit display.
    """
    dataset_root = Path(dataset_root)
    samples = []

    for class_name in class_names:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        imgs = list(class_dir.glob("*.*"))
        if not imgs:
            continue
        selected = random.sample(imgs, min(num_samples_per_class, len(imgs)))
        for img_path in selected:
            img = Image.open(img_path).convert("RGB")
            samples.append((img, class_name))
    return samples

# ============================================================
#                PREDICTION FUNCTION
# ============================================================
def predict_image(model_type, img_file):
    """
    Predict disease class using CNN or ResNet on a single uploaded image.
    Returns predicted class, confidence %, PIL image.
    """
    img = Image.open(img_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model = cnn_model if model_type.lower() == "cnn" else resnet_model

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence_score = float(confidence.item() * 100)
    return predicted_class, confidence_score, img
