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
module_dir = Path(__file__).parent

# Resolve weight file paths relative to this module's location (CNN/)
cnn_weights_path = module_dir / "CNN_Plant_best.pth"
resnet_weights_path = module_dir / "ResNet_Plant_best.pth"

# Models will be loaded lazily on first prediction to avoid import-time failures
cnn_model = None
resnet_model = None

def _load_models_if_needed():
    """Load CNN and ResNet weights into globals if they are not already loaded.
    This is safe to call multiple times and will silently continue if weights
    are missing — prediction functions will raise a clear error instead.
    """
    global cnn_model, resnet_model
    # Load CNN
    if cnn_model is None:
        try:
            m = ImprovedCNN(num_classes).to(device)
            state = torch.load(str(cnn_weights_path), map_location=device)
            # Allow checkpoints that store either state_dict or the raw dict
            if isinstance(state, dict) and "model_state_dict" in state:
                m.load_state_dict(state["model_state_dict"])
            else:
                m.load_state_dict(state)
            m.eval()
            cnn_model = m
        except Exception as e:
            print(f"Warning: failed to load CNN weights from {cnn_weights_path}: {e}")
            cnn_model = None

    # Load ResNet
    if resnet_model is None:
        try:
            r = models.resnet50(weights=None)
            r.fc = nn.Linear(r.fc.in_features, num_classes)
            state = torch.load(str(resnet_weights_path), map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                r.load_state_dict(state["model_state_dict"])
            else:
                r.load_state_dict(state)
            r.to(device)
            r.eval()
            resnet_model = r
        except Exception as e:
            print(f"Warning: failed to load ResNet weights from {resnet_weights_path}: {e}")
            resnet_model = None

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
    # Ensure models are loaded (lazy load)
    _load_models_if_needed()

    img = Image.open(img_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model = cnn_model if model_type.lower() == "cnn" else resnet_model

    if model is None:
        raise RuntimeError(f"Requested model ('{model_type}') is not available because weights were not found.\n"
                           f"Expected files: {cnn_weights_path} and/or {resnet_weights_path} in the `CNN/` folder.")

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence_score = float(confidence.item() * 100)
    return predicted_class, confidence_score, img
