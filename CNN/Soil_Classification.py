# ============================================================
# IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CNN MODEL
# ============================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ============================================================
# SINGLE IMAGE PREDICTION (THIS IS ALL STREAMLIT USES)
# ============================================================
def predict_image(model_path, image_path, class_names, model_type="cnn"):
    """
    Loads a trained CNN/ResNet model and performs prediction on a single image.
    Safe for Streamlit.
    """

    num_classes = len(class_names)

    # Load model
    if model_type.lower() == "cnn":
        model = ImprovedCNN(num_classes)
    elif model_type.lower() == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Model type must be 'cnn' or 'resnet50'")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Process image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, index = torch.max(probs, 1)

    predicted_class = class_names[index.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score, img


# ============================================================
# TRAINING + EVALUATION CODE (ONLY RUNS WHEN CALLED DIRECTLY)
# ============================================================
if __name__ == "__main__":

    print("Running Soil Model Training Mode (not used by Streamlit)")

    # ----------------------------------------------------------
    # TRANSFORMS
    # ----------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # ----------------------------------------------------------
    # DATASET
    # ----------------------------------------------------------
    root_path = r"C:\Final_Project\Soil_Moisture"
    full_dataset = datasets.ImageFolder(root=root_path, transform=train_transform)

    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset.imgs[i][1] for i in all_indices]

    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(datasets.ImageFolder(root=root_path, transform=test_transform), test_indices)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    # ----------------------------------------------------------
    # MODEL INITIALIZATION
    # ----------------------------------------------------------
    cnn_model = ImprovedCNN(num_classes).to(device)

    resnet_model = models.resnet50(pretrained=True)
    for param in resnet_model.parameters():
        param.requires_grad = False
    for param in resnet_model.layer4.parameters():
        param.requires_grad = True

    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    resnet_model = resnet_model.to(device)

    cnn_weights_path = r"C:\Final_Project\cnn_soil_best.pth"
    resnet_weights_path = r"C:\Final_Project\resnet50_soil_best.pth"

    # ----------------------------------------------------------
    # TRAINING FUNCTION (unchanged)
    # ----------------------------------------------------------
    def train_model(model, train_loader, val_loader, num_epochs=20, save_path=None, patience=5, lr=1e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total, correct = 0, 0
            running_loss = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            total, correct = 0, 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    val_loss += criterion(outputs, labels).item()

                    _, preds = outputs.max(1)
                    total += labels.size(0)
                    correct += preds.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * correct / total

            print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save({"model_state_dict": model.state_dict()}, save_path)
                    print("Checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break

    # ----------------------------------------------------------
    # START TRAINING
    # ----------------------------------------------------------
    train_model(cnn_model, train_loader, test_loader, save_path=cnn_weights_path)
    train_model(resnet_model, train_loader, test_loader, save_path=resnet_weights_path)

