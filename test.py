import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------------------------------------------------
# 1. Setup Device & Hyperparameters
# --------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 72  # Match your dataset's number of classes
BATCH_SIZE = 32

# --------------------------------------------------------------------------------------------
# 2. Test Transforms & Dataset Loader
# --------------------------------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # must match training resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset (or validation folder)
test_dataset = ImageFolder(root='archive/synthetic_test_set', transform=test_transform) # or your test folder path
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes

# --------------------------------------------------------------------------------------------
# 3. Load Trained Model
# --------------------------------------------------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

# Load saved weights from training
model.load_state_dict(torch.load('model.pt', map_location=device))
model = model.to(device)
model.eval()

# --------------------------------------------------------------------------------------------
# 4. Gather Predictions & Ground Truth
# --------------------------------------------------------------------------------------------
all_preds = []
all_targets = []

print(f"Running evaluation on {device}...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())

# --------------------------------------------------------------------------------------------
# 5. Print Classification Report
# --------------------------------------------------------------------------------------------
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_targets, all_preds, target_names=class_names))

# --------------------------------------------------------------------------------------------
# 6. Plot Confusion Matrix
# --------------------------------------------------------------------------------------------
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()