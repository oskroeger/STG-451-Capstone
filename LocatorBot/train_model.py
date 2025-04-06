import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from PIL import Image
from collections import defaultdict

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)  # Increased resolution
TOP_N = 50  # Number of top classes to keep
ORIG_DATA_DIR = "./dataset/compressed_dataset"
TRIMMED_DATA_DIR = "./dataset/top50_dataset"


def trim_dataset(source_dir, dest_dir, top_n):
    """
    Creates a trimmed dataset containing only the top_n classes
    (based on number of images) from the source_dir.
    """
    # Dictionary to hold counts for each class (folder)
    class_counts = defaultdict(int)
    # Iterate over class folders
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            # Count the number of image files in this folder
            num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = num_files

    # Sort classes by count (descending) and select the top_n
    top_classes = sorted(class_counts, key=class_counts.get, reverse=True)[:top_n]
    print("Top classes:", top_classes)

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Copy only the top classes to the new dataset folder
    for class_name in top_classes:
        src_path = os.path.join(source_dir, class_name)
        dst_path = os.path.join(dest_dir, class_name)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
        print(f"Copied {class_name} with {class_counts[class_name]} images.")

    print("Dataset trimming complete!")


def save_model_and_classes(model, class_names, path="models/country_classifier_improved.pth"):
    """
    Save the model and class names to a file.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, path)
    print(f"Model and class names saved to {path}")


def load_model_and_classes(path="models/top50_classifier.pth", device=DEVICE):
    """
    Load the trained EfficientNetV2-S model and class names.
    """
    checkpoint = torch.load(path, map_location=device)
    num_classes = len(checkpoint["class_names"])

    # Use EfficientNetV2-S, matching training architecture
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint["class_names"]

def preprocess_image(image, image_size=IMAGE_SIZE):
    """
    Preprocess an image for EfficientNetV2-S inference.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform_pipeline(image).unsqueeze(0)

def predict_country(model, class_names, image_tensor):
    """
    Predict the top-1 country.
    """
    from country_coordinates import country_coords
    with torch.no_grad():
        outputs = model(image_tensor.to(DEVICE))
        _, pred_index = torch.max(outputs, 1)
        predicted_country = class_names[pred_index.item()]
        coordinates = country_coords.get(predicted_country, (0.0, 0.0))
    return predicted_country, coordinates

def predict_country_ranked(model, class_names, image_tensor, top_k=50):
    """
    Predict top-k countries with probabilities and coordinates.
    """
    from country_coordinates import country_coords
    with torch.no_grad():
        outputs = model(image_tensor.to(DEVICE))
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            country = class_names[idx.item()]
            coords = country_coords.get(country, (0.0, 0.0))
            results.append((country.lower(), prob.item(), coords))
        return results


# Define FocalLoss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Main block for training logic
if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets

    # First, trim the dataset if not already trimmed
    if not os.path.exists(TRIMMED_DATA_DIR) or len(os.listdir(TRIMMED_DATA_DIR)) == 0:
        print("Trimming dataset to top", TOP_N, "classes...")
        trim_dataset(ORIG_DATA_DIR, TRIMMED_DATA_DIR, TOP_N)
    else:
        print("Trimmed dataset already exists at", TRIMMED_DATA_DIR)

    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load trimmed dataset
    print("Loading trimmed dataset...")
    dataset = datasets.ImageFolder(TRIMMED_DATA_DIR, transform=transform)
    class_names = dataset.classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")

    # Compute sample weights for train_dataset only using the indices from the subset
    all_labels = np.array([s[1] for s in dataset.samples])
    class_sample_count = np.array([np.sum(all_labels == t) for t in np.unique(all_labels)])
    weight_per_class = 1. / class_sample_count
    train_indices = train_dataset.indices
    samples_weight = np.array([weight_per_class[all_labels[idx]] for idx in train_indices])
    samples_weight = torch.from_numpy(samples_weight).float()
    train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model with ImageNet weights
    print("Initializing model...")
    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, len(class_names))
    )
    model = model.to(DEVICE)

    # Define loss function and optimizer
    criterion = FocalLoss()  # Using focal loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

    # Training function
    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(Loss=running_loss / total, Accuracy=100 * correct / total)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / len(train_loader.dataset)
        return epoch_loss, epoch_acc

    # Validation function
    def validate():
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100 * correct / len(val_loader.dataset)
        return val_loss, val_acc

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate()
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    save_model_and_classes(model, class_names)
