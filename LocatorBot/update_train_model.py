import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from PIL import Image
import numpy as np
from tqdm import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_and_classes(model, class_names, path="models/country_classifier.pth"):
    """
    Save the model and class names to a file.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, path)
    print(f"Model and class names saved to {path}")


def load_model_and_classes(path="models/country_classifier.pth", device=DEVICE):
    """
    Load the trained model and class names from a file.
    Automatically detects the number of classes from the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    num_classes = len(checkpoint["class_names"])

    # Initialize the EfficientNetV2 model without pretrained weights and update the classifier layer
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["class_names"]


def preprocess_image(image, image_size=(128, 128)):
    """
    Preprocess an image for input into the model.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Add batch dimension
    return transform(image).unsqueeze(0)


def predict_country(model, class_names, image_tensor):
    """
    Predict the country and its coordinates from an image tensor.
    """
    from country_coordinates import country_coords
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred_index = torch.max(outputs, 1)
        predicted_country = class_names[pred_index.item()]
        coordinates = country_coords.get(predicted_country, (0.0, 0.0))
    return predicted_country, coordinates


def get_class_balanced_sampler(dataset):
    """
    Automatically balance class sampling by computing weights inversely proportional
    to class frequencies and return a WeightedRandomSampler.
    """
    class_counts = np.bincount(dataset.targets)
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()
    sample_weights = [class_weights[label] for label in dataset.targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


if __name__ == '__main__':
    # Define constants
    DATA_DIR = "./dataset/compressed_dataset"
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 0.001
    IMAGE_SIZE = (128, 128)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = dataset.classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")

    # Create a balanced sampler for training data
    balanced_sampler = get_class_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=balanced_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize the model with EfficientNetV2
    print("Initializing model...")
    model = efficientnet_v2_s(weights="IMAGENET1K_V1")
    # Replace the classifier layer to match the number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_one_epoch(epoch):
        """
        Training function for one epoch.
        """
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

            # Update progress bar with current loss and accuracy
            pbar.set_postfix(Loss=running_loss / total, Accuracy=100 * correct / total)
        return running_loss / len(train_loader.dataset), 100 * correct / len(train_loader.dataset)

    def validate():
        """
        Validation function to evaluate the model on the validation set.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update metrics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / len(val_loader.dataset), 100 * correct / len(val_loader.dataset)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate()
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    save_model_and_classes(model, class_names)
