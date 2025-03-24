import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_and_classes(model, class_names, path="models/country_classifier.pth"):
    """
    Save the model and class names to a file.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,  # Ensure class names are saved correctly
    }, path)
    print(f"Model and class names saved to {path}")


def load_model_and_classes(path="models/efficientnet_b0.pth", device=DEVICE):
    """
    Load the trained model and class names from a file.
    Automatically detects the number of classes from the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    num_classes = len(checkpoint["class_names"])  # Detect the number of classes

    # Adjusted architecture to match the saved model
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]
    return model, class_names


def preprocess_image(image, image_size=(128, 128)):
    """
    Preprocess an image for input into the model.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


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


def predict_country_ranked(model, class_names, image_tensor, top_k=5):
    """
    Predict the top-k countries with probabilities and their coordinates.
    Returns a list of (country, probability, (lat, lon)) tuples.
    """
    from country_coordinates import country_coords

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            country = class_names[idx.item()]
            coords = country_coords.get(country, (0.0, 0.0))
            results.append((country.lower(), prob.item(), coords))  # lowercase for matching
        return results


# Main block for training logic
if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets
    from tqdm import tqdm
    import pandas as pd

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = dataset.classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Split dataset into train, val, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize the model
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

            # Update progress bar
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

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update metrics
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

        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    save_model_and_classes(model, class_names)
