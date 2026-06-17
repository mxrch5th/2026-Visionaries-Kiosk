from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.0003
CLASS_NAMES = ["under50", "over50"]


def build_loaders() -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    print("클래스:", train_dataset.classes)
    if train_dataset.classes != CLASS_NAMES:
        print("권장 폴더명은 under50, over50 입니다. 현재 폴더명을 확인하세요.")

    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    )


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 장치:", device)

    train_loader, val_loader = build_loaders()
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"[{epoch:02d}/{EPOCHS}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": best_acc,
                },
                MODEL_DIR / "age_group_resnet18.pt",
            )

    print("최고 검증 정확도:", round(best_acc, 4))
    print("저장 위치:", MODEL_DIR / "age_group_resnet18.pt")


if __name__ == "__main__":
    main()
