import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

from nbb import UpsampleConcatClassifier

if __name__ == "__main__":
    # ----------------------------
    # 데이터셋 로드 및 전처리
    # ----------------------------
    train_ds = load_dataset("food101", split="train").select(range(40000))
    val_ds = load_dataset("food101", split="validation").select(range(10000))


    def transform_example(example):
        from PIL import Image
        try:
            image = example["image"].convert("RGB")
        except Exception:
            return None

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        example["pixel_values"] = transform(image)
        return example


    train_ds = train_ds.map(transform_example, num_proc=4)
    val_ds = val_ds.map(transform_example, num_proc=4)
    train_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'label'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # ----------------------------
    # 모델/옵티마이저/로스 설정
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UpsampleConcatClassifier(num_classes=101, in_channels=3, concat_stride=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # ----------------------------
    # 학습 루프
    # ----------------------------
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Acc": f"{(total_correct / total_samples):.4f}"
            })

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy


    def eval_epoch(model, dataloader, criterion, device):
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        start_time = time.perf_counter()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in pbar:
                inputs = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)

                pbar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Acc": f"{(total_correct / total_samples):.4f}"
                })

        end_time = time.perf_counter()
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        inference_time = end_time - start_time
        return avg_loss, accuracy, inference_time


    # ----------------------------
    # 전체 학습 실행
    # ----------------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_time = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Inference Time: {val_time:.2f}s")
