import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm
import csv

from byol import ViTBackbone, BYOL
from data import ContrastiveTinyImageNet, ContrastiveTransform


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return int(os.environ["RANK"]) == 0


def train_model(model, dataloader, method="byol", num_epochs=10,
                lr=0.05, device="cuda", save_path="checkpoints", log_interval=10, sampler=None):

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = open(f"logs/loss_{method}.csv", mode="w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "avg_loss"])

    for epoch in range(num_epochs):
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[{method.upper()}] Epoch {epoch}")
        for i, (x1, x2) in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % log_interval == 0:
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")
        log_writer.writerow([epoch, avg_loss])

        #체크포인트 저장
        torch.save(model.state_dict(), os.path.join(save_path, f"{method}_epoch{epoch}.pth"))

    log_file.close()


def linear_probe(weight_path, method, data_root, batch_size=128, num_epochs=30, device="cuda"):
    #validation augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val/structured"), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    backbone = ViTBackbone().to(device)
    feature_dim = backbone.get_feature_dim()

    model = BYOL(backbone, feature_dim).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    encoder = model.online_encoder 

    # freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    num_classes = 200
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/acc_{method}.csv"
    with open(log_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "loss"])

        # training classifier on training set
        for epoch in range(num_epochs):
            classifier.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    features = encoder(images)

                logits = classifier(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            
            # evaluation on validation set
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    features = encoder(images)
                    outputs = classifier(features)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            acc = 100 * correct / total
            print(f"[Epoch {epoch}] Accuracy: {acc:.2f}% | Loss: {avg_loss:.4f}")
            writer.writerow([epoch, acc, avg_loss])
    
def main():
    parser = argparse.ArgumentParser(description="DDP ViT-Contrastive Learning")
    parser.add_argument("--data_root", type=str, default="/home/hrkim/hw/tiny-imagenet-200")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--mode", default="train")
    args = parser.parse_args()
    parser.add_argument("--weight", type=str, default="/home/hrkim/hw/checkpoints/byol_epoch19.pth")
    args = parser.parse_args()

    if args.mode == "test":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Evaluating")
        linear_probe(
            weight_path=args.weight,
            method="byol",
            data_root=args.data_root,
            batch_size=128,
            num_epochs=30,
            device=device
        )
        exit
    
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    transform = ContrastiveTransform(image_size=224)
    dataset = ContrastiveTinyImageNet(args.data_root, split='train', transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                            pin_memory=True, sampler=sampler, drop_last=True)


    backbone = ViTBackbone().to(device)
    feature_dim = backbone.get_feature_dim()
    model = BYOL(backbone, feature_dim).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    train_model(model, dataloader, method="byol", num_epochs=args.epochs,
                lr=args.lr, device=device, save_path="checkpoints")

    cleanup_ddp()


if __name__ == "__main__":
    main()