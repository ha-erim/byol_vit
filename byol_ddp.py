import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets
import timm
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
IMAGE_SIZE = 224
BATCH_SIZE = 64
PROJECTION_DIM = 256
PREDICTION_DIM = 4096
LEARNING_RATE = 0.05
EPOCHS = 12
EVAL_EPOCHS = 30

# =========================
# DATA AUGMENTATIONS
# =========================
class BYOLDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform):
        self.dataset = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        x1, x2 = self.transform(img)
        return x1, x2
    
class BYOLTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# BACKBONE + PROJECTOR + PREDICTOR
# =========================
from transformers import ViTModel, ViTConfig
class ViTBackbone(nn.Module):
    def __init__(self, model_name = "facebook/deit-tiny-patch16-224"):
        super(ViTBackbone, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_dim = self.vit.config.hidden_size

    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:,0]
        return cls_token

    def get_feature_dim(self):
        return self.feature_dim
    
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, output_dim=256):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# BYOL MAIN MODULE
# =========================
import copy

class BYOL(nn.Module):
    def __init__(self, backbone, feature_dim, hidden_dim=4096, proj_dim=256, momentum=0.996):
        super(BYOL, self).__init__()

        # online encoder
        self.online_encoder = backbone
        self.online_projector = MLPHead(feature_dim, hidden_dim, proj_dim)
        self.online_predictor = MLPHead(proj_dim, hidden_dim // 2, proj_dim)

        # target encoder
        self.target_encoder = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(self.online_projector)

        # EMA decay hyperparameter
        self.momentum = momentum

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def _update_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data

        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data

    def forward(self, x1, x2):
        # Online network forward
        z1_online = self.online_projector(self.online_encoder(x1))  # projection
        p1 = self.online_predictor(z1_online)  # prediction

        z2_online = self.online_projector(self.online_encoder(x2))
        p2 = self.online_predictor(z2_online)

        with torch.no_grad():
            self._update_target()
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))

        # cosine similarity loss
        loss = (self._loss_fn(p1, z2_target.detach()) + self._loss_fn(p2, z1_target.detach())) * 0.5
        return loss
    
    def _loss_fn(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return - (p * z).sum(dim=1).mean()


# =========================
# DDP TRAINING LOOP
# =========================
def train_byol_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dataset = datasets.ImageFolder("/home/hrkim/hw/tiny-imagenet-200/train", transform=None)
    sim_dataset = BYOLDataset(dataset, BYOLTransform(train_transform))
    sampler = DistributedSampler(sim_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(sim_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    backbone = ViTBackbone()
    feature_dim = backbone.get_feature_dim()
    model = BYOL(backbone, feature_dim).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    loss_history = []

    model.train()
    for epoch in range(EPOCHS):
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
        for _, (x1, x2) in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
            loss_history.append(avg_loss)
            torch.save(model.module.state_dict(), "byol_vit_checkpoint.pth")

    if rank == 0:
        plt.figure()
        plt.plot(range(1, EPOCHS + 1), loss_history, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('BYOL Training Loss')
        plt.grid(True)
        plt.savefig('byol_training_loss.png')
        plt.close()

    dist.destroy_process_group()
    
# =========================
# LINEAR PROBE MODULE
# =========================
class LinearProbe(nn.Module):
    def __init__(self, backbone, feature_dim=768, num_classes=200):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.fc(feat)

# =========================
# LINEAR PROBE DDP EVALUATION
# =========================
def evaluate_byol_ddp():
    val_path = "/home/hrkim/hw/tiny-imagenet-200/val/structured"
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    train_dataset = datasets.ImageFolder("/home/hrkim/hw/tiny-imagenet-200/train", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,  num_workers=4, pin_memory=True)

    backbone = ViTBackbone()
    feature_dim = backbone.get_feature_dim()
    model = BYOL(backbone, feature_dim)
    ckpt = torch.load("byol_vit_checkpoint.pth")
    model.load_state_dict(ckpt)
    encoder = model.online_encoder
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False
    encoder = encoder.cuda()

    classifier = nn.Linear(feature_dim, 200)
    classifier.cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    acc_history = []

    for epoch in range(EVAL_EPOCHS):
        model.train()
        classifier.train()
        total_train, correct_train, total_loss_train = 0, 0, 0.0

        train_pbar = tqdm(train_loader, desc=f"[Linear Probe Train] Epoch {epoch}")
        for x, labels in train_pbar:
            x = x.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                feats = encoder(x)

            outputs = classifier(feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            total_loss_train += loss.item()
            train_pbar.set_postfix(acc=100 * correct_train / total_train, loss=loss.item())

        # Evaluate on validation set
        classifier.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[Linear Probe Val] Epoch {epoch}")
            for x, labels in val_pbar:
                x = x.cuda()
                labels = labels.cuda()
                feats = encoder(x)
                outputs = classifier(feats)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val
        print(f"Epoch {epoch} — Val Accuracy: {val_acc:.2f}%")
        acc_history.append(val_acc)

    plt.figure()
    plt.plot(range(1, EVAL_EPOCHS + 1), acc_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Val Accuracy (%)')
    plt.title('Linear Probe Validation Accuracy Curve')
    plt.grid(True)
    plt.savefig('linear_probe_accuracy_curve_ddp.png')
    plt.close()

# =========================
# ENTRY POINT FOR DDP
# =========================
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_byol_ddp, args=(world_size,), nprocs=world_size, join=True)
    # evaluate_byol_ddp()
