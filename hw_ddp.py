# ddp_train_simsiam.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== CONFIGURATION =====
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS_PRETRAIN = 10
LEARNING_RATE = 1e-4

# ===== DATA AUGMENTATION =====
class SimSiamTransform:
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

# ===== DATASET WRAPPER =====
class SimSiamDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_dataset, transform):
        self.dataset = image_folder_dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        x1, x2 = self.transform(img)
        return x1, x2

    def __len__(self):
        return len(self.dataset)

# ===== MODEL COMPONENTS =====
class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.backbone = self.vit.forward_features
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        feats = self.vit.forward_features(x)
        return feats[:, 0]

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, out_dim)
        # )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 추가
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)      # 추가
        )

    def forward(self, x):
        return self.net(x)

class PredictionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 추가
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)      # 추가
        )

    def forward(self, x):
        return self.net(x)

class SimSiamViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTBackbone()
        self.projector = ProjectionHead()
        self.predictor = PredictionHead()

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1, z2

# ===== LOSS =====
def negative_cosine_similarity(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return - (p * z).sum(dim=1).mean()

# ===== TRAINING =====
def train(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 포트는 충돌 없게 설정
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    dataset = datasets.ImageFolder("/home/hrkim/hw/tiny-imagenet-200/train", transform=None)
    simsiam_dataset = SimSiamDataset(dataset, SimSiamTransform(train_transform))
    sampler = DistributedSampler(simsiam_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(simsiam_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    model = SimSiamViT().to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_history = []
    model.train()
    for epoch in range(EPOCHS_PRETRAIN):
        sampler.set_epoch(epoch)
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Rank {rank})", disable=rank != 0)

        for x1, x2 in pbar:
            x1, x2 = x1.to(rank), x2.to(rank)
            p1, p2, z1, z2 = model(x1, x2)
            loss = 0.5 * (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1))
            
            print("cos(p1, z2):", F.cosine_similarity(p1, z2).mean().item())
            print("cos(p2, z1):", F.cosine_similarity(p2, z1).mean().item())
            print("std(p1):", p1.std().item(), "std(p2):", p2.std().item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")
            loss_history.append(total_loss / len(dataloader))

    if rank == 0:
        torch.save({
            'backbone': model.module.backbone.state_dict(),
            'projector': model.module.projector.state_dict()
        }, "simsiam_vit_checkpoint.pth")
        plt.figure()
        plt.plot(range(1, EPOCHS_PRETRAIN + 1), loss_history, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SimSiam Training Loss per Epoch')
        plt.grid(True)
        plt.savefig('simsiam_training_loss_ddp.png')
        plt.close()

    dist.destroy_process_group()


# ===== ENTRY POINT =====
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
