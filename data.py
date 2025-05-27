import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class ContrastiveTransform:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

class ContrastiveTinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        assert split in ['train', 'val']
        self.dataset = datasets.ImageFolder(os.path.join(root, split))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 동일한 이미지에 augmentation을 각각 적용하여 2개의 이미지 생성해서 사용
        img, _ = self.dataset[idx]
        view1, view2 = self.transform(img)
        return view1, view2
    
def get_dataloader(data_root="/home/hrkim/hw/tiny-imagenet-200", batch_size=128, num_workers=0, persistent_workers=False, prefetch_factor=2
, image_size=224, sampler=None):
    #이미지에 augmentation 적용
    transform = ContrastiveTransform(image_size=image_size)
    dataset = ContrastiveTinyImageNet(data_root, split='train', transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    return loader