# BYOL with Vision Transformer (ViT)

이 레포지토리는 **Bootstrap Your Own Latent (BYOL)** 프레임워크에서 기존 ResNet 백본을 **Vision Transformer (ViT)** 로 대체하여 구현한 자기지도학습 모델입니다.  
데이터셋으로는 **Tiny-ImageNet**을 사용합니다.

## 🔍 주요 특징
- BYOL 구조에서 백본을 Vision Transformer로 변경
- HuggingFace `ViTModel` 사용
- BatchNorm 제거 및 MLP 구조 수정
- Tiny-ImageNet을 기반으로 자기지도 사전학습
- 선형 프로빙을 통한 성능 평가 지원

## 📁 데이터셋 준비
- [Tiny-ImageNet](https://tiny-imagenet.herokuapp.com/) 다운로드 후 `tiny-imagenet-200` 디렉토리 구성 필요
- **`val/` 디렉토리는 클래스별 하위 폴더로 분류되어 있어야 합니다**

예시 구조:
tiny-imagenet-200/
├── train/
│ └── n01443537/
│ └── images/
├── val/
│ └── n01443537/
│ └── images/
└── wnids.txt

> `val_annotations.txt` 파일을 사용하여 validation 이미지를 클래스별 폴더로 정리해야 합니다.

## 🛠 설치
```bash
git clone https://github.com/ha-erim/byol_vit.git
cd byol_vit
pip install -r requirements.txt

## 🚀 사용법

### 1. BYOL 사전학습
```bash
python train_byol.py \
    --data_path /path/to/tiny-imagenet-200 \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 128

### 2. 선형 프로빙 (Linear Evaluation)
```bash
python linear_probe.py \
    --data_path /path/to/tiny-imagenet-200 \
    --checkpoint ./checkpoints/byol_vit.pth \
    --epochs 50 \
    --batch_size 128
