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
- Tiny-ImageNet 다운로드 후 `tiny-imagenet-200` 디렉토리 구성 필요
  
  ```bash
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  ```
- **`val/` 디렉토리는 클래스별 하위 폴더로 분류되어 있어야 합니다**
> `val_annotations.txt` 파일을 사용하여 validation 이미지를 클래스별 폴더로 정리해야 합니다.

## 🛠 설치
```bash
git clone https://github.com/ha-erim/byol_vit.git
cd byol_vit
pip install -r requirements.txt
```

## 🚀 사용법

### 1. BYOL 사전학습
```bash
torchrun --nproc_per_node=2 main.py \
    --data_root /path/to/tiny-imagenet-200 \
    --batch_size 128 \
    --epochs 30 \
    --lr 0.05 \
    --mode train
```

### 2. 선형 프로빙 (Linear Evaluation)
```bash
python main.py \
    --mode test \
    --data_root /path/to/tiny-imagenet-200 \
    --weight /path/to/checkpoints/byol_epoch19.pth
```
