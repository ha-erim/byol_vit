import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import ViTModel, ViTConfig

class ViTBackbone(nn.Module):

    def __init__(self, model_name = "facebook/deit-tiny-patch16-224"):
        super(ViTBackbone, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_dim = self.vit.config.hidden_size

    def forward(self, x):
        # input : (B, C, H, W) image
        # output : (B, T, D) sequence
        outputs = self.vit(x)
        # using cls_token as representative embedding
        cls_token = outputs.last_hidden_state[:,0]
        return cls_token

    def get_feature_dim(self):
        return self.feature_dim

# MLP for prediction and projection head
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

class BYOL(nn.Module):
    def __init__(self, backbone, feature_dim, hidden_dim=4096, proj_dim=256, momentum=0.996):
        super(BYOL, self).__init__()
        
        # Online network
        self.online_encoder = backbone
        self.online_projector = MLPHead(feature_dim, hidden_dim, proj_dim)
        self.online_predictor = MLPHead(proj_dim, hidden_dim // 2, proj_dim)

        # Target network (copy online network's initial value)
        self.target_encoder = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(self.online_projector)

        self.momentum = momentum
        
        # set target network is disabled from training
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target(self):
        # update target network using EMA
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data

        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data

    def forward(self, x1, x2):
        z1_online = self.online_projector(self.online_encoder(x1))
        p1 = self.online_predictor(z1_online)

        z2_online = self.online_projector(self.online_encoder(x2))
        p2 = self.online_predictor(z2_online)

        with torch.no_grad():
            self._update_target()
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))

        # BYOL loss : cross-view prediction
        # detach로 학습 안되게 설정
        loss = (self.negative_cosine_similarity_loss(p1, z2_target.detach()) + self.negative_cosine_similarity_loss(p2, z1_target.detach())) * 0.5
        return loss
    
    def negative_cosine_similarity_loss(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return - (p * z).sum(dim=1).mean()
