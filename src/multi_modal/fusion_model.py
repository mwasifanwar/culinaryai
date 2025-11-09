import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class MultiModalFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, flavor_dim: int, fusion_dim: int):
        super(MultiModalFusion, self).__init__()
        
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.flavor_projection = nn.Linear(flavor_dim, fusion_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )
        
        self.output_layer = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor, flavor_features: torch.Tensor) -> torch.Tensor:
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        flavor_proj = self.flavor_projection(flavor_features)
        
        combined = torch.cat([text_proj, image_proj, flavor_proj], dim=1)
        fused = self.fusion_layer(combined)
        output = self.sigmoid(self.output_layer(fused))
        
        return output

class MultiModalModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        text_dim = 768
        image_dim = 2048
        flavor_dim = 7
        fusion_dim = self.config.get('multi_modal.fusion_dim')
        
        self.model = MultiModalFusion(text_dim, image_dim, flavor_dim, fusion_dim).to(self.device)
    
    def predict_recipe_quality(self, text_features: np.ndarray, image_features: np.ndarray, flavor_features: np.ndarray) -> float:
        text_tensor = torch.FloatTensor(text_features).unsqueeze(0).to(self.device)
        image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)
        flavor_tensor = torch.FloatTensor(flavor_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            quality_score = self.model(text_tensor, image_tensor, flavor_tensor)
        
        return float(quality_score.cpu().numpy()[0])
    
    def fuse_modalities_for_recipe_generation(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        text_features = modalities.get('text', np.zeros(768))
        image_features = modalities.get('image', np.zeros(2048))
        flavor_features = modalities.get('flavor', np.zeros(7))
        
        text_tensor = torch.FloatTensor(text_features).unsqueeze(0).to(self.device)
        image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)
        flavor_tensor = torch.FloatTensor(flavor_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            text_proj = self.model.text_projection(text_tensor)
            image_proj = self.model.image_projection(image_tensor)
            flavor_proj = self.model.flavor_projection(flavor_tensor)
        
        fused = torch.cat([text_proj, image_proj, flavor_proj], dim=1)
        return fused.cpu().numpy()[0]