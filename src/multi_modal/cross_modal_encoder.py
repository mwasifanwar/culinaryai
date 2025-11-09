import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class CrossModalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(CrossModalEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class CrossModalSystem:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.text_encoder = CrossModalEncoder(768, 512, 256).to(self.device)
        self.image_encoder = CrossModalEncoder(2048, 512, 256).to(self.device)
        self.flavor_encoder = CrossModalEncoder(7, 128, 256).to(self.device)
    
    def encode_text(self, text_features: np.ndarray) -> np.ndarray:
        tensor = torch.FloatTensor(text_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.text_encoder(tensor)
        return encoded.cpu().numpy()[0]
    
    def encode_image(self, image_features: np.ndarray) -> np.ndarray:
        tensor = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.image_encoder(tensor)
        return encoded.cpu().numpy()[0]
    
    def encode_flavor(self, flavor_features: np.ndarray) -> np.ndarray:
        tensor = torch.FloatTensor(flavor_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.flavor_encoder(tensor)
        return encoded.cpu().numpy()[0]
    
    def compute_cross_modal_similarity(self, modality1: str, features1: np.ndarray, modality2: str, features2: np.ndarray) -> float:
        if modality1 == 'text':
            encoded1 = self.encode_text(features1)
        elif modality1 == 'image':
            encoded1 = self.encode_image(features1)
        else:
            encoded1 = self.encode_flavor(features1)
        
        if modality2 == 'text':
            encoded2 = self.encode_text(features2)
        elif modality2 == 'image':
            encoded2 = self.encode_image(features2)
        else:
            encoded2 = self.encode_flavor(features2)
        
        similarity = np.dot(encoded1, encoded2) / (np.linalg.norm(encoded1) * np.linalg.norm(encoded2))
        return float(similarity)