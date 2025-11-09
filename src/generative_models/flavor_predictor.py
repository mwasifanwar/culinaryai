import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

class FlavorPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(FlavorPredictor, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class FlavorModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = 256
        hidden_dim = self.config.get('generative_models.flavor_predictor.hidden_dim')
        output_dim = 7
        num_layers = self.config.get('generative_models.flavor_predictor.num_layers')
        
        self.model = FlavorPredictor(input_dim, hidden_dim, output_dim, num_layers).to(self.device)
        self.flavor_dimensions = ['sweet', 'savory', 'bitter', 'sour', 'spicy', 'herbal', 'earthy']
    
    def predict_flavor_profile(self, ingredient_embeddings: List[np.ndarray]) -> Dict[str, float]:
        if not ingredient_embeddings:
            return {flavor: 0.0 for flavor in self.flavor_dimensions}
        
        avg_embedding = np.mean(ingredient_embeddings, axis=0)
        input_tensor = torch.FloatTensor(avg_embedding).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        flavor_scores = predictions.cpu().numpy()[0]
        
        return {self.flavor_dimensions[i]: float(flavor_scores[i]) for i in range(len(self.flavor_dimensions))}
    
    def recommend_ingredient_adjustments(self, current_flavors: Dict[str, float], target_flavors: Dict[str, float]) -> List[Tuple[str, float]]:
        adjustments = []
        
        for flavor in self.flavor_dimensions:
            current = current_flavors.get(flavor, 0.0)
            target = target_flavors.get(flavor, 0.0)
            difference = target - current
            
            if abs(difference) > 0.1:
                direction = "increase" if difference > 0 else "decrease"
                adjustments.append((flavor, difference, direction))
        
        adjustments.sort(key=lambda x: abs(x[1]), reverse=True)
        return adjustments