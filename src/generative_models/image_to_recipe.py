import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, List

class ImageToRecipeModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()
        self.image_model.to(self.device)
        self.image_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.config.get('multi_modal.image_size'))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.ingredient_classes = [
            'tomato', 'onion', 'garlic', 'chicken', 'beef', 'fish', 'rice', 'pasta',
            'potato', 'carrot', 'broccoli', 'cheese', 'egg', 'milk', 'flour', 'sugar'
        ]
    
    def extract_image_features(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.image_model(image_tensor)
        
        return features.cpu().numpy()[0]
    
    def predict_ingredients_from_image(self, image_path: str) -> List[str]:
        features = self.extract_image_features(image_path)
        
        simulated_predictions = np.random.rand(len(self.ingredient_classes))
        threshold = 0.5
        
        predicted_ingredients = []
        for i, prob in enumerate(simulated_predictions):
            if prob > threshold:
                predicted_ingredients.append(self.ingredient_classes[i])
        
        if not predicted_ingredients:
            predicted_ingredients = ['tomato', 'onion', 'garlic']
        
        return predicted_ingredients[:5]
    
    def generate_recipe_from_image(self, image_path: str, cuisine: str = "") -> Dict[str, any]:
        ingredients = self.predict_ingredients_from_image(image_path)
        
        from .recipe_generator import RecipeGenerator
        generator = RecipeGenerator()
        recipe_text = generator.generate_recipe(ingredients, cuisine)
        
        return {
            'predicted_ingredients': ingredients,
            'generated_recipe': recipe_text,
            'image_features': self.extract_image_features(image_path).tolist()
        }