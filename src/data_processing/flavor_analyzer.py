import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class FlavorAnalyzer:
    def __init__(self):
        self.config = Config()
        self.flavor_profiles = self.load_flavor_profiles()
        self.compound_similarity = {}
    
    def load_flavor_profiles(self) -> Dict:
        base_profiles = {
            'sweet': ['sugar', 'honey', 'maple', 'vanilla', 'fruit'],
            'savory': ['salt', 'soy', 'mushroom', 'umami', 'yeast'],
            'bitter': ['coffee', 'dark_chocolate', 'kale', 'hops'],
            'sour': ['lemon', 'vinegar', 'yogurt', 'tamarind'],
            'spicy': ['chili', 'pepper', 'ginger', 'wasabi'],
            'herbal': ['basil', 'mint', 'rosemary', 'thyme'],
            'earthy': ['beet', 'potato', 'truffle', 'mushroom']
        }
        return base_profiles
    
    def analyze_ingredient_flavor(self, ingredient: str) -> Dict[str, float]:
        ingredient_lower = ingredient.lower()
        flavor_scores = {flavor: 0.0 for flavor in self.flavor_profiles.keys()}
        
        for flavor, keywords in self.flavor_profiles.items():
            for keyword in keywords:
                if keyword in ingredient_lower:
                    flavor_scores[flavor] += 0.3
        
        total = sum(flavor_scores.values())
        if total > 0:
            for flavor in flavor_scores:
                flavor_scores[flavor] /= total
        
        return flavor_scores
    
    def compute_flavor_compatibility(self, ingredient1: str, ingredient2: str) -> float:
        flavors1 = self.analyze_ingredient_flavor(ingredient1)
        flavors2 = self.analyze_ingredient_flavor(ingredient2)
        
        vector1 = np.array(list(flavors1.values()))
        vector2 = np.array(list(flavors2.values()))
        
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return float(similarity)
    
    def get_flavor_pairing_recommendations(self, base_ingredient: str, available_ingredients: List[str]) -> List[Tuple[str, float]]:
        pairings = []
        for ingredient in available_ingredients:
            if ingredient != base_ingredient:
                compatibility = self.compute_flavor_compatibility(base_ingredient, ingredient)
                pairings.append((ingredient, compatibility))
        
        pairings.sort(key=lambda x: x[1], reverse=True)
        return pairings[:10]