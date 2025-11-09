from typing import List, Dict, Tuple
from ..data_processing.flavor_analyzer import FlavorAnalyzer

class PairingEngine:
    def __init__(self):
        self.config = Config()
        self.flavor_analyzer = FlavorAnalyzer()
        self.pairing_rules = self._load_pairing_rules()
    
    def _load_pairing_rules(self) -> Dict[str, List[str]]:
        return {
            'tomato': ['basil', 'mozzarella', 'garlic', 'onion', 'oregano'],
            'chicken': ['thyme', 'rosemary', 'lemon', 'garlic', 'ginger'],
            'beef': ['thyme', 'red_wine', 'mushroom', 'garlic', 'pepper'],
            'fish': ['lemon', 'dill', 'parsley', 'capers', 'white_wine'],
            'chocolate': ['vanilla', 'orange', 'coffee', 'caramel', 'raspberry'],
            'strawberry': ['cream', 'chocolate', 'mint', 'lemon', 'basil'],
            'pork': ['apple', 'sage', 'fennel', 'mustard', 'garlic'],
            'lamb': ['mint', 'rosemary', 'garlic', 'thyme', 'red_wine']
        }
    
    def get_optimal_pairings(self, main_ingredient: str, available_ingredients: List[str]) -> List[Tuple[str, float]]:
        main_ingredient = main_ingredient.lower()
        
        rule_based_pairings = []
        if main_ingredient in self.pairing_rules:
            for pairing in self.pairing_rules[main_ingredient]:
                if pairing in available_ingredients:
                    rule_based_pairings.append((pairing, 1.0))
        
        flavor_based_pairings = self.flavor_analyzer.get_flavor_pairing_recommendations(
            main_ingredient, available_ingredients
        )
        
        combined_pairings = {}
        for ingredient, score in rule_based_pairings:
            combined_pairings[ingredient] = score * 0.6
        
        for ingredient, score in flavor_based_pairings:
            if ingredient in combined_pairings:
                combined_pairings[ingredient] += score * 0.4
            else:
                combined_pairings[ingredient] = score * 0.4
        
        sorted_pairings = sorted(combined_pairings.items(), key=lambda x: x[1], reverse=True)
        return [(ing, score) for ing, score in sorted_pairings if score > self.config.get('food_science.pairing_threshold')]
    
    def generate_flavor_combinations(self, base_ingredients: List[str], max_combinations: int = 5) -> List[List[str]]:
        if not base_ingredients:
            return []
        
        combinations = [base_ingredients.copy()]
        
        for ingredient in base_ingredients:
            pairings = self.get_optimal_pairings(ingredient, [])
            
            for pairing, score in pairings[:3]:
                new_combination = base_ingredients + [pairing]
                if new_combination not in combinations:
                    combinations.append(new_combination)
                
                if len(combinations) >= max_combinations:
                    break
            
            if len(combinations) >= max_combinations:
                break
        
        return combinations[:max_combinations]
    
    def analyze_cuisine_patterns(self, cuisine: str) -> List[Tuple[str, float]]:
        cuisine_patterns = {
            'italian': [('tomato', 0.9), ('basil', 0.8), ('garlic', 0.9), ('olive_oil', 0.9), ('pasta', 0.8)],
            'mexican': [('chili', 0.9), ('lime', 0.8), ('cilantro', 0.8), ('avocado', 0.7), ('corn', 0.7)],
            'indian': [('turmeric', 0.9), ('cumin', 0.8), ('coriander', 0.8), ('ginger', 0.8), ('garlic', 0.8)],
            'chinese': [('soy_sauce', 0.9), ('ginger', 0.8), ('garlic', 0.8), ('green_onion', 0.7), ('sesame_oil', 0.7)],
            'french': [('butter', 0.8), ('wine', 0.7), ('thyme', 0.7), ('shallot', 0.6), ('cream', 0.6)]
        }
        
        return cuisine_patterns.get(cuisine.lower(), [])