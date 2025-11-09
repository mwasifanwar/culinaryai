import numpy as np
from typing import Dict, List, Tuple
from ..data_processing.molecular_processor import MolecularProcessor

class ChemistryAnalyzer:
    def __init__(self):
        self.config = Config()
        self.molecular_processor = MolecularProcessor()
        self.flavor_compounds = self.molecular_processor.get_common_flavor_compounds()
    
    def analyze_recipe_chemistry(self, ingredients: List[str]) -> Dict[str, any]:
        molecular_interactions = []
        total_compatibility = 0.0
        pairs_analyzed = 0
        
        for i in range(len(ingredients)):
            for j in range(i + 1, len(ingredients)):
                ing1 = ingredients[i].lower()
                ing2 = ingredients[j].lower()
                
                compound1 = self._get_compound_for_ingredient(ing1)
                compound2 = self._get_compound_for_ingredient(ing2)
                
                if compound1 and compound2:
                    interaction = self.molecular_processor.analyze_compound_interactions(compound1, compound2)
                    molecular_interactions.append({
                        'ingredient1': ing1,
                        'ingredient2': ing2,
                        'interaction': interaction
                    })
                    total_compatibility += interaction['compatibility_score']
                    pairs_analyzed += 1
        
        avg_compatibility = total_compatibility / pairs_analyzed if pairs_analyzed > 0 else 0.0
        
        return {
            'molecular_interactions': molecular_interactions,
            'average_compatibility': avg_compatibility,
            'chemistry_score': self._calculate_overall_chemistry_score(avg_compatibility, len(ingredients))
        }
    
    def _get_compound_for_ingredient(self, ingredient: str) -> str:
        for key, compound in self.flavor_compounds.items():
            if key in ingredient:
                return compound
        return list(self.flavor_compounds.values())[hash(ingredient) % len(self.flavor_compounds)]
    
    def _calculate_overall_chemistry_score(self, compatibility: float, num_ingredients: int) -> float:
        base_score = compatibility * 0.7
        complexity_bonus = min(num_ingredients * 0.05, 0.3)
        return min(base_score + complexity_bonus, 1.0)
    
    def predict_chemical_reactions(self, ingredients: List[str], cooking_method: str) -> List[str]:
        reactions = []
        
        if 'baking' in cooking_method.lower() or 'oven' in cooking_method.lower():
            if any(ing in ['flour', 'sugar', 'baking_powder'] for ing in ingredients):
                reactions.append("Maillard reaction - browning and flavor development")
            
            if 'yeast' in ' '.join(ingredients).lower():
                reactions.append("Fermentation - carbon dioxide production for rising")
        
        if any(ing in ['lemon', 'vinegar'] for ing in ingredients) and any(ing in ['baking_soda'] for ing in ingredients):
            reactions.append("Acid-base reaction - leavening through gas production")
        
        if 'grilling' in cooking_method.lower() or 'searing' in cooking_method.lower():
            reactions.append("Caramelization - sugar breakdown creating complex flavors")
        
        return reactions[:3]