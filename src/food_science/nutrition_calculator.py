from typing import Dict, List, Tuple

class NutritionCalculator:
    def __init__(self):
        self.config = Config()
        self.ingredient_nutrition = self._load_nutrition_database()
    
    def _load_nutrition_database(self) -> Dict[str, Dict]:
        return {
            'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
            'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15},
            'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 13},
            'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
            'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1},
            'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2},
            'onion': {'calories': 40, 'protein': 1.1, 'carbs': 9.3, 'fat': 0.1},
            'garlic': {'calories': 149, 'protein': 6.4, 'carbs': 33, 'fat': 0.5},
            'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1},
            'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2},
            'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4},
            'cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33},
            'egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11},
            'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1},
            'flour': {'calories': 364, 'protein': 10, 'carbs': 76, 'fat': 1},
            'sugar': {'calories': 387, 'protein': 0, 'carbs': 100, 'fat': 0}
        }
    
    def estimate_recipe_nutrition(self, ingredients: List[str], servings: int = 4) -> Dict[str, float]:
        total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            matched = False
            
            for key, nutrition in self.ingredient_nutrition.items():
                if key in ingredient_lower:
                    for nutrient in total_nutrition:
                        total_nutrition[nutrient] += nutrition.get(nutrient, 0)
                    matched = True
                    break
            
            if not matched:
                base_calories = self.config.get('food_science.nutrition_calories_per_gram') * 100
                total_nutrition['calories'] += base_calories
        
        for nutrient in total_nutrition:
            total_nutrition[nutrient] = round(total_nutrition[nutrient] / servings, 1)
        
        return total_nutrition
    
    def calculate_nutrition_score(self, nutrition: Dict[str, float]) -> float:
        ideal_ratios = {'protein': 0.3, 'carbs': 0.5, 'fat': 0.2}
        total_calories = nutrition['calories']
        
        if total_calories == 0:
            return 0.0
        
        actual_ratios = {}
        for nutrient in ['protein', 'carbs', 'fat']:
            nutrient_calories = nutrition[nutrient] * (4 if nutrient == 'protein' or nutrient == 'carbs' else 9)
            actual_ratios[nutrient] = nutrient_calories / total_calories
        
        score = 0.0
        for nutrient in ideal_ratios:
            difference = abs(actual_ratios.get(nutrient, 0) - ideal_ratios[nutrient])
            score += max(0, 1 - difference * 3)
        
        return round(score / len(ideal_ratios), 2)
    
    def suggest_healthier_alternatives(self, ingredients: List[str]) -> List[Tuple[str, str]]:
        alternatives = {
            'sugar': 'honey or maple syrup',
            'butter': 'olive oil or avocado',
            'white_flour': 'whole wheat flour',
            'cream': 'greek yogurt',
            'beef': 'chicken or turkey',
            'potato': 'sweet potato',
            'white_rice': 'brown rice or quinoa',
            'pasta': 'zucchini noodles or whole wheat pasta'
        }
        
        suggestions = []
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            for unhealthy, healthy in alternatives.items():
                if unhealthy in ingredient_lower:
                    suggestions.append((ingredient, healthy))
                    break
        
        return suggestions