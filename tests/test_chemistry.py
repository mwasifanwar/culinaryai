import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.food_science.chemistry_analyzer import ChemistryAnalyzer
from src.food_science.nutrition_calculator import NutritionCalculator

class TestFoodScience(unittest.TestCase):
    def test_chemistry_analyzer(self):
        analyzer = ChemistryAnalyzer()
        ingredients = ["tomato", "basil", "garlic"]
        analysis = analyzer.analyze_recipe_chemistry(ingredients)
        self.assertIn('chemistry_score', analysis)
        self.assertIn('molecular_interactions', analysis)
    
    def test_nutrition_calculator(self):
        calculator = NutritionCalculator()
        ingredients = ["chicken", "rice", "broccoli"]
        nutrition = calculator.estimate_recipe_nutrition(ingredients)
        self.assertIn('calories', nutrition)
        self.assertIn('protein', nutrition)
        self.assertIn('carbs', nutrition)
        self.assertIn('fat', nutrition)

if __name__ == '__main__':
    unittest.main()