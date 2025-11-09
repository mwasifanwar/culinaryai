import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generative_models.recipe_generator import RecipeGenerator
from src.data_processing.recipe_parser import RecipeParser

class TestRecipeGenerator(unittest.TestCase):
    def test_generator_creation(self):
        generator = RecipeGenerator()
        self.assertIsNotNone(generator)
    
    def test_recipe_generation(self):
        generator = RecipeGenerator()
        ingredients = ["chicken", "tomato", "onion"]
        recipe = generator.generate_recipe(ingredients)
        self.assertIsInstance(recipe, str)
        self.assertGreater(len(recipe), 0)
    
    def test_recipe_parsing(self):
        parser = RecipeParser()
        sample_recipe = """
        Title: Test Recipe
        Ingredients:
        - 2 chicken breasts
        - 1 tomato
        Instructions:
        - Cook chicken
        - Add tomato
        """
        parsed = parser.parse_recipe_text(sample_recipe)
        self.assertEqual(parsed['title'], 'Test Recipe')
        self.assertEqual(len(parsed['ingredients']), 2)

if __name__ == '__main__':
    unittest.main()