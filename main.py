import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generative_models.recipe_generator import RecipeGenerator
from src.generative_models.image_to_recipe import ImageToRecipeModel
from src.food_science.chemistry_analyzer import ChemistryAnalyzer
from src.food_science.nutrition_calculator import NutritionCalculator
from src.api.server import app
import uvicorn

def generate_sample_recipe():
    print("Generating sample recipe with CulinaryAI...")
    
    ingredients = ["chicken", "tomato", "onion", "garlic", "basil"]
    generator = RecipeGenerator()
    recipe = generator.generate_recipe(ingredients, "italian", "main course")
    
    print("Generated Recipe:")
    print(recipe)
    
    chemistry_analyzer = ChemistryAnalyzer()
    chemistry = chemistry_analyzer.analyze_recipe_chemistry(ingredients)
    print(f"\nChemistry Score: {chemistry['chemistry_score']:.2f}")
    
    nutrition_calculator = NutritionCalculator()
    nutrition = nutrition_calculator.estimate_recipe_nutrition(ingredients)
    print(f"Nutrition per serving: {nutrition}")
    
    return recipe

def run_api():
    from src.utils.config import Config
    config = Config()
    print(f"Starting CulinaryAI API server on {config.get('api.host')}:{config.get('api.port')}")
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))

def main():
    parser = argparse.ArgumentParser(description='CulinaryAI: Recipe Generation & Food Science')
    parser.add_argument('--mode', choices=['api', 'generate', 'test'], default='api', help='Operation mode')
    parser.add_argument('--ingredients', nargs='+', help='List of ingredients for recipe generation')
    parser.add_argument('--cuisine', type=str, help='Cuisine style for recipe generation')
    parser.add_argument('--image', type=str, help='Image path for image-to-recipe generation')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        run_api()
    elif args.mode == 'generate':
        if args.ingredients:
            generator = RecipeGenerator()
            recipe = generator.generate_recipe(args.ingredients, args.cuisine or "")
            print(recipe)
        elif args.image:
            image_model = ImageToRecipeModel()
            result = image_model.generate_recipe_from_image(args.image, args.cuisine or "")
            print(result)
        else:
            generate_sample_recipe()
    else:
        print("CulinaryAI system ready - mwasifanwar")

if __name__ == "__main__":
    main()