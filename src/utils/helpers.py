import numpy as np
import re

def normalize_ingredient_name(ingredient):
    ingredient = ingredient.lower().strip()
    ingredient = re.sub(r'[^a-zA-Z0-9\s]', '', ingredient)
    return ingredient

def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def format_recipe_output(recipe_dict):
    output = f"Recipe: {recipe_dict.get('title', 'Unknown')}\n\n"
    output += "Ingredients:\n"
    for ingredient in recipe_dict.get('ingredients', []):
        output += f"- {ingredient}\n"
    output += "\nInstructions:\n"
    for i, instruction in enumerate(recipe_dict.get('instructions', []), 1):
        output += f"{i}. {instruction}\n"
    return output