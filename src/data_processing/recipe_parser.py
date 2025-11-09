import json
import re
from typing import Dict, List, Any

class RecipeParser:
    def __init__(self):
        self.config = Config()
    
    def parse_recipe_text(self, recipe_text: str) -> Dict[str, Any]:
        lines = recipe_text.strip().split('\n')
        parsed = {'title': '', 'ingredients': [], 'instructions': []}
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'title' in line.lower() or not parsed['title']:
                if ':' in line:
                    parsed['title'] = line.split(':', 1)[1].strip()
                else:
                    parsed['title'] = line
            elif any(keyword in line.lower() for keyword in ['ingredient', 'materials']):
                current_section = 'ingredients'
            elif any(keyword in line.lower() for keyword in ['instruction', 'method', 'directions']):
                current_section = 'instructions'
            elif line.startswith('-') or line.startswith('*'):
                content = line[1:].strip()
                if current_section == 'ingredients':
                    parsed['ingredients'].append(content)
                elif current_section == 'instructions':
                    parsed['instructions'].append(content)
            elif re.match(r'^\d+\.', line):
                content = re.sub(r'^\d+\.', '', line).strip()
                parsed['instructions'].append(content)
        
        return parsed
    
    def extract_ingredients(self, recipe_dict: Dict) -> List[str]:
        return recipe_dict.get('ingredients', [])
    
    def extract_cooking_steps(self, recipe_dict: Dict) -> List[str]:
        return recipe_dict.get('instructions', [])
    
    def validate_recipe(self, recipe_dict: Dict) -> bool:
        required_fields = ['title', 'ingredients', 'instructions']
        return all(field in recipe_dict and recipe_dict[field] for field in required_fields)