import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class RecipeGenerator:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_name = self.config.get('generative_models.recipe_generator.model_name')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
    
    def generate_recipe(self, ingredients: List[str], cuisine: str = "", dish_type: str = "") -> str:
        prompt = self._build_prompt(ingredients, cuisine, dish_type)
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=self.config.get('generative_models.recipe_generator.max_length'),
                temperature=self.config.get('generative_models.recipe_generator.temperature'),
                top_p=self.config.get('generative_models.recipe_generator.top_p'),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._post_process_generation(generated_text, prompt)
    
    def _build_prompt(self, ingredients: List[str], cuisine: str, dish_type: str) -> str:
        prompt = "Generate a recipe"
        
        if cuisine:
            prompt += f" in the {cuisine} cuisine style"
        if dish_type:
            prompt += f" for a {dish_type}"
        
        prompt += f" using these ingredients: {', '.join(ingredients)}.\n\nRecipe:"
        return prompt
    
    def _post_process_generation(self, generated_text: str, original_prompt: str) -> str:
        recipe_text = generated_text[len(original_prompt):].strip()
        
        if 'Ingredients:' not in recipe_text and 'Instructions:' not in recipe_text:
            recipe_text = self._format_unstructured_recipe(recipe_text)
        
        return recipe_text
    
    def _format_unstructured_recipe(self, text: str) -> str:
        lines = text.split('\n')
        formatted = "Ingredients:\n"
        in_instructions = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(keyword in line.lower() for keyword in ['instruction', 'method', 'step']):
                if not in_instructions:
                    formatted += "\nInstructions:\n"
                    in_instructions = True
                formatted += line + "\n"
            elif not in_instructions and ('cup' in line.lower() or 'tbsp' in line.lower() or 'gram' in line.lower()):
                formatted += "- " + line + "\n"
            elif in_instructions:
                if line[0].isdigit() and '.' in line:
                    formatted += line + "\n"
                else:
                    formatted += "- " + line + "\n"
        
        return formatted