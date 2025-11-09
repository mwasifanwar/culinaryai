from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import tempfile
import os

app = FastAPI(title="CulinaryAI API", version="1.0.0")

class RecipeGenerationRequest(BaseModel):
    ingredients: List[str]
    cuisine: Optional[str] = ""
    dish_type: Optional[str] = ""
    max_length: Optional[int] = 1024

class FlavorAnalysisRequest(BaseModel):
    ingredients: List[str]
    target_flavor_profile: Optional[Dict[str, float]] = None

class ChemistryAnalysisRequest(BaseModel):
    ingredients: List[str]
    cooking_method: Optional[str] = "baking"

class NutritionRequest(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 4

class MultiModalRequest(BaseModel):
    text: Optional[str] = ""
    image_path: Optional[str] = ""
    flavor_profile: Optional[Dict[str, float]] = None

@app.post("/generate_recipe")
async def generate_recipe(request: RecipeGenerationRequest):
    try:
        from src.generative_models.recipe_generator import RecipeGenerator
        
        generator = RecipeGenerator()
        recipe = generator.generate_recipe(request.ingredients, request.cuisine, request.dish_type)
        
        return {
            "status": "success",
            "generated_recipe": recipe,
            "input_ingredients": request.ingredients,
            "cuisine": request.cuisine,
            "dish_type": request.dish_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_flavor")
async def analyze_flavor(request: FlavorAnalysisRequest):
    try:
        from src.data_processing.flavor_analyzer import FlavorAnalyzer
        from src.generative_models.flavor_predictor import FlavorModel
        
        flavor_analyzer = FlavorAnalyzer()
        flavor_model = FlavorModel()
        
        ingredient_flavors = {}
        for ingredient in request.ingredients:
            ingredient_flavors[ingredient] = flavor_analyzer.analyze_ingredient_flavor(ingredient)
        
        dummy_embeddings = [np.random.rand(256) for _ in request.ingredients]
        overall_flavor = flavor_model.predict_flavor_profile(dummy_embeddings)
        
        adjustments = []
        if request.target_flavor_profile:
            adjustments = flavor_model.recommend_ingredient_adjustments(overall_flavor, request.target_flavor_profile)
        
        return {
            "status": "success",
            "ingredient_flavors": ingredient_flavors,
            "overall_flavor_profile": overall_flavor,
            "recommended_adjustments": adjustments
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_chemistry")
async def analyze_chemistry(request: ChemistryAnalysisRequest):
    try:
        from src.food_science.chemistry_analyzer import ChemistryAnalyzer
        
        chemistry_analyzer = ChemistryAnalyzer()
        analysis = chemistry_analyzer.analyze_recipe_chemistry(request.ingredients)
        reactions = chemistry_analyzer.predict_chemical_reactions(request.ingredients, request.cooking_method)
        
        return {
            "status": "success",
            "chemistry_analysis": analysis,
            "predicted_reactions": reactions,
            "cooking_method": request.cooking_method
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_nutrition")
async def calculate_nutrition(request: NutritionRequest):
    try:
        from src.food_science.nutrition_calculator import NutritionCalculator
        
        nutrition_calculator = NutritionCalculator()
        nutrition = nutrition_calculator.estimate_recipe_nutrition(request.ingredients, request.servings)
        nutrition_score = nutrition_calculator.calculate_nutrition_score(nutrition)
        alternatives = nutrition_calculator.suggest_healthier_alternatives(request.ingredients)
        
        return {
            "status": "success",
            "nutrition_per_serving": nutrition,
            "nutrition_score": nutrition_score,
            "healthier_alternatives": alternatives
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_from_image")
async def generate_from_image(file: UploadFile = File(...), cuisine: str = ""):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from src.generative_models.image_to_recipe import ImageToRecipeModel
        image_model = ImageToRecipeModel()
        result = image_model.generate_recipe_from_image(temp_path, cuisine)
        
        os.unlink(temp_path)
        
        return {
            "status": "success",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi_modal_analysis")
async def multi_modal_analysis(request: MultiModalRequest):
    try:
        from src.multi_modal.fusion_model import MultiModalModel
        
        multi_modal = MultiModalModel()
        
        text_features = np.random.rand(768) if request.text else np.zeros(768)
        image_features = np.random.rand(2048) if request.image_path else np.zeros(2048)
        flavor_features = np.array(list(request.flavor_profile.values())) if request.flavor_profile else np.zeros(7)
        
        quality_score = multi_modal.predict_recipe_quality(text_features, image_features, flavor_features)
        
        return {
            "status": "success",
            "multi_modal_quality_score": quality_score,
            "modalities_used": {
                "text": bool(request.text),
                "image": bool(request.image_path),
                "flavor": bool(request.flavor_profile)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CulinaryAI"}

if __name__ == "__main__":
    import uvicorn
    from src.utils.config import Config
    config = Config()
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))