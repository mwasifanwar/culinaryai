import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import Dict, List, Optional

class MolecularProcessor:
    def __init__(self):
        self.config = Config()
    
    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except:
            return None
    
    def calculate_molecular_similarity(self, smiles1: str, smiles2: str) -> float:
        fp1 = self.smiles_to_fingerprint(smiles1)
        fp2 = self.smiles_to_fingerprint(smiles2)
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        return float(np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2)))
    
    def analyze_compound_interactions(self, compound1: str, compound2: str) -> Dict[str, float]:
        similarity = self.calculate_molecular_similarity(compound1, compound2)
        
        interaction_strength = similarity * 0.8 + 0.2
        
        return {
            'similarity': similarity,
            'interaction_strength': interaction_strength,
            'compatibility_score': min(similarity * 1.5, 1.0)
        }
    
    def get_common_flavor_compounds(self) -> Dict[str, str]:
        return {
            'vanilla': 'C=CC1=CC(=C(C=C1)O)OC',
            'cinnamon': 'C1=CC(=C(C=C1C=CC=O)O)OC',
            'garlic': 'C=CCS(=O)CCS(=O)C=CC',
            'ginger': 'CC1CCC2C1(CCC3C2CCC4(C3CCC4C(=C)C)C)C',
            'lemon': 'CC1=CC(=O)CC(C1)C(C)C',
            'chocolate': 'C1=CC(=CC=C1C=CC(=O)O)O'
        }