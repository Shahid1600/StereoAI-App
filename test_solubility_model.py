import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load model
model = joblib.load("solubility_model.pkl")

# Descriptor calculator
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
    }

# Test SMILES
test_smiles = ["CCO", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]  # Ethanol, Ibuprofen

for smi in test_smiles:
    desc = get_descriptors(smi)
    df = pd.DataFrame([desc])
    pred = model.predict(df)[0]
    print(f"SMILES: {smi}")
    print(f"📦 Predicted log(solubility): {pred:.2f}\n")
