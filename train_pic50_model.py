import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("pic50_data.csv")

# Keep only valid SMILES and pIC50 columns
df = df[['canonical_smiles', 'pIC50']]
df = df[df['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)].reset_index(drop=True)

# Generate descriptors from SMILES
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
    }

desc_df = df['canonical_smiles'].apply(get_descriptors).apply(pd.Series)
X = desc_df
y = df['pIC50']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save model
joblib.dump(model, "pic50_model.pkl")
print("✅ pIC50 model saved as 'pic50_model.pkl'")
