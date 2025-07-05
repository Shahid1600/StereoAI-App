import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 🔹 Step 1: Load the ESOL Dataset
df = pd.read_csv("esol.csv")
df = df[['smiles', 'measured log solubility in mols per litre']]
df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)].reset_index(drop=True)

# 🔹 Step 2: Generate Descriptors
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
    }

desc_df = df['smiles'].apply(get_descriptors).apply(pd.Series)
X = desc_df
y = df['measured log solubility in mols per litre']

# 🔹 Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 4: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Step 5: Evaluate the Model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 🔹 Step 6: Save the Model
joblib.dump(model, "solubility_model.pkl")
print("✅ Solubility model saved as 'solubility_model.pkl'")
