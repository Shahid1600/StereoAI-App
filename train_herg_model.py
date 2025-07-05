import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load your file
df = pd.read_csv("herg.csv")

# Rename columns to match expected format
df = df.rename(columns={"USED_AS": "Activity"})

# Filter out invalid SMILES
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        }
    return None

df['descriptors'] = df['SMILES'].apply(get_descriptors)
df = df[df['descriptors'].notnull()]
desc_df = df['descriptors'].apply(pd.Series)

# Input/output
X = desc_df
y = df['Activity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc:.4f}")

# Save model
joblib.dump(model, "herg_model.pkl")
print("✅ hERG model saved as 'herg_model.pkl'")
