import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import gzip

# STEP 1: Load compressed CSV
with gzip.open("tox21.csv.gz", "rt") as f:
    df = pd.read_csv(f)

# STEP 2: Use 1 label: SR-MMP
df = df[['smiles', 'SR-MMP']].dropna()

# STEP 3: Filter only valid SMILES
df = df[df['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
df.reset_index(drop=True, inplace=True)

# STEP 4: Calculate molecular descriptors
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
y = df['SR-MMP'].astype(int)

# STEP 5: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 6: Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# STEP 7: Save model
joblib.dump(model, "toxicity_model.pkl")
print("✅ Model saved as 'toxicity_model.pkl'")
