import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdMolDescriptors, Crippen
from rdkit import DataStructs
import py3Dmol
import requests
from chempy import balance_stoichiometry
from itertools import permutations
import re  

# Set page config
st.set_page_config(page_title="StereoAI Chem Pro", layout="wide")

# ---------------------------------------------
# 📊 Stereochemistry Analyzer
# ---------------------------------------------
st.title("📊 StereoAI: Stereochemistry Analyzer")
st.write("Paste any molecule in SMILES format to analyze chiral centers (R/S).")

smiles = st.text_input("Enter SMILES:", "CC(C)F")

if smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES entered.")
        else:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

            st.image(Draw.MolToImage(mol, size=(400, 400)), caption="Structure Preview")

            formula = rdMolDescriptors.CalcMolFormula(mol)
            st.subheader("📜 Molecular Formula")
            st.write(f"📟 {formula}")

            try:
                response = requests.get(f'https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name')
                if response.status_code == 200:
                    iupac_name = response.text
                    st.subheader("💼 Full IUPAC Name")
                    st.write(f"💻 {iupac_name}")
                else:
                    st.warning("⚠️ IUPAC name could not be retrieved.")
            except Exception as e:
                st.warning(f"⚠️ Error getting IUPAC name: {e}")

            try:
                mol_with_H = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_with_H, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol_with_H)
                mol_block = Chem.MolToMolBlock(mol_with_H)

                st.subheader("🔬 3D Structure Viewer")
                viewer = py3Dmol.view(width=500, height=400)
                viewer.addModel(mol_block, 'mol')
                viewer.setStyle({'stick': {}})
                viewer.setBackgroundColor('white')
                viewer.zoomTo()
                st.components.v1.html(viewer._make_html(), height=400)
            except Exception as e:
                st.warning(f"⚠️ Could not render 3D structure: {e}")

            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            if chiral_centers:
                st.subheader("🧽 Chiral Centers Found:")
                for idx, chirality in chiral_centers:
                    st.write(f"Atom Index: {idx}, Configuration: {chirality}")
            else:
                st.warning("❌ No chiral centers found in this molecule.")

            double_bonds = []
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    stereo = bond.GetStereo()
                    stereo_str = "Z (cis)" if stereo == Chem.rdchem.BondStereo.STEREOZ else "E (trans)"
                    double_bonds.append((i, j, stereo_str))

            if double_bonds:
                st.subheader("🔁 E/Z Stereochemistry Found:")
                for i, j, stereo_str in double_bonds:
                    st.write(f"Double bond between atoms {i}-{j}: {stereo_str}")
            else:
                st.info("ℹ️ No E/Z stereochemistry found.")

    except Exception as e:
        st.error(f"❌ Error parsing molecule: {e}")

# ---------------------------------------------
# 🔁 Reaction Visualizer
# ---------------------------------------------
st.header("🔁 Reaction Visualizer (SMILES Format)")
reaction_smiles = st.text_input("Enter Reaction SMILES:", "CCBr.CN>>CCN")

if reaction_smiles:
    try:
        reactants_str, products_str = reaction_smiles.split(">>")
        reactant_smiles = reactants_str.split(".")
        product_smiles = products_str.split(".")
        reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles]
        product_mols = [Chem.MolFromSmiles(smi) for smi in product_smiles]

        st.subheader("📊 Reactants")
        reactant_cols = st.columns(len(reactant_mols))
        for i, mol in enumerate(reactant_mols):
            if mol:
                img = Draw.MolToImage(mol, size=(200, 200))
                reactant_cols[i].image(img, caption=reactant_smiles[i])
            else:
                reactant_cols[i].error(f"Invalid SMILES: {reactant_smiles[i]}")

        st.subheader("➡️ Products")
        product_cols = st.columns(len(product_mols))
        for i, mol in enumerate(product_mols):
            if mol:
                img = Draw.MolToImage(mol, size=(200, 200))
                product_cols[i].image(img, caption=product_smiles[i])
            else:
                product_cols[i].error(f"Invalid SMILES: {product_smiles[i]}")

    except Exception as e:
        st.error(f"⚠️ Error parsing reaction: {e}")

# ---------------------------------------------
# 🧪 Functional Group Identifier
# ---------------------------------------------
st.header("🧪 Functional Group Identifier")
st.write("Paste a molecule in SMILES format to detect common functional groups.")

functional_groups = {
    "Alcohol": "[OX2H]",  # -OH group
    "Carboxylic Acid": "C(=O)[OH]", 
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[CX3](=O)[#6]",
    "Amine": "[NX3;H2,H1;!$(NC=O)]",
    "Ether": "[OD2]([#6])[#6]",
    "Ester": "C(=O)O[#6]",
    "Amide": "C(=O)N",
    "Halide": "[F,Cl,Br,I]",
    "Nitrile": "C#N",
    "Phenol": "c1ccccc1O",
    "Alkene": "C=C",
    "Alkyne": "C#C"
}

smiles_fg = st.text_input("Enter SMILES to identify functional groups:", "", key="fg_input")

if smiles_fg:
    try:
        mol = Chem.MolFromSmiles(smiles_fg)
        if mol is None:
            st.error("❌ Invalid SMILES entered.")
        else:
            present_groups = []
            for name, smarts in functional_groups.items():
                patt = Chem.MolFromSmarts(smarts)
                if mol.HasSubstructMatch(patt):
                    present_groups.append(name)
            if present_groups:
                st.success(f"Functional Groups found: {', '.join(present_groups)}")
            else:
                st.info("ℹ️ No common functional groups detected.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------------------------------------------
# 🔬 Molecular Property Calculator
# ---------------------------------------------
st.header("🔬 Molecular Property Calculator")
st.write("Enter any molecule in SMILES format to calculate its basic properties.")

prop_smiles = st.text_input("Enter SMILES for Property Calculation:", "", key="prop_input")

if prop_smiles:
    try:
        mol = Chem.MolFromSmiles(prop_smiles)
        if mol:
            mol_weight = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            h_acceptors = rdMolDescriptors.CalcNumHBA(mol)
            h_donors = rdMolDescriptors.CalcNumHBD(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)

            st.subheader("📊 Molecular Properties")
            st.write(f"**🧪 Molecular Weight:** {mol_weight:.2f} g/mol")
            st.write(f"**💧 LogP (Hydrophobicity):** {logp:.2f}")
            st.write(f"**🔹 H-bond Acceptors:** {h_acceptors}")
            st.write(f"**🔸 H-bond Donors:** {h_donors}")
            st.write(f"**🌀 Rotatable Bonds:** {rot_bonds}")
            st.write(f"**🌐 Topological Polar Surface Area (TPSA):** {tpsa:.2f} Å²")
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"❌ Error calculating properties: {e}")

# ---------------------------------------------
# 🧬 Molecule Similarity Checker
# ---------------------------------------------
st.header("🧬 Molecule Similarity Checker")
st.write("Compare two molecules (by SMILES) to check their similarity.")

smiles1 = st.text_input("Enter first SMILES:", "", key="smiles1")
smiles2 = st.text_input("Enter second SMILES:", "", key="smiles2")

if smiles1 and smiles2:
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            st.error("❌ One or both SMILES are invalid.")
        else:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            st.success(f"🔍 Similarity Score: {similarity:.2f}")
    except Exception as e:
        st.error(f"⚠️ Error calculating similarity: {e}")

# ---------------------------------------------
# 🔍 Molecule Search with PubChem
# ---------------------------------------------
st.header("🔍 Molecule Search with PubChem")
st.write("Enter a molecule name and get its SMILES, IUPAC name, and structure from PubChem.")

mol_name = st.text_input("Enter Molecule Name (e.g., aspirin):", "", key="pubchem_input")

if mol_name:
    try:
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{mol_name}/property/IUPACName,CanonicalSMILES/JSON"
        response = requests.get(search_url)

        if response.status_code == 200:
            data = response.json()
            props = data['PropertyTable']['Properties'][0]
            smiles = props['CanonicalSMILES']
            iupac = props['IUPACName']

            st.success("✅ Molecule Found!")
            st.write(f"**IUPAC Name:** {iupac}")
            st.write(f"**SMILES:** `{smiles}`")

            mol = Chem.MolFromSmiles(smiles)
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption=iupac)

        else:
            st.error("❌ Molecule not found on PubChem.")
    except Exception as e:
        st.error(f"⚠️ Error searching PubChem: {e}")

# ---------------------------------------------
# ⚖️ Chemical Reaction Balancer
# ---------------------------------------------
st.header("⚖️ Chemical Reaction Balancer")
st.write("Enter an unbalanced chemical reaction (e.g. `Fe + O2 -> Fe2O3`) to balance it.")

reaction_input = st.text_input("Enter Reaction:", "Fe + O2 -> Fe2O3", key="balance_input")

if reaction_input:
    try:
        reactants_part, products_part = reaction_input.split("->")
        reactants = [r.strip() for r in reactants_part.split("+")]
        products = [p.strip() for p in products_part.split("+")]

        reac, prod = balance_stoichiometry(set(reactants), set(products))

        st.success("✅ Balanced Reaction:")
        reaction_str = ' + '.join([f"{reac[r]} {r}" for r in reac]) + " → " + ' + '.join([f"{prod[p]} {p}" for p in prod])
        st.latex(reaction_str)

    except Exception as e:
        st.error(f"⚠️ Error balancing reaction: {e}")

# ---------------------------------------------
# 🧿 Stereoisomer Classifier
# ---------------------------------------------
st.header("🧿 Stereoisomer Classifier")
st.write("Enter a SMILES to detect and classify stereoisomers (R/S or E/Z).")

stereo_smiles = st.text_input("Enter SMILES for stereoisomer check:", "F[C@](Br)(Cl)I", key="stereo_input")

if stereo_smiles:
    try:
        mol = Chem.MolFromSmiles(stereo_smiles)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        st.image(Draw.MolToImage(mol, size=(400, 400)), caption="Structure")

        # Check for R/S centers
        rs_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if rs_centers:
            st.subheader("🌀 R/S Chiral Centers:")
            for idx, chiral in rs_centers:
                st.write(f"Atom Index: {idx}, Configuration: {chiral}")
        else:
            st.info("ℹ No R/S stereocenters found.")

        # Check for E/Z double bonds
        ez_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                stereo = bond.GetStereo()
                stereo_str = "Z (cis)" if stereo == Chem.rdchem.BondStereo.STEREOZ else "E (trans)"
                ez_bonds.append((i, j, stereo_str))

        if ez_bonds:
            st.subheader("🪞 E/Z Double Bonds:")
            for i, j, label in ez_bonds:
                st.write(f"Double bond between atoms {i}-{j}: {label}")
        else:
            st.info("ℹ No E/Z stereochemistry found.")

    except Exception as e:
        st.error(f"❌ Error analyzing stereoisomerism: {e}")
# ✅ Feature added to replace Structural Isomer Generator
# 🔬 Elemental Composition Analyzer

import streamlit as st
import re
from rdkit.Chem import Descriptors

st.header("🔬 Elemental Composition Analyzer")
st.write("Enter a molecular formula (e.g., C8H10N4O2) to calculate atomic composition and total molecular weight.")

formula_input = st.text_input("Enter Molecular Formula for Composition:", "C8H10N4O2")

if formula_input:
    try:
        # Extract element counts
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula_input)
        elements = {}
        total_weight = 0.0

        # Atomic weights (approximate)
        atomic_weights = {
            "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.01,
            "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18, "Na": 22.99, "Mg": 24.31,
            "Al": 26.98, "Si": 28.09, "P": 30.97, "S": 32.07, "Cl": 35.45, "K": 39.10,
            "Ar": 39.95, "Ca": 40.08, "Fe": 55.85, "Zn": 65.38, "Br": 79.90, "I": 126.90
        }

        for (element, count) in matches:
            count = int(count) if count else 1
            if element in atomic_weights:
                elements[element] = elements.get(element, 0) + count
                total_weight += atomic_weights[element] * count
            else:
                st.warning(f"⚠ Unknown element: {element}")

        st.subheader("📘 Atomic Composition")
        for element, count in elements.items():
            st.write(f"{element}: {count} atoms")

        st.subheader("⚖️ Estimated Molecular Weight")
        st.write(f"{total_weight:.2f} g/mol")

    except Exception as e:
        st.error(f"❌ Error analyzing formula: {e}")
# ---------------------------------------------
# 💧 Molecule Polarity Predictor
# ---------------------------------------------
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen

st.header("💧 Molecule Polarity Predictor")
st.write("Enter a SMILES string to estimate whether the molecule is likely polar or non-polar based on LogP and TPSA.")

polarity_smiles = st.text_input("Enter SMILES for Polarity Prediction:", "CCO")

if polarity_smiles:
    try:
        mol = Chem.MolFromSmiles(polarity_smiles)
        if mol:
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)

            st.subheader("🔍 Polarity Metrics")
            st.write(f"**💧 LogP:** {logp:.2f} (lower means more polar)")
            st.write(f"**🌐 TPSA:** {tpsa:.2f} Å² (higher means more polar)")

            st.subheader("📘 Polarity Estimation")
            if logp < 0.5 or tpsa > 40:
                st.success("🌊 This molecule is likely **polar**.")
            else:
                st.info("🌫️ This molecule is likely **non-polar** or weakly polar.")
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error analyzing polarity: {e}")
        # ✅ Test Set for Polarity Predictor and Drug-Likeness Checker

# 📘 Use these SMILES strings to test your app's polarity and drug-likeness prediction modules.

# ------------------------------
# ✅ Polar and Non-Polar Molecule Test Set
polarity_test_set = [
    {"compound": "Water", "smiles": "O", "expected": "Polar"},
    {"compound": "Methanol", "smiles": "CO", "expected": "Polar"},
    {"compound": "Acetic Acid", "smiles": "CC(=O)O", "expected": "Polar"},
    {"compound": "Ammonia", "smiles": "N", "expected": "Polar"},
    {"compound": "Formaldehyde", "smiles": "C=O", "expected": "Polar"},
    {"compound": "Benzene", "smiles": "c1ccccc1", "expected": "Non-polar"},
    {"compound": "Hexane", "smiles": "CCCCCC", "expected": "Non-polar"},
    {"compound": "Toluene", "smiles": "Cc1ccccc1", "expected": "Slightly polar"},
    {"compound": "Chloroform", "smiles": "ClC(Cl)Cl", "expected": "Polar"},
    {"compound": "Carbon Tetrachloride", "smiles": "ClC(Cl)(Cl)Cl", "expected": "Non-polar"},
]

# ------------------------------
# 💊 Drug-Likeness & Lipinski Rule Test Set
drug_likeness_test_set = [
    {
        "compound": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "mw": "~180",
        "logp": "<5",
        "hba": 4,
        "hbd": 1,
        "lipinski_pass": True,
    },
    {
        "compound": "Paracetamol",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "mw": "~151",
        "logp": "<5",
        "hba": 3,
        "hbd": 2,
        "lipinski_pass": True,
    },
    {
        "compound": "Morphine",
        "smiles": "CN1CCC23C4C1CC(C2)C5=C3C(=C(C=C5)O)O4",
        "mw": "~285",
        "logp": "<5",
        "hba": 5,
        "hbd": 3,
        "lipinski_pass": True,
    },
    {
        "compound": "Erythromycin",
        "smiles": "CCC(C)C1CCC(C(C1C)OC2CC(C(C(O2)C)OC)O)OC",
        "mw": ">500",
        "logp": ">5",
        "hba": ">10",
        "hbd": ">5",
        "lipinski_pass": False,
    },
    {
        "compound": "Methotrexate",
        "smiles": "CN(C)C(=O)CN1C=NC2=C1N=CN=C2N",
        "mw": "~454",
        "logp": "<5",
        "hba": ">10",
        "hbd": 5,
        "lipinski_pass": False,
    },
]
# 🧪 Toxicity & Drug-Likeness Analyzer
import streamlit as st
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

st.subheader("☠️ Toxicity & Drug-Likeness Analyzer")
smiles = st.text_input("Enter SMILES for Analysis:")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # QED Score
        qed_score = QED.qed(mol)

        # Lipinski Rule of 5
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
        hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
        lipinski_pass = (mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)

        # Toxicity Flags (simple substructure check)
        toxic_alerts = {
            "Nitro group (mutagenic)": "[NX3](=O)=O",
            "Aromatic amine": "cN",
            "Halogen (Cl, Br, F, I)": "[Cl,Br,F,I]",
            "Aldehyde": "[CX3H1](=O)[#6]",
        }

        flagged = []
        for name, smarts in toxic_alerts.items():
            patt = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(patt):
                flagged.append(name)

        st.markdown(f"**💊 QED Score:** {qed_score:.2f} → {'Drug-like' if qed_score > 0.5 else 'Low drug-likeness'}")
        st.markdown(f"**🧬 Lipinski Rule of 5:** {'✅ Pass' if lipinski_pass else '❌ Fail'}")
        st.markdown("**☢️ Toxicity Alerts:**")
        if flagged:
            for alert in flagged:
                st.markdown(f"- ⚠️ {alert}")
        else:
            st.markdown("- ✅ No major toxicity substructures found")
    else:
        st.error("Invalid SMILES string.")
#import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import math

st.header("💧 Solubility Estimator")

smiles = st.text_input("Enter SMILES for Solubility Prediction:")

def estimate_logS(mol):
    logP = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # Simple LogS approximation formula (based on Delaney, 2004)
    logS = 0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * hbd - 0.74
    return logS, logP, mw, hbd, hba, tpsa

def classify_solubility(logS):
    if logS > -2:
        return "💧 Soluble"
    elif logS > -4:
        return "🌥️ Moderately soluble"
    else:
        return "❌ Poorly soluble"

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        logS, logP, mw, hbd, hba, tpsa = estimate_logS(mol)

        st.subheader("🔍 Solubility Metrics")
        st.markdown(f"**LogP**: {logP:.2f}")
        st.markdown(f"**Molecular Weight**: {mw:.2f} g/mol")
        st.markdown(f"**HBD**: {hbd}, **HBA**: {hba}, **TPSA**: {tpsa:.2f} Å²")

        st.subheader("📘 Solubility Estimation")
        st.markdown(f"**Estimated LogS**: {logS:.2f}")
        st.success(classify_solubility(logS))
    else:
        st.error("❌ Invalid SMILES string.")
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


#import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

st.markdown("""
## 💊 Bioavailability Checker
Enter a SMILES string to check if the compound is likely orally bioavailable using **Lipinski's Rule of Five**.
""")

smiles = st.text_input("SMILES Input", "")

if smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string.")
        else:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)

            rule1 = mw <= 500
            rule2 = logp <= 5
            rule3 = hbd <= 5
            rule4 = hba <= 10
            passed_rules = sum([rule1, rule2, rule3, rule4])

            st.subheader("🔍 Molecular Properties")
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol")
            st.write(f"**LogP:** {logp:.2f}")
            st.write(f"**HBD:** {hbd}, **HBA:** {hba}")

            st.subheader("📘 Lipinski Rule Evaluation")
            st.write(f"Passed {passed_rules}/4 rules")

            if passed_rules == 4:
                st.success("✅ Likely orally bioavailable")
            elif passed_rules >= 2:
                st.warning("⚠️ Possibly bioavailable, but not optimal")
            else:
                st.error("❌ Unlikely to be orally bioavailable")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

st.markdown("## 🧪 Blood-Brain Barrier (BBB) Permeability Predictor")
st.write("Estimate if a compound can cross the BBB based on LogP and Polar Surface Area (PSA).")

smiles = st.text_input("Enter SMILES for BBB Prediction")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)

        st.subheader("🔍 BBB Prediction Metrics")
        st.write(f"**LogP:** {logp:.2f}")
        st.write(f"**TPSA:** {tpsa:.2f} Å²")

        # Heuristic criteria for BBB permeability:
        if logp > 0.9 and tpsa < 90:
            st.success("🧠 Likely to cross the blood-brain barrier.")
        else:
            st.warning("❌ Unlikely to cross the blood-brain barrier.")
    else:
        st.error("❌ Invalid SMILES string.")
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

st.markdown("## ⚗️ Chemical Stability Estimator")
st.write("Enter a SMILES string to estimate chemical stability based on molecular descriptors.")

smiles = st.text_input("SMILES Input", key="stability_smiles")

def estimate_stability(mol):
    # Example: Use simple rules based on LogP and number of rotatable bonds
    logp = Crippen.MolLogP(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Simple heuristic:
    # Higher LogP (>5) means more hydrophobic, often less stable in aqueous environments
    # More rotatable bonds (>10) may indicate less stability (flexibility)
    stability_score = 10 - (logp + rot_bonds * 0.5)  # arbitrary scoring for demo
    return stability_score, logp, rot_bonds

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        stability_score, logp, rot_bonds = estimate_stability(mol)

        st.subheader("🔍 Stability Metrics")
        st.write(f"**LogP:** {logp:.2f}")
        st.write(f"**Rotatable Bonds:** {rot_bonds}")

        st.subheader("📊 Stability Estimation")
        if stability_score > 5:
            st.success(f"✅ Chemical stability is likely good (Score: {stability_score:.2f})")
        else:
            st.warning(f"⚠️ Chemical stability may be low (Score: {stability_score:.2f})")
    else:
        st.error("❌ Invalid SMILES string.")
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pickle

st.title("🧬 QSAR Property Estimator")

st.write("""
Enter a SMILES string to predict a molecular property using a QSAR model.
(Currently, this demo uses a mock prediction — replace with your trained model.)
""")

smiles = st.text_input("Enter SMILES:")

def calculate_descriptors(mol):
    # Calculate a few common molecular descriptors
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]

# Uncomment and set your model path here once you have a real model
# MODEL_PATH = "qsar_model.pkl"
# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES string. Please enter a valid molecule.")
    else:
        descriptors = calculate_descriptors(mol)
        st.subheader("Molecular Descriptors")
        st.write(f"Molecular Weight: {descriptors[0]:.2f}")
        st.write(f"LogP: {descriptors[1]:.2f}")
        st.write(f"H-bond Donors: {descriptors[2]}")
        st.write(f"H-bond Acceptors: {descriptors[3]}")

        # MOCK prediction - replace with your model's predict method
        predicted_property = np.random.uniform(0, 1)
        st.subheader("Predicted Property (Mock)")
        st.write(f"Value: {predicted_property:.3f}")

        # Example usage with a real model:
        # prediction = model.predict([descriptors])
        # st.write(f"Predicted Property: {prediction[0]:.3f}")#  import streamlit as st
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib

# Load models once at the top
toxicity_model = joblib.load('toxicity_model.pkl')
sol_model = joblib.load('solubility_model.pkl')
pic50_model = joblib.load('pic50_model.pkl')
herg_model = joblib.load('herg_model.pkl')

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
    else:
        return None

def batch_predict(df):
    results = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = get_descriptors(smiles)
            input_df = pd.DataFrame([desc])

            tox_pred = toxicity_model.predict(input_df)[0]
            tox_prob = toxicity_model.predict_proba(input_df)[0][1]

            sol_pred = sol_model.predict(input_df)[0]

            pic50_pred = pic50_model.predict(input_df)[0]

            herg_pred = herg_model.predict(input_df)[0]
            herg_prob = herg_model.predict_proba(input_df)[0][1]

            results.append({
                "SMILES": smiles,
                "Toxicity": tox_pred,
                "Toxicity_Prob": tox_prob,
                "Solubility_LogS": sol_pred,
                "pIC50": pic50_pred,
                "hERG": herg_pred,
                "hERG_Prob": herg_prob,
            })
        else:
            results.append({
                "SMILES": smiles,
                "Toxicity": None,
                "Toxicity_Prob": None,
                "Solubility_LogS": None,
                "pIC50": None,
                "hERG": None,
                "hERG_Prob": None,
            })
    return pd.DataFrame(results)

import streamlit as st
import pandas as pd
from rdkit import Chem

def batch_predict(df):
    results = []
    for smiles in df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Replace ... with your actual prediction logic
                # For example:
                results.append({
                    "SMILES": smiles,
                    "Toxicity": "Predicted_value",  # placeholder
                    # Add other prediction outputs here
                })
            else:
                results.append({
                    "SMILES": smiles,
                    "Toxicity": None,
                    # other fields None
                })
        except Exception as e:
            results.append({
                "SMILES": smiles,
                "Error": str(e),
            })
    return pd.DataFrame(results)

#

import streamlit as st
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load trained model
model = joblib.load("toxicity_model.pkl")

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

# Streamlit UI
st.title("🧪 Toxicity Predictor (SR-MMP Assay)")

smiles = st.text_input("Enter a SMILES string:")

if st.button("Predict Toxicity"):
    if Chem.MolFromSmiles(smiles):
        desc = get_descriptors(smiles)
        input_df = pd.DataFrame([desc])
        result = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if result == 1:
            st.error(f"⚠️ Toxic with probability {prob:.2f}")
        else:
            st.success(f"✅ Non-toxic with probability {1 - prob:.2f}")
    else:
        st.warning("❌ Invalid SMILES string.")
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load solubility model
sol_model = joblib.load("solubility_model.pkl")

# Descriptor generator
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
    }

# 💧 Solubility UI
st.subheader("💧 Solubility Predictor")

smiles_input = st.text_input("Enter SMILES for Solubility Prediction")

if st.button("Predict Solubility"):
    try:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            desc = get_descriptors(smiles_input)
            input_df = pd.DataFrame([desc])
            pred = sol_model.predict(input_df)[0]
            st.success(f"📦 Predicted Log(Solubility): {pred:.2f}")
        else:
            st.warning("❌ Invalid SMILES format")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load pIC50 model
pic50_model = joblib.load("pic50_model.pkl")

# Descriptor generator (if not already defined globally)
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
    }

# 🧪 pIC50 Predictor
st.subheader("🧪 pIC50 Activity Predictor")
smiles_input = st.text_input("Enter SMILES to Predict pIC50")

if st.button("Predict pIC50"):
    try:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            desc = get_descriptors(smiles_input)
            input_df = pd.DataFrame([desc])
            prediction = pic50_model.predict(input_df)[0]
            st.success(f"📈 Predicted pIC50: {prediction:.2f}")
        else:
            st.warning("❌ Invalid SMILES format")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import joblib

# Load trained models
toxicity_model = joblib.load('toxicity_model.pkl')
sol_model = joblib.load('solubility_model.pkl')
pic50_model = joblib.load('pic50_model.pkl')
herg_model = joblib.load('herg_model.pkl')

# Custom dark/light mode toggle (simulated)
mode = st.sidebar.radio("Choose Theme", ["🌙 Dark Mode", "☀️ Light Mode"], key="theme_selector")

if mode == "🌙 Dark Mode":
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .stTextInput > div > div > input {
                background-color: #262730;
                color: #FAFAFA;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("🧪 StereoAI: Smart Molecular Property Predictor")

# Helper function to compute descriptors
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
    else:
        return None

# Setup Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚠️ Toxicity", "💧 Solubility", "📊 pIC50", "🧬 hERG Cardiotoxicity", "💊 Lipinski Rule Checker"])

with tab1:
    st.header("Toxicity Predictor")
    smiles = st.text_input("Enter SMILES for Toxicity Prediction", key="tox_input")
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol), caption="Molecule Structure")
            desc = get_descriptors(smiles)
            if desc:
                df = pd.DataFrame([desc])
                pred = toxicity_model.predict(df)[0]
                prob = toxicity_model.predict_proba(df)[0][1]

                st.write("Prediction:", "Toxic" if pred else "Non-toxic")
                st.progress(prob)
                if prob > 0.7:
                    st.markdown(f"**🔴 High Risk:** {prob:.2f}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**🟢 Low Risk:** {prob:.2f}", unsafe_allow_html=True)
        else:
            st.error("❌ Invalid SMILES")

with tab2:
    st.header("Solubility Estimator")
    smiles2 = st.text_input("Enter SMILES for Solubility Prediction", key="sol_input")
    if smiles2:
        mol = Chem.MolFromSmiles(smiles2)
        if mol:
            st.image(Draw.MolToImage(mol), caption="Molecule Structure")
            desc = get_descriptors(smiles2)
            if desc:
                df = pd.DataFrame([desc])
                sol = sol_model.predict(df)[0]
                st.success(f"📦 Predicted Log(Solubility): {sol:.2f}")

with tab3:
    st.header("pIC50 QSAR Predictor")
    smiles3 = st.text_input("Enter SMILES for pIC50 Prediction", key="pic50_input")
    if smiles3:
        mol = Chem.MolFromSmiles(smiles3)
        if mol:
            st.image(Draw.MolToImage(mol), caption="Molecule Structure")
            desc = get_descriptors(smiles3)
            if desc:
                df = pd.DataFrame([desc])
                prediction = pic50_model.predict(df)[0]
                st.info(f"🔬 Predicted pIC50: {prediction:.2f}")

with tab4:
    st.header("hERG Cardiotoxicity Predictor")
    smiles4 = st.text_input("Enter SMILES for hERG Prediction", key="herg_input")
    if smiles4:
        mol = Chem.MolFromSmiles(smiles4)
        if mol:
            st.image(Draw.MolToImage(mol), caption="Molecule Structure")
            desc = get_descriptors(smiles4)
            if desc:
                df = pd.DataFrame([desc])
                pred = herg_model.predict(df)[0]
                prob = herg_model.predict_proba(df)[0][1]
                if pred:
                    st.error(f"☠️ Likely Blocker (Toxic) — Risk Score: {prob:.2f}")
                else:
                    st.success(f"✅ Likely Safe (Non-blocker) — Risk Score: {prob:.2f}")
                st.progress(prob)

with tab5:
    st.header("Lipinski Rule of Five Checker")
    smiles5 = st.text_input("Enter SMILES to check drug-likeness", key="lipinski_input")
    if smiles5:
        mol = Chem.MolFromSmiles(smiles5)
        if mol:
            st.image(Draw.MolToImage(mol), caption="Molecule Structure")
            desc = get_descriptors(smiles5)
            if desc:
                st.write("### Lipinski's Rule Results:")
                st.write(f"Molecular Weight: {desc['MolWt']:.2f} — {'✔️' if desc['MolWt'] <= 500 else '❌'}")
                st.write(f"LogP: {desc['LogP']:.2f} — {'✔️' if desc['LogP'] <= 5 else '❌'}")
                st.write(f"H-bond Donors: {desc['NumHDonors']} — {'✔️' if desc['NumHDonors'] <= 5 else '❌'}")
                st.write(f"H-bond Acceptors: {desc['NumHAcceptors']} — {'✔️' if desc['NumHAcceptors'] <= 10 else '❌'}")

                all_passed = (desc['MolWt'] <= 500 and desc['LogP'] <= 5 and
                              desc['NumHDonors'] <= 5 and desc['NumHAcceptors'] <= 10)
                if all_passed:
                    st.success("✅ Likely Orally Bioavailable (Passes Lipinski's Rule)")
                else:
                    st.warning("⚠️ May Violate Drug-Likeness Rules")
        else:
            st.error("❌ Invalid SMILES")
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import pandas as pd
import joblib

# Load your trained models
toxicity_model = joblib.load('toxicity_model.pkl')
sol_model = joblib.load('solubility_model.pkl')
pic50_model = joblib.load('pic50_model.pkl')
herg_model = joblib.load('herg_model.pkl')

# Descriptor generator
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol)
        }
    return None

# Simple SMILES extractor from input
def extract_smiles(text):
    tokens = text.split()
    for token in tokens:
        if Chem.MolFromSmiles(token):
            return token
    return None

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 StereoAI Chat Assistant")

# Chat input
user_input = st.chat_input("Ask about toxicity, solubility, pIC50, or hERG")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    smiles = extract_smiles(user_input)
    if not smiles:
        st.session_state.messages.append({"role": "assistant", "content": "❌ No valid SMILES found in your message."})
    else:
        mol = Chem.MolFromSmiles(smiles)
        desc = get_descriptors(smiles)
        img = Draw.MolToImage(mol)

        # Default response
        response = f"Here's what I found for **{smiles}**:\n"

        # Predict toxicity
        if "toxic" in user_input.lower():
            df = pd.DataFrame([desc])
            pred = toxicity_model.predict(df)[0]
            prob = toxicity_model.predict_proba(df)[0][1]
            response += f"\n☠️ Toxicity: {'Toxic' if pred else 'Non-toxic'} (Confidence: {prob:.2f})"

        # Predict solubility
        if "solubility" in user_input.lower():
            df = pd.DataFrame([desc])
            sol = sol_model.predict(df)[0]
            response += f"\n💧 Solubility (logS): {sol:.2f}"

        # Predict pIC50
        if "pic50" in user_input.lower():
            df = pd.DataFrame([desc])
            pic50 = pic50_model.predict(df)[0]
            response += f"\n📊 pIC50: {pic50:.2f}"

        # Predict hERG
        if "herg" in user_input.lower():
            df = pd.DataFrame([desc])
            pred = herg_model.predict(df)[0]
            prob = herg_model.predict_proba(df)[0][1]
            result = "Blocker ☠️" if pred else "Non-blocker ✅"
            response += f"\n🧬 hERG Prediction: {result} (Risk Score: {prob:.2f})"

        # Add to message history
        st.session_state.messages.append({"role": "assistant", "content": response, "image": img})

# Render chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"], caption="Molecule Structure")
