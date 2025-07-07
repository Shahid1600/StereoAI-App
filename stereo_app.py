import streamlit as st
from PIL import Image
import base64
import os

# Logo display
logo_path = "logo.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=180)

# Optional App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>StereoAI: Stereochemistry & Drug Analysis</h1>", unsafe_allow_html=True)
st.markdown("---")

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdMolDescriptors, Crippen
from rdkit import DataStructs
import py3Dmol
import requests
from chempy import balance_stoichiometry
import re
import pandas as pd
import numpy as np
from io import BytesIO

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

            # Use BytesIO to handle image in memory
            img = Draw.MolToImage(mol, size=(400, 400))
            st.image(img, caption="Structure Preview")

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

# ---------------------------------------------
# 🔬 Elemental Composition Analyzer
# ---------------------------------------------
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

# ---------------------------------------------
# 🧪 Toxicity & Drug-Likeness Analyzer
# ---------------------------------------------
st.header("☠️ Toxicity & Drug-Likeness Analyzer")
st.write("Enter a SMILES string to analyze toxicity and drug-likeness properties.")

toxicity_smiles = st.text_input("Enter SMILES for Toxicity Analysis:", "CC(=O)OC1=CC=CC=C1C(=O)O")

if toxicity_smiles:
    try:
        mol = Chem.MolFromSmiles(toxicity_smiles)
        if mol:
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            # Lipinski's Rule of Five
            lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
            
            # Toxicity alerts (simple substructure matching)
            toxic_patterns = {
                "Nitro group": "[NX3](=O)=O",
                "Aromatic amine": "cN",
                "Michael acceptor": "C=CC=O",
                "Epoxide": "C1OC1",
                "Thiol": "SH"
            }
            
            toxic_alerts = []
            for name, smarts in toxic_patterns.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt is not None and mol.HasSubstructMatch(patt):
                    toxic_alerts.append(name)
            
            st.subheader("📊 Molecular Properties")
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol")
            st.write(f"**LogP:** {logp:.2f}")
            st.write(f"**H-bond Donors:** {hbd}")
            st.write(f"**H-bond Acceptors:** {hba}")
            st.write(f"**Rotatable Bonds:** {rot_bonds}")
            st.write(f"**TPSA:** {tpsa:.2f} Å²")
            
            st.subheader("💊 Drug-Likeness")
            st.write(f"**Lipinski's Rule of Five:** {'✅ Pass' if lipinski_pass else '❌ Fail'}")
            
            st.subheader("☠️ Toxicity Alerts")
            if toxic_alerts:
                for alert in toxic_alerts:
                    st.warning(f"⚠️ {alert}")
            else:
                st.success("✅ No major toxicity alerts detected")
                
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error in analysis: {e}")


# ---------------------------------------------
# 💧 Solubility Estimator
# ---------------------------------------------
st.header("💧 Solubility Estimator")
st.write("Enter a SMILES string to estimate aqueous solubility.")

solubility_smiles = st.text_input("Enter SMILES for Solubility Estimation:", "CCO")

if solubility_smiles:
    try:
        mol = Chem.MolFromSmiles(solubility_smiles)
        if mol:
            # Simple solubility estimation based on descriptors
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            # Simplified solubility model (for demonstration)
            logS = 0.5 - 0.75*logp - 0.01*mw + 0.1*hbd + 0.005*tpsa
            
            st.subheader("🔍 Solubility Metrics")
            st.write(f"**LogP:** {logp:.2f}")
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol")
            st.write(f"**H-bond Donors:** {hbd}")
            st.write(f"**TPSA:** {tpsa:.2f} Å²")
            
            st.subheader("📘 Solubility Estimation")
            st.write(f"**Predicted LogS:** {logS:.2f}")
            
            if logS > -4:
                st.success("💧 Good solubility (LogS > -4)")
            elif logS > -6:
                st.warning("🌫 Moderate solubility (-6 < LogS ≤ -4)")
            else:
                st.error("❌ Poor solubility (LogS ≤ -6)")
                
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error in solubility estimation: {e}")

# ---------------------------------------------
# 💊 Bioavailability Checker
# ---------------------------------------------
st.header("💊 Bioavailability Checker")
st.write("Check if a molecule is likely to be orally bioavailable using Lipinski's Rule of Five.")

bioavailability_smiles = st.text_input("Enter SMILES for Bioavailability Check:", "CC(=O)NC1=CC=C(C=C1)O")

if bioavailability_smiles:
    try:
        mol = Chem.MolFromSmiles(bioavailability_smiles)
        if mol:
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            
            # Lipinski's Rule of Five
            rule1 = mw <= 500
            rule2 = logp <= 5
            rule3 = hbd <= 5
            rule4 = hba <= 10
            
            passed_rules = sum([rule1, rule2, rule3, rule4])
            
            st.subheader("📊 Molecular Properties")
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol {'✅' if rule1 else '❌'}")
            st.write(f"**LogP:** {logp:.2f} {'✅' if rule2 else '❌'}")
            st.write(f"**H-bond Donors:** {hbd} {'✅' if rule3 else '❌'}")
            st.write(f"**H-bond Acceptors:** {hba} {'✅' if rule4 else '❌'}")
            
            st.subheader("💊 Bioavailability Prediction")
            if passed_rules == 4:
                st.success("✅ Likely orally bioavailable (passes all 4 rules)")
            elif passed_rules >= 3:
                st.warning("⚠️ Possibly bioavailable (passes 3/4 rules)")
            else:
                st.error("❌ Unlikely to be orally bioavailable (fails 2+ rules)")
                
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error in analysis: {e}")

# ---------------------------------------------
# 🧠 Blood-Brain Barrier (BBB) Permeability Predictor
# ---------------------------------------------
st.header("🧠 Blood-Brain Barrier (BBB) Permeability Predictor")
st.write("Estimate if a compound can cross the blood-brain barrier based on molecular properties.")

bbb_smiles = st.text_input("Enter SMILES for BBB Prediction:", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)C3C4=O")

if bbb_smiles:
    try:
        mol = Chem.MolFromSmiles(bbb_smiles)
        if mol:
            # Calculate properties
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            # Simple BBB permeability rules
            bbb_permeable = (logp > 0.9) and (tpsa < 90) and (mw < 450)
            
            st.subheader("🔍 BBB Permeability Metrics")
            st.write(f"**LogP:** {logp:.2f} (should be > 0.9)")
            st.write(f"**TPSA:** {tpsa:.2f} Å² (should be < 90)")
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol (should be < 450)")
            
            st.subheader("🧠 BBB Permeability Prediction")
            if bbb_permeable:
                st.success("✅ Likely to cross the blood-brain barrier")
            else:
                st.warning("⚠️ Unlikely to cross the blood-brain barrier")
                
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error in analysis: {e}")

# ---------------------------------------------
# ⚗️ Chemical Stability Estimator
# ---------------------------------------------
st.header("⚗️ Chemical Stability Estimator")
st.write("Estimate the chemical stability of a molecule based on molecular properties.")

stability_smiles = st.text_input("Enter SMILES for Stability Estimation:", "C1=CC=CC=C1")

if stability_smiles:
    try:
        mol = Chem.MolFromSmiles(stability_smiles)
        if mol:
            # Calculate properties
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            # Simple stability score (higher is more stable)
            stability_score = 0.5*(5 - logp) + 0.2*(100 - tpsa)/10 + 0.2*(10 - rot_bonds) + 0.1*aromatic_rings*2
            
            st.subheader("🔍 Stability Metrics")
            st.write(f"**LogP:** {logp:.2f} (moderate values preferred)")
            st.write(f"**TPSA:** {tpsa:.2f} Å² (lower may be more stable)")
            st.write(f"**Rotatable Bonds:** {rot_bonds} (fewer may be more stable)")
            st.write(f"**Aromatic Rings:** {aromatic_rings} (more may increase stability)")
            
            st.subheader("⚗️ Stability Estimation")
            st.write(f"**Stability Score:** {stability_score:.2f}/10")
            
            if stability_score > 7:
                st.success("✅ Likely chemically stable")
            elif stability_score > 5:
                st.warning("⚠️ Moderate stability")
            else:
                st.error("❌ Potentially unstable")
                
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error in analysis: {e}")

# ---------------------------------------------
# 🧬 QSAR Property Estimator
# ---------------------------------------------
st.header("🧬 QSAR Property Estimator")
st.write("Estimate molecular properties using quantitative structure-activity relationship models.")

qsar_smiles = st.text_input("Enter SMILES for QSAR Estimation:", "CC(=O)NC1=CC=C(C=C1)O")

if qsar_smiles:
    try:
        mol = Chem.MolFromSmiles(qsar_smiles)
        if mol:
            # Calculate descriptors
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            # Mock QSAR predictions (replace with actual models)
            pIC50 = 5.2 + 0.1*logp - 0.01*mw + 0.5*hbd - 0.2*hba + 0.05*tpsa
            solubility = 0.5 - 0.75*logp - 0.01*mw + 0.1*hbd + 0.005*tpsa
            permeability = 0.6 + 0.2*logp - 0.001*tpsa - 0.005*mw
            
            st.subheader("📊 Molecular Descriptors")
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol")
            st.write(f"**LogP:** {logp:.2f}")
            st.write(f"**H-bond Donors:** {hbd}")
            st.write(f"**H-bond Acceptors:** {hba}")
            st.write(f"**TPSA:** {tpsa:.2f} Å²")
            st.write(f"**Aromatic Rings:** {aromatic_rings}")
            
            st.subheader("📈 QSAR Predictions")
            st.write(f"**Predicted pIC50:** {pIC50:.2f}")
            st.write(f"**Predicted LogS (solubility):** {solubility:.2f}")
            st.write(f"**Predicted Permeability:** {permeability:.2f}")
            
        else:
            st.error("❌ Invalid SMILES string.")
    except Exception as e:
        st.error(f"⚠️ Error in analysis: {e}")
        st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "🔬 Developed by <b>Shahid Ul Hassan</b> | 🚀 Powered by RDKit & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
