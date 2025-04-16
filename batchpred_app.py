import streamlit as st
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from io import BytesIO

# Set Streamlit page config
st.set_page_config(page_title="Thrombin Drug Activity Predictor", layout="centered")

# Load the pre-trained model
try:
    model = joblib.load('model_smote.pkl')  # Ensure this file is in the same directory
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

# Function to generate a fingerprint from the SMILES string
def generate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        fp = mfpgen.GetFingerprint(mol)
        return np.array(fp)
    else:
        return None

# Excel export helper
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    return output.getvalue()

# --- Streamlit App UI ---
st.title("🧪 Activity of Drug Against Thrombin Target")
st.markdown("Predict whether a compound is **Active** or **Inactive** against the thrombin target.")

# === Manual Input ===
st.subheader("🔬 Predict a Single Compound")
smiles_input = st.text_input("Enter SMILES string:")

if st.button("Predict Single SMILES"):
    if smiles_input:
        fingerprint = generate_fingerprint(smiles_input)
        if fingerprint is not None:
            fingerprint = fingerprint.reshape(1, -1)
            prediction = model.predict(fingerprint)
            result = "Active" if prediction[0] == 1 else "Inactive"
            st.success(f"Prediction: **{result}**")
        else:
            st.error("Invalid SMILES string.")
    else:
        st.warning("Please enter a SMILES string.")

# === Batch Upload ===
st.subheader("📂 Batch Prediction from File")
uploaded_file = st.file_uploader("Upload a CSV or Excel file with a column named 'smiles'", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Validate column
    if 'smiles' not in df.columns:
        st.error("The file must contain a column named 'smiles'.")
    else:
        st.info("Processing predictions...")

        predictions = []
        for smi in df['smiles']:
            fp = generate_fingerprint(smi)
            if fp is not None:
                pred = model.predict(fp.reshape(1, -1))[0]
                predictions.append("Active" if pred == 1 else "Inactive")
            else:
                predictions.append("Invalid SMILES")

        df['Prediction'] = predictions

        st.success("✅ Predictions completed. Preview below:")
        st.dataframe(df)

        st.download_button(
            label="📥 Download Results as Excel",
            data=to_excel(df),
            file_name="thrombin_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
