import streamlit as st
import joblib  # To load the pre-trained model
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Function to generate a fingerprint from the SMILES string
def generate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        fp = mfpgen.GetFingerprint(mol)
        return np.array(fp)  # Convert fingerprint to numpy array
    else:
        return None  # In case of invalid SMILES

# Load the pre-trained model
model = joblib.load('model_smote.pkl')  # Make sure your model file is in the correct path

# Streamlit App
st.title("Activity of Drug Against Thrombin Target")
st.markdown("Enter a SMILES string below to predict whether the drug is Active or Inactive for the thrombin target.")

# Take input for the SMILES string
smiles_input = st.text_input("Enter SMILES string:")

# Button to trigger prediction
if st.button("Predict"):
    if smiles_input:
        # Generate fingerprint for the new SMILES string
        fingerprint = generate_fingerprint(smiles_input)
        
        if fingerprint is not None:
            # Reshape to match the input shape of the model
            fingerprint = fingerprint.reshape(1, -1)
            
            # Make the prediction
            prediction = model.predict(fingerprint)
            
            # Display the result
            if prediction == 1:
                st.subheader("Compound Prediction: Active")
                st.write("This compound is **Active** against the thrombin target.")
                st.write("You can explore other active compounds or modify your input.")
            else:
                st.subheader("Compound Prediction: Inactive")
                st.write("This compound is **Inactive** against the thrombin target.")
                st.write("You can explore other inactive compounds or modify your input.")
        else:
            st.error("Invalid SMILES string. Please check your input.")
    else:
        st.warning("Please enter a SMILES string.")
