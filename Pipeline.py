import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Draw import IPythonConsole
from sklearn.preprocessing import StandardScaler
import string
import joblib

file_name = "REAL_lead-like"
model = joblib.load('location of the model')
# for i in ['a','b','c','d']:
#     for j in range(ord('a'), ord('z') + 1):
#         file_name = 'REAL_lead-like'  + i + chr(j)
#         if(file_name == 'REAL_lead-likedb'):
#             break
#         with open(file_name, 'r') as file:
#             first_elements = []
#             for line in file:
#                 parts = line.split()
#                 if parts:
#                     first_elements.append(parts[0])

with open('REAL_lead-likeae', 'r') as file:
    first_elements = []
    for line in file:
        parts = line.split()
        if parts:
            first_elements.append(parts[0])

df = pd.DataFrame(first_elements)
df.rename(columns={0:'Smiles'},inplace=True)
df = df.drop_duplicates()
df = df.dropna()
df.reset_index(inplace=True, drop=True)

logP_values = []
hydrogen_donors = []
hydrogen_acceptors = []
tpsa_values = []
mw_values = []
hac_values = []
rot_bonds_values = []
fsp3_values = []
qed_values = []
slogp_values = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Convert SMILES to molecule object
    mol = Chem.MolFromSmiles(row['Smiles'])
    
    # Check if conversion was successful
    if mol is None:
        print(f"Failed to convert SMILES to molecule for index {index}. Skipping.")
        continue  # Skip this iteration and move to the next one
    
    # Calculate properties
    logP = Descriptors.MolLogP(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    hac = Descriptors.HeavyAtomCount(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    qed = QED.qed(mol)
    slogp = Descriptors.MolLogP(mol)
    
    # Append calculated properties to lists
    logP_values.append(logP)
    hydrogen_donors.append(num_h_donors)
    hydrogen_acceptors.append(num_h_acceptors)
    tpsa_values.append(tpsa)
    mw_values.append(mw)
    hac_values.append(hac)
    rot_bonds_values.append(rot_bonds)
    fsp3_values.append(fsp3)
    qed_values.append(qed)
    slogp_values.append(slogp)

# Create a new DataFrame with only the rows that had successful conversions
valid_rows_df = df[df['Smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

# Add calculated columns to the valid rows DataFrame
valid_rows_df['logP'] = logP_values[:len(valid_rows_df)]
valid_rows_df['num_H_Donors'] = hydrogen_donors[:len(valid_rows_df)]
valid_rows_df['num_H_Acceptors'] = hydrogen_acceptors[:len(valid_rows_df)]
valid_rows_df['TPSA'] = tpsa_values[:len(valid_rows_df)]
valid_rows_df['MW'] = mw_values[:len(valid_rows_df)]
valid_rows_df['HAC'] = hac_values[:len(valid_rows_df)]
valid_rows_df['RotBonds'] = rot_bonds_values[:len(valid_rows_df)]
valid_rows_df['FSP3'] = fsp3_values[:len(valid_rows_df)]
valid_rows_df['QED'] = qed_values[:len(valid_rows_df)]
valid_rows_df['SLogP'] = slogp_values[:len(valid_rows_df)]


df=valid_rows_df

morgan = []
for smiles in df['Smiles']:
      mol = Chem.MolFromSmiles(smiles)
      
      #Morgan fingerprint
      morgan.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

df['morgan_fp'] = morgan
scaler = StandardScaler()
# Scale the Morgan fingerprints and replace the original column
df['morgan_fp'] = scaler.fit_transform(df['morgan_fp'].tolist()).tolist()

# Assuming you have already created additional_features and X
additional_features = df[['logP', 'num_H_Donors', 'num_H_Acceptors', 'TPSA', 'MW', 'HAC', 'RotBonds', 'FSP3', 'QED', 'SLogP']].values
X_morgan = np.array(df['morgan_fp'].tolist())
X = np.hstack((X_morgan, additional_features))

# Predicting using the model
prediction = model.predict(X)
predictions_prob = model.predict_proba(X)

# Adding predictions and probabilities to the original dataset
df['predicted_activity'] = prediction
df['probability_class_Activity'] = predictions_prob[:, 0]
# df['probability_class_1'] = predictions_prob[:, 1]

label_mapping = {0: "Active", 1: "Inactive"}

# Convert predictions to labels using the mapping
predictions_labels = [label_mapping[pred] for pred in prediction]

# Add the converted predictions to the original DataFrame
df['predicted_activity'] = predictions_labels
df.drop('morgan_fp', axis=1, inplace=True)
df.drop('logP', axis=1, inplace=True)
df.drop('num_H_Donors', axis=1, inplace=True)
df.drop('num_H_Acceptors', axis=1, inplace=True)
df.drop('TPSA', axis=1, inplace=True)
df.drop('MW', axis=1, inplace=True)
df.drop('HAC', axis=1, inplace=True)
df.drop('RotBonds', axis=1, inplace=True)
df.drop('FSP3', axis=1, inplace=True)
df.drop('QED', axis=1, inplace=True)
df.drop('SLogP', axis=1, inplace=True)

shortlisted_df = df[df['probability_class_Activity'] > 0.97]




