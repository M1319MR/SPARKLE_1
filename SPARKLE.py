import streamlit as st
import re
import rdkit
import mordred
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
import base64

from collections import defaultdict
from scscore.scscore import standalone_model_numpy as smn
from rdkit import Chem

scscore = smn.SCScorer()
scscore.restore()

def calculate_scs_plus_unique_atoms_ta(smi):
    m = Chem.MolFromSmiles(smi)
    if m:
        num_of_atoms = set()
        num_of_atoms = m.GetNumAtoms()
        scs = set()
        scs = scscore.get_score_from_smi(smi)[1]

        equivs = defaultdict(set)
        matches = m.GetSubstructMatches(m, uniquify=False)
        for match in matches:
            for idx1, idx2 in enumerate(match):
                equivs[idx1].add(idx2)
        classes = set()
        for s in equivs.values():
            classes.add(tuple(s))
        # Remove duplicates by sorting each tuple
        unique_data = set(tuple(sorted(item)) for item in classes)

        # Convert back to the original format
        result = set(tuple(sorted(item, reverse=True)) for item in unique_data)
        count_sets = sum(1 for item in result if len(item) > 1)
        count_single = sum(1 for item in result if len(item) == 1)

        # Calculate SCS+Unique atoms/TA
        scs_plus_unique_atoms_ta = (num_of_atoms/(scs+count_single))#((scs + count_single) / num_of_atoms) if num_of_atoms > 0 else 0
        return smi, scs_plus_unique_atoms_ta
    else:
        print(f"Invalid SMILES: {smi}")
        return None

def calculate_scores_for_list(smiles_list):
    return [calculate_scs_plus_unique_atoms_ta(smi)[1] for smi in smiles_list]

def calculate_scores(input_data):
    if isinstance(input_data, str):
        return calculate_scs_plus_unique_atoms_ta(input_data)
    elif isinstance(input_data, list):
        return calculate_scores_for_list(input_data)
    else:
        raise ValueError("Input data must be either a single SMILES string or a list of SMILES strings.")




#Checking valididy of string
def contains_only_numbers(string):
	return bool(re.match(r"^[0-9]+$", string))

desc_list = [descriptors.Autocorrelation.ATS(1,'Z'),
             descriptors.Autocorrelation.ATS(2,'Z'), 
             descriptors.BaryszMatrix.BaryszMatrix('are','VR2'),
             descriptors.BaryszMatrix.BaryszMatrix('p','VR2'),
	     descriptors.Autocorrelation.GATS(2,'d'),
	     descriptors.MolecularId.MolecularId('N',False,1e-10),
	     descriptors.Autocorrelation.GATS(3,'p')]


#Process the input data using the textbox
def process_input(input_data, desc_list):
	calc = Calculator(desc_list)
	mols = Chem.MolFromSmiles(input_data)
	desc = calc(mols)
	try:
		result = 1717.161781 * (desc['VR2_Dzare'] / (desc['VR2_Dzp'] * (desc['ATS1Z'] + desc['ATS2Z']))) + 0.02741337642
		result_specific_energy = round(result,4)
		result1 = -0.0008272568654 *((desc['ATS1Z']/desc['GATS2d'])*(desc['MID_N']+desc['GATS3p'])) -0.09311602313
		result_solvation_energy = round(result1,4)
		synthesis_score = calculate_scores(input_data)
		processed_data = "Specific Energy of molecule: " + str(result_specific_energy) + "Wh/kg"+ " and Solvation Energy of molecule: " + str(result_solvation_energy) + "Kcal/mol" +" and Synthesis Accessibility Score:" + str(synthesis_score[1])
		
		return processed_data
	except Exception as e:
		return "Error occured during calculating descriptors value which resulted in NaN or string values. Please refer to Mordred descriptors documentation."


##Mordred descriptors calculation

def Mordred_descriptors(data,desc_list):
  calc = Calculator(desc_list)
  mols = [Chem.MolFromSmiles(smi) for smi in data]
  df1=calc.pandas(mols)
  numeric_df = df1.apply(pd.to_numeric, errors='coerce')
  cleaned_df = numeric_df.dropna()
  return cleaned_df


#Process the file uploaded
def process_file(uploaded_file):
	df =pd.read_csv(uploaded_file)
	synthesis_score = calculate_scores(df.iloc[:,0].tolist())
	df['SASA'] = synthesis_score
	desc = Mordred_descriptors(df.iloc[:,0],desc_list)
	st.write("NOTE: If you see the difference in number of molecules submitted with the molecules in downloaded file, that is because we removed molecules for which we encountered errors in descriptors calculation.")
	Specific_Energy = 1717.161781 * (desc['VR2_Dzare'] / (desc['VR2_Dzp'] * (desc['ATS1Z'] + desc['ATS2Z']))) + 0.02741337642
	df['Specific Energy'] = Specific_Energy
	Solvation_Energy = -0.0008272568654 *((desc['ATS1Z']/desc['GATS2d'])*(desc['MID_N']+desc['GATS3p'])) -0.09311602313
	df['Solvation Energy']=Solvation_Energy
	# Provide a downloadable link to the modified CSV file
	csv_file = df.to_csv(index=False)
	b64 = base64.b64encode(csv_file.encode()).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="predicted_properties.csv">predicted_properties</a>'
	st.markdown(href, unsafe_allow_html=True)


st.title("SPARKLE: Molecule Property Prediction")

# Add input elements with a placeholder
input_data = st.text_input("SMILE string of molecule:", placeholder="SMILE string of molecule")

# Add submit button and reset button side by side
col1, col2 = st.columns(2)
with col1:
	if st.button("Submit"):
		if input_data != "" and contains_only_numbers(input_data) == False:
			mol = Chem.MolFromSmiles(input_data)
			if mol is not None:
				processed_data = process_input(input_data, desc_list)
				st.write("Output: ", processed_data)
			else:
				st.write('Wrong representation of molecule in SMILE format!!, unable to make a molecule representation using rdkit package')
		else:
			st.write('Invalid SMILE representation or empty data.')
		
with col2:
	if st.button("Structure of Molecule"):
		st.write("SMILE:", input_data)
		st.write('Structure of the Molecule')
		if input_data != "" and contains_only_numbers(input_data) == False:
			mol = Chem.MolFromSmiles(input_data)
			if mol is not None:
				img = Draw.MolToImage(mol)
				st.image(img, use_column_width=True)
			else:
				st.write('Wrong representation of molecule in SMILE format!!, unable to make a structure using rdkit package')
		else:
			st.write('Invalid SMILE representation or empty data.')

#Checking for validity of SMILE string
def contains_only_numbers(string):
	return bool(re.match(r"^[0-9]+$", string))

uploaded_file = st.file_uploader("Choose a file")

if st.button("Upload File"):
	process_file(uploaded_file)

