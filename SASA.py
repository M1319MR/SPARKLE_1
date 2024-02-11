from collections import defaultdict
from scscore-master.scscore import standalone_model_numpy as smn
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

'''

# Example usage:
single_smiles = "CCO"
smiles_list = ["CCO", "CC", "C=C"]
result_single = calculate_scores(single_smiles)
result_list = calculate_scores(smiles_list)

print("Result for single SMILES:")
print(result_single[1])
print("Result for list of SMILES:")
print(result_list)
'''
