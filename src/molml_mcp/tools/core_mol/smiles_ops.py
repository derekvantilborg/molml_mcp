from rdkit.Chem import MolFromSmiles, MolToSmiles

def canonicalize_smiles(smiles: list[str]) -> list[str]: 
    """ Convert a SMILES string to its canonical form. """

    canonic = []
    for smi in smiles:
        mol = MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES string {smi}")
        
        canonic.append(MolToSmiles(mol, canonical=True))
        
    return canonic
