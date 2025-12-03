from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.logging import loggable


def _canonicalize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]: 
    """ Convert a SMILES string to its canonical form. Failed conversions are treated as None. Return both canonical SMILES and comments."""

    canonic = []
    comment = []
    for smi in smiles:
        mol = MolFromSmiles(smi)

        if mol is None:
            canonic.append(None)
            comment.append("Failed: Invalid SMILES string")
            continue

        try:
            smi_canon = MolToSmiles(mol, canonical=True)
            canonic.append(smi_canon)
            comment.append("Passed")
        except Exception as e:
            canonic.append(None)
            comment.append(f"Failed: {str(e)}")

    return canonic, comment



def _remove_pattern(smiles: list[str], smarts_pattern: str) -> tuple[list[str], list[str]]: 
    """ Remove some pattern from a SMILES string using the specified SMARTS.

    :param smiles: list of SMILES strings
    :param smarts_pattern: SMARTS pattern (e.g., "[Cl,Na,Mg]")
    :return: cleaned SMILES without pattern, comments
    """
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover

    # Create SaltRemover with the provided SMARTS pattern
    # defnData format: "SMARTS<tab>name" per line
    remover = SaltRemover(defnData=f"{smarts_pattern}\tsalts")

    new_smi, comment = [], []
    for smi in smiles:
        # If no fragments, no salts to remove
        if '.' not in smi:
            new_smi.append(smi)
            comment.append("Passed")
            continue

        # Try to parse the SMILES with fragments
        try:
            mol = Chem.MolFromSmiles(smi)
            
            if mol is None:
                new_smi.append(None)
                comment.append("Failed: Invalid SMILES string")
                continue

            # Remove salts
            cleaned_mol = remover.StripMol(mol, dontRemoveEverything=True)
            
            if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
                new_smi.append(None)
                comment.append("Failed: All fragments were salts")
                continue
            
            cleaned_smi = Chem.MolToSmiles(cleaned_mol)
            
            # If still has fragments after salt removal, keep the largest
            if '.' in cleaned_smi:
                frags = cleaned_smi.split('.')
                # Sort by length and take longest
                largest_frag = max(frags, key=len)
                new_smi.append(largest_frag)
                comment.append("Passed")
            else:
                new_smi.append(cleaned_smi)
                comment.append("Passed")
                
        except Exception as e:
            new_smi.append(None)
            comment.append(f"Failed: {str(e)}")

    return new_smi, comment

