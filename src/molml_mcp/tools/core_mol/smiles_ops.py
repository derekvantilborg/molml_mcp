from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.constants import COMMON_SOLVENTS


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
            mol = MolFromSmiles(smi)
            
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



def _is_common_solvent_fragment(smiles_frag: str) -> bool:
    """
    Return True if this standalone fragment is one of the known common solvents.
    Uses canonical SMILES matching; independent of any broader cleaning pipeline.
    """
    mol = MolFromSmiles(smiles_frag)
    if mol is None:
        return False
    can = MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return can in COMMON_SOLVENTS


def _strip_common_solvent_fragments(smiles: list[str]) -> tuple[list[str], list[str]]: 
    """
    Remove known common solvent fragments from a fragmented SMILES string.

    Behavior:
    - If the SMILES has no '.' (single component), it is returned unchanged.
    - If some fragments match the solvent list and at least one fragment
      does NOT match, solvent fragments are dropped and the rest is joined.
    - If *all* fragments would be removed as solvents, the original SMILES
      is returned unchanged (assumed main molecule of interest).
    """

    new_smiles = [] 
    comments = []   

    for smi in smiles:
        # Only act on fragmented SMILES
        if '.' not in smi:
            new_smiles.append(smi)
            comments.append('Pass')
            continue    

        try:
        
            frags = [f.strip() for f in smi.split('.') if f.strip()]
            
            kept: list[str] = []
            any_removed = False

            for frag in frags:
                if _is_common_solvent_fragment(frag):
                    any_removed = True
                else:
                    kept.append(frag)

            # Case 1: nothing matched as solvent → return original
            if not any_removed:
                new_smiles.append(smi)
                comments.append('SMILES string is fragmented, but found no common solvents')
                continue    

            # Case 2: everything would be removed → keep original (your preference)
            if not kept:
                new_smiles.append(smi)
                comments.append('SMILES string is fragmented, but all fragments are common solvents. Kept original SMILES')
                continue    

            # Case 3: we removed some solvents but kept at least one fragment

            new_smiles.append('.'.join(kept))
            comments.append('Pass, removed solvents')
            continue    

        except Exception as e:
            new_smiles.append(smi)
            comments.append(f"Failed: {str(e)}")
            continue

    return new_smiles, comments



