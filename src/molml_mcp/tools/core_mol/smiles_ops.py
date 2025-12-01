from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.logging import loggable


@loggable
def canonicalize_smiles(smiles: list[str]) -> list[str]: 
    """ Convert a SMILES string to its canonical form. Failed conversions are treated as None. Return canonicalized SMILES."""

    canonic, comment = _canonicalize_smiles(smiles)

    return canonic


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
