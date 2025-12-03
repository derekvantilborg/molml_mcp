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

    :param smiles: SMILES string
    :param smarts_pattern: SMARTS pattern
    :return: cleaned SMILES without pattern, comments
    """
    from rdkit.Chem.SaltRemover import SaltRemover

    remover = SaltRemover(defnData=smarts_pattern)

    new_smi, comment = [], []
    for smi in smiles:

        if '.' not in smi:
            new_smi.append(smi)
            comment.append("Passed")
            continue

        mol = MolFromSmiles(smi)

        if mol is None:
            new_smi.append(None)
            comment.append("Failed: Invalid SMILES string")
            continue

        try:
            new_smi = Chem.MolToSmiles(remover.StripMol(mol))
            new_smi.append(new_smi)
            comment.append("Passed")
        except Exception as e:
            new_smi.append(None)
            comment.append(f"Failed: {str(e)}")

    return new_smi, comment

