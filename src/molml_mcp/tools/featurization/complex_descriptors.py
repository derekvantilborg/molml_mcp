
from molml_mcp.tools.featurization.supported.ecfps import smiles_to_ecfp_dataset
from molml_mcp.tools.featurization.supported.maccs import smiles_to_maccs_dataset
from molml_mcp.tools.featurization.supported.rdkit import smiles_to_rdkit_fp_dataset


def get_all_complex_descriptor_tools():
    """Return a list of all complex descriptor tools for MCP registration."""
    return [
        smiles_to_ecfp_dataset,
        smiles_to_maccs_dataset,
        smiles_to_rdkit_fp_dataset
    ]

