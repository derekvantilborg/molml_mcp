from rdkit.Chem import Descriptors, MolFromSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource



DESCRIPTOR_REGISTRY: dict[str, callable] = {
    name: fn for name, fn in Descriptors._descList
}

def list_rdkit_descriptors() -> list[dict]:
    """
    List all available RDKit 2D molecular descriptors.
    
    Returns
    -------
    list[dict]
        Dicts with 'name' (descriptor function name) and 'explanation' (brief description).
    """
    out = []
    for name, fn in Descriptors._descList:
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        out.append({
            "descriptor name": name,
            "explanation": doc,
        })
    return out


def calculate_simple_descriptors(input_filename: str, smiles_column: str, descriptor_names: list[str], project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Calculate RDKit molecular descriptors for molecules in a dataset.
    
    Common descriptors:
    - TPSA: Topological Polar Surface Area (Å²)
    - MolWt: Molecular weight (g/mol)
    - MolLogP: Octanol-water partition coefficient (log P)
    - NumHDonors: Number of hydrogen bond donors
    - NumHAcceptors: Number of hydrogen bond acceptors
    - HeavyAtomCount: Number of heavy (non-hydrogen) atoms
    - NumRotatableBonds: Number of rotatable bonds
    - NumAromaticRings: Number of aromatic rings
    - RingCount: Total number of rings
    - FractionCSP3: Fraction of sp³ hybridized carbons
    - qed: Quantitative Estimate of Drug-likeness
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    smiles_column : str
        Column containing SMILES strings.
    descriptor_names : list[str]
        RDKit descriptor names to calculate. Use list_rdkit_descriptors() to see all 210+ available names.
    project_manifest_path : str
        Path to manifest.json.
    output_filename : str
        Base name for output file.
    explanation : str
        Description for manifest.
    
    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, descriptors_added, n_failed, note, preview.
    
    Raises
    ------
    ValueError
        If SMILES column not found or invalid descriptor names provided.
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate inputs
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Validate descriptor names
    invalid_descriptors = [name for name in descriptor_names if name not in DESCRIPTOR_REGISTRY]
    if invalid_descriptors:
        available = list(DESCRIPTOR_REGISTRY.keys())
        raise ValueError(
            f"Invalid descriptor names: {invalid_descriptors}. "
            f"Use list_rdkit_descriptors() to see all available descriptors. "
            f"Examples: {available}"
        )
    
    # Calculate descriptors for each molecule
    n_failed = {name: 0 for name in descriptor_names}
    descriptor_data = {name: [] for name in descriptor_names}
    
    for smiles in df[smiles_column]:
        mol = MolFromSmiles(smiles) if pd.notna(smiles) else None
        
        for desc_name in descriptor_names:
            try:
                if mol is not None:
                    value = DESCRIPTOR_REGISTRY[desc_name](mol)
                else:
                    value = None
                    n_failed[desc_name] += 1
            except Exception:
                value = None
                n_failed[desc_name]  += 1
            
            descriptor_data[desc_name].append(value)
    
    # Add descriptor columns to dataframe
    for desc_name, values in descriptor_data.items():
        df[desc_name] = values
    
    # Store the updated dataset
    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "descriptors_added": descriptor_names,
        "n_failed": n_failed,
        "note": "If molecules fails a descriptor calculation, a None value is assigned for that descriptor.",
        "preview": df.head(5).to_dict(orient="records"),
    }

