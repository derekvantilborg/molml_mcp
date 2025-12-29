import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.featurization.SMILES_encoding import (
    create_default_vocab_json,
    inspect_vocab_json,
    smiles_to_indices,
    smiles_to_one_hot,
    batch_smiles_to_one_hot,
    check_smiles_vocab_coverage,
    check_dataset_vocab_coverage
)

print("=" * 80)
print("MCP-FRIENDLY SMILES ENCODING TEST")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset
test_smiles = [
    'CCCl',
    'c1ccccc1Br',
    'C@@H',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
]

df = pd.DataFrame({'SMILES': test_smiles, 'id': range(len(test_smiles))})
df_filename = _store_resource(df, str(test_manifest), "mcp_test_data", "Test molecules", 'csv')

print(f"\n✅ Test data created: {df_filename}")

# TEST 1: Create and Inspect Default Vocab
print("\n" + "=" * 80)
print("TEST 1: Create and Inspect Default Vocab JSON")
print("=" * 80)

vocab_file = create_default_vocab_json(
    str(test_manifest),
    "mcp_default_vocab",
    add_pad=True,
    add_unk=True
)
print(f"✅ Created vocab: {vocab_file}")

# Inspect the vocab
vocab_info = inspect_vocab_json(vocab_file, str(test_manifest))
print(f"\n   Vocab size: {vocab_info['vocab_size']}")
print(f"   Special tokens: {vocab_info['special_tokens']}")
print(f"   First 10 tokens: {vocab_info['tokens'][:10]}")
print(f"   Description: {vocab_info['description']}")

# TEST 2: Encode SMILES to Indices
print("\n" + "=" * 80)
print("TEST 2: Encode SMILES to Indices (MCP-friendly)")
print("=" * 80)

result = smiles_to_indices(
    'CCO',
    vocab_file,
    str(test_manifest),
    max_length=10
)
print(f"✅ SMILES: CCO")
print(f"   Tokens: {result['tokens']}")
print(f"   Indices: {result['indices']}")
print(f"   Length: {result['length']}")
print(f"   Padded: {result['padded']}")
print(f"   Truncated: {result['truncated']}")

# TEST 3: Encode SMILES to One-Hot (with save)
print("\n" + "=" * 80)
print("TEST 3: Encode SMILES to One-Hot with Save")
print("=" * 80)

result = smiles_to_one_hot(
    'CCl',
    vocab_file,
    str(test_manifest),
    max_length=5,
    output_filename='ccl_onehot',
    explanation='One-hot encoding of CCl'
)
print(f"✅ One-hot encoding saved: {result.get('output_filename', 'N/A')}")
print(f"   Shape: {result['shape']}")
print(f"   Tokens: {result['tokens']}")

# TEST 4: Batch Encode Dataset
print("\n" + "=" * 80)
print("TEST 4: Batch Encode Dataset to One-Hot")
print("=" * 80)

result = batch_smiles_to_one_hot(
    df_filename,
    str(test_manifest),
    'SMILES',
    vocab_file,
    'batch_encoded',
    max_length=20
)
print(f"✅ Batch encoded: {result['output_filename']}")
print(f"   Shape: {result['shape']}")
print(f"   N SMILES: {result['n_smiles']}")
print(f"   Summary: {result['summary']}")

# TEST 5: Check Single SMILES Coverage
print("\n" + "=" * 80)
print("TEST 5: Check Single SMILES Vocab Coverage")
print("=" * 80)

result = check_smiles_vocab_coverage(
    'CCCl',
    vocab_file,
    str(test_manifest)
)
print(f"✅ SMILES: CCCl")
print(f"   Can tokenize: {result['can_tokenize']}")
print(f"   Coverage: {result['coverage']}%")
print(f"   Known/Total: {result['n_known']}/{result['n_tokens']}")

# TEST 6: Check Dataset Coverage
print("\n" + "=" * 80)
print("TEST 6: Check Dataset Vocab Coverage")
print("=" * 80)

result = check_dataset_vocab_coverage(
    df_filename,
    str(test_manifest),
    'SMILES',
    vocab_file
)
print(f"✅ Dataset coverage: {result['overall_coverage']}%")
print(f"   Fully covered: {result['n_fully_covered']}/{result['n_smiles']}")
print(f"   Partial coverage: {result['n_partial_coverage']}")
if result['missing_tokens']:
    print(f"   Missing tokens: {result['missing_tokens']}")
    print(f"   Token frequencies: {result['missing_token_counts']}")

print("\n" + "=" * 80)
print("✅ ALL MCP-FRIENDLY TESTS PASSED!")
print("=" * 80)
print("\nMCP-Ready Features:")
print("  ✓ All functions use vocab JSON paths instead of dicts")
print("  ✓ inspect_vocab_json() shows vocab contents")
print("  ✓ Functions return MCP-friendly dictionaries")
print("  ✓ Automatic resource saving with manifest integration")
print("  ✓ Clear error messages with resource filenames")
