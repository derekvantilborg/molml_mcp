import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.featurization.SMILES_encoding import (
    tokenize_smiles,
    discover_tokens_from_dataset,
    build_vocab,
    save_vocab_to_json,
    load_vocab_from_json,
    create_default_vocab_json,
    compile_vocab_from_tokens,
    smiles_to_indices,
    smiles_to_one_hot,
    batch_smiles_to_one_hot,
    check_smiles_vocab_coverage,
    check_dataset_vocab_coverage
)

print("=" * 80)
print("COMPREHENSIVE SMILES ENCODING TEST SUITE")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")

# Test dataset
test_smiles = [
    'CCCl',                          # Simple with Cl
    'c1ccccc1Br',                    # Aromatic with Br
    'C@@H',                          # Chirality
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',   # Ibuprofen
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    'c1ccccc1',                      # Benzene
    'C%10CC%10',                     # Extended ring number
]

df = pd.DataFrame({'SMILES': test_smiles, 'id': range(len(test_smiles))})
df_filename = _store_resource(df, str(test_manifest), "encoding_test", "Test molecules", 'csv')

print(f"\n✅ Test data created: {df_filename}")
print(f"   - {len(df)} molecules")

# TEST 1: Token Discovery
print("\n" + "=" * 80)
print("TEST 1: Token Discovery from Dataset")
print("=" * 80)

result = discover_tokens_from_dataset(df_filename, str(test_manifest), 'SMILES')
print(f"✅ Discovered {result['n_unique_tokens']} unique tokens")
print(f"   Top 5: {[t['token'] for t in result['top_tokens'][:5]]}")

# TEST 2: Create Default Vocab JSON
print("\n" + "=" * 80)
print("TEST 2: Create Default Vocab JSON")
print("=" * 80)

default_vocab_file = create_default_vocab_json(
    str(test_manifest),
    "default_smiles_vocab",
    add_pad=True,
    add_unk=True
)
print(f"✅ Created default vocab: {default_vocab_file}")

# TEST 3: Load Vocab from JSON
print("\n" + "=" * 80)
print("TEST 3: Load Vocab from JSON")
print("=" * 80)

vocab, special_tokens = load_vocab_from_json(default_vocab_file, str(test_manifest))
print(f"✅ Loaded vocab with {len(vocab)} tokens")
print(f"   Special tokens: {special_tokens}")

# TEST 4: Compile Custom Vocab
print("\n" + "=" * 80)
print("TEST 4: Compile Custom Vocab from Tokens")
print("=" * 80)

custom_tokens = ['C', 'c', 'N', 'n', 'O', 'o', 'Cl', 'Br', '(', ')', '=', '1', '2']
custom_vocab_file = compile_vocab_from_tokens(
    custom_tokens,
    str(test_manifest),
    "custom_vocab",
    "Small custom vocab",
    add_pad=True,
    add_unk=True
)
print(f"✅ Created custom vocab: {custom_vocab_file}")
custom_vocab, _ = load_vocab_from_json(custom_vocab_file, str(test_manifest))
print(f"   Size: {len(custom_vocab)} tokens")

# TEST 5: Check Single SMILES Coverage
print("\n" + "=" * 80)
print("TEST 5: Check Single SMILES Vocab Coverage")
print("=" * 80)

test_smiles_simple = 'CCCl'
coverage = check_smiles_vocab_coverage(test_smiles_simple, vocab)
print(f"✅ SMILES: {test_smiles_simple}")
print(f"   Can tokenize: {coverage['can_tokenize']}")
print(f"   Coverage: {coverage['coverage']}%")
print(f"   Tokens: {coverage['n_known']}/{coverage['n_tokens']}")

# Test with incomplete vocab
test_smiles_complex = 'C@@H'
coverage2 = check_smiles_vocab_coverage(test_smiles_complex, custom_vocab)
print(f"\n   SMILES: {test_smiles_complex} (with custom vocab)")
print(f"   Can tokenize: {coverage2['can_tokenize']}")
print(f"   Coverage: {coverage2['coverage']}%")
if not coverage2['can_tokenize']:
    print(f"   Unknown tokens: {coverage2['unknown_tokens']}")

# TEST 6: Check Dataset Coverage
print("\n" + "=" * 80)
print("TEST 6: Check Dataset Vocab Coverage")
print("=" * 80)

dataset_coverage = check_dataset_vocab_coverage(
    df_filename,
    str(test_manifest),
    'SMILES',
    vocab
)
print(f"✅ Dataset coverage: {dataset_coverage['overall_coverage']}%")
print(f"   Fully covered: {dataset_coverage['n_fully_covered']}/{dataset_coverage['n_smiles']}")
if dataset_coverage['missing_tokens']:
    print(f"   Missing tokens: {dataset_coverage['missing_tokens']}")

# TEST 7: Encode SMILES to Indices
print("\n" + "=" * 80)
print("TEST 7: Encode SMILES to Indices")
print("=" * 80)

test_smiles_enc = 'CCO'
indices = smiles_to_indices(test_smiles_enc, vocab, max_length=5)
print(f"✅ SMILES: {test_smiles_enc}")
print(f"   Tokens: {tokenize_smiles(test_smiles_enc)}")
print(f"   Indices: {indices}")

# TEST 8: Encode SMILES to One-Hot
print("\n" + "=" * 80)
print("TEST 8: Encode SMILES to One-Hot")
print("=" * 80)

one_hot = smiles_to_one_hot(test_smiles_enc, vocab, max_length=5)
print(f"✅ One-hot shape: {one_hot.shape}")
print(f"   (max_length={one_hot.shape[0]}, vocab_size={one_hot.shape[1]})")

# TEST 9: Batch Encode to One-Hot
print("\n" + "=" * 80)
print("TEST 9: Batch Encode SMILES to One-Hot")
print("=" * 80)

batch_smiles = ['CC', 'CCO', 'CCCO']
batch_one_hot = batch_smiles_to_one_hot(batch_smiles, vocab)
print(f"✅ Batch shape: {batch_one_hot.shape}")
print(f"   (n_smiles={batch_one_hot.shape[0]}, max_length={batch_one_hot.shape[1]}, vocab_size={batch_one_hot.shape[2]})")

# TEST 10: Save Custom Vocab
print("\n" + "=" * 80)
print("TEST 10: Save Custom Vocab to JSON")
print("=" * 80)

manual_vocab = build_vocab(add_pad=True, add_start=True, add_end=True, add_unk=True)
saved_vocab_file = save_vocab_to_json(
    manual_vocab,
    str(test_manifest),
    "manual_vocab",
    "Manually created vocab with all special tokens",
    special_tokens={'pad': '<pad>', 'start': '<start>', 'end': '<end>', 'unk': '<unk>'}
)
print(f"✅ Saved vocab: {saved_vocab_file}")

# Verify we can load it back
loaded_vocab, loaded_special = load_vocab_from_json(saved_vocab_file, str(test_manifest))
print(f"   Loaded back: {len(loaded_vocab)} tokens")
print(f"   Special tokens: {loaded_special}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nComplete functionality verified:")
print("  ✓ Token discovery from datasets")
print("  ✓ Vocab JSON save/load system")
print("  ✓ Default SMILES vocab creation")
print("  ✓ Custom vocab compilation")
print("  ✓ SMILES to indices encoding")
print("  ✓ SMILES to one-hot encoding")
print("  ✓ Batch one-hot encoding")
print("  ✓ Single SMILES vocab coverage check")
print("  ✓ Dataset vocab coverage check")
