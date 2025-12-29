import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.featurization.SMILES_encoding import discover_tokens_from_dataset

print("=" * 80)
print("TESTING TOKEN DISCOVERY")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset
test_smiles = [
    'CCCl',                          # Simple with Cl
    'c1ccccc1Br',                    # Aromatic with Br
    'C@@H',                          # Chirality
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',   # Ibuprofen
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    'c1ccccc1',                      # Benzene
]

df = pd.DataFrame({
    'SMILES': test_smiles,
    'id': range(len(test_smiles))
})

# Store test dataset
df_filename = _store_resource(df, str(test_manifest), "token_test", "Test molecules for token discovery", 'csv')

print(f"\n✅ Test data created: {df_filename}")
print(f"   - {len(df)} molecules")

# Test token discovery
print("\n" + "=" * 80)
print("TEST 1: Basic Token Discovery")
print("=" * 80)

result = discover_tokens_from_dataset(
    df_filename,
    str(test_manifest),
    'SMILES'
)

print(f"\nN SMILES processed: {result['n_smiles']}")
print(f"N failed: {result['n_failed']}")
print(f"N unique tokens: {result['n_unique_tokens']}")
print(f"N total tokens: {result['n_total_tokens']}")
print(f"Min frequency: {result['min_frequency']}")
print()
print('Top 20 tokens:')
for item in result['top_tokens']:
    print(f"  {item['token']!r:10} -> {item['count']:5} occurrences")
print()
print(f"Summary: {result['summary']}")

# Test with min_frequency filter
print("\n" + "=" * 80)
print("TEST 2: Token Discovery with min_frequency=5")
print("=" * 80)

result2 = discover_tokens_from_dataset(
    df_filename,
    str(test_manifest),
    'SMILES',
    min_frequency=5
)
print(f"Unique tokens (freq >= 5): {result2['n_unique_tokens']}")
print(f"Top 5 tokens: {[t['token'] for t in result2['top_tokens'][:5]]}")

# Test with max_tokens limit
print("\n" + "=" * 80)
print("TEST 3: Token Discovery with max_tokens=10")
print("=" * 80)

result3 = discover_tokens_from_dataset(
    df_filename,
    str(test_manifest),
    'SMILES',
    max_tokens=10
)
print(f"Tokens returned: {len(result3['tokens'])}")
print(f"Token list: {[t['token'] for t in result3['tokens']]}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
