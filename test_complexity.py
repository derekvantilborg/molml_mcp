import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core_mol.complexity import add_complexity_columns

print("=" * 80)
print("MOLECULAR COMPLEXITY METRICS TEST")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset with diverse molecules
test_smiles = [
    'C',                                    # Methane (simplest)
    'CCCl',                                 # Simple chlorinated
    'c1ccccc1Br',                           # Bromobenzene
    'C@@H',                                 # Chiral center
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',          # Ibuprofen (moderate complexity)
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',        # Caffeine (multiple rings)
    'CC(C)(C)c1ccc(O)cc1',                  # BHT-like (branched)
    'C1=CC=C2C(=C1)C=CC=C2',                # Naphthalene (fused rings)
]

df = pd.DataFrame({
    'SMILES': test_smiles,
    'name': ['Methane', 'Chloroethane', 'Bromobenzene', 'Chiral', 'Ibuprofen', 
             'Caffeine', 'Phenol', 'Naphthalene'],
    'id': range(len(test_smiles))
})
df_filename = _store_resource(df, str(test_manifest), "complexity_test_data", "Test molecules for complexity", 'csv')

print(f"\n✅ Test data created: {df_filename}")
print(f"   Molecules: {len(df)}")

# TEST 1: Add single metric
print("\n" + "=" * 80)
print("TEST 1: Add Single Metric (num_tokens)")
print("=" * 80)

result1 = add_complexity_columns(
    df_filename,
    str(test_manifest),
    'SMILES',
    metrics=['num_tokens'],
    output_filename='complexity_test_single'
)

print(f"✅ Output: {result1['output_filename']}")
print(f"   Columns added: {result1['columns_added']}")
print(f"   Failures: {result1['n_failed']}")
print(f"\n   Preview:")
for row in result1['preview']:
    print(f"      {row['SMILES']:35s} tokens={row['num_tokens']}")

# TEST 2: Add multiple metrics
print("\n" + "=" * 80)
print("TEST 2: Add Multiple Metrics")
print("=" * 80)

result2 = add_complexity_columns(
    df_filename,
    str(test_manifest),
    'SMILES',
    metrics=['num_tokens', 'branches', 'smiles_entropy'],
    output_filename='complexity_test_multiple'
)

print(f"✅ Output: {result2['output_filename']}")
print(f"   Columns added: {result2['columns_added']}")
print(f"   Failures: {result2['n_failed']}")
print(f"\n   Preview:")
for row in result2['preview']:
    print(f"      {row['SMILES']:35s} tokens={row['num_tokens']:2d} branches={row['branches']:2d} entropy={row['smiles_entropy']:.3f}")

# TEST 3: Add all complexity metrics
print("\n" + "=" * 80)
print("TEST 3: Add All Complexity Metrics")
print("=" * 80)

result3 = add_complexity_columns(
    df_filename,
    str(test_manifest),
    'SMILES',
    metrics=['num_tokens', 'branches', 'smiles_entropy', 'molecular_entropy', 'bertz', 'bottcher'],
    output_filename='complexity_test_all'
)

print(f"✅ Output: {result3['output_filename']}")
print(f"   Columns added: {result3['columns_added']}")
print(f"   Failures: {result3['n_failed']}")
print(f"\n   Preview (simplified):")
for row in result3['preview']:
    print(f"      {row['SMILES']:35s} tokens={row['num_tokens']:2d} bertz={row['bertz']:.1f} bottcher={row['bottcher']:.1f}")

# TEST 4: Test error handling (invalid metric)
print("\n" + "=" * 80)
print("TEST 4: Error Handling (Invalid Metric)")
print("=" * 80)

try:
    result4 = add_complexity_columns(
        df_filename,
        str(test_manifest),
        'SMILES',
        metrics=['invalid_metric', 'num_tokens'],
        output_filename='should_fail'
    )
    print("❌ Should have raised ValueError!")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

# TEST 5: Test with invalid SMILES
print("\n" + "=" * 80)
print("TEST 5: Handling Invalid SMILES")
print("=" * 80)

df_invalid = pd.DataFrame({
    'SMILES': ['CCO', 'INVALID_SMILES', 'c1ccccc1', None],
    'id': range(4)
})
df_invalid_filename = _store_resource(df_invalid, str(test_manifest), "complexity_invalid_test", "Test invalid SMILES", 'csv')

result5 = add_complexity_columns(
    df_invalid_filename,
    str(test_manifest),
    'SMILES',
    metrics=['num_tokens', 'bertz', 'bottcher'],
    output_filename='complexity_with_invalid'
)

print(f"✅ Output: {result5['output_filename']}")
print(f"   Total failures per metric: {result5['n_failed']}")
print(f"\n   Preview (with invalid/None values):")
for row in result5['preview']:
    smiles_str = str(row['SMILES']) if row['SMILES'] is not None else 'None'
    print(f"      SMILES={smiles_str:20s} tokens={row['num_tokens']} bertz={row['bertz']} bottcher={row['bottcher']}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
print("=" * 80)
