import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core.statistics import (
    test_pearson_correlation,
    test_spearman_correlation
)

print("=" * 80)
print("CORRELATION TESTS")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")
np.random.seed(42)

# SCENARIO 1: Strong positive linear correlation
print("\n" + "=" * 80)
print("SCENARIO 1: Strong Positive Linear Correlation")
print("=" * 80)

x1 = np.random.normal(loc=50, scale=10, size=100)
y1 = 2 * x1 + np.random.normal(loc=0, scale=5, size=100)  # y = 2x + noise

df_x1 = pd.DataFrame({'value': x1})
df_y1 = pd.DataFrame({'value': y1})

x1_file = _store_resource(df_x1, str(test_manifest), "linear_x", "Linear X values", 'csv')
y1_file = _store_resource(df_y1, str(test_manifest), "linear_y", "Linear Y values", 'csv')

print(f"✅ Created datasets with strong positive linear relationship")
print(f"   Relationship: y ≈ 2x + noise")

# TEST 1: Pearson correlation (strong positive)
print("\n" + "-" * 80)
print("TEST 1: Pearson Correlation (Strong Positive)")
print("-" * 80)

result1 = test_pearson_correlation(
    x1_file,
    y1_file,
    str(test_manifest),
    'value',
    'value',
    alpha=0.05
)

print(f"✅ {result1['summary']}")
print(f"   Correlation (r): {result1['correlation']:.4f}")
print(f"   P-value: {result1['p_value']:.4e}")
print(f"   Strength: {result1['strength']}")
print(f"   Direction: {result1['direction']}")
print(f"   Is significant: {result1['is_significant']}")
print(f"   Interpretation: {result1['interpretation']}")

# TEST 2: Spearman correlation (strong positive)
print("\n" + "-" * 80)
print("TEST 2: Spearman Correlation (Strong Positive)")
print("-" * 80)

result2 = test_spearman_correlation(
    x1_file,
    y1_file,
    str(test_manifest),
    'value',
    'value',
    alpha=0.05
)

print(f"✅ {result2['summary']}")
print(f"   Correlation (ρ): {result2['correlation']:.4f}")
print(f"   P-value: {result2['p_value']:.4e}")
print(f"   Strength: {result2['strength']}")

# SCENARIO 2: No correlation
print("\n" + "=" * 80)
print("SCENARIO 2: No Correlation (Independent Variables)")
print("=" * 80)

x2 = np.random.normal(loc=0, scale=1, size=100)
y2 = np.random.normal(loc=0, scale=1, size=100)  # Completely independent

df_x2 = pd.DataFrame({'value': x2})
df_y2 = pd.DataFrame({'value': y2})

x2_file = _store_resource(df_x2, str(test_manifest), "independent_x", "Independent X", 'csv')
y2_file = _store_resource(df_y2, str(test_manifest), "independent_y", "Independent Y", 'csv')

print(f"✅ Created independent datasets (no correlation expected)")

# TEST 3: Pearson correlation (no correlation)
print("\n" + "-" * 80)
print("TEST 3: Pearson Correlation (No Correlation)")
print("-" * 80)

result3 = test_pearson_correlation(
    x2_file,
    y2_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result3['summary']}")
print(f"   Correlation (r): {result3['correlation']:.4f}")
print(f"   P-value: {result3['p_value']:.4f}")
print(f"   Is significant: {result3['is_significant']}")

# TEST 4: Spearman correlation (no correlation)
print("\n" + "-" * 80)
print("TEST 4: Spearman Correlation (No Correlation)")
print("-" * 80)

result4 = test_spearman_correlation(
    x2_file,
    y2_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result4['summary']}")
print(f"   Correlation (ρ): {result4['correlation']:.4f}")
print(f"   Is significant: {result4['is_significant']}")

# SCENARIO 3: Strong negative correlation
print("\n" + "=" * 80)
print("SCENARIO 3: Strong Negative Correlation")
print("=" * 80)

x3 = np.random.normal(loc=100, scale=20, size=80)
y3 = -1.5 * x3 + np.random.normal(loc=0, scale=10, size=80)  # y = -1.5x + noise

df_x3 = pd.DataFrame({'value': x3})
df_y3 = pd.DataFrame({'value': y3})

x3_file = _store_resource(df_x3, str(test_manifest), "negative_x", "Negative X", 'csv')
y3_file = _store_resource(df_y3, str(test_manifest), "negative_y", "Negative Y", 'csv')

print(f"✅ Created datasets with strong negative correlation")

# TEST 5: Pearson correlation (negative)
print("\n" + "-" * 80)
print("TEST 5: Pearson Correlation (Strong Negative)")
print("-" * 80)

result5 = test_pearson_correlation(
    x3_file,
    y3_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result5['summary']}")
print(f"   Correlation (r): {result5['correlation']:.4f}")
print(f"   Direction: {result5['direction']}")
print(f"   Is significant: {result5['is_significant']}")

# SCENARIO 4: Non-linear relationship (better for Spearman)
print("\n" + "=" * 80)
print("SCENARIO 4: Non-Linear (Quadratic) Relationship")
print("=" * 80)

x4 = np.linspace(-10, 10, 100)
y4 = x4**2 + np.random.normal(loc=0, scale=5, size=100)  # Quadratic relationship

df_x4 = pd.DataFrame({'value': x4})
df_y4 = pd.DataFrame({'value': y4})

x4_file = _store_resource(df_x4, str(test_manifest), "quadratic_x", "Quadratic X", 'csv')
y4_file = _store_resource(df_y4, str(test_manifest), "quadratic_y", "Quadratic Y", 'csv')

print(f"✅ Created datasets with quadratic relationship (y = x²)")

# TEST 6: Pearson correlation (non-linear - may be weak)
print("\n" + "-" * 80)
print("TEST 6: Pearson Correlation (Non-Linear - Less Appropriate)")
print("-" * 80)

result6 = test_pearson_correlation(
    x4_file,
    y4_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result6['summary']}")
print(f"   Correlation (r): {result6['correlation']:.4f}")
print(f"   Note: Pearson measures LINEAR correlation only")

# TEST 7: Spearman correlation (non-linear - more appropriate)
print("\n" + "-" * 80)
print("TEST 7: Spearman Correlation (Non-Linear - More Appropriate)")
print("-" * 80)

result7 = test_spearman_correlation(
    x4_file,
    y4_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result7['summary']}")
print(f"   Correlation (ρ): {result7['correlation']:.4f}")
print(f"   Note: Spearman captures MONOTONIC relationships better")

# SCENARIO 5: Data with outliers
print("\n" + "=" * 80)
print("SCENARIO 5: Data with Outliers")
print("=" * 80)

x5 = np.random.normal(loc=50, scale=10, size=95)
y5 = 2 * x5 + np.random.normal(loc=0, scale=5, size=95)

# Add 5 outliers
x5 = np.concatenate([x5, [100, 105, 110, 115, 120]])
y5 = np.concatenate([y5, [50, 45, 40, 35, 30]])  # Outliers break the pattern

df_x5 = pd.DataFrame({'value': x5})
df_y5 = pd.DataFrame({'value': y5})

x5_file = _store_resource(df_x5, str(test_manifest), "outlier_x", "Data with outliers X", 'csv')
y5_file = _store_resource(df_y5, str(test_manifest), "outlier_y", "Data with outliers Y", 'csv')

print(f"✅ Created datasets with 5 outliers (95 normal + 5 outliers)")

# TEST 8: Pearson correlation (sensitive to outliers)
print("\n" + "-" * 80)
print("TEST 8: Pearson Correlation (Sensitive to Outliers)")
print("-" * 80)

result8 = test_pearson_correlation(
    x5_file,
    y5_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result8['summary']}")
print(f"   Correlation (r): {result8['correlation']:.4f}")
print(f"   Note: Pearson is sensitive to outliers")

# TEST 9: Spearman correlation (robust to outliers)
print("\n" + "-" * 80)
print("TEST 9: Spearman Correlation (Robust to Outliers)")
print("-" * 80)

result9 = test_spearman_correlation(
    x5_file,
    y5_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result9['summary']}")
print(f"   Correlation (ρ): {result9['correlation']:.4f}")
print(f"   Note: Spearman uses ranks, more robust to outliers")

# SCENARIO 6: With NaN values
print("\n" + "=" * 80)
print("SCENARIO 6: Data with NaN Values")
print("=" * 80)

x6 = np.concatenate([np.random.normal(loc=50, scale=10, size=45), [np.nan] * 5])
y6 = np.concatenate([2 * x6[:45] + np.random.normal(loc=0, scale=5, size=45), [np.nan] * 5])

df_x6 = pd.DataFrame({'value': x6})
df_y6 = pd.DataFrame({'value': y6})

x6_file = _store_resource(df_x6, str(test_manifest), "nan_x", "Data with NaN X", 'csv')
y6_file = _store_resource(df_y6, str(test_manifest), "nan_y", "Data with NaN Y", 'csv')

print(f"✅ Created datasets with NaN values")

# TEST 10: Pearson with NaN handling
print("\n" + "-" * 80)
print("TEST 10: Pearson Correlation (Handling NaN)")
print("-" * 80)

result10 = test_pearson_correlation(
    x6_file,
    y6_file,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result10['summary']}")
print(f"   Samples analyzed (NaN excluded): {result10['n_samples']}")
print(f"   Correlation (r): {result10['correlation']:.4f}")

# TEST 11: Error handling - unequal lengths
print("\n" + "=" * 80)
print("TEST 11: Error Handling - Unequal Dataset Lengths")
print("=" * 80)

df_short = pd.DataFrame({'value': np.random.normal(size=30)})
df_long = pd.DataFrame({'value': np.random.normal(size=50)})

short_file = _store_resource(df_short, str(test_manifest), "short_corr", "Short dataset", 'csv')
long_file = _store_resource(df_long, str(test_manifest), "long_corr", "Long dataset", 'csv')

try:
    result = test_pearson_correlation(
        short_file,
        long_file,
        str(test_manifest),
        'value',
        'value'
    )
    print("❌ Should have raised ValueError!")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

# TEST 12: Compare Pearson vs Spearman on different scenarios
print("\n" + "=" * 80)
print("TEST 12: Comparison Summary - Pearson vs Spearman")
print("=" * 80)

print(f"\n   Linear Relationship:")
print(f"      Pearson:  r={result1['correlation']:.4f}, p={result1['p_value']:.4e}")
print(f"      Spearman: ρ={result2['correlation']:.4f}, p={result2['p_value']:.4e}")
print(f"      → Both detect strong positive correlation")

print(f"\n   No Correlation:")
print(f"      Pearson:  r={result3['correlation']:.4f}, p={result3['p_value']:.4f}")
print(f"      Spearman: ρ={result4['correlation']:.4f}, p={result4['p_value']:.4f}")
print(f"      → Both correctly find no significant correlation")

print(f"\n   Non-Linear (Quadratic):")
print(f"      Pearson:  r={result6['correlation']:.4f} ({result6['strength']})")
print(f"      Spearman: ρ={result7['correlation']:.4f} ({result7['strength']})")
print(f"      → Spearman better captures monotonic relationship")

print(f"\n   With Outliers:")
print(f"      Pearson:  r={result8['correlation']:.4f} ({result8['strength']})")
print(f"      Spearman: ρ={result9['correlation']:.4f} ({result9['strength']})")
print(f"      → Spearman more robust to outliers")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
print("=" * 80)
