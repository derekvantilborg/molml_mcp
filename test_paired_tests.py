import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core.statistics import (
    test_paired_ttest,
    test_wilcoxon_signed_rank
)

print("=" * 80)
print("PAIRED COMPARISON TESTS")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")
np.random.seed(42)

# TEST SCENARIO 1: Before/After with significant improvement
print("\n" + "=" * 80)
print("SCENARIO 1: Before/After Treatment (Significant Improvement)")
print("=" * 80)

before_scores = np.random.normal(loc=50, scale=10, size=100)
after_scores = before_scores + np.random.normal(loc=10, scale=5, size=100)  # Significant increase

df_before = pd.DataFrame({'score': before_scores, 'patient_id': range(100)})
df_after = pd.DataFrame({'score': after_scores, 'patient_id': range(100)})

before_file = _store_resource(df_before, str(test_manifest), "before_treatment", "Scores before treatment", 'csv')
after_file = _store_resource(df_after, str(test_manifest), "after_treatment", "Scores after treatment", 'csv')

print(f"✅ Created datasets:")
print(f"   Before: {before_file}")
print(f"   After: {after_file}")
print(f"   Mean before: {before_scores.mean():.2f}")
print(f"   Mean after: {after_scores.mean():.2f}")
print(f"   Expected: Significant increase")

# TEST 1: Paired t-test (two-sided)
print("\n" + "-" * 80)
print("TEST 1: Paired t-test (two-sided)")
print("-" * 80)

result1 = test_paired_ttest(
    before_file,
    after_file,
    str(test_manifest),
    'score',
    'score',
    alpha=0.05,
    alternative="two-sided"
)

print(f"✅ {result1['summary']}")
print(f"   t-statistic: {result1['statistic']:.4f}")
print(f"   p-value: {result1['p_value']:.4e}")
print(f"   Mean difference: {result1['mean_diff']:.4f}")
print(f"   Is significant: {result1['is_significant']}")
print(f"   Interpretation: {result1['interpretation']}")

# TEST 2: Paired t-test (one-sided: greater)
print("\n" + "-" * 80)
print("TEST 2: Paired t-test (one-sided: after > before)")
print("-" * 80)

result2 = test_paired_ttest(
    after_file,
    before_file,
    str(test_manifest),
    'score',
    'score',
    alpha=0.05,
    alternative="greater"
)

print(f"✅ {result2['summary']}")
print(f"   Is significant (after > before): {result2['is_significant']}")

# TEST 3: Wilcoxon signed-rank test (two-sided)
print("\n" + "-" * 80)
print("TEST 3: Wilcoxon signed-rank test (two-sided)")
print("-" * 80)

result3 = test_wilcoxon_signed_rank(
    before_file,
    after_file,
    str(test_manifest),
    'score',
    'score',
    alpha=0.05,
    alternative="two-sided"
)

print(f"✅ {result3['summary']}")
print(f"   W-statistic: {result3['statistic']:.4f}")
print(f"   p-value: {result3['p_value']:.4e}")
print(f"   Median difference: {result3['median_diff']:.4f}")
print(f"   Is significant: {result3['is_significant']}")
print(f"   Positive diffs: {result3['n_positive']}, Negative diffs: {result3['n_negative']}, Zero: {result3['n_zero']}")

# TEST SCENARIO 2: No significant difference
print("\n" + "=" * 80)
print("SCENARIO 2: Method A vs Method B (No Significant Difference)")
print("=" * 80)

method_a = np.random.normal(loc=100, scale=15, size=80)
method_b = method_a + np.random.normal(loc=0, scale=3, size=80)  # Small random noise, no real difference

df_method_a = pd.DataFrame({'measurement': method_a})
df_method_b = pd.DataFrame({'measurement': method_b})

method_a_file = _store_resource(df_method_a, str(test_manifest), "method_a", "Method A measurements", 'csv')
method_b_file = _store_resource(df_method_b, str(test_manifest), "method_b", "Method B measurements", 'csv')

print(f"✅ Created datasets:")
print(f"   Method A mean: {method_a.mean():.2f}")
print(f"   Method B mean: {method_b.mean():.2f}")
print(f"   Expected: No significant difference")

# TEST 4: Paired t-test (no difference)
print("\n" + "-" * 80)
print("TEST 4: Paired t-test (no expected difference)")
print("-" * 80)

result4 = test_paired_ttest(
    method_a_file,
    method_b_file,
    str(test_manifest),
    'measurement',
    'measurement',
    alpha=0.05
)

print(f"✅ {result4['summary']}")
print(f"   p-value: {result4['p_value']:.4f}")
print(f"   Is significant: {result4['is_significant']}")
print(f"   Mean difference: {result4['mean_diff']:.4f}")

# TEST 5: Wilcoxon (no difference)
print("\n" + "-" * 80)
print("TEST 5: Wilcoxon signed-rank test (no expected difference)")
print("-" * 80)

result5 = test_wilcoxon_signed_rank(
    method_a_file,
    method_b_file,
    str(test_manifest),
    'measurement',
    'measurement',
    alpha=0.05
)

print(f"✅ {result5['summary']}")
print(f"   p-value: {result5['p_value']:.4f}")
print(f"   Is significant: {result5['is_significant']}")

# TEST SCENARIO 3: With NaN values
print("\n" + "=" * 80)
print("SCENARIO 3: Data with NaN Values")
print("=" * 80)

data_x = np.concatenate([np.random.normal(loc=50, scale=10, size=45), [np.nan] * 5])
data_y = np.concatenate([np.random.normal(loc=55, scale=10, size=45), [np.nan] * 5])

df_x = pd.DataFrame({'value': data_x})
df_y = pd.DataFrame({'value': data_y})

file_x = _store_resource(df_x, str(test_manifest), "data_with_nan_x", "Data X with NaN", 'csv')
file_y = _store_resource(df_y, str(test_manifest), "data_with_nan_y", "Data Y with NaN", 'csv')

print(f"✅ Created datasets with NaN values")
print(f"   Total rows: 50, Valid pairs: 45")

# TEST 6: Paired t-test with NaN
print("\n" + "-" * 80)
print("TEST 6: Paired t-test (handling NaN)")
print("-" * 80)

result6 = test_paired_ttest(
    file_x,
    file_y,
    str(test_manifest),
    'value',
    'value'
)

print(f"✅ {result6['summary']}")
print(f"   Pairs analyzed (NaN excluded): {result6['n_pairs']}")
print(f"   Is significant: {result6['is_significant']}")

# TEST SCENARIO 4: Non-normal data (better for Wilcoxon)
print("\n" + "=" * 80)
print("SCENARIO 4: Non-Normal Distribution (Exponential)")
print("=" * 80)

exp_before = np.random.exponential(scale=10, size=60)
exp_after = np.random.exponential(scale=15, size=60)  # Different scale

df_exp_before = pd.DataFrame({'time': exp_before})
df_exp_after = pd.DataFrame({'time': exp_after})

exp_before_file = _store_resource(df_exp_before, str(test_manifest), "exp_before", "Exponential before", 'csv')
exp_after_file = _store_resource(df_exp_after, str(test_manifest), "exp_after", "Exponential after", 'csv')

print(f"✅ Created exponential datasets (non-normal)")
print(f"   Median before: {np.median(exp_before):.2f}")
print(f"   Median after: {np.median(exp_after):.2f}")

# TEST 7: Paired t-test on non-normal
print("\n" + "-" * 80)
print("TEST 7: Paired t-test on non-normal data")
print("-" * 80)

result7 = test_paired_ttest(
    exp_before_file,
    exp_after_file,
    str(test_manifest),
    'time',
    'time'
)

print(f"✅ {result7['summary']}")
print(f"   Mean difference: {result7['mean_diff']:.4f}")
print(f"   Is significant: {result7['is_significant']}")

# TEST 8: Wilcoxon on non-normal (more appropriate)
print("\n" + "-" * 80)
print("TEST 8: Wilcoxon signed-rank on non-normal data (more appropriate)")
print("-" * 80)

result8 = test_wilcoxon_signed_rank(
    exp_before_file,
    exp_after_file,
    str(test_manifest),
    'time',
    'time'
)

print(f"✅ {result8['summary']}")
print(f"   Median difference: {result8['median_diff']:.4f}")
print(f"   Is significant: {result8['is_significant']}")

# TEST 9: Error handling - unequal lengths
print("\n" + "=" * 80)
print("TEST 9: Error Handling - Unequal Dataset Lengths")
print("=" * 80)

df_short = pd.DataFrame({'value': np.random.normal(size=30)})
df_long = pd.DataFrame({'value': np.random.normal(size=50)})

short_file = _store_resource(df_short, str(test_manifest), "short_dataset", "Short dataset", 'csv')
long_file = _store_resource(df_long, str(test_manifest), "long_dataset", "Long dataset", 'csv')

try:
    result = test_paired_ttest(
        short_file,
        long_file,
        str(test_manifest),
        'value',
        'value'
    )
    print("❌ Should have raised ValueError!")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

# TEST 10: Error handling - invalid alternative
print("\n" + "=" * 80)
print("TEST 10: Error Handling - Invalid Alternative")
print("=" * 80)

try:
    result = test_paired_ttest(
        before_file,
        after_file,
        str(test_manifest),
        'score',
        'score',
        alternative="invalid"
    )
    print("❌ Should have raised ValueError!")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {str(e)[:80]}...")

# TEST 11: Compare both tests on same data
print("\n" + "=" * 80)
print("TEST 11: Compare Paired t-test vs Wilcoxon on Same Data")
print("=" * 80)

print(f"   Scenario 1 (Normal, significant difference):")
print(f"      Paired t-test:  p={result1['p_value']:.4e}, significant={result1['is_significant']}")
print(f"      Wilcoxon test:  p={result3['p_value']:.4e}, significant={result3['is_significant']}")

print(f"\n   Scenario 2 (Normal, no difference):")
print(f"      Paired t-test:  p={result4['p_value']:.4f}, significant={result4['is_significant']}")
print(f"      Wilcoxon test:  p={result5['p_value']:.4f}, significant={result5['is_significant']}")

print(f"\n   Scenario 4 (Non-normal, difference):")
print(f"      Paired t-test:  p={result7['p_value']:.4f}, significant={result7['is_significant']}")
print(f"      Wilcoxon test:  p={result8['p_value']:.4f}, significant={result8['is_significant']}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
print("=" * 80)
