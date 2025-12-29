import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core.statistics import (
    test_shapiro_wilk,
    test_kolmogorov_smirnov,
    test_anderson_darling
)

print("=" * 80)
print("NORMALITY TESTS")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")

# Create test datasets with different distributions
np.random.seed(42)

# Dataset 1: Normal distribution (should pass normality tests)
normal_data = np.random.normal(loc=100, scale=15, size=500)

# Dataset 2: Uniform distribution (should fail normality tests)
uniform_data = np.random.uniform(low=50, high=150, size=500)

# Dataset 3: Exponential distribution (should fail normality tests)
exponential_data = np.random.exponential(scale=50, size=500)

# Dataset 4: Small sample normal (edge case)
small_normal = np.random.normal(loc=50, scale=10, size=20)

# Dataset 5: Mixed with NaN values
normal_with_nan = np.concatenate([
    np.random.normal(loc=100, scale=15, size=480),
    [np.nan] * 20
])

# Create dataframe
df = pd.DataFrame({
    'normal': normal_data,
    'uniform': uniform_data,
    'exponential': exponential_data,
    'small_normal': list(small_normal) + [np.nan] * 480,
    'normal_with_nan': normal_with_nan,
})

df_filename = _store_resource(df, str(test_manifest), "normality_test_data", "Test data for normality tests", 'csv')
print(f"\n✅ Test data created: {df_filename}")
print(f"   Columns: {list(df.columns)}")

# TEST 1: Shapiro-Wilk on Normal Distribution
print("\n" + "=" * 80)
print("TEST 1: Shapiro-Wilk on Normal Distribution")
print("=" * 80)

result1 = test_shapiro_wilk(
    df_filename,
    str(test_manifest),
    'normal',
    alpha=0.05
)

print(f"✅ {result1['summary']}")
print(f"   Statistic: {result1['statistic']:.4f}")
print(f"   P-value: {result1['p_value']:.4f}")
print(f"   Is normal: {result1['is_normal']}")
print(f"   Interpretation: {result1['interpretation']}")

# TEST 2: Shapiro-Wilk on Uniform Distribution
print("\n" + "=" * 80)
print("TEST 2: Shapiro-Wilk on Uniform Distribution (Should Fail)")
print("=" * 80)

result2 = test_shapiro_wilk(
    df_filename,
    str(test_manifest),
    'uniform',
    alpha=0.05
)

print(f"✅ {result2['summary']}")
print(f"   Statistic: {result2['statistic']:.4f}")
print(f"   P-value: {result2['p_value']:.4e}")
print(f"   Is normal: {result2['is_normal']}")

# TEST 3: Kolmogorov-Smirnov on Normal Distribution
print("\n" + "=" * 80)
print("TEST 3: Kolmogorov-Smirnov on Normal Distribution")
print("=" * 80)

result3 = test_kolmogorov_smirnov(
    df_filename,
    str(test_manifest),
    'normal',
    alpha=0.05
)

print(f"✅ {result3['summary']}")
print(f"   Statistic: {result3['statistic']:.4f}")
print(f"   P-value: {result3['p_value']:.4f}")
print(f"   Is normal: {result3['is_normal']}")
print(f"   Mean: {result3['mean']:.2f}, Std: {result3['std']:.2f}")

# TEST 4: Kolmogorov-Smirnov on Exponential Distribution
print("\n" + "=" * 80)
print("TEST 4: Kolmogorov-Smirnov on Exponential Distribution (Should Fail)")
print("=" * 80)

result4 = test_kolmogorov_smirnov(
    df_filename,
    str(test_manifest),
    'exponential',
    alpha=0.05
)

print(f"✅ {result4['summary']}")
print(f"   Statistic: {result4['statistic']:.4f}")
print(f"   P-value: {result4['p_value']:.4e}")
print(f"   Is normal: {result4['is_normal']}")

# TEST 5: Anderson-Darling on Normal Distribution
print("\n" + "=" * 80)
print("TEST 5: Anderson-Darling on Normal Distribution")
print("=" * 80)

result5 = test_anderson_darling(
    df_filename,
    str(test_manifest),
    'normal',
    significance_level="5%"
)

print(f"✅ {result5['summary']}")
print(f"   Statistic: {result5['statistic']:.4f}")
print(f"   Critical value at 5%: {result5['critical_value']:.4f}")
print(f"   Is normal: {result5['is_normal']}")
print(f"   All critical values: {result5['critical_values']}")

# TEST 6: Anderson-Darling on Uniform Distribution
print("\n" + "=" * 80)
print("TEST 6: Anderson-Darling on Uniform Distribution (Should Fail)")
print("=" * 80)

result6 = test_anderson_darling(
    df_filename,
    str(test_manifest),
    'uniform',
    significance_level="5%"
)

print(f"✅ {result6['summary']}")
print(f"   Statistic: {result6['statistic']:.4f}")
print(f"   Critical value at 5%: {result6['critical_value']:.4f}")
print(f"   Is normal: {result6['is_normal']}")

# TEST 7: Anderson-Darling with different significance levels
print("\n" + "=" * 80)
print("TEST 7: Anderson-Darling with Different Significance Levels")
print("=" * 80)

for level in ["15%", "10%", "5%", "2.5%", "1%"]:
    result = test_anderson_darling(
        df_filename,
        str(test_manifest),
        'normal',
        significance_level=level
    )
    print(f"   α={level:5s}: is_normal={result['is_normal']}, critical={result['critical_value']:.4f}")

# TEST 8: Handling NaN values
print("\n" + "=" * 80)
print("TEST 8: Handling NaN Values")
print("=" * 80)

result8 = test_shapiro_wilk(
    df_filename,
    str(test_manifest),
    'normal_with_nan',
    alpha=0.05
)

print(f"✅ {result8['summary']}")
print(f"   Samples tested (NaN excluded): {result8['n_samples']}")
print(f"   Is normal: {result8['is_normal']}")

# TEST 9: Small sample size
print("\n" + "=" * 80)
print("TEST 9: Small Sample Size")
print("=" * 80)

result9 = test_shapiro_wilk(
    df_filename,
    str(test_manifest),
    'small_normal',
    alpha=0.05
)

print(f"✅ {result9['summary']}")
print(f"   Samples tested: {result9['n_samples']}")
print(f"   Is normal: {result9['is_normal']}")

# TEST 10: Compare all three tests on same data
print("\n" + "=" * 80)
print("TEST 10: Compare All Three Tests on Normal Data")
print("=" * 80)

sw = test_shapiro_wilk(df_filename, str(test_manifest), 'normal')
ks = test_kolmogorov_smirnov(df_filename, str(test_manifest), 'normal')
ad = test_anderson_darling(df_filename, str(test_manifest), 'normal')

print(f"   Shapiro-Wilk:      p={sw['p_value']:.4f}, normal={sw['is_normal']}")
print(f"   Kolmogorov-Smirnov: p={ks['p_value']:.4f}, normal={ks['is_normal']}")
print(f"   Anderson-Darling:  stat={ad['statistic']:.4f}, normal={ad['is_normal']}")

print("\n" + "=" * 80)
print("TEST 11: Compare All Three Tests on Non-Normal Data")
print("=" * 80)

sw = test_shapiro_wilk(df_filename, str(test_manifest), 'exponential')
ks = test_kolmogorov_smirnov(df_filename, str(test_manifest), 'exponential')
ad = test_anderson_darling(df_filename, str(test_manifest), 'exponential')

print(f"   Shapiro-Wilk:      p={sw['p_value']:.4e}, normal={sw['is_normal']}")
print(f"   Kolmogorov-Smirnov: p={ks['p_value']:.4e}, normal={ks['is_normal']}")
print(f"   Anderson-Darling:  stat={ad['statistic']:.4f}, normal={ad['is_normal']}")

# TEST 12: Error handling (invalid column)
print("\n" + "=" * 80)
print("TEST 12: Error Handling (Invalid Column)")
print("=" * 80)

try:
    result = test_shapiro_wilk(
        df_filename,
        str(test_manifest),
        'nonexistent_column'
    )
    print("❌ Should have raised ValueError!")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

# TEST 13: Error handling (invalid significance level for Anderson-Darling)
print("\n" + "=" * 80)
print("TEST 13: Error Handling (Invalid Significance Level)")
print("=" * 80)

try:
    result = test_anderson_darling(
        df_filename,
        str(test_manifest),
        'normal',
        significance_level="3%"
    )
    print("❌ Should have raised ValueError!")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {str(e)[:80]}...")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
print("=" * 80)
