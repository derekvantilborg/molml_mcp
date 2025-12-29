"""
Test script to showcase effect size calculations.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core.statistics import (
    test_independent_ttest,
    test_mann_whitney_u,
    test_one_way_anova,
    test_kruskal_wallis,
    test_chi_square,
)

# Setup
test_manifest = Path("tests/data/test_manifest.json")
TEST_MANIFEST = str(test_manifest)
np.random.seed(42)

print("=" * 80)
print("EFFECT SIZE CALCULATIONS")
print("=" * 80)

# ============================================================================
# TEST 1: Cohen's d - Small, Medium, Large effects
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Cohen's d Effect Sizes")
print("=" * 80)

# Small effect (d ≈ 0.2)
df_control = pd.DataFrame({"value": np.random.normal(100, 15, 50)})
df_small = pd.DataFrame({"value": np.random.normal(103, 15, 50)})
control_file = _store_resource(df_control, TEST_MANIFEST, "control", "Control group", "csv")
small_file = _store_resource(df_small, TEST_MANIFEST, "small_effect", "Small effect", "csv")

result = test_independent_ttest(control_file, small_file, TEST_MANIFEST, "value", "value")
print(f"Small effect:")
print(f"  Cohen's d: {result['cohens_d']:.4f}")
print(f"  Effect size: {result['effect_size']}")
print(f"  p-value: {result['p_value']:.4f}")

# Medium effect (d ≈ 0.5)
df_medium = pd.DataFrame({"value": np.random.normal(107.5, 15, 50)})
medium_file = _store_resource(df_medium, TEST_MANIFEST, "medium_effect", "Medium effect", "csv")

result = test_independent_ttest(control_file, medium_file, TEST_MANIFEST, "value", "value")
print(f"\nMedium effect:")
print(f"  Cohen's d: {result['cohens_d']:.4f}")
print(f"  Effect size: {result['effect_size']}")
print(f"  p-value: {result['p_value']:.4f}")

# Large effect (d ≈ 0.8+)
df_large = pd.DataFrame({"value": np.random.normal(112, 15, 50)})
large_file = _store_resource(df_large, TEST_MANIFEST, "large_effect", "Large effect", "csv")

result = test_independent_ttest(control_file, large_file, TEST_MANIFEST, "value", "value")
print(f"\nLarge effect:")
print(f"  Cohen's d: {result['cohens_d']:.4f}")
print(f"  Effect size: {result['effect_size']}")
print(f"  p-value: {result['p_value']:.4f}")

print(f"\nNote: Cohen's d thresholds - small: 0.2, medium: 0.5, large: 0.8")

# ============================================================================
# TEST 2: Cliff's Delta - Non-parametric effect size
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Cliff's Delta Effect Sizes")
print("=" * 80)

# Create ordinal data (ranks)
df_rank_low = pd.DataFrame({"rank": np.random.randint(1, 5, 40)})
df_rank_high = pd.DataFrame({"rank": np.random.randint(3, 8, 40)})

rank_low_file = _store_resource(df_rank_low, TEST_MANIFEST, "rank_low", "Low ranks", "csv")
rank_high_file = _store_resource(df_rank_high, TEST_MANIFEST, "rank_high", "High ranks", "csv")

result = test_mann_whitney_u(rank_low_file, rank_high_file, TEST_MANIFEST, "rank", "rank")
print(f"Cliff's delta: {result['cliffs_delta']:.4f}")
print(f"Effect size: {result['effect_size']}")
print(f"Interpretation: {result['interpretation']}")
print(f"\nNote: Cliff's delta ranges from -1 to 1")
print(f"  -1: All values in A < all values in B")
print(f"   0: Random overlap")
print(f"  +1: All values in A > all values in B")

# ============================================================================
# TEST 3: Eta-squared - ANOVA effect size
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Eta-Squared (η²) for ANOVA")
print("=" * 80)

# Create groups with large variance between means
df_group1 = pd.DataFrame({"score": np.random.normal(50, 10, 30)})
df_group2 = pd.DataFrame({"score": np.random.normal(65, 10, 30)})
df_group3 = pd.DataFrame({"score": np.random.normal(80, 10, 30)})

group1_file = _store_resource(df_group1, TEST_MANIFEST, "anova_g1", "ANOVA group 1", "csv")
group2_file = _store_resource(df_group2, TEST_MANIFEST, "anova_g2", "ANOVA group 2", "csv")
group3_file = _store_resource(df_group3, TEST_MANIFEST, "anova_g3", "ANOVA group 3", "csv")

result = test_one_way_anova(
    [group1_file, group2_file, group3_file],
    TEST_MANIFEST,
    ["score", "score", "score"]
)

print(f"Eta-squared (η²): {result['eta_squared']:.4f}")
print(f"Effect size: {result['effect_size']}")
print(f"F-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Group means: {result['group_means']}")
print(f"\nNote: η² represents proportion of variance explained by groups")
print(f"  < 0.01: negligible, < 0.06: small, < 0.14: medium, ≥ 0.14: large")

# ============================================================================
# TEST 4: Epsilon-squared - Kruskal-Wallis effect size
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Epsilon-Squared (ε²) for Kruskal-Wallis")
print("=" * 80)

result = test_kruskal_wallis(
    [group1_file, group2_file, group3_file],
    TEST_MANIFEST,
    ["score", "score", "score"]
)

print(f"Epsilon-squared (ε²): {result['epsilon_squared']:.4f}")
print(f"Effect size: {result['effect_size']}")
print(f"H-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Group medians: {result['group_medians']}")
print(f"\nNote: ε² is non-parametric alternative to η² (similar thresholds)")

# ============================================================================
# TEST 5: Cramér's V - Chi-square effect size
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Cramér's V for Chi-Square")
print("=" * 80)

# Create categorical data with association
np.random.seed(42)
# Treatment strongly affects outcome
treatment = ["Control"] * 50 + ["Treatment"] * 50
outcome = (
    ["Success"] * 15 + ["Failure"] * 35 +  # Control: low success
    ["Success"] * 40 + ["Failure"] * 10    # Treatment: high success
)

df_categorical = pd.DataFrame({
    "treatment": treatment,
    "outcome": outcome
})

categorical_file = _store_resource(
    df_categorical, TEST_MANIFEST, "categorical", "Categorical data", "csv"
)

result = test_chi_square(categorical_file, TEST_MANIFEST, "treatment", "outcome")

print(f"Cramér's V: {result['cramers_v']:.4f}")
print(f"Effect size: {result['effect_size']}")
print(f"Chi-square: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Contingency table:")
for row_label, row_data in result['contingency_table'].items():
    print(f"  {row_label}: {row_data}")
print(f"\nInterpretation: {result['interpretation']}")
print(f"\nNote: Cramér's V ranges from 0 to 1")
print(f"  < 0.1: negligible, < 0.3: small, < 0.5: medium, ≥ 0.5: large")

print("\n" + "=" * 80)
print("ALL EFFECT SIZE TESTS COMPLETED!")
print("=" * 80)
print("\nSummary of Effect Size Metrics Implemented:")
print("  ✓ Cohen's d - standardized mean difference (t-tests)")
print("  ✓ Cliff's delta - non-parametric effect size (Mann-Whitney U)")
print("  ✓ Eta-squared (η²) - variance explained (ANOVA)")
print("  ✓ Epsilon-squared (ε²) - non-parametric variance (Kruskal-Wallis)")
print("  ✓ Cramér's V - association strength (Chi-square)")
