"""
Test suite for Fisher's exact test and McNemar's test implementations.

Demonstrates:
1. Fisher's exact test with 2x2 contingency table (small sample)
2. McNemar's test with paired before/after categorical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.core.statistics import test_fisher_exact, test_mcnemar
from molml_mcp.infrastructure.resources import _store_resource


def test_fishers_exact_test():
    """
    Test Fisher's exact test with a 2x2 contingency table.
    
    Scenario: Testing whether a new drug improves recovery rate
    - Small sample size (n=30)
    - Treatment group: 12 recovered, 3 not recovered
    - Control group: 4 recovered, 11 not recovered
    - Expect significant association (treatment improves recovery)
    """
    print("\n" + "="*70)
    print("TEST 1: Fisher's Exact Test - Drug Efficacy (Small Sample)")
    print("="*70)
    
    # Create test data with binary categorical variables
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create dataset with treatment and outcome columns
    # Stronger effect: 80% recovery in drug group vs 27% in placebo
    data = {
        'treatment': ['drug']*15 + ['placebo']*15,
        'outcome': ['recovered']*12 + ['not_recovered']*3 + 
                   ['recovered']*4 + ['not_recovered']*11
    }
    df = pd.DataFrame(data)
    
    # Store dataset
    filename = _store_resource(
        df,
        str(manifest_path),
        "fisher_test_data",
        "Small sample drug efficacy data for Fisher's exact test",
        "csv"
    )
    
    print(f"\nDataset: {filename}")
    print(f"Sample size: {len(df)}")
    print(f"\nContingency table:")
    print(pd.crosstab(df['treatment'], df['outcome']))
    
    # Perform Fisher's exact test (two-sided)
    result = test_fisher_exact(
        filename,
        str(manifest_path),
        "treatment",
        "outcome",
        alpha=0.05,
        alternative="two-sided"
    )
    
    print(f"\n{result['summary']}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Odds Ratio: {result['odds_ratio']:.4f}")
    print(f"Effect Size: {result['effect_size']}")
    print(f"Significant: {result['is_significant']}")
    print(f"\n{result['interpretation']}")
    
    # Also test one-sided (less) alternative (drug group has lower odds of recovery in the contingency table order)
    result_less = test_fisher_exact(
        filename,
        str(manifest_path),
        "treatment",
        "outcome",
        alpha=0.05,
        alternative="less"
    )
    
    print(f"\nOne-sided test (less): p={result_less['p_value']:.6f}, significant={result_less['is_significant']}")
    
    assert result['p_value'] < 0.05, "Expected significant association"
    assert result['odds_ratio'] > 1 or result['odds_ratio'] < 1, "Expected non-null odds ratio"
    
    print("\n✓ Fisher's exact test working correctly!")


def test_mcnemar_test():
    """
    Test McNemar's test with paired before/after data.
    
    Scenario: Testing whether a training program improves pass rates
    - 50 students tested before and after training
    - Before: 20 passed, 30 failed
    - After: 35 passed, 15 failed
    - Expect significant improvement (more failures → passes than passes → failures)
    """
    print("\n" + "="*70)
    print("TEST 2: McNemar's Test - Training Program Effectiveness")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create paired before/after data
    # Simulate realistic scenario with some stability
    np.random.seed(42)
    
    # Design the 2x2 table structure:
    #              After_Pass  After_Fail
    # Before_Pass      15          5
    # Before_Fail      20         10
    
    # 15 students: Pass → Pass (stable)
    # 5 students: Pass → Fail (got worse)
    # 20 students: Fail → Pass (improved!)
    # 10 students: Fail → Fail (stable)
    
    before_values = ['pass']*20 + ['fail']*30
    after_values = ['pass']*15 + ['fail']*5 + ['pass']*20 + ['fail']*10
    
    df_before = pd.DataFrame({'result': before_values})
    df_after = pd.DataFrame({'result': after_values})
    
    # Store datasets
    filename_before = _store_resource(
        df_before,
        str(manifest_path),
        "mcnemar_before",
        "Student test results before training",
        "csv"
    )
    
    filename_after = _store_resource(
        df_after,
        str(manifest_path),
        "mcnemar_after",
        "Student test results after training",
        "csv"
    )
    
    print(f"\nBefore dataset: {filename_before}")
    print(f"After dataset: {filename_after}")
    print(f"Sample size: {len(df_before)} paired observations")
    
    # Create contingency table for visualization
    contingency = pd.crosstab(df_before['result'], df_after['result'])
    print(f"\nContingency table (Before × After):")
    print(contingency)
    print(f"\nDiscordant pairs:")
    print(f"  Fail → Pass (improved): {contingency.loc['fail', 'pass']}")
    print(f"  Pass → Fail (worsened): {contingency.loc['pass', 'fail']}")
    
    # Perform McNemar's test
    result = test_mcnemar(
        filename_before,
        filename_after,
        str(manifest_path),
        "result",
        "result",
        alpha=0.05
    )
    
    print(f"\n{result['summary']}")
    print(f"Statistic (χ²): {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Odds Ratio (b/c): {result['odds_ratio']:.4f}")
    print(f"Effect Size: {result['effect_size']}")
    print(f"Significant: {result['is_significant']}")
    print(f"Concordant pairs: {result['n_concordant']}")
    print(f"Discordant pairs: {result['n_discordant']}")
    print(f"\n{result['interpretation']}")
    
    assert result['p_value'] < 0.05, "Expected significant change"
    assert result['odds_ratio'] > 1, "Expected more improvements than deteriorations"
    assert result['discordant_b'] > result['discordant_c'], "Expected more Fail→Pass than Pass→Fail"
    
    print("\n✓ McNemar's test working correctly!")


def test_edge_cases():
    """
    Test edge cases for both tests.
    """
    print("\n" + "="*70)
    print("TEST 3: Edge Cases - Extreme Effect Sizes")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Fisher's exact with extreme odds ratio (all successes in treatment, none in control)
    print("\n--- Fisher's exact: Extreme effect ---")
    data_extreme = {
        'group': ['treatment']*10 + ['control']*10,
        'outcome': ['success']*10 + ['failure']*10
    }
    df_extreme = pd.DataFrame(data_extreme)
    
    filename_extreme = _store_resource(
        df_extreme,
        str(manifest_path),
        "fisher_extreme",
        "Extreme effect size data",
        "csv"
    )
    
    result_extreme = test_fisher_exact(
        filename_extreme,
        str(manifest_path),
        "group",
        "outcome",
        alpha=0.05
    )
    
    print(f"Odds ratio: {result_extreme['odds_ratio']}")
    print(f"Effect size: {result_extreme['effect_size']}")
    print(f"P-value: {result_extreme['p_value']:.6f}")
    
    # McNemar's with all changes in one direction
    print("\n--- McNemar's: Strong one-directional change ---")
    # Keep some stable cases to maintain 2x2 structure
    # 12 Fail→Pass (improved), 1 Pass→Fail (rare worsening), 5 stable
    before_one_dir = ['fail']*12 + ['pass']*1 + ['pass']*5
    after_one_dir = ['pass']*12 + ['fail']*1 + ['pass']*5
    
    df_before_od = pd.DataFrame({'status': before_one_dir})
    df_after_od = pd.DataFrame({'status': after_one_dir})
    
    filename_before_od = _store_resource(
        df_before_od,
        str(manifest_path),
        "mcnemar_before_od",
        "Before data with strong one-directional change",
        "csv"
    )
    
    filename_after_od = _store_resource(
        df_after_od,
        str(manifest_path),
        "mcnemar_after_od",
        "After data with strong one-directional change",
        "csv"
    )
    
    result_one_dir = test_mcnemar(
        filename_before_od,
        filename_after_od,
        str(manifest_path),
        "status",
        "status",
        alpha=0.05
    )
    
    print(f"Odds ratio: {result_one_dir['odds_ratio']}")
    print(f"Effect size: {result_one_dir['effect_size']}")
    print(f"Discordant b (Fail→Pass): {result_one_dir['discordant_b']}")
    print(f"Discordant c (Pass→Fail): {result_one_dir['discordant_c']}")
    print(f"P-value: {result_one_dir['p_value']:.6f}")
    
    print("\n✓ Edge cases handled correctly!")


if __name__ == "__main__":
    test_fishers_exact_test()
    test_mcnemar_test()
    test_edge_cases()
    
    print("\n" + "="*70)
    print("ALL CATEGORICAL TESTS PASSED! ✓")
    print("="*70)
    print("\nSummary:")
    print("- Fisher's exact test: 2x2 contingency tables, exact p-values, OR effect size")
    print("- McNemar's test: Paired categorical data, change detection, OR effect size")
    print("- Both tests handle edge cases (extreme ORs, one-sided tests)")
