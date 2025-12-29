"""
Test suite for outlier detection methods.

Demonstrates all 5 outlier detection approaches:
1. Z-score method (±3σ)
2. Modified Z-score method (MAD-based, robust)
3. IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
4. Grubbs' test (single outlier, parametric)
5. Generalized ESD test (multiple outliers, parametric)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.core.outliers import (
    detect_outliers_zscore,
    detect_outliers_modified_zscore,
    detect_outliers_iqr,
    detect_outliers_grubbs,
    detect_outliers_gesd
)
from molml_mcp.infrastructure.resources import _store_resource


def create_test_dataset_with_outliers():
    """
    Create a test dataset with known outliers.
    
    Returns dataset with:
    - normal_data: ~N(50, 10) with 3 extreme outliers
    - skewed_data: Exponential distribution with 2 extreme values
    """
    np.random.seed(42)
    
    # Generate normal data with outliers
    n = 100
    normal_data = np.random.normal(50, 10, n)
    # Add 3 extreme outliers
    normal_data[0] = 100  # 5 std deviations above mean
    normal_data[1] = 0    # 5 std deviations below mean
    normal_data[2] = 95   # 4.5 std deviations above mean
    
    # Generate skewed data with outliers
    skewed_data = np.random.exponential(20, n)
    # Add 2 extreme outliers
    skewed_data[50] = 150  # Very high value
    skewed_data[51] = 140  # Another high value
    
    df = pd.DataFrame({
        'normal_data': normal_data,
        'skewed_data': skewed_data,
        'id': range(n)
    })
    
    return df


def test_zscore_method():
    """
    Test Z-score outlier detection.
    
    Expected: Should detect extreme values beyond ±3σ
    """
    print("\n" + "="*70)
    print("TEST 1: Z-Score Method (±3σ threshold)")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    df = create_test_dataset_with_outliers()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Normal data - Mean: {df['normal_data'].mean():.2f}, Std: {df['normal_data'].std():.2f}")
    print(f"Known outliers in normal_data: indices [0, 1, 2] with values {df.loc[[0,1,2], 'normal_data'].values}")
    
    # Store dataset
    filename = _store_resource(
        df,
        str(manifest_path),
        "outlier_test_data",
        "Test dataset with known outliers",
        "csv"
    )
    
    # Run Z-score detection
    result = detect_outliers_zscore(
        filename,
        str(manifest_path),
        ["normal_data", "skewed_data"],
        "zscore_filtered",
        "Z-score outlier detection test",
        threshold=3.0,
    )
    
    print(f"\n{result['summary']}")
    print(f"Outliers per column: {result['outliers_per_column']}")
    print(f"Total rows with outliers: {result['total_outliers']}")
    print(f"Columns added: {result['outlier_columns_added']}")
    
    # Verify outliers detected in normal_data
    from molml_mcp.infrastructure.resources import _load_resource
    df_result = _load_resource(str(manifest_path), result['output_filename'])
    
    outliers_normal = df_result[~df_result['normal_data_zscore_pass']]
    print(f"\nDetected outliers in normal_data:")
    print(outliers_normal[['id', 'normal_data', 'normal_data_zscore_pass']])
    
    assert result['outliers_per_column']['normal_data'] >= 2, "Should detect at least 2 outliers in normal_data"
    print("\n✓ Z-score method working correctly!")


def test_modified_zscore_method():
    """
    Test Modified Z-score outlier detection (robust).
    
    Expected: More robust to existing outliers, uses MAD instead of std
    """
    print("\n" + "="*70)
    print("TEST 2: Modified Z-Score Method (MAD-based, threshold=3.5)")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    df = create_test_dataset_with_outliers()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Skewed data - Median: {df['skewed_data'].median():.2f}, MAD: {np.median(np.abs(df['skewed_data'] - df['skewed_data'].median())):.2f}")
    print(f"Known extreme values in skewed_data: indices [50, 51] with values {df.loc[[50,51], 'skewed_data'].values}")
    
    filename = _store_resource(
        df,
        str(manifest_path),
        "outlier_test_modified",
        "Test dataset for modified Z-score",
        "csv"
    )
    
    # Run Modified Z-score detection
    result = detect_outliers_modified_zscore(
        filename,
        str(manifest_path),
        ["skewed_data"],
        "modified_zscore_filtered",
        "Modified Z-score test on skewed data",
        threshold=3.5,
    )
    
    print(f"\n{result['summary']}")
    print(f"Outliers detected: {result['outliers_per_column']}")
    
    from molml_mcp.infrastructure.resources import _load_resource
    df_result = _load_resource(str(manifest_path), result['output_filename'])
    
    outliers_skewed = df_result[~df_result['skewed_data_modified_zscore_pass']]
    print(f"\nDetected outliers in skewed_data:")
    print(outliers_skewed[['id', 'skewed_data', 'skewed_data_modified_zscore_pass']].head(10))
    
    assert result['outliers_per_column']['skewed_data'] >= 2, "Should detect at least 2 outliers"
    print("\n✓ Modified Z-score method working correctly!")


def test_iqr_method():
    """
    Test IQR outlier detection.
    
    Expected: Non-parametric, detects values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
    """
    print("\n" + "="*70)
    print("TEST 3: IQR Method (1.5× multiplier)")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    df = create_test_dataset_with_outliers()
    
    print(f"\nDataset shape: {df.shape}")
    q1 = df['normal_data'].quantile(0.25)
    q3 = df['normal_data'].quantile(0.75)
    iqr = q3 - q1
    print(f"Normal data - Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"Expected bounds: [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}]")
    
    filename = _store_resource(
        df,
        str(manifest_path),
        "outlier_test_iqr",
        "Test dataset for IQR method",
        "csv"
    )
    
    # Run IQR detection
    result = detect_outliers_iqr(
        filename,
        str(manifest_path),
        ["normal_data"],
        "iqr_filtered",
        "IQR outlier detection test",
        multiplier=1.5,
    )
    
    print(f"\n{result['summary']}")
    print(f"Outliers detected: {result['outliers_per_column']}")
    print(f"Bounds per column: {result['bounds_per_column']}")
    
    from molml_mcp.infrastructure.resources import _load_resource
    df_result = _load_resource(str(manifest_path), result['output_filename'])
    
    outliers = df_result[~df_result['normal_data_iqr_pass']]
    print(f"\nDetected outliers:")
    print(outliers[['id', 'normal_data', 'normal_data_iqr_pass']])
    
    assert result['outliers_per_column']['normal_data'] >= 2, "Should detect at least 2 outliers"
    print("\n✓ IQR method working correctly!")


def test_grubbs_test():
    """
    Test Grubbs' test for single outlier.
    
    Expected: Detects the single most extreme outlier (parametric)
    """
    print("\n" + "="*70)
    print("TEST 4: Grubbs' Test (single outlier, α=0.05)")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create dataset with one clear outlier
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, 50)
    normal_data[0] = 100  # Extreme outlier
    
    df = pd.DataFrame({
        'value': normal_data,
        'id': range(50)
    })
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")
    print(f"Most extreme value: {df['value'].max():.2f} at index {df['value'].idxmax()}")
    
    filename = _store_resource(
        df,
        str(manifest_path),
        "outlier_test_grubbs",
        "Test dataset for Grubbs test",
        "csv"
    )
    
    # Run Grubbs' test
    result = detect_outliers_grubbs(
        filename,
        str(manifest_path),
        ["value"],
        "grubbs_filtered",
        "Grubbs test for single outlier",
        alpha=0.05,
    )
    
    print(f"\n{result['summary']}")
    print(f"Outlier detection results:")
    for col, info in result['outliers_per_column'].items():
        print(f"  {col}: {info}")
    
    from molml_mcp.infrastructure.resources import _load_resource
    df_result = _load_resource(str(manifest_path), result['output_filename'])
    
    outliers = df_result[~df_result['value_grubbs_pass']]
    print(f"\nDetected outlier:")
    print(outliers[['id', 'value', 'value_grubbs_pass']])
    
    assert result['outliers_per_column']['value']['outlier_detected'], "Should detect the extreme outlier"
    assert result['outliers_per_column']['value']['n_outliers'] == 1, "Grubbs detects only one outlier"
    print("\n✓ Grubbs' test working correctly!")


def test_gesd_test():
    """
    Test Generalized ESD test for multiple outliers.
    
    Expected: Detects up to max_outliers extreme values (parametric)
    """
    print("\n" + "="*70)
    print("TEST 5: Generalized ESD Test (max=5 outliers, α=0.05)")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create dataset with multiple outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, 60)
    # Add 4 outliers at different extremes
    normal_data[0] = 85
    normal_data[1] = 15
    normal_data[2] = 80
    normal_data[3] = 20
    
    df = pd.DataFrame({
        'value': normal_data,
        'id': range(60)
    })
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")
    print(f"Extreme values added at indices [0, 1, 2, 3]: {normal_data[[0,1,2,3]]}")
    
    filename = _store_resource(
        df,
        str(manifest_path),
        "outlier_test_gesd",
        "Test dataset for GESD test",
        "csv"
    )
    
    # Run Generalized ESD test
    result = detect_outliers_gesd(
        filename,
        str(manifest_path),
        ["value"],
        "gesd_filtered",
        "GESD test for multiple outliers",
        max_outliers=5,
        alpha=0.05,
    )
    
    print(f"\n{result['summary']}")
    print(f"Outlier detection results:")
    for col, info in result['outliers_per_column'].items():
        print(f"  {col}:")
        for key, val in info.items():
            if key in ['outlier_indices', 'outlier_values']:
                print(f"    {key}: {val[:5] if len(str(val)) > 50 else val}")
            else:
                print(f"    {key}: {val}")
    
    from molml_mcp.infrastructure.resources import _load_resource
    df_result = _load_resource(str(manifest_path), result['output_filename'])
    
    outliers = df_result[~df_result['value_gesd_pass']]
    print(f"\nDetected outliers:")
    print(outliers[['id', 'value', 'value_gesd_pass']])
    
    assert result['outliers_per_column']['value']['n_outliers'] >= 3, "Should detect at least 3 outliers"
    assert result['outliers_per_column']['value']['n_outliers'] <= 5, "Should not exceed max_outliers"
    print("\n✓ Generalized ESD test working correctly!")


if __name__ == "__main__":
    test_zscore_method()
    test_modified_zscore_method()
    test_iqr_method()
    test_grubbs_test()
    test_gesd_test()
    
    print("\n" + "="*70)
    print("ALL OUTLIER DETECTION TESTS PASSED! ✓")
    print("="*70)
    print("\nSummary:")
    print("- Z-score: Detects values beyond ±3σ (assumes normality)")
    print("- Modified Z-score: Uses MAD, robust to existing outliers")
    print("- IQR: Non-parametric, Q1 - 1.5×IQR to Q3 + 1.5×IQR")
    print("- Grubbs: Statistical test for single outlier (parametric)")
    print("- GESD: Detects multiple outliers iteratively (parametric)")
    print("\nAll methods add boolean pass/fail columns for filtering!")
