"""
Outlier Detection Usage Examples

This document demonstrates how to use the 5 outlier detection methods
and how to filter datasets based on the pass/fail columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.core.outliers import (
    detect_outliers_zscore,
    detect_outliers_iqr,
    detect_outliers_gesd
)
from molml_mcp.tools.core.dataset_ops import keep_from_dataset
from molml_mcp.infrastructure.resources import _store_resource, _load_resource


def example_basic_usage():
    """
    Example 1: Basic usage - detect outliers and add pass/fail columns
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Outlier Detection")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'compound_id': [f'MOL_{i:03d}' for i in range(100)],
        'pIC50': np.random.normal(6.5, 1.2, 100),
        'logP': np.random.normal(2.5, 0.8, 100),
    })
    
    # Add some outliers
    df.loc[0, 'pIC50'] = 12.0  # Extreme outlier
    df.loc[1, 'logP'] = 8.0    # Extreme outlier
    
    print(f"\nOriginal dataset: {df.shape}")
    print(df.head())
    
    # Store dataset
    filename = _store_resource(
        df,
        str(manifest_path),
        "compounds_with_outliers",
        "Compound dataset with some outliers",
        "csv"
    )
    
    # Detect outliers using Z-score method
    result = detect_outliers_zscore(
        filename,
        str(manifest_path),
        columns=["pIC50", "logP"],
        output_filename="compounds_outliers_detected",
        explanation="Z-score outlier detection on pIC50 and logP",
        threshold=3.0,
    )
    
    print(f"\nOutlier detection completed!")
    print(f"Output: {result['output_filename']}")
    print(f"Outliers per column: {result['outliers_per_column']}")
    print(f"Total rows with outliers: {result['total_outliers']}")
    print(f"New columns: {result['outlier_columns_added']}")
    
    # Load result
    df_result = _load_resource(str(manifest_path), result['output_filename'])
    print(f"\nDataset with outlier flags: {df_result.shape}")
    print(df_result.head())


def example_filtering_outliers():
    """
    Example 2: Detect outliers and filter them out
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Detect and Filter Outliers")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'compound_id': [f'MOL_{i:03d}' for i in range(50)],
        'activity': np.random.normal(100, 15, 50),
    })
    df.loc[0, 'activity'] = 200  # Outlier
    df.loc[1, 'activity'] = 10   # Outlier
    
    print(f"\nOriginal dataset: {df.shape}")
    print(f"Activity range: [{df['activity'].min():.1f}, {df['activity'].max():.1f}]")
    
    filename = _store_resource(
        df,
        str(manifest_path),
        "activity_data",
        "Activity dataset",
        "csv"
    )
    
    # Step 1: Detect outliers using IQR method
    outlier_result = detect_outliers_iqr(
        filename,
        str(manifest_path),
        columns=["activity"],
        output_filename="activity_with_flags",
        explanation="IQR outlier detection",
        multiplier=1.5,
    )
    
    print(f"\nOutliers detected: {outlier_result['outliers_per_column']['activity']}")
    print(f"Bounds: {outlier_result['bounds_per_column']['activity']}")
    
    # Step 2: Filter out outliers using keep_from_dataset
    # Keep only rows where activity_iqr_pass is True
    filter_result = keep_from_dataset(
        outlier_result['output_filename'],
        str(manifest_path),
        "activity_filtered",
        "Dataset with outliers removed",
        column="activity_iqr_pass",
        values=[True]  # Keep only True values (not outliers)
    )
    
    print(f"\nFiltered dataset: {filter_result['n_rows']} rows")
    print(f"Removed: {df.shape[0] - filter_result['n_rows']} outliers")
    
    # Load and display
    df_filtered = _load_resource(str(manifest_path), filter_result['output_filename'])
    print(f"New activity range: [{df_filtered['activity'].min():.1f}, {df_filtered['activity'].max():.1f}]")


def example_multiple_methods():
    """
    Example 3: Apply multiple outlier detection methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Outlier Detection Methods")
    print("="*70)
    
    manifest_path = Path(__file__).parent / "data" / "test_manifest.json"
    
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'value': np.random.normal(50, 10, 80)
    })
    df.loc[0, 'value'] = 120  # Extreme outlier
    df.loc[1, 'value'] = 110  # Moderate outlier
    df.loc[2, 'value'] = 5    # Extreme outlier
    
    print(f"\nDataset: {df.shape}")
    print(f"Value range: [{df['value'].min():.1f}, {df['value'].max():.1f}]")
    
    filename = _store_resource(
        df,
        str(manifest_path),
        "multi_method_test",
        "Dataset for multiple detection methods",
        "csv"
    )
    
    # Method 1: Z-score
    result_z = detect_outliers_zscore(
        filename, str(manifest_path), ["value"],
        "multi_zscore", "Z-score detection",
        threshold=3.0
    )
    
    # Method 2: IQR (on original data)
    result_iqr = detect_outliers_iqr(
        filename, str(manifest_path), ["value"],
        "multi_iqr", "IQR detection",
        multiplier=1.5
    )
    
    # Method 3: GESD (on original data)
    result_gesd = detect_outliers_gesd(
        filename, str(manifest_path), ["value"],
        "multi_gesd", "GESD detection",
        max_outliers=5, alpha=0.05
    )
    
    print(f"\nResults comparison:")
    print(f"Z-score:  {result_z['outliers_per_column']['value']} outliers")
    print(f"IQR:      {result_iqr['outliers_per_column']['value']} outliers")
    print(f"GESD:     {result_gesd['outliers_per_column']['value']['n_outliers']} outliers")
    
    print("\n💡 Different methods may detect different numbers of outliers!")
    print("   - Z-score: Assumes normality, fixed threshold")
    print("   - IQR: Non-parametric, robust to distribution")
    print("   - GESD: Statistical test, detects multiple iteratively")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OUTLIER DETECTION USAGE EXAMPLES")
    print("="*70)
    
    example_basic_usage()
    example_filtering_outliers()
    example_multiple_methods()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nAvailable Methods:")
    print("1. detect_outliers_zscore() - ±3σ threshold (parametric)")
    print("2. detect_outliers_modified_zscore() - MAD-based (robust)")
    print("3. detect_outliers_iqr() - Q1-1.5×IQR to Q3+1.5×IQR (non-parametric)")
    print("4. detect_outliers_grubbs() - Single outlier test (parametric)")
    print("5. detect_outliers_gesd() - Multiple outliers test (parametric)")
    print("\nAll methods add boolean pass/fail columns:")
    print("  - False = outlier detected")
    print("  - True = not an outlier")
    print("\nUse keep_from_dataset() to filter based on pass/fail columns!")
