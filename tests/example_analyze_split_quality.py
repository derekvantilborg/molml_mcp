"""
Example usage of analyze_split_quality() function.

This demonstrates how to run a comprehensive data splitting quality analysis
on molecular datasets.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.reports.data_splitting import analyze_split_quality
from molml_mcp.infrastructure.resources import _store_resource, _load_resource


# Example manifest path
MANIFEST_PATH = Path(__file__).parent / 'data' / 'test_manifest.json'


def example_basic_usage():
    """Basic usage with train/test split."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage (Train/Test Split)")
    print("="*70 + "\n")
    
    # Create example datasets
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1', 'CC(=O)C', 'CCO', 'CCN', 'c1ccccc1C',
            'c1cccnc1', 'CCC(=O)C', 'CCCO', 'CCCN', 'c1ccc(C)cc1'
        ] * 5,  # 50 molecules
        'activity': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5
    })
    
    test_data = pd.DataFrame({
        'smiles': [
            'c1ccc(CC)cc1', 'CC(=O)CC', 'CCCCO', 'CC(C)N', 'c1ccc(O)cc1'
        ] * 4,  # 20 molecules
        'activity': [1, 0, 1, 0, 1] * 4
    })
    
    # Store datasets
    train_file = _store_resource(train_data, str(MANIFEST_PATH), 'example_train', 'Training data', 'csv')
    test_file = _store_resource(test_data, str(MANIFEST_PATH), 'example_test', 'Test data', 'csv')
    
    # Run comprehensive analysis
    result = analyze_split_quality(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(MANIFEST_PATH),
        smiles_col='smiles',
        label_col='activity',
        output_filename='split_quality_report'
    )
    
    print(f"Report saved to: {result['output_filename']}")
    print(f"Overall severity: {result['overall_severity']}")
    print(f"Severity breakdown: {result['severity_summary']}")
    print(f"Execution time: {result['execution_time_seconds']:.2f}s")
    print(f"\nKey issues found:")
    for issue, count in result['issues_found'].items():
        if isinstance(count, (int, float)) and count > 0:
            print(f"  - {issue}: {count}")
    
    # Load and display part of the report
    report = _load_resource(str(MANIFEST_PATH), result['output_filename'])
    print(f"\nReport contains {len(report)} top-level sections")
    print(f"Sections: {list(report.keys())}")


def example_with_validation():
    """Usage with train/test/val split."""
    print("\n" + "="*70)
    print("Example 2: Train/Test/Val Split")
    print("="*70 + "\n")
    
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO', 'CCN'] * 10,
        'activity': [0, 1, 0, 1] * 10
    })
    
    test_data = pd.DataFrame({
        'smiles': ['c1ccc(C)cc1', 'CCC(=O)C', 'CCCO', 'CCCN'] * 5,
        'activity': [1, 0, 1, 0] * 5
    })
    
    val_data = pd.DataFrame({
        'smiles': ['c1ccc(CC)cc1', 'CC(=O)CC', 'CCCCO', 'CC(C)N'] * 3,
        'activity': [0, 1, 0, 1] * 3
    })
    
    # Store datasets
    train_file = _store_resource(train_data, str(MANIFEST_PATH), 'val_train', 'Training data', 'csv')
    test_file = _store_resource(test_data, str(MANIFEST_PATH), 'val_test', 'Test data', 'csv')
    val_file = _store_resource(val_data, str(MANIFEST_PATH), 'val_val', 'Validation data', 'csv')
    
    # Run analysis with validation split
    result = analyze_split_quality(
        train_path=train_file,
        test_path=test_file,
        val_path=val_file,
        project_manifest_path=str(MANIFEST_PATH),
        smiles_col='smiles',
        label_col='activity',
        output_filename='three_way_split_report'
    )
    
    print(f"Report saved to: {result['output_filename']}")
    print(f"Overall severity: {result['overall_severity']}")
    print(f"Checks performed: {result['n_checks_performed']}")
    print(f"Execution time: {result['execution_time_seconds']:.2f}s")


def example_problematic_split():
    """Example with data leakage issues."""
    print("\n" + "="*70)
    print("Example 3: Problematic Split (With Data Leakage)")
    print("="*70 + "\n")
    
    # Create dataset with exact duplicates (CRITICAL issue)
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1', 'CC(=O)C', 'CCO', 'CCN', 'c1ccccc1C',
            'c1cccnc1', 'CCC(=O)C', 'CCCO'
        ],
        'activity': [0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    test_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # DUPLICATE from train!
            'CC(=O)C',   # DUPLICATE from train!
            'c1ccc(O)cc1'
        ],
        'activity': [1, 0, 1]
    })
    
    # Store datasets
    train_file = _store_resource(train_data, str(MANIFEST_PATH), 'problem_train', 'Training data', 'csv')
    test_file = _store_resource(test_data, str(MANIFEST_PATH), 'problem_test', 'Test data', 'csv')
    
    # Run analysis
    result = analyze_split_quality(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(MANIFEST_PATH),
        smiles_col='smiles',
        label_col='activity',
        output_filename='problematic_split_report'
    )
    
    print(f"Report saved to: {result['output_filename']}")
    print(f"⚠️  Overall severity: {result['overall_severity']} (CRITICAL!)")
    print(f"Severity breakdown: {result['severity_summary']}")
    print(f"\n🔴 Issues detected:")
    print(f"  - Exact duplicates: {result['issues_found']['exact_duplicates']}")
    print(f"  - High similarity pairs: {result['issues_found']['high_similarity_pairs']}")
    print(f"  - Scaffold overlap: {result['issues_found']['scaffold_overlap_pct']:.1f}%")
    
    # Load report and show duplicate details
    report = _load_resource(str(MANIFEST_PATH), result['output_filename'])
    if report['exact_duplicates']['train_test_duplicates']['n_duplicates'] > 0:
        print(f"\n  Duplicate molecules found:")
        for example in report['exact_duplicates']['train_test_duplicates']['examples'][:3]:
            print(f"    - {example['smiles']}")


def example_custom_parameters():
    """Example with custom parameters."""
    print("\n" + "="*70)
    print("Example 4: Custom Parameters")
    print("="*70 + "\n")
    
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'] * 20,
        'activity': [0, 1, 0] * 20
    })
    
    test_data = pd.DataFrame({
        'smiles': ['c1ccc(C)cc1', 'CCC(=O)C', 'CCCO'] * 10,
        'activity': [1, 0, 1] * 10
    })
    
    train_file = _store_resource(train_data, str(MANIFEST_PATH), 'custom_train', 'Training', 'csv')
    test_file = _store_resource(test_data, str(MANIFEST_PATH), 'custom_test', 'Test', 'csv')
    
    # Run with custom parameters
    result = analyze_split_quality(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(MANIFEST_PATH),
        smiles_col='smiles',
        label_col='activity',
        output_filename='custom_params_report',
        # Custom parameters
        similarity_threshold=0.85,  # More lenient (default 0.9)
        alpha=0.01,  # More strict (default 0.05)
        max_examples=5,  # Fewer examples (default 10)
        min_occurrence_threshold=3  # Higher threshold (default 2)
    )
    
    print(f"Report saved to: {result['output_filename']}")
    print(f"Custom parameters used:")
    
    report = _load_resource(str(MANIFEST_PATH), result['output_filename'])
    params = report['metadata']['parameters']
    print(f"  - Similarity threshold: {params['similarity_threshold']}")
    print(f"  - Alpha (significance): {params['alpha']}")
    print(f"  - Max examples: {params['max_examples']}")
    print(f"  - Min occurrence threshold: {params['min_occurrence_threshold']}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("COMPREHENSIVE SPLIT QUALITY ANALYSIS EXAMPLES")
    print("="*70)
    
    example_basic_usage()
    example_with_validation()
    example_problematic_split()
    example_custom_parameters()
    
    print("\n" + "="*70)
    print("✓ All examples completed successfully!")
    print("="*70 + "\n")
