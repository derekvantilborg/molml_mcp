"""
Test suite for analyze_split_quality() aggregator function.

This function combines all 8 helper functions into a comprehensive report.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.reports.data_splitting import analyze_split_quality
from molml_mcp.infrastructure.resources import _store_resource, _load_resource


# Test manifest path
TEST_MANIFEST = Path(__file__).parent / 'data' / 'test_manifest.json'


def test_basic_aggregation():
    """Test that all 8 helpers are called and results aggregated."""
    # Create simple datasets
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # benzene
            'CC(=O)C',  # acetone
            'CCO',  # ethanol
            'CCN',  # ethylamine
            'c1ccccc1C',  # toluene
        ],
        'activity': [0, 1, 0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': [
            'c1ccc(C)cc1',  # toluene (similar to train)
            'CCC(=O)C',  # 2-butanone
        ],
        'activity': [1, 0]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'agg_basic_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'agg_basic_test', 'test', 'csv')
    
    # Run aggregator
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'basic_report'
    )
    
    # Check return structure
    assert 'output_filename' in result
    assert 'overall_severity' in result
    assert 'severity_summary' in result
    assert 'n_checks_performed' in result
    assert result['n_checks_performed'] == 8
    assert 'execution_time_seconds' in result
    assert 'issues_found' in result
    
    # Check that file was saved
    assert result['output_filename'].endswith('.json')
    
    # Load the saved report
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    # Check all 8 sections present
    assert 'metadata' in report
    assert 'split_characteristics' in report
    assert 'exact_duplicates' in report
    assert 'similarity_leakage' in report
    assert 'scaffold_leakage' in report
    assert 'stereoisomer_tautomer_leakage' in report
    assert 'property_distributions' in report
    assert 'activity_distributions' in report
    assert 'functional_groups' in report
    assert 'overall_severity' in report
    assert 'severity_summary' in report
    
    print(f"✓ Basic aggregation: {result['n_checks_performed']} checks, saved to {result['output_filename']}")


def test_overall_severity_calculation():
    """Test that overall severity is the highest found across all checks."""
    # Create datasets with exact duplicates (CRITICAL)
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'],
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCCC'],  # First is duplicate
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'sev_crit_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'sev_crit_test', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'severity_report'
    )
    
    # Should be CRITICAL due to exact duplicate
    assert result['overall_severity'] == 'CRITICAL'
    assert result['severity_summary']['CRITICAL'] >= 1
    assert result['issues_found']['exact_duplicates'] >= 1
    
    print(f"✓ Overall severity: {result['overall_severity']} (found {result['issues_found']['exact_duplicates']} duplicates)")


def test_severity_summary():
    """Test that severity summary counts issues correctly."""
    # Create clean dataset (should be mostly OK)
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1cccnc1', 'c1ccco1'] * 10,  # 30 molecules
        'activity': [0, 1, 0] * 10
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'c1ccc(C)nc1', 'c1ccc(C)o1'] * 5,  # 15 molecules, similar but not identical
        'activity': [1, 0, 1] * 5
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'clean_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'clean_test', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'clean_report'
    )
    
    # Check severity summary
    assert 'CRITICAL' in result['severity_summary']
    assert 'HIGH' in result['severity_summary']
    assert 'MEDIUM' in result['severity_summary']
    assert 'LOW' in result['severity_summary']
    assert 'OK' in result['severity_summary']
    
    # Sum should equal number of checks (or close - some may not have severity if errors)
    total_checks = sum(result['severity_summary'].values())
    assert total_checks >= 6  # At least most checks should complete
    assert total_checks <= 8  # Max 8 checks
    
    # Should have mostly OK or LOW severity
    assert result['severity_summary']['OK'] + result['severity_summary']['LOW'] >= 3
    
    print(f"✓ Severity summary: {result['severity_summary']} (total: {total_checks})")


def test_metadata_fields():
    """Test that metadata contains all expected fields."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO'],
        'activity': [0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCCO'],
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'meta_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'meta_test', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'metadata_report'
    )
    
    # Load report
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    # Check metadata
    meta = report['metadata']
    assert 'analysis_type' in meta
    assert meta['analysis_type'] == 'data_splitting_quality'
    assert 'timestamp' in meta
    assert 'train_file' in meta
    assert 'test_file' in meta
    assert 'val_file' in meta
    assert 'smiles_column' in meta
    assert 'label_column' in meta
    assert 'parameters' in meta
    assert 'execution_time_seconds' in meta
    assert 'completed_at' in meta
    
    # Check parameters
    params = meta['parameters']
    assert 'min_split_size' in params
    assert 'imbalance_threshold' in params
    assert 'similarity_threshold' in params
    assert 'alpha' in params
    assert 'max_examples' in params
    
    print(f"✓ Metadata: {len(meta)} fields, execution time {meta['execution_time_seconds']:.3f}s")


def test_three_way_split():
    """Test with train/test/val splits."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'] * 5,
        'activity': [0, 1, 0] * 5
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCC(=O)C', 'CCCO'] * 3,
        'activity': [1, 0, 1] * 3
    })
    val_data = pd.DataFrame({
        'smiles': ['c1ccc(C)cc1', 'CC(=O)CC', 'CCCCO'] * 2,
        'activity': [0, 1, 0] * 2
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'three_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'three_test', 'test', 'csv')
    val_file = _store_resource(val_data, str(TEST_MANIFEST), 'three_val', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, val_file, str(TEST_MANIFEST),
        'smiles', 'activity', 'three_way_report'
    )
    
    # Load report
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    # Check that val was included
    assert report['metadata']['val_file'] == val_file
    
    # Check that all helpers handled val split
    # Many helpers will have val-specific fields
    assert 'train_val_duplicates' in report['exact_duplicates']
    assert 'val_vs_train' in report['similarity_leakage']
    assert 'train_vs_val' in report['property_distributions']
    
    print(f"✓ Three-way split: train/test/val all processed")


def test_issues_found_summary():
    """Test that issues_found summarizes key problems."""
    # Create dataset with various issues
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # benzene
            'CC(C)C',  # isobutane
            'CCO',  # ethanol
            'c1ccccc1C',  # toluene
        ],
        'activity': [0, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # benzene - DUPLICATE!
            'CC(=O)C',  # acetone
        ],
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'issues_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'issues_test', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'issues_report'
    )
    
    # Check issues_found structure
    issues = result['issues_found']
    assert 'exact_duplicates' in issues
    assert 'high_similarity_pairs' in issues
    assert 'activity_cliffs' in issues
    assert 'scaffold_overlap_pct' in issues
    assert 'stereoisomer_pairs' in issues
    assert 'tautomer_pairs' in issues
    assert 'significant_property_diffs' in issues
    assert 'activity_distribution_different' in issues
    assert 'unique_functional_groups_test' in issues
    
    # Should find duplicate
    assert issues['exact_duplicates'] >= 1
    
    print(f"✓ Issues found: {issues}")


def test_custom_parameters():
    """Test that custom parameters are passed to helpers."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'] * 10,
        'activity': [0, 1, 0] * 10
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCC(=O)C', 'CCCO'] * 5,
        'activity': [1, 0, 1] * 5
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'params_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'params_test', 'test', 'csv')
    
    # Run with custom parameters
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'params_report',
        min_split_size=5,
        similarity_threshold=0.85,
        alpha=0.01,
        max_examples=5
    )
    
    # Load report and check parameters were recorded
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    params = report['metadata']['parameters']
    
    assert params['min_split_size'] == 5
    assert params['similarity_threshold'] == 0.85
    assert params['alpha'] == 0.01
    assert params['max_examples'] == 5
    
    print(f"✓ Custom parameters: {params}")


def test_json_serialization():
    """Test that the saved JSON is valid and can be loaded."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO'],
        'activity': [0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCCO'],
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'json_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'json_test', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'json_test_report'
    )
    
    # Load using resource system
    report1 = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    # Also try loading directly with json
    manifest_dir = Path(TEST_MANIFEST).parent
    report_path = manifest_dir / result['output_filename']
    with open(report_path, 'r') as f:
        report2 = json.load(f)
    
    # Both should be identical
    assert report1['overall_severity'] == report2['overall_severity']
    assert report1['metadata']['analysis_type'] == report2['metadata']['analysis_type']
    
    print(f"✓ JSON serialization: Valid JSON, {len(str(report1))} chars")


def test_execution_time_reasonable():
    """Test that execution time is recorded and reasonable."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'] * 3,
        'activity': [0, 1, 0] * 3
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCC(=O)C', 'CCCO'],
        'activity': [1, 0, 1]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'time_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'time_test', 'test', 'csv')
    
    result = analyze_split_quality(
        train_file, test_file, None, str(TEST_MANIFEST),
        'smiles', 'activity', 'time_report'
    )
    
    # Check execution time
    assert result['execution_time_seconds'] > 0
    assert result['execution_time_seconds'] < 60  # Should be well under 1 minute for small dataset
    
    # Load report and verify metadata
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    assert 'timestamp' in report['metadata']
    assert 'completed_at' in report['metadata']
    
    print(f"✓ Execution time: {result['execution_time_seconds']:.3f}s")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Testing analyze_split_quality() aggregator function")
    print("="*70 + "\n")
    
    test_basic_aggregation()
    test_overall_severity_calculation()
    test_severity_summary()
    test_metadata_fields()
    test_three_way_split()
    test_issues_found_summary()
    test_custom_parameters()
    test_json_serialization()
    test_execution_time_reasonable()
    
    print("\n" + "="*70)
    print("✓ All 9 tests passed!")
    print("="*70)
