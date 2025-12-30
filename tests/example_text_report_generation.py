"""
Example usage of generate_split_quality_text_report()

Demonstrates the complete workflow:
1. Create train/test datasets
2. Generate JSON analysis (analyze_split_quality)
3. Generate human-readable text report (generate_split_quality_text_report)
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.reports.data_splitting import (
    analyze_split_quality,
    generate_split_quality_text_report
)

# Test manifest path
TEST_MANIFEST = Path(__file__).parent / 'data' / 'test_manifest.json'


def example_complete_workflow():
    """
    Complete end-to-end example:
    1. Store datasets
    2. Generate JSON quality analysis
    3. Generate text report
    4. Display the report
    """
    print("\n" + "="*80)
    print("EXAMPLE: Complete Data Splitting Quality Analysis Workflow")
    print("="*80 + "\n")
    
    # 1. Create datasets with some intentional issues
    print("Step 1: Creating train/test datasets...")
    train_df = pd.DataFrame({
        'smiles': [
            'c1ccccc1',      # benzene
            'CCO',           # ethanol
            'CC(=O)C',       # acetone
            'CCN',           # ethylamine
            'CCC',           # propane
            'CCCC',          # butane
            'c1ccc(C)cc1',   # toluene
            'CC(C)O',        # isopropanol
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    test_df = pd.DataFrame({
        'smiles': [
            'CCCCC',         # pentane
            'c1ccc(CC)cc1',  # ethylbenzene (similar to toluene)
            'CC(=O)CC',      # butanone (similar to acetone)
        ],
        'label': [1, 0, 1]
    })
    
    # Store datasets
    train_file = _store_resource(
        train_df, 
        str(TEST_MANIFEST), 
        "example_train", 
        "Training set for workflow example",
        "csv"
    )
    test_file = _store_resource(
        test_df, 
        str(TEST_MANIFEST), 
        "example_test", 
        "Test set for workflow example",
        "csv"
    )
    print(f"  ✓ Train set: {train_file} ({len(train_df)} molecules)")
    print(f"  ✓ Test set:  {test_file} ({len(test_df)} molecules)")
    
    # 2. Generate JSON quality analysis
    print("\nStep 2: Running comprehensive quality analysis...")
    json_result = analyze_split_quality(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(TEST_MANIFEST),
        smiles_col='smiles',
        label_col='label',
        output_filename='example_quality_analysis',
        explanation="Quality analysis for workflow example"
    )
    
    print(f"  ✓ JSON report saved: {json_result['output_filename']}")
    print(f"  ✓ Overall severity: {json_result['overall_severity']}")
    print(f"  ✓ Checks performed: {json_result['n_checks_performed']}")
    print(f"  ✓ Execution time: {json_result['execution_time_seconds']:.2f}s")
    
    # 3. Generate text report
    print("\nStep 3: Generating human-readable text report...")
    text_result = generate_split_quality_text_report(
        json_report_filename=json_result['output_filename'],
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='example_quality_report_text',
        explanation="Text report for workflow example"
    )
    
    print(f"  ✓ Text report saved: {text_result['output_filename']}")
    print(f"  ✓ Report lines: {text_result['n_lines']}")
    print(f"  ✓ Report sections: {len(text_result['report_sections'])}")
    
    # 4. Display the text report
    print("\nStep 4: Displaying the generated text report...")
    print("\n" + "="*80)
    text_content = _load_resource(str(TEST_MANIFEST), text_result['output_filename'])
    print(text_content)
    print("="*80)
    
    print("\n✅ Workflow complete!")
    print(f"   - JSON report: {json_result['output_filename']}")
    print(f"   - Text report: {text_result['output_filename']}")
    
    return json_result, text_result


def example_with_critical_issues():
    """
    Example with CRITICAL issues (exact duplicates).
    """
    print("\n" + "="*80)
    print("EXAMPLE: Detecting CRITICAL Issues")
    print("="*80 + "\n")
    
    print("Creating datasets with exact duplicates...")
    
    # Train set
    train_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO', 'CC(=O)C', 'CCN'],
        'label': [0, 1, 0, 1]
    })
    
    # Test set with duplicates
    test_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCCC'],  # benzene is duplicate!
        'label': [1, 0]
    })
    
    train_file = _store_resource(train_df, str(TEST_MANIFEST), "critical_train", "Train", "csv")
    test_file = _store_resource(test_df, str(TEST_MANIFEST), "critical_test", "Test", "csv")
    
    print(f"  ⚠️  Train and test sets share molecule: c1ccccc1 (benzene)")
    
    # Analyze
    print("\nRunning analysis...")
    json_result = analyze_split_quality(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(TEST_MANIFEST),
        smiles_col='smiles',
        label_col='label',
        output_filename='critical_analysis'
    )
    
    # Generate report
    text_result = generate_split_quality_text_report(
        json_report_filename=json_result['output_filename'],
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='critical_report_text'
    )
    
    # Display key findings
    print(f"\n🔴 Overall Severity: {json_result['overall_severity']}")
    print(f"   Exact duplicates found: {json_result['issues_found']['exact_duplicates']}")
    
    print("\n📄 Text report preview (first 50 lines):")
    text_content = _load_resource(str(TEST_MANIFEST), text_result['output_filename'])
    lines = text_content.split('\n')[:50]
    print('\n'.join(lines))
    print("...")
    
    print(f"\n✅ Full report saved: {text_result['output_filename']}")
    
    return json_result, text_result


if __name__ == '__main__':
    print("\n" + "🧪 Data Splitting Quality Analysis Examples" + "\n")
    
    # Example 1: Complete workflow
    example_complete_workflow()
    
    # Example 2: Critical issues
    example_with_critical_issues()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)
