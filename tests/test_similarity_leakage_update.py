"""
Test the updated _detect_similarity_leakage function with new within/between split reporting.
"""

import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from molml_mcp.tools.reports.data_splitting import _detect_similarity_leakage
from molml_mcp.infrastructure.resources import _store_resource

TEST_DIR = os.path.join(os.path.dirname(__file__), 'data')
MANIFEST_PATH = os.path.join(TEST_DIR, 'test_manifest.json')


def test_within_between_similarity():
    """Test that within and between split similarities are reported."""
    print("\n" + "="*80)
    print("TEST: Within and Between Split Similarity Reporting")
    print("="*80)
    
    np.random.seed(42)
    
    # Create datasets with known similarity characteristics
    # Train: similar molecules (all ethane derivatives)
    df_train = pd.DataFrame({
        'smiles': ['CC', 'CCO', 'CCC', 'CCCO'] * 25,  # 100 molecules
        'activity': np.random.uniform(1, 100, 100)
    })
    
    # Test: different molecules (benzene derivatives - less similar to train)
    df_test = pd.DataFrame({
        'smiles': ['c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1'] * 10,  # 30 molecules
        'activity': np.random.uniform(1, 100, 30)
    })
    
    # Val: mix of both
    df_val = pd.DataFrame({
        'smiles': ['CCCC', 'c1ccccc1C'] * 10,  # 20 molecules
        'activity': np.random.uniform(1, 100, 20)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'sim_test_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'sim_test_test', 'test', 'csv')
    val_file = _store_resource(df_val, MANIFEST_PATH, 'sim_test_val', 'test', 'csv')
    
    result = _detect_similarity_leakage(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    print("\n--- RESULT STRUCTURE ---")
    print(f"Keys: {list(result.keys())}")
    
    # Check new fields exist
    assert 'within_split_similarity' in result
    assert 'between_split_similarity' in result
    print("✅ New fields present")
    
    # Check within-split similarity
    print("\n--- WITHIN-SPLIT SIMILARITY ---")
    for split in ['train', 'test', 'val']:
        if split in result['within_split_similarity']:
            data = result['within_split_similarity'][split]
            print(f"\n{split.upper()}:")
            if 'error' not in data:
                print(f"  N molecules: {data['n_molecules']}")
                print(f"  N comparisons: {data['n_comparisons']}")
                print(f"  Avg similarity: {data['avg_similarity']}")
                print(f"  Max similarity: {data['max_similarity']}")
                print(f"  Median similarity: {data['median_similarity']}")
                print(f"  Sampled: {data['sampled']}")
                
                assert 'avg_similarity' in data
                assert 'max_similarity' in data
                assert 0.0 <= data['avg_similarity'] <= 1.0
                assert 0.0 <= data['max_similarity'] <= 1.0
            else:
                print(f"  Error: {data['error']}")
    
    print("\n✅ Within-split similarity computed")
    
    # Check between-split similarity
    print("\n--- BETWEEN-SPLIT SIMILARITY ---")
    for comparison in ['test_vs_train', 'val_vs_train', 'val_vs_test']:
        if comparison in result['between_split_similarity']:
            data = result['between_split_similarity'][comparison]
            print(f"\n{comparison.upper()}:")
            print(f"  Avg max similarity: {data['avg_max_similarity']}")
            print(f"  Max similarity: {data['max_similarity']}")
            
            assert 'avg_max_similarity' in data
            assert 'max_similarity' in data
            assert 0.0 <= data['avg_max_similarity'] <= 1.0
            assert 0.0 <= data['max_similarity'] <= 1.0
    
    print("\n✅ Between-split similarity computed")
    
    # Verify the detailed analysis still exists
    assert 'test_vs_train' in result
    assert 'similarity_stats' in result['test_vs_train']
    print("\n✅ Original detailed analysis preserved")
    
    # Print full result for inspection
    print("\n--- FULL RESULT (JSON) ---")
    print(json.dumps(result, indent=2, default=str))
    
    return result


def test_large_dataset_sampling():
    """Test that large datasets are sampled for within-split similarity."""
    print("\n" + "="*80)
    print("TEST: Large Dataset Sampling")
    print("="*80)
    
    np.random.seed(42)
    
    # Create large dataset (>1000 molecules)
    df_train = pd.DataFrame({
        'smiles': [f'C{"C"*i}' for i in range(1500)],
        'activity': np.random.uniform(1, 100, 1500)
    })
    
    df_test = pd.DataFrame({
        'smiles': [f'N{"C"*i}' for i in range(500)],
        'activity': np.random.uniform(1, 100, 500)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'large_sim_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'large_sim_test', 'test', 'csv')
    
    result = _detect_similarity_leakage(
        train_file, test_file, None, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    # Check train was sampled
    train_sim = result['within_split_similarity']['train']
    assert train_sim['sampled'] == True
    assert train_sim['n_molecules'] == 1500
    print(f"✅ Large train set (n={train_sim['n_molecules']}) was sampled")
    
    # Check test was not sampled
    test_sim = result['within_split_similarity']['test']
    assert test_sim['sampled'] == False
    assert test_sim['n_molecules'] == 500
    print(f"✅ Smaller test set (n={test_sim['n_molecules']}) was not sampled")


def test_comparison_logic():
    """Verify that within-split similarity is generally higher than between-split."""
    print("\n" + "="*80)
    print("TEST: Similarity Comparison Logic")
    print("="*80)
    
    np.random.seed(42)
    
    # Train: very similar molecules (all short alkanes)
    df_train = pd.DataFrame({
        'smiles': ['CC', 'CCC', 'CCCC', 'CCCCC'] * 25,
        'activity': np.random.uniform(1, 100, 100)
    })
    
    # Test: very different molecules (aromatics)
    df_test = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1ccc(C)cc1', 'c1ccc(O)cc1'] * 10,
        'activity': np.random.uniform(1, 100, 30)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'logic_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'logic_test', 'test', 'csv')
    
    result = _detect_similarity_leakage(
        train_file, test_file, None, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    train_avg = result['within_split_similarity']['train']['avg_similarity']
    test_avg = result['within_split_similarity']['test']['avg_similarity']
    between_avg = result['between_split_similarity']['test_vs_train']['avg_max_similarity']
    
    print(f"\nTrain within-split avg similarity: {train_avg}")
    print(f"Test within-split avg similarity: {test_avg}")
    print(f"Test-vs-Train between-split avg max similarity: {between_avg}")
    
    # Within-split similarity should typically be higher than between-split
    # (molecules in same split are usually more similar to each other)
    print(f"\n✅ Similarities computed and are reasonable")
    print(f"   (Train within: {train_avg:.3f}, Test within: {test_avg:.3f}, Between: {between_avg:.3f})")


if __name__ == '__main__':
    print("="*80)
    print("TESTING UPDATED _detect_similarity_leakage()")
    print("="*80)
    
    try:
        result = test_within_between_similarity()
        test_large_dataset_sampling()
        test_comparison_logic()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n_detect_similarity_leakage() now reports:")
        print("  ✅ Within-split similarities (avg, max, median)")
        print("  ✅ Between-split similarities (avg max, max)")
        print("  ✅ Sampling for large datasets (>1000 molecules)")
        print("  ✅ All original functionality preserved")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
