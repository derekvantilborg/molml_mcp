"""Tests for substructure matching functions."""

import pytest
from molml_mcp.tools.core_mol.substructure_matching import (
    get_available_structural_patterns,
    get_available_functional_group_patterns,
    smiles_has_structural_pattern,
    find_structural_patterns_in_smiles,
    find_functional_group_patterns_in_smiles,
    find_functional_group_patterns_in_list_of_smiles,
    find_structural_patterns_in_list_of_smiles
)


def test_get_available_structural_patterns():
    """Test that structural patterns dict is returned correctly."""
    patterns = get_available_structural_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check that patterns have expected structure
    for pattern_name, pattern_data in patterns.items():
        assert 'pattern' in pattern_data
        assert 'comment' in pattern_data
        assert isinstance(pattern_data['pattern'], str)
        assert isinstance(pattern_data['comment'], str)
    
    # Check for some known patterns
    assert 'Rotatable bond' in patterns
    assert 'Ring atom' in patterns


def test_get_available_functional_group_patterns():
    """Test that functional group patterns dict is returned correctly."""
    patterns = get_available_functional_group_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check that patterns have expected structure
    for pattern_name, pattern_data in patterns.items():
        assert 'pattern' in pattern_data
        assert 'comment' in pattern_data
        assert isinstance(pattern_data['pattern'], str)
        assert isinstance(pattern_data['comment'], str)
    
    # Check for some known functional groups
    assert 'Carbonyl group' in patterns
    assert 'Hydroxyl' in patterns


def test_smiles_has_structural_pattern():
    """Test detection of structural patterns in molecules."""
    # Benzene should match aromatic benzene pattern
    assert smiles_has_structural_pattern('c1ccccc1', 'c1ccccc1') is True
    
    # Alkane should not match benzene pattern
    assert smiles_has_structural_pattern('CCCC', 'c1ccccc1') is False
    
    # Test invalid SMILES
    assert smiles_has_structural_pattern('invalid', 'c1ccccc1') is False
    
    # Test invalid SMARTS (should return False, not crash)
    assert smiles_has_structural_pattern('CCO', 'invalid[[[smarts') is False


def test_find_structural_patterns_in_smiles():
    """Test finding all structural patterns in a molecule."""
    # Benzene should have ring atoms and benzene ring
    result = find_structural_patterns_in_smiles('c1ccccc1')
    assert isinstance(result, str)
    assert 'Ring atom' in result or 'Unfused benzene ring' in result
    
    # Simple alkane should have rotatable bonds
    result = find_structural_patterns_in_smiles('CCCC')
    assert isinstance(result, str)
    # May or may not have rotatable bonds depending on definition
    
    # Invalid SMILES should return empty string
    result = find_structural_patterns_in_smiles('invalid')
    assert result == ''
    
    # Empty result should be string
    result = find_structural_patterns_in_smiles('C')
    assert isinstance(result, str)


def test_find_functional_group_patterns_in_smiles():
    """Test finding all functional groups in a molecule."""
    # Ethanol should have hydroxyl
    result = find_functional_group_patterns_in_smiles('CCO')
    assert isinstance(result, str)
    assert 'Hydroxyl' in result
    
    # Acetone should have carbonyl and ketone
    result = find_functional_group_patterns_in_smiles('CC(=O)C')
    assert isinstance(result, str)
    assert 'Carbonyl' in result or 'Ketone' in result
    
    # Ethyl acetate should have ester
    result = find_functional_group_patterns_in_smiles('CC(=O)OCC')
    assert isinstance(result, str)
    assert 'Ester' in result or 'Carbonyl' in result
    
    # Invalid SMILES should return empty string
    result = find_functional_group_patterns_in_smiles('invalid')
    assert result == ''
    
    # Methane may have no functional groups
    result = find_functional_group_patterns_in_smiles('C')
    assert isinstance(result, str)


def test_find_functional_group_patterns_in_list_of_smiles():
    """Test batch finding of functional groups."""
    smiles_list = ['CCO', 'CC(=O)C', 'CCCC']
    results = find_functional_group_patterns_in_list_of_smiles(smiles_list)
    
    assert isinstance(results, list)
    assert len(results) == len(smiles_list)
    
    # All results should be strings
    for result in results:
        assert isinstance(result, str)
    
    # First should have hydroxyl
    assert 'Hydroxyl' in results[0]
    
    # Second should have carbonyl/ketone
    assert 'Carbonyl' in results[1] or 'Ketone' in results[1]
    
    # Test with invalid SMILES
    results = find_functional_group_patterns_in_list_of_smiles(['invalid', 'CCO'])
    assert len(results) == 2
    assert results[0] == ''  # Invalid returns empty
    assert 'Hydroxyl' in results[1]


def test_find_structural_patterns_in_list_of_smiles():
    """Test batch finding of structural patterns."""
    smiles_list = ['c1ccccc1', 'CCCC', 'C']
    results = find_structural_patterns_in_list_of_smiles(smiles_list)
    
    assert isinstance(results, list)
    assert len(results) == len(smiles_list)
    
    # All results should be strings
    for result in results:
        assert isinstance(result, str)
    
    # Benzene should have ring patterns
    assert 'Ring' in results[0] or len(results[0]) > 0
    
    # Test with invalid SMILES
    results = find_structural_patterns_in_list_of_smiles(['invalid', 'c1ccccc1'])
    assert len(results) == 2
    assert results[0] == ''  # Invalid returns empty
    assert len(results[1]) > 0  # Benzene has patterns
