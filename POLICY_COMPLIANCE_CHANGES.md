# Data Traceability Policy Compliance Changes

## Summary
Removed all "inplace" functionality from outlier detection, molecular complexity, SMILES encoding, and similarity tools to enforce data traceability policy. All operations now create new versioned datasets instead of modifying existing ones.

## Policy Requirement
**No inplace operations allowed** - Every transformation must create a new resource with a unique ID for complete audit trail and data lineage tracking.

## Changes Made

### 1. Source Code (`src/molml_mcp/tools/core/outliers.py`)
All 5 outlier detection functions updated:

#### Functions Modified:
- `detect_outliers_zscore()`
- `detect_outliers_modified_zscore()`
- `detect_outliers_iqr()`
- `detect_outliers_grubbs()`
- `detect_outliers_gesd()`

#### Changes:
1. **Removed parameter**: `inplace: bool = False` removed from all function signatures
2. **Simplified logic**: Removed conditional logic:
   ```python
   # OLD (REMOVED)
   if inplace:
       df_result = df
   else:
       df_result = df.copy()
   
   # NEW (ENFORCED)
   # Always create copy for traceability
   df_result = df.copy()
   ```
3. **Updated docstrings**: Removed references to "same as input if inplace=True"
4. **Behavior**: All functions now ALWAYS create new resource with unique ID

### 2. Test Files

#### `tests/test_outliers.py`
- Removed `test_inplace_modification()` function entirely
- Removed all `inplace=False` parameter lines from function calls
- Updated `__main__` block to exclude inplace test

#### `tests/test_outliers_usage_examples.py`
- Removed `example_inplace_modification()` function entirely
- Removed all `inplace=False` and `inplace=True` parameter lines
- Updated `__main__` block to exclude inplace example

### 3. Verification
✅ All tests pass without inplace functionality
✅ No "inplace" references remain in outliers.py
✅ No "inplace" references remain in test files
✅ All 5 methods correctly detect outliers and create new resources

## Impact
- **Breaking Change**: Any code calling these functions with `inplace=True` will fail
- **Data Policy**: Now enforces traceability - every operation creates auditable resource
- **Storage**: More disk space used (creates copies instead of modifying)
- **Benefit**: Complete data lineage tracking in manifest.json

## Testing Results
All 5 core tests pass successfully:
- ✅ Z-score: 3 outliers detected correctly
- ✅ Modified Z-score: 6 outliers detected correctly
- ✅ IQR: 4 outliers detected correctly
- ✅ Grubbs' test: 1 outlier detected correctly
- ✅ Generalized ESD: 4 outliers detected correctly

## Migration Guide
If you have existing code using `inplace=True`:

```python
# OLD (NO LONGER WORKS)
result = detect_outliers_zscore(
    "data.csv",
    "manifest.json",
    ["column"],
    "ignored_name",
    "Description",
    inplace=True  # ❌ This parameter no longer exists
)

# NEW (CORRECT)
result = detect_outliers_zscore(
    "data.csv",
    "manifest.json",
    ["column"],
    "data_with_zscore_flags",  # Meaningful output name
    "Z-score outlier detection"
)
# Result will have new filename: "data_with_zscore_flags_A3F2B1D4.csv"
```

### Complexity Module Migration
If you have code using `add_complexity_columns()` with `inplace=True`:

```python
# OLD (NO LONGER WORKS)
result = add_complexity_columns(
    "data.csv",
    "manifest.json",
    "SMILES",
    ["bertz", "bottcher"],
    inplace=True  # ❌ This parameter no longer exists
)

# NEW (CORRECT)
result = add_complexity_columns(
    "data.csv",
    "manifest.json",
    "SMILES",
    ["bertz", "bottcher"],
    "data_with_complexity",  # Now required
    "Added Bertz and Böttcher complexity metrics"
)
# Result will have new filename: "data_with_complexity_A3F2B1D4.csv"
```

### SMILES Encoding Module Migration
If you have code using `flag_smiles_vocab_fit()` with `inplace=True`:

```python
# OLD (NO LONGER WORKS)
result = flag_smiles_vocab_fit(
    "molecules.csv",
    "manifest.json",
    "SMILES",
    "vocab.json",
    add_coverage_column=True,
    inplace=True  # ❌ This parameter no longer exists
)

# NEW (CORRECT)
result = flag_smiles_vocab_fit(
    "molecules.csv",
    "manifest.json",
    "SMILES",
    "vocab.json",
    add_coverage_column=True,
    output_filename="molecules_vocab_flagged"  # Now required
)
# Result will have new filename: "molecules_vocab_flagged_A3F2B1D4.csv"
```

### Similarity Module Migration
If you have code using `add_similarity_statistics()` with `inplace=True`:

```python
# OLD (NO LONGER WORKS)
result = add_similarity_statistics(
    "molecules.csv",
    "manifest.json",
    "SMILES",
    "fingerprints.pkl",
    "ignored_name",
    "Description",
    inplace=True  # ❌ This parameter no longer exists
)

# NEW (CORRECT)
result = add_similarity_statistics(
    "molecules.csv",
    "manifest.json",
    "SMILES",
    "fingerprints.pkl",
    "molecules_with_similarity",  # Now always used
    "Added similarity statistics"
)
# Result will have new filename: "molecules_with_similarity_A3F2B1D4.csv"
```

## Files Changed
1. `src/molml_mcp/tools/core/outliers.py` - 5 functions updated
2. `tests/test_outliers.py` - Inplace test removed
3. `tests/test_outliers_usage_examples.py` - Inplace example removed
4. `src/molml_mcp/tools/core_mol/complexity.py` - 1 function updated
5. `test_complexity.py` - Inplace test removed
6. `src/molml_mcp/tools/featurization/SMILES_encoding.py` - 1 function updated
7. `src/molml_mcp/tools/core_mol/similarity.py` - 1 function updated
