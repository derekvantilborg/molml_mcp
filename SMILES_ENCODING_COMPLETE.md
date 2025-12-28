# SMILES Encoding Module - Complete MCP-Friendly Implementation

## Overview
The `SMILES_encoding.py` module provides comprehensive MCP-ready tools for tokenizing, encoding, and managing vocabularies for SMILES strings in molecular machine learning workflows. All functions use JSON vocab paths instead of vocab dictionaries for seamless MCP integration.

## Core Features

### 1. Token Discovery ✅
**Function:** `discover_tokens_from_dataset()`
- Discovers all tokens from SMILES strings in a dataset column
- Properly handles multi-character tokens (Cl, Br, @@, bracketed atoms)
- Counts token frequencies
- Supports min_frequency filtering and max_tokens limit
- Returns comprehensive statistics

**Example:**
```python
result = discover_tokens_from_dataset(
    'molecules.csv',
    'manifest.json',
    'SMILES',
    min_frequency=10
)
print(f"Found {result['n_unique_tokens']} tokens")
```

### 2. Vocabulary JSON System ✅
**Save Function:** `save_vocab_to_json()`
- Saves vocabulary to JSON as a project resource
- Includes metadata and special token information
- Integrates with manifest system

**Load Function:** `load_vocab_from_json()`
- Loads vocabulary from JSON resource
- Returns vocab dict and special tokens dict

**Inspect Function:** `inspect_vocab_json()` **[NEW - MCP Ready]**
- Shows vocab contents without loading into memory
- Returns vocab size, special tokens, all tokens, and description
- Perfect for MCP server to display vocab information

**Example:**
```python
# Inspect vocab
info = inspect_vocab_json('my_vocab.json', 'manifest.json')
print(f"Vocab has {info['vocab_size']} tokens")
print(f"Special tokens: {info['special_tokens']}")
print(f"All tokens: {info['tokens']}")
```

### 3. Default SMILES Vocab JSON ✅
**Function:** `create_default_vocab_json()`
- Creates default vocabulary for drug-like molecules
- 44 base tokens: C, c, N, n, O, o, S, s, P, p, F, Cl, Br, I, H, B, bonds, rings, charges, stereochemistry
- Configurable special tokens (pad, start, end, unk)
- Saves as JSON resource

**Example:**
```python
vocab_file = create_default_vocab_json(
    'manifest.json',
    add_pad=True,
    add_unk=True
)
```

### 4. Compile Custom Vocab ✅
**Function:** `compile_vocab_from_tokens()`
- Creates vocabulary from user-specified token list
- Adds optional special tokens
- Saves as JSON resource

**Example:**
```python
custom_tokens = ['C', 'N', 'O', 'Cl', '(', ')', '=', '1']
vocab_file = compile_vocab_from_tokens(
    custom_tokens,
    'manifest.json',
    'small_vocab',
    add_pad=True,
    add_unk=True
)
```

### 5. Encode SMILES to Indices ✅ **[MCP-Friendly]**
**Function:** `smiles_to_indices()`
- Converts SMILES to integer indices using vocab JSON path
- Supports padding and truncation
- Handles unknown tokens with <unk>
- Returns detailed dictionary with indices, tokens, and metadata

**Example:**
```python
result = smiles_to_indices(
    'CCCl',
    'my_vocab.json',
    'manifest.json',
    max_length=10
)
print(result['indices'])  # [2, 2, 13, 0, 0, ...]
print(result['tokens'])   # ['C', 'C', 'Cl']
print(result['padded'])   # True
```

### 6. Encode SMILES to One-Hot ✅ **[MCP-Friendly]**
**Function:** `smiles_to_one_hot()`
- Converts SMILES to one-hot encoded matrix using vocab JSON path
- Optionally saves output as joblib resource
- Returns shape, tokens, and optional saved filename
- Shape: (max_length, vocab_size)

**Example:**
```python
result = smiles_to_one_hot(
    'CCO',
    'my_vocab.json',
    'manifest.json',
    max_length=5,
    output_filename='cco_onehot'
)
print(result['shape'])             # (5, 46)
print(result['output_filename'])   # 'cco_onehot_A1B2C3D4.joblib'
```

### 7. Batch Encode to One-Hot ✅ **[MCP-Friendly]**
**Function:** `batch_smiles_to_one_hot()`
- Converts dataset of SMILES to 3D one-hot array using vocab JSON path
- Auto-detects max_length if not specified
- Saves output as joblib resource
- Shape: (n_smiles, max_length, vocab_size)

**Example:**
```python
result = batch_smiles_to_one_hot(
    'molecules.csv',
    'manifest.json',
    'SMILES',
    'my_vocab.json',
    'batch_onehot'
)
print(result['shape'])             # (100, 50, 46)
print(result['output_filename'])   # 'batch_onehot_A1B2C3D4.joblib'
```

### 8. Check SMILES Vocab Coverage ✅ **[MCP-Friendly]**
**Single SMILES:** `check_smiles_vocab_coverage()`
- Checks if SMILES can be tokenized with vocab from JSON path
- Returns coverage percentage
- Lists unknown tokens

**Dataset Coverage:** `check_dataset_vocab_coverage()`
- Checks coverage across entire dataset using vocab JSON path
- Returns overall statistics
- Lists all missing tokens with frequencies

**Example:**
```python
# Single SMILES
result = check_smiles_vocab_coverage(
    'CCCl',
    'my_vocab.json',
    'manifest.json'
)
if not result['can_tokenize']:
    print(f"Missing: {result['unknown_tokens']}")

# Dataset
result = check_dataset_vocab_coverage(
    'molecules.csv',
    'manifest.json',
    'SMILES',
    'my_vocab.json'
)
print(f"Coverage: {result['overall_coverage']}%")
```

## MCP-Ready Design

### Key MCP Improvements
1. **Vocab JSON Paths**: All functions accept `vocab_filename` instead of vocab dict
2. **Inspect Function**: `inspect_vocab_json()` allows MCP server to show vocab contents
3. **Dict Returns**: All functions return dictionaries (not raw arrays) with metadata
4. **Resource Integration**: Outputs automatically saved as manifest resources
5. **Clear Errors**: Error messages include resource filenames for debugging

### Function Signatures
```python
# OLD (Not MCP-friendly)
def smiles_to_indices(smiles, vocab: Dict, max_length) -> List[int]

# NEW (MCP-friendly)
def smiles_to_indices(smiles, vocab_filename: str, 
                      project_manifest_path: str, 
                      max_length) -> Dict
```

## Token Handling

### Multi-Character Tokens
The tokenizer properly handles:
- **2-char elements:** Cl, Br
- **Chirality:** @@
- **Extended rings:** %10, %11, %12, etc.
- **Bracketed atoms:** [nH], [C@@H], [O-], etc.

### Regex Pattern
```python
r'(\[[^\]]+\]|@@|%\d{2}|Br|Cl|[BCNOPSFIHbcnopsfi]|[=#@+\-\\\/().\[\]]|\d)'
```

## Default Token List (44 tokens)
- **Elements:** C, c, N, n, O, o, S, s, P, p, F, Cl, Br, I, H, B
- **Bonds:** =, #, -, /, \\
- **Structure:** (, ), [, ], .
- **Rings:** 1-9, %10-%15
- **Charges:** +, -
- **Chirality:** @, @@

## Special Tokens
- **<pad>:** Padding token (typically index 0)
- **<start>:** Start of sequence token
- **<end>:** End of sequence token
- **<unk>:** Unknown token for out-of-vocab tokens

## Resource Types
- **Vocabularies:** Saved as `.json` files
- **One-hot arrays:** Saved as `.joblib` files
- **Datasets:** CSV format

## Testing
Comprehensive test suite in `test_mcp_friendly_encoding.py` covers:
- Vocab creation and inspection
- MCP-friendly encoding functions
- Resource saving and loading
- Coverage checking
- All tests pass ✅

## Summary
All functions are fully MCP-ready with:
- JSON vocab path inputs
- Dictionary outputs with metadata
- Automatic resource management
- Clear error messages
- Inspection capabilities

## Core Features

### 1. Token Discovery ✅
**Function:** `discover_tokens_from_dataset()`
- Discovers all tokens from SMILES strings in a dataset column
- Properly handles multi-character tokens (Cl, Br, @@, bracketed atoms)
- Counts token frequencies
- Supports min_frequency filtering and max_tokens limit
- Returns comprehensive statistics

**Example:**
```python
result = discover_tokens_from_dataset(
    'molecules.csv',
    'manifest.json',
    'SMILES',
    min_frequency=10
)
print(f"Found {result['n_unique_tokens']} tokens")
```

### 2. Vocabulary JSON System ✅
**Save Function:** `save_vocab_to_json()`
- Saves vocabulary to JSON as a project resource
- Includes metadata and special token information
- Integrates with manifest system

**Load Function:** `load_vocab_from_json()`
- Loads vocabulary from JSON resource
- Returns vocab dict and special tokens dict

**Example:**
```python
# Save
vocab = build_vocab(add_pad=True, add_unk=True)
filename = save_vocab_to_json(vocab, 'manifest.json', 'my_vocab')

# Load
vocab, special_tokens = load_vocab_from_json(filename, 'manifest.json')
```

### 3. Default SMILES Vocab JSON ✅
**Function:** `create_default_vocab_json()`
- Creates default vocabulary for drug-like molecules
- 44 base tokens: C, c, N, n, O, o, S, s, P, p, F, Cl, Br, I, H, B, bonds, rings, charges, stereochemistry
- Configurable special tokens (pad, start, end, unk)
- Saves as JSON resource

**Example:**
```python
vocab_file = create_default_vocab_json(
    'manifest.json',
    add_pad=True,
    add_unk=True
)
```

### 4. Compile Custom Vocab ✅
**Function:** `compile_vocab_from_tokens()`
- Creates vocabulary from user-specified token list
- Adds optional special tokens
- Saves as JSON resource

**Example:**
```python
custom_tokens = ['C', 'N', 'O', 'Cl', '(', ')', '=', '1']
vocab_file = compile_vocab_from_tokens(
    custom_tokens,
    'manifest.json',
    'small_vocab',
    add_pad=True,
    add_unk=True
)
```

### 5. Encode SMILES to Indices ✅
**Function:** `smiles_to_indices()`
- Converts SMILES to integer indices using vocabulary
- Supports padding and truncation
- Handles unknown tokens with <unk>

**Example:**
```python
vocab, _ = load_vocab_from_json('my_vocab.json', 'manifest.json')
indices = smiles_to_indices('CCCl', vocab, max_length=10)
```

### 6. Encode SMILES to One-Hot ✅
**Function:** `smiles_to_one_hot()`
- Converts SMILES to one-hot encoded matrix
- Shape: (max_length, vocab_size)
- Uses vocabulary for encoding

**Example:**
```python
one_hot = smiles_to_one_hot('CCO', vocab, max_length=5)
# Returns: (5, vocab_size) matrix
```

### 7. Batch Encode to One-Hot ✅
**Function:** `batch_smiles_to_one_hot()`
- Converts batch of SMILES to 3D one-hot array
- Auto-detects max_length if not specified
- Shape: (n_smiles, max_length, vocab_size)

**Example:**
```python
batch = ['CC', 'CCO', 'CCCO']
one_hot_batch = batch_smiles_to_one_hot(batch, vocab)
# Returns: (3, max_length, vocab_size) array
```

### 8. Check SMILES Vocab Coverage ✅
**Single SMILES:** `check_smiles_vocab_coverage()`
- Checks if SMILES can be tokenized with vocab
- Returns coverage percentage
- Lists unknown tokens

**Dataset Coverage:** `check_dataset_vocab_coverage()`
- Checks coverage across entire dataset
- Returns overall statistics
- Lists all missing tokens with frequencies

**Example:**
```python
# Single SMILES
result = check_smiles_vocab_coverage('CCCl', vocab)
if not result['can_tokenize']:
    print(f"Missing: {result['unknown_tokens']}")

# Dataset
result = check_dataset_vocab_coverage(
    'molecules.csv',
    'manifest.json',
    'SMILES',
    vocab
)
print(f"Coverage: {result['overall_coverage']}%")
```

## Token Handling

### Multi-Character Tokens
The tokenizer properly handles:
- **2-char elements:** Cl, Br
- **Chirality:** @@
- **Extended rings:** %10, %11, %12, etc.
- **Bracketed atoms:** [nH], [C@@H], [O-], etc.

### Regex Pattern
```python
r'(\[[^\]]+\]|@@|%\d{2}|Br|Cl|[BCNOPSFIHbcnopsfi]|[=#@+\-\\\/().\[\]]|\d)'
```

## Default Token List (44 tokens)
- **Elements:** C, c, N, n, O, o, S, s, P, p, F, Cl, Br, I, H, B
- **Bonds:** =, #, -, /, \\
- **Structure:** (, ), [, ], .
- **Rings:** 1-9, %10-%15
- **Charges:** +, -
- **Chirality:** @, @@

## Special Tokens
- **<pad>:** Padding token (typically index 0)
- **<start>:** Start of sequence token
- **<end>:** End of sequence token
- **<unk>:** Unknown token for out-of-vocab tokens

## Integration with MCP Server
All functions integrate with the project manifest system:
- Resources tracked with unique IDs
- JSON vocabs saved as project resources
- Consistent with other MCP tools

## Testing
Comprehensive test suite in `test_smiles_encoding_full.py` covers:
- Token discovery
- Vocab creation and persistence
- Encoding functions
- Coverage checking
- Edge cases

All tests pass ✅
