# MolML MCP Server - AI Coding Agent Instructions

## Project Overview
This is an **MCP (Model Context Protocol) server** that enables LLMs to perform molecular machine learning tasks. Built with FastMCP, it exposes tools for molecular structure manipulation, dataset operations, and ML workflows.

## Architecture

### Core Components
- **`server.py`**: FastMCP server registration point - tools added via `mcp.add_tool()`
- **`tools/`**: Domain-organized tool modules (core_mol, ml, protein)
- **`resources/`**: Persistent resource management with unique ID system
- **`config.py`**: Currently empty - configuration through environment variables

### Resource Management System
The project uses a **unique resource ID system** for stateful operations:

- **ID Format**: `YYYYMMDDTHHMMSS_TYPE_RANDOM.ext` (e.g., `20251125T143022_csv_A3F2B1D4.csv`)
- **Storage**: Resources stored in `~/.molml_mcp/` (or `$MOLML_MCP_DATA_DIR`)
- **Registry**: `TYPE_REGISTRY` in `supported_resource_types.py` defines save/load handlers for each type (csv, model, json)
- **Pattern**: Tools return `resource_id` strings; subsequent operations accept them as input

Example flow:
```python
# Store: returns resource_id
rid = _store_resource(dataframe, "csv")  # → "20251125T143022_csv_A3F2B1D4.csv"

# Load: accepts resource_id
df = _load_resource(rid)
```

### Tool Organization
Tools follow a **domain-based namespace pattern**:
- `tools/core_mol/`: Molecular operations (SMILES canonicalization, cleaning, descriptors, scaffolds)
- `tools/ml/`: Machine learning (training, evaluation)
- `tools/protein/`: Protein-related operations
- Individual tools registered in `server.py` and exported through `tools/__init__.py`

## Key Patterns

### 1. MCP Tool Registration
Tools must be explicitly added to FastMCP in `server.py`:
```python
from molml_mcp.tools.core_mol.smiles_ops import canonicalize_smiles
mcp.add_tool(canonicalize_smiles)
```

### 2. Function Logging with @loggable
Use the `@loggable` decorator (in `resources/logistics.py`) to automatically log:
- Function name and first line of docstring
- **Original** input arguments (before any mutation)
- Full return values
- Writes to `~/.molml_mcp/history.log`

Example:
```python
@loggable
def my_tool(resource_id: str, param: int) -> dict:
    """Process dataset with parameter."""
    # ... implementation
    return {"resource_id": new_id, "result": value}
```

### 3. Dataset Tool Return Pattern
Dataset manipulation tools follow a consistent return schema:
```python
return {
    "resource_id": str,        # New or same resource_id
    "n_rows": int,             # Row count
    "columns": list[str],      # Column names
    "preview": list[dict],     # First 5 rows as records
}
```

### 4. Inplace Operations
Many dataset tools support `inplace: bool = False`:
- `inplace=False`: Creates new resource, returns new `resource_id`
- `inplace=True`: Modifies existing resource, returns same `resource_id`

## Development Workflow

### Deployment
Use `deploy_mcp_server.sh` to install/update the server in Claude Desktop:
```bash
./deploy_mcp_server.sh
```
This script:
1. Runs `uv mcp install src/molml_mcp/server.py`
2. Updates Claude Desktop config JSON with jq
3. Restarts Claude Desktop

### Adding New Tools
1. Create function in appropriate `tools/` subdirectory
2. Export from `tools/__init__.py`
3. Register in `server.py` with `mcp.add_tool()`
4. Add `@loggable` decorator for automatic logging
5. Follow resource_id pattern if stateful

### Adding New Resource Types
Edit `supported_resource_types.py`:
```python
TYPE_REGISTRY["newtype"] = {
    "ext": ".extension",
    "save": _save_newtype,  # (obj, path: Path) -> None
    "load": _load_newtype,  # (path: Path) -> Any
}
```

## Dependencies & Tech Stack
- **FastMCP**: MCP server framework
- **RDKit**: Molecular structure manipulation (SMILES, molecule objects)
- **pandas**: Dataset operations
- **scikit-learn**: ML models
- **joblib**: Model serialization
- Python 3.13+ required

## Important Notes
- Test data available in `tests/data/canonicalization.csv` (includes NaN handling examples)
- Many tool files are empty placeholders for future implementation
- No formal test suite yet - manual testing workflow
- Environment variable `MOLML_MCP_DATA_DIR` overrides default data directory
