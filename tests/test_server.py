"""Tests for server.py initialization."""
import sys
import subprocess
from pathlib import Path
import pytest


@pytest.mark.slow
@pytest.mark.server
def test_server_imports_and_initializes():
    """Test that server.py can be imported and initialized without errors."""
    # Run a Python script that imports the server module to check it initializes
    server_path = Path(__file__).parent.parent / "src" / "molml_mcp" / "server.py"
    
    # Try to find the virtual environment python
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = sys.executable
    
    # Import the server module to check if it initializes without errors
    # Use -c to run Python code that imports and exits immediately
    result = subprocess.run(
        [python_cmd, "-c", f"import sys; sys.path.insert(0, '{server_path.parent.parent}'); from molml_mcp import server; print('OK')"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Check that import succeeded
    assert result.returncode == 0, f"Server import failed: {result.stderr}"
    assert "OK" in result.stdout, f"Server import did not complete: {result.stdout}"
