from mcp.server.fastmcp import FastMCP

# All tools we want to expose via the MCP server
from molml_mcp.tools.core import get_all_dataset_tools
from molml_mcp.tools import canonicalize_smiles
from molml_mcp.tools.core_mol.cleaning import canonicalize_smiles_dataset
from molml_mcp.tools.core_mol.descriptors import list_rdkit_descriptors, calculate_descriptors

# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add dataset management tools
for tool_func in get_all_dataset_tools():
    mcp.add_tool(tool_func)

# Add molecular tools
mcp.add_tool(canonicalize_smiles)
mcp.add_tool(canonicalize_smiles_dataset)
mcp.add_tool(list_rdkit_descriptors)
mcp.add_tool(calculate_descriptors)

