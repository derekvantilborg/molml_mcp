from mcp.server.fastmcp import FastMCP

# All tools we want to expose via the MCP server
from molml_mcp.tools.core import get_all_dataset_tools
from molml_mcp.tools import canonicalize_smiles

# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add dataset management tools
for tool_func in get_all_dataset_tools():
    mcp.add_tool(tool_func)

# Add tools
mcp.add_tool(canonicalize_smiles)
# mcp.add_tool(canonicalize_smiles_dataset)

