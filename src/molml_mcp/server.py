# from .tools import get_all_tools


# server.py
from mcp.server.fastmcp import FastMCP
from molml_mcp.tools import canonicalize_smiles


# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add tools
mcp.add_tool(canonicalize_smiles)

