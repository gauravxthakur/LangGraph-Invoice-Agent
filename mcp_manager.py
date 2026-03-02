import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

class MCPManager:
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGO_URI")
        if not self.mongodb_uri:
            raise ValueError("MONGO_URI not found in environment variables.")

        self.client = MultiServerMCPClient({
            "mongo": {
                "command": "npx",
                "args": ["-y", "mongodb-mcp-server@latest", self.mongodb_uri],
                "transport": "stdio",
            }
        })
        

    async def connect_mongo(self):
        
        """Returns the tools. The client connects automatically."""
        print(f"Fetching tools from MongoDB MCP Server...")
        try:
            mongo_tools = await self.client.get_tools()
            print(f"[MCP] Successfully loaded {len(mongo_tools)} MongoDB tools.")
            return mongo_tools
        except Exception as e:
            print(f"[MCP ERROR] Failed to load tools: {e}")
            return []

    async def disconnect_all(self):
        """Closes all connections."""
        if hasattr(self.client, "close"):
            await self.client.close()