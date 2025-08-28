# GitHub Copilot MCP Server Configuration

This document explains how GitHub Copilot is configured to use MCP (Model Context Protocol) servers in this workspace.

## Configuration Files

### Workspace MCP Configuration (`.vscode/mcp.json`)
This file configures MCP servers specifically for this workspace. All configured servers are available to GitHub Copilot when using agent mode.

### VS Code User Settings
The following settings in `settings.json` enable MCP support:
- `"chat.mcp.discovery.enabled": true` - Enables automatic discovery of MCP servers
- `"github.copilot.enable"` - Ensures Copilot is enabled for all file types
- `"github.copilot.editor.enableAutoCompletions": true` - Enables Copilot auto-completions

## Available MCP Servers

1. **gemini-coding** - Gemini AI assistant for coding tasks
2. **context7** - Code documentation and context server
3. **n8n-mcp** - n8n workflow automation server
4. **n8n-workflows** - n8n workflow templates and examples
5. **singlestore-notebooks** - SingleStore notebook examples and templates
6. **singlestore-tutorials** - SingleStore tutorials and documentation
7. **firecrawl** - Web scraping and content extraction
8. **playwright** - Browser automation and testing
9. **puppeteer** - Browser automation alternative
10. **magic** - UI component generation (@21st-dev/magic)
11. **s2-mcp** - SingleStore MCP server for database operations
12. **sequential-thinking** - Advanced reasoning and thinking server
13. **ref** - Reference and documentation lookup
14. **semgrep** - Code security analysis
15. **exasearch** - Advanced web search and research

## How to Use with GitHub Copilot

1. **Open Copilot Chat** in VS Code (Ctrl+Shift+I or Cmd+Shift+I)

2. **Enable Agent Mode**:
   - Click the "Ask" dropdown in the Copilot Chat
   - Select "Agent"
   - You should see MCP tools become available

3. **Select Tools**:
   - Click the "Tools" button in the chat interface
   - Toggle on the MCP servers you want to use for your task

4. **Use Natural Language**:
   - "Search for React components using firecrawl"
   - "Run security analysis with semgrep"
   - "Query the database using s2-mcp"
   - "Generate UI components with magic"

## Troubleshooting

### Check MCP Server Status
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Run "MCP: List Servers"
3. Select a server and choose "Show Output" to view logs

### Common Issues
- **Server not starting**: Check if required dependencies (Node.js, Python) are installed
- **Permission errors**: Ensure API keys and tokens are correctly configured
- **Tool not available**: Verify the server is running in the MCP server list

### Dependencies Required
- **Node.js and npm** - For npx-based servers
- **Python and uv** - For uvx-based servers (semgrep, s2-mcp)
- **Docker** - For containerized servers (if any)

## API Keys and Environment Variables

The following servers require API keys (configured in the MCP configuration):
- **magic**: Uses API key for UI component generation
- **ref**: Uses REF_API_KEY for documentation access
- **semgrep**: Uses SEMGREP_APP_TOKEN for security analysis
- **exasearch**: Uses EXA_API_KEY for web search
- **firecrawl**: Uses SSE endpoint with embedded credentials

## Server Types

- **stdio**: Servers that communicate via standard input/output
- **http**: Servers accessible via HTTP endpoints
- **sse**: Servers using Server-Sent Events for real-time communication

## Next Steps

1. Test each MCP server individually in Copilot Chat
2. Create workspace-specific tool combinations for common tasks
3. Set up additional MCP servers as needed for your workflow
4. Configure organization-level MCP policies if in an enterprise environment