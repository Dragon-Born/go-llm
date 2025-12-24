package ai

// ═══════════════════════════════════════════════════════════════════════════
// OpenAI Responses API - Built-in Tools
// ═══════════════════════════════════════════════════════════════════════════
//
// These tools use OpenAI's Responses API (/v1/responses) which provides
// built-in capabilities like web search, file search, code execution, and MCP.
//
// Usage:
//   resp, _ := ai.GPT5().WebSearch().User("Latest AI news?").Send()
//   resp, _ := ai.GPT5().FileSearch("vs_abc123").User("What's our policy?").Send()
//   resp, _ := ai.GPT5().CodeInterpreter().User("Calculate factorial of 50").Send()
//   resp, _ := ai.GPT5().MCP("dice", "https://example.com/mcp").User("Roll 2d6").Send()
//
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Tool Types
// ═══════════════════════════════════════════════════════════════════════════

// BuiltinTool represents a built-in tool for the Responses API
type BuiltinTool struct {
	Type string `json:"type"` // "web_search", "file_search", "code_interpreter", "mcp"

	// Web Search options
	UserLocation *UserLocation `json:"user_location,omitempty"`
	SearchFilter *SearchFilter `json:"search_filters,omitempty"` // domain filters for web search

	// File Search options
	VectorStoreIDs []string `json:"vector_store_ids,omitempty"`
	MaxNumResults  int      `json:"max_num_results,omitempty"`
	FileFilter     any      `json:"file_filters,omitempty"` // attribute filters for file search

	// Code Interpreter options
	Container any `json:"container,omitempty"` // string ID or ContainerConfig

	// MCP options
	ServerLabel       string   `json:"server_label,omitempty"`
	ServerURL         string   `json:"server_url,omitempty"`
	ServerDescription string   `json:"server_description,omitempty"`
	ConnectorID       string   `json:"connector_id,omitempty"`
	Authorization     string   `json:"authorization,omitempty"`
	RequireApproval   any      `json:"require_approval,omitempty"` // "always", "never", or ApprovalConfig
	AllowedTools      []string `json:"allowed_tools,omitempty"`
}

// UserLocation for web search geo-targeting
type UserLocation struct {
	Type     string `json:"type"`               // "approximate"
	Country  string `json:"country,omitempty"`  // ISO country code (e.g., "US")
	City     string `json:"city,omitempty"`     // City name
	Region   string `json:"region,omitempty"`   // State/region
	Timezone string `json:"timezone,omitempty"` // IANA timezone
}

// SearchFilter for web search domain filtering
type SearchFilter struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}

// ContainerConfig for code interpreter
type ContainerConfig struct {
	Type        string   `json:"type"`                   // "auto"
	MemoryLimit string   `json:"memory_limit,omitempty"` // "1g", "4g", "16g", "64g"
	FileIDs     []string `json:"file_ids,omitempty"`
}

// ApprovalConfig for MCP tool approval
type ApprovalConfig struct {
	Never  *ToolNameFilter `json:"never,omitempty"`
	Always *ToolNameFilter `json:"always,omitempty"`
}

// ToolNameFilter for MCP allowed/approval tools
type ToolNameFilter struct {
	ToolNames []string `json:"tool_names,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Option Types for Builder Methods
// ═══════════════════════════════════════════════════════════════════════════

// WebSearchOptions configures web search behavior
type WebSearchOptions struct {
	// Location for geo-targeted results
	Country  string // ISO country code (e.g., "US", "GB")
	City     string // City name
	Region   string // State/region
	Timezone string // IANA timezone (e.g., "America/New_York")

	// Domain filtering
	AllowedDomains []string // Limit search to these domains
}

// FileSearchOptions configures file search behavior
type FileSearchOptions struct {
	VectorStoreIDs []string // Vector store IDs to search
	MaxNumResults  int      // Max results (default 10, max 50)
	Filters        any      // Attribute filters for metadata
}

// CodeInterpreterOptions configures code interpreter
type CodeInterpreterOptions struct {
	ContainerID string   // Existing container ID (optional)
	MemoryLimit string   // "1g", "4g", "16g", "64g" (default "1g")
	FileIDs     []string // Files to make available
}

// MCPOptions configures MCP server connection
type MCPOptions struct {
	Label           string   // Unique label for this server
	URL             string   // Server URL (for remote MCP)
	ConnectorID     string   // Connector ID (for built-in connectors)
	Description     string   // Description for the model
	Authorization   string   // OAuth token or API key
	RequireApproval string   // "always", "never", or use ApprovalConfig
	AllowedTools    []string // Limit to specific tools
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods
// ═══════════════════════════════════════════════════════════════════════════

// WebSearch enables web search for this request
func (b *Builder) WebSearch() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "web_search",
	})
	return b
}

// WebSearchWith enables web search with custom options
func (b *Builder) WebSearchWith(opts WebSearchOptions) *Builder {
	tool := BuiltinTool{Type: "web_search"}

	// Set location if any location field is provided
	if opts.Country != "" || opts.City != "" || opts.Region != "" || opts.Timezone != "" {
		tool.UserLocation = &UserLocation{
			Type:     "approximate",
			Country:  opts.Country,
			City:     opts.City,
			Region:   opts.Region,
			Timezone: opts.Timezone,
		}
	}

	// Set domain filter
	if len(opts.AllowedDomains) > 0 {
		tool.SearchFilter = &SearchFilter{
			AllowedDomains: opts.AllowedDomains,
		}
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// FileSearch enables file search with the specified vector stores
func (b *Builder) FileSearch(vectorStoreIDs ...string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:           "file_search",
		VectorStoreIDs: vectorStoreIDs,
	})
	return b
}

// FileSearchWith enables file search with custom options
func (b *Builder) FileSearchWith(opts FileSearchOptions) *Builder {
	tool := BuiltinTool{
		Type:           "file_search",
		VectorStoreIDs: opts.VectorStoreIDs,
		MaxNumResults:  opts.MaxNumResults,
		FileFilter:     opts.Filters,
	}
	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// CodeInterpreter enables Python code execution
func (b *Builder) CodeInterpreter() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "code_interpreter",
		Container: ContainerConfig{
			Type: "auto",
		},
	})
	return b
}

// CodeInterpreterWith enables code interpreter with custom options
func (b *Builder) CodeInterpreterWith(opts CodeInterpreterOptions) *Builder {
	tool := BuiltinTool{Type: "code_interpreter"}

	if opts.ContainerID != "" {
		tool.Container = opts.ContainerID
	} else {
		tool.Container = ContainerConfig{
			Type:        "auto",
			MemoryLimit: opts.MemoryLimit,
			FileIDs:     opts.FileIDs,
		}
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// MCP connects to a remote MCP server
func (b *Builder) MCP(label, url string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:            "mcp",
		ServerLabel:     label,
		ServerURL:       url,
		RequireApproval: "never",
	})
	return b
}

// MCPWith connects to an MCP server with custom options
func (b *Builder) MCPWith(opts MCPOptions) *Builder {
	tool := BuiltinTool{
		Type:              "mcp",
		ServerLabel:       opts.Label,
		ServerURL:         opts.URL,
		ConnectorID:       opts.ConnectorID,
		ServerDescription: opts.Description,
		Authorization:     opts.Authorization,
		AllowedTools:      opts.AllowedTools,
	}

	if opts.RequireApproval != "" {
		tool.RequireApproval = opts.RequireApproval
	} else {
		tool.RequireApproval = "never" // sensible default
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// MCPConnector connects to a built-in OpenAI connector (Dropbox, Gmail, etc.)
func (b *Builder) MCPConnector(label, connectorID, authToken string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:            "mcp",
		ServerLabel:     label,
		ConnectorID:     connectorID,
		Authorization:   authToken,
		RequireApproval: "never",
	})
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Connector IDs (for MCPConnector)
// ═══════════════════════════════════════════════════════════════════════════

const (
	ConnectorDropbox         = "connector_dropbox"
	ConnectorGmail           = "connector_gmail"
	ConnectorGoogleCalendar  = "connector_googlecalendar"
	ConnectorGoogleDrive     = "connector_googledrive"
	ConnectorMicrosoftTeams  = "connector_microsoftteams"
	ConnectorOutlookCalendar = "connector_outlookcalendar"
	ConnectorOutlookEmail    = "connector_outlookemail"
	ConnectorSharePoint      = "connector_sharepoint"
)

// ═══════════════════════════════════════════════════════════════════════════
// Response Types (Responses API Output)
// ═══════════════════════════════════════════════════════════════════════════

// ResponsesOutput represents the parsed output from Responses API
type ResponsesOutput struct {
	// Text content from the model
	Text string

	// Citations from web/file search
	Citations []Citation

	// Sources from web search (all URLs consulted)
	Sources []Source

	// Tool calls made (web_search_call, file_search_call, mcp_call, etc.)
	ToolCalls []ResponsesToolCall

	// Raw output items (for advanced use)
	OutputItems []any
}

// Citation represents a URL or file citation in the response
type Citation struct {
	Type       string `json:"type"` // "url_citation" or "file_citation"
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	FileID     string `json:"file_id,omitempty"`
	Filename   string `json:"filename,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
}

// Source represents a web source consulted
type Source struct {
	URL   string `json:"url"`
	Title string `json:"title,omitempty"`
}

// ResponsesToolCall represents a tool call from Responses API
type ResponsesToolCall struct {
	ID          string `json:"id"`
	Type        string `json:"type"` // "web_search_call", "file_search_call", "mcp_call", "code_interpreter_call"
	Status      string `json:"status"`
	ServerLabel string `json:"server_label,omitempty"` // for MCP
	Name        string `json:"name,omitempty"`         // tool name
	Arguments   string `json:"arguments,omitempty"`
	Output      string `json:"output,omitempty"`
	Error       string `json:"error,omitempty"`
}

// HasBuiltinTools returns true if any built-in tools are configured
func (b *Builder) HasBuiltinTools() bool {
	return len(b.builtinTools) > 0
}
