package ai

// ═══════════════════════════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════════════════════════

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []ContentPart for vision

	// Tool calling fields
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ContentPart represents a part of multimodal content
type ContentPart struct {
	Type     string       `json:"type"` // "text", "image_url", or "document"
	Text     string       `json:"text,omitempty"`
	ImageURL *ImageURL    `json:"image_url,omitempty"`
	Document *DocumentRef `json:"document,omitempty"`
}

// DocumentRef represents a document reference in a message
type DocumentRef struct {
	Data     string `json:"data,omitempty"`      // base64-encoded content
	URL      string `json:"url,omitempty"`       // or URL
	MimeType string `json:"mime_type,omitempty"` // e.g., "application/pdf"
	Name     string `json:"name,omitempty"`      // optional filename
}

// ImageURL represents an image URL in a message
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// ═══════════════════════════════════════════════════════════════════════════
// Thinking Level (Reasoning Effort)
// ═══════════════════════════════════════════════════════════════════════════

// ThinkingLevel controls reasoning/thinking effort
// Gemini Flash: minimal, low, medium, high
// Gemini Pro: low, high
type ThinkingLevel string

const (
	ThinkingNone    ThinkingLevel = ""
	ThinkingMinimal ThinkingLevel = "minimal" // Gemini Flash only - fastest
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
)

// ═══════════════════════════════════════════════════════════════════════════
// Request Types (for legacy compatibility)
// ═══════════════════════════════════════════════════════════════════════════

// Request is the OpenRouter API request (legacy, kept for compatibility)
type Request struct {
	Model       string        `json:"model"`
	Messages    []Message     `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	Reasoning   ThinkingLevel `json:"reasoning,omitempty"`

	// Tool calling
	Tools      []Tool `json:"tools,omitempty"`
	ToolChoice any    `json:"tool_choice,omitempty"` // "auto", "none", or specific

	// Structured output
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat for structured output
type ResponseFormat struct {
	Type       string `json:"type"` // "json_object" or "json_schema"
	JSONSchema any    `json:"json_schema,omitempty"`
}

// Response is the OpenRouter API response (legacy, kept for compatibility)
type Response struct {
	ID      string `json:"id"`
	Choices []struct {
		Message      ResponseMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

// ResponseMessage is the message in a response (different from request Message)
type ResponseMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// SendOptions contains optional parameters for Send
type SendOptions struct {
	// Temperature is the sampling temperature. If nil, the provider default is used.
	Temperature *float64
	Thinking    ThinkingLevel
}
