package ai

// ═══════════════════════════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════════════════════════

// Message represents a single message in a conversation.
// Role can be "user", "assistant", "system", or "tool".
type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []ContentPart for vision

	// Tool calling fields
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ContentPart represents a segment of a multimodal message.
// Used for combining text, images, and documents in a single message.
type ContentPart struct {
	Type     string       `json:"type"` // "text", "image_url", or "document"
	Text     string       `json:"text,omitempty"`
	ImageURL *ImageURL    `json:"image_url,omitempty"`
	Document *DocumentRef `json:"document,omitempty"`
}

// DocumentRef represents a reference to a document file (e.g., PDF).
// Supported by some models for document analysis.
type DocumentRef struct {
	Data     string `json:"data,omitempty"`      // base64-encoded content
	URL      string `json:"url,omitempty"`       // or URL
	MimeType string `json:"mime_type,omitempty"` // e.g., "application/pdf"
	Name     string `json:"name,omitempty"`      // optional filename
}

// ImageURL represents an image source for vision capabilities.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// ═══════════════════════════════════════════════════════════════════════════
// Thinking Level (Reasoning Effort)
// ═══════════════════════════════════════════════════════════════════════════

// ThinkingLevel controls the effort level for reasoning models.
// Higher levels typically result in better reasoning but higher latency and cost.
// Supported by models like Gemini Pro/Flash.
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

// Request represents a standard API request payload.
// Used internally by providers to structure the request to the AI service.
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

// ResponseFormat defines the structure for enforced output formats (e.g., JSON).
type ResponseFormat struct {
	Type       string `json:"type"` // "json_object" or "json_schema"
	JSONSchema any    `json:"json_schema,omitempty"`
}

// Response represents a standard API response.
// Used internally to parse the response from the AI service.
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

// ResponseMessage is the internal representation of a message in the response.
// It differs from Message by having specific fields for tool calls as returned by APIs.
type ResponseMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// SendOptions contains runtime options for a request.
// Passed to Send() functions to override defaults.
type SendOptions struct {
	// Temperature is the sampling temperature. If nil, the provider default is used.
	Temperature *float64
	Thinking    ThinkingLevel
}
