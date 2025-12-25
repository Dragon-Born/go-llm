package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Anthropic Provider
// ═══════════════════════════════════════════════════════════════════════════

const anthropicBaseURL = "https://api.anthropic.com/v1"
const anthropicAPIVersion = "2023-06-01"

// AnthropicProvider implements Provider for Anthropic's API.
type AnthropicProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewAnthropicProvider creates an AnthropicProvider.
func NewAnthropicProvider(config ProviderConfig) *AnthropicProvider {
	if config.BaseURL == "" {
		config.BaseURL = anthropicBaseURL
	}
	if config.APIKey == "" {
		config.APIKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	client := http.DefaultClient
	if config.Timeout > 0 {
		client = &http.Client{Timeout: config.Timeout}
	}
	return &AnthropicProvider{config: config, httpClient: client}
}

// Name returns the provider identifier ("anthropic").
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

// Capabilities reports the features supported by this provider implementation.
func (p *AnthropicProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		Tools:     true,
		Vision:    true,
		Streaming: true,
		JSON:      true,
		Thinking:  true, // Claude supports extended thinking
		PDF:       true, // Claude supports PDF input
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Send
// ═══════════════════════════════════════════════════════════════════════════

// Send executes a non-streaming request.
func (p *AnthropicProvider) Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "ANTHROPIC_API_KEY not set",
		}
	}

	anthropicReq := p.buildRequest(req)

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s\n", colorDim("→"), p.Name(), "/messages")
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to read response", Err: err}
	}

	return p.parseResponse(respBody)
}

// ═══════════════════════════════════════════════════════════════════════════
// SendStream
// ═══════════════════════════════════════════════════════════════════════════

// SendStream executes a streaming request and invokes callback for each chunk.
func (p *AnthropicProvider) SendStream(ctx context.Context, req *ProviderRequest, callback StreamCallback) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "ANTHROPIC_API_KEY not set",
		}
	}

	anthropicReq := p.buildRequest(req)
	anthropicReq.Stream = true

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s (stream)\n", colorDim("→"), p.Name(), "/messages")
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     fmt.Sprintf("%d", resp.StatusCode),
			Message:  string(body),
		}
	}

	var fullContent strings.Builder
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, &ProviderError{Provider: p.Name(), Message: "stream read error", Err: err}
		}

		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		data := bytes.TrimPrefix(line, []byte("data: "))

		// Anthropic stream events
		var event struct {
			Type  string `json:"type"`
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}

		if err := json.Unmarshal(data, &event); err != nil {
			continue
		}

		// Handle content_block_delta events
		if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
			fullContent.WriteString(event.Delta.Text)
			callback(event.Delta.Text)
		}

		// message_stop indicates end
		if event.Type == "message_stop" {
			break
		}
	}

	completionTokens := len(fullContent.String()) / 4

	return &ProviderResponse{
		Content:          fullContent.String(),
		CompletionTokens: completionTokens,
		TotalTokens:      completionTokens,
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

// anthropicRequest is Anthropic's API format
type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Stream      bool               `json:"stream,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	// Extended thinking (Claude)
	Thinking *anthropicThinking `json:"thinking,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []anthropicContent
}

type anthropicContent struct {
	Type   string       `json:"type"`
	Text   string       `json:"text,omitempty"`
	Source *mediaSource `json:"source,omitempty"`
}

type mediaSource struct {
	Type      string `json:"type"`       // "base64" or "url"
	MediaType string `json:"media_type"` // e.g., "image/png", "application/pdf"
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

func (p *AnthropicProvider) buildRequest(req *ProviderRequest) *anthropicRequest {
	// Convert our messages to Anthropic format
	var system string
	var messages []anthropicMessage

	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Anthropic takes system as a separate field
			if content, ok := msg.Content.(string); ok {
				system = content
			}
			continue
		}

		// Convert role names
		role := msg.Role
		if role == "assistant" {
			role = "assistant"
		} else {
			role = "user"
		}

		// Handle content (string or multimodal)
		var content any
		switch c := msg.Content.(type) {
		case string:
			content = c
		case []ContentPart:
			// Convert to Anthropic's content format
			var parts []anthropicContent
			for _, part := range c {
				if part.Type == "text" {
					parts = append(parts, anthropicContent{
						Type: "text",
						Text: part.Text,
					})
				} else if part.Type == "image_url" && part.ImageURL != nil {
					// Parse data URI
					url := part.ImageURL.URL
					if strings.HasPrefix(url, "data:") {
						// Extract media type and base64 data
						dataParts := strings.SplitN(url, ",", 2)
						if len(dataParts) == 2 {
							mediaType := strings.TrimPrefix(strings.Split(dataParts[0], ";")[0], "data:")
							parts = append(parts, anthropicContent{
								Type: "image",
								Source: &mediaSource{
									Type:      "base64",
									MediaType: mediaType,
									Data:      dataParts[1],
								},
							})
						}
					}
				} else if part.Type == "document" && part.Document != nil {
					// Handle PDF/document content
					doc := part.Document
					if doc.Data != "" {
						// Base64-encoded document
						parts = append(parts, anthropicContent{
							Type: "document",
							Source: &mediaSource{
								Type:      "base64",
								MediaType: doc.MimeType,
								Data:      doc.Data,
							},
						})
					} else if doc.URL != "" {
						// URL-based document
						parts = append(parts, anthropicContent{
							Type: "document",
							Source: &mediaSource{
								Type:      "url",
								MediaType: doc.MimeType,
								URL:       doc.URL,
							},
						})
					}
				}
			}
			content = parts
		default:
			content = msg.Content
		}

		messages = append(messages, anthropicMessage{
			Role:    role,
			Content: content,
		})
	}

	anthropicReq := &anthropicRequest{
		Model:     resolveModel(ProviderAnthropic, Model(req.Model)),
		MaxTokens: 8192, // Default max tokens
		System:    system,
		Messages:  messages,
	}

	if req.Temperature != nil {
		anthropicReq.Temperature = req.Temperature
	}

	// Claude extended thinking
	if req.Thinking != "" {
		budgetTokens := 1024
		switch req.Thinking {
		case ThinkingLow:
			budgetTokens = 1024
		case ThinkingMedium:
			budgetTokens = 4096
		case ThinkingHigh:
			budgetTokens = 16384
		}
		anthropicReq.Thinking = &anthropicThinking{
			Type:         "enabled",
			BudgetTokens: budgetTokens,
		}
	}

	// Convert tools to Anthropic format
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			anthropicReq.Tools = append(anthropicReq.Tools, anthropicTool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: tool.Function.Parameters,
			})
		}
	}

	return anthropicReq
}

func (p *AnthropicProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.config.APIKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)

	for k, v := range p.config.Headers {
		req.Header.Set(k, v)
	}
}

func (p *AnthropicProvider) parseResponse(body []byte) (*ProviderResponse, error) {
	var result struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type  string         `json:"type"`
			Text  string         `json:"text,omitempty"`
			ID    string         `json:"id,omitempty"`
			Name  string         `json:"name,omitempty"`
			Input map[string]any `json:"input,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("parse error: %v\nBody: %s", err, string(body)),
		}
	}

	if result.Error != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     result.Error.Type,
			Message:  result.Error.Message,
		}
	}

	// Extract text content and tool calls
	var content strings.Builder
	var toolCalls []ToolCall

	for _, block := range result.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "tool_use":
			// Convert to our ToolCall format
			argsJSON, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      block.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	return &ProviderResponse{
		Content:          content.String(),
		ToolCalls:        toolCalls,
		PromptTokens:     result.Usage.InputTokens,
		CompletionTokens: result.Usage.OutputTokens,
		TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
		FinishReason:     result.StopReason,
	}, nil
}
