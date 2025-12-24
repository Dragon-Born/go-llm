package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Ollama Provider (Local)
// ═══════════════════════════════════════════════════════════════════════════

const ollamaDefaultURL = "http://localhost:11434"

// OllamaProvider implements Provider for local Ollama
type OllamaProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewOllamaProvider creates an Ollama provider
func NewOllamaProvider(config ProviderConfig) *OllamaProvider {
	if config.BaseURL == "" {
		config.BaseURL = ollamaDefaultURL
	}
	// Ollama doesn't need an API key by default
	client := http.DefaultClient
	if config.Timeout > 0 {
		client = &http.Client{Timeout: config.Timeout}
	}
	return &OllamaProvider{config: config, httpClient: client}
}

func (p *OllamaProvider) Name() string {
	return "ollama"
}

func (p *OllamaProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		Tools:     true, // Ollama supports tools with some models
		Vision:    true, // LLaVA and other vision models
		Streaming: true,
		JSON:      true,  // JSON mode supported
		Thinking:  false, // No built-in thinking mode
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Send
// ═══════════════════════════════════════════════════════════════════════════

func (p *OllamaProvider) Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	ollamaReq := p.buildRequest(req)

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s (model: %s)\n", colorDim("→"), p.Name(), "/api/chat", req.Model)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("request failed (is Ollama running at %s?): %v", p.config.BaseURL, err),
			Err:      err,
		}
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

func (p *OllamaProvider) SendStream(ctx context.Context, req *ProviderRequest, callback StreamCallback) (*ProviderResponse, error) {
	ollamaReq := p.buildRequest(req)
	ollamaReq.Stream = true

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s (stream, model: %s)\n", colorDim("→"), p.Name(), "/api/chat", req.Model)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("request failed (is Ollama running at %s?): %v", p.config.BaseURL, err),
			Err:      err,
		}
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
	var promptTokens, completionTokens int

	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, &ProviderError{Provider: p.Name(), Message: "stream read error", Err: err}
		}

		var chunk struct {
			Model   string `json:"model"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Done            bool `json:"done"`
			PromptEvalCount int  `json:"prompt_eval_count"`
			EvalCount       int  `json:"eval_count"`
		}

		if err := json.Unmarshal(line, &chunk); err != nil {
			continue
		}

		if chunk.Message.Content != "" {
			fullContent.WriteString(chunk.Message.Content)
			callback(chunk.Message.Content)
		}

		if chunk.Done {
			promptTokens = chunk.PromptEvalCount
			completionTokens = chunk.EvalCount
			break
		}
	}

	return &ProviderResponse{
		Content:          fullContent.String(),
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  *ollamaOptions  `json:"options,omitempty"`
	Format   string          `json:"format,omitempty"` // "json" for JSON mode
	Tools    []ollamaTool    `json:"tools,omitempty"`
}

type ollamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"` // base64 encoded images
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
}

type ollamaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
	} `json:"function"`
}

func (p *OllamaProvider) buildRequest(req *ProviderRequest) *ollamaRequest {
	// Convert messages to Ollama format
	var messages []ollamaMessage

	for _, msg := range req.Messages {
		ollamaMsg := ollamaMessage{
			Role: msg.Role,
		}

		// Handle content (string or multimodal)
		switch c := msg.Content.(type) {
		case string:
			ollamaMsg.Content = c
		case []ContentPart:
			// Extract text and images
			var textParts []string
			for _, part := range c {
				if part.Type == "text" {
					textParts = append(textParts, part.Text)
				} else if part.Type == "image_url" && part.ImageURL != nil {
					// Extract base64 from data URI
					url := part.ImageURL.URL
					if strings.HasPrefix(url, "data:") {
						parts := strings.SplitN(url, ",", 2)
						if len(parts) == 2 {
							ollamaMsg.Images = append(ollamaMsg.Images, parts[1])
						}
					}
				}
			}
			ollamaMsg.Content = strings.Join(textParts, "\n")
		}

		messages = append(messages, ollamaMsg)
	}

	ollamaReq := &ollamaRequest{
		Model:    req.Model, // Ollama uses raw model names
		Messages: messages,
		Stream:   false, // Set in SendStream
	}

	if req.Temperature != nil {
		ollamaReq.Options = &ollamaOptions{
			Temperature: *req.Temperature,
		}
	}

	if req.JSONMode {
		ollamaReq.Format = "json"
	}

	// Convert tools
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			ollamaReq.Tools = append(ollamaReq.Tools, ollamaTool{
				Type: "function",
				Function: struct {
					Name        string         `json:"name"`
					Description string         `json:"description"`
					Parameters  map[string]any `json:"parameters"`
				}{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			})
		}
	}

	return ollamaReq
}

func (p *OllamaProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")

	// API key if set (some Ollama deployments use auth)
	if p.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	}

	for k, v := range p.config.Headers {
		req.Header.Set(k, v)
	}
}

func (p *OllamaProvider) parseResponse(body []byte) (*ProviderResponse, error) {
	var result struct {
		Model   string `json:"model"`
		Message struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Function struct {
					Name      string         `json:"name"`
					Arguments map[string]any `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
		Done            bool   `json:"done"`
		PromptEvalCount int    `json:"prompt_eval_count"`
		EvalCount       int    `json:"eval_count"`
		Error           string `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("parse error: %v\nBody: %s", err, string(body)),
		}
	}

	if result.Error != "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  result.Error,
		}
	}

	// Convert tool calls
	var toolCalls []ToolCall
	for i, tc := range result.Message.ToolCalls {
		argsJSON, _ := json.Marshal(tc.Function.Arguments)
		toolCalls = append(toolCalls, ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{
				Name:      tc.Function.Name,
				Arguments: string(argsJSON),
			},
		})
	}

	return &ProviderResponse{
		Content:          result.Message.Content,
		ToolCalls:        toolCalls,
		PromptTokens:     result.PromptEvalCount,
		CompletionTokens: result.EvalCount,
		TotalTokens:      result.PromptEvalCount + result.EvalCount,
	}, nil
}
