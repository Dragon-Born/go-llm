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
// OpenRouter Provider
// ═══════════════════════════════════════════════════════════════════════════

const openRouterBaseURL = "https://openrouter.ai/api/v1"

// OpenRouterProvider implements Provider for OpenRouter
type OpenRouterProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewOpenRouterProvider creates an OpenRouter provider
func NewOpenRouterProvider(config ProviderConfig) *OpenRouterProvider {
	if config.BaseURL == "" {
		config.BaseURL = openRouterBaseURL
	}
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENROUTER_API_KEY")
	}
	client := http.DefaultClient
	if config.Timeout > 0 {
		client = &http.Client{Timeout: config.Timeout}
	}
	return &OpenRouterProvider{config: config, httpClient: client}
}

func (p *OpenRouterProvider) Name() string {
	return "openrouter"
}

func (p *OpenRouterProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		Tools:     true,
		Vision:    true,
		Streaming: true,
		JSON:      true,
		Thinking:  true,
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Send
// ═══════════════════════════════════════════════════════════════════════════

func (p *OpenRouterProvider) Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "OPENROUTER_API_KEY not set",
		}
	}

	// Build OpenRouter request
	orReq := p.buildRequest(req)

	body, err := json.Marshal(orReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s\n", colorDim("→"), p.Name(), "/chat/completions")
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

func (p *OpenRouterProvider) SendStream(ctx context.Context, req *ProviderRequest, callback StreamCallback) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "OPENROUTER_API_KEY not set",
		}
	}

	// Build request with streaming enabled
	orReq := p.buildRequest(req)
	orReq.Stream = true

	body, err := json.Marshal(orReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s (stream)\n", colorDim("→"), p.Name(), "/chat/completions")
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
		if string(data) == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			fullContent.WriteString(content)
			callback(content)
		}
	}

	// Approximate token count for streaming (no usage data in stream)
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

// openRouterRequest is the OpenRouter API request format
type openRouterRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Stream         bool            `json:"stream,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	Reasoning      ThinkingLevel   `json:"reasoning,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

func (p *OpenRouterProvider) buildRequest(req *ProviderRequest) *openRouterRequest {
	orReq := &openRouterRequest{
		Model:    resolveModel(ProviderOpenRouter, Model(req.Model)),
		Messages: req.Messages,
	}

	if req.Temperature != nil {
		orReq.Temperature = req.Temperature
	}
	if req.Thinking != "" {
		orReq.Reasoning = req.Thinking
	}
	if len(req.Tools) > 0 {
		orReq.Tools = req.Tools
		orReq.ToolChoice = "auto"
	}
	if req.JSONMode {
		orReq.ResponseFormat = &ResponseFormat{Type: "json_object"}
	}

	return orReq
}

func (p *OpenRouterProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	req.Header.Set("HTTP-Referer", "gopkg.in/dragon-born/go-llm.v1")
	req.Header.Set("X-Title", "dragon-born/go-llm")

	for k, v := range p.config.Headers {
		req.Header.Set(k, v)
	}
}

func (p *OpenRouterProvider) parseResponse(body []byte) (*ProviderResponse, error) {
	var result struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Role      string     `json:"role"`
				Content   string     `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
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

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("parse error: %v\nBody: %s", err, string(body)),
		}
	}

	if result.Error != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     result.Error.Code,
			Message:  result.Error.Message,
		}
	}

	if len(result.Choices) == 0 {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "no response choices",
		}
	}

	choice := result.Choices[0]
	return &ProviderResponse{
		Content:          choice.Message.Content,
		ToolCalls:        choice.Message.ToolCalls,
		PromptTokens:     result.Usage.PromptTokens,
		CompletionTokens: result.Usage.CompletionTokens,
		TotalTokens:      result.Usage.TotalTokens,
		FinishReason:     choice.FinishReason,
	}, nil
}
