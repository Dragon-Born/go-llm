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
// Google / Gemini Provider
// ═══════════════════════════════════════════════════════════════════════════

const googleBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// GoogleProvider implements Provider for Google's Gemini API.
type GoogleProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewGoogleProvider creates a GoogleProvider.
func NewGoogleProvider(config ProviderConfig) *GoogleProvider {
	if config.BaseURL == "" {
		config.BaseURL = googleBaseURL
	}
	if config.APIKey == "" {
		config.APIKey = getEnvWithFallback("GOOGLE_API_KEY", "GEMINI_API_KEY")
	}
	client := http.DefaultClient
	if config.Timeout > 0 {
		client = &http.Client{Timeout: config.Timeout}
	}
	return &GoogleProvider{config: config, httpClient: client}
}

// Name returns the provider identifier ("google").
func (p *GoogleProvider) Name() string {
	return "google"
}

// Capabilities reports the features supported by this provider implementation.
func (p *GoogleProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		Tools:     true,
		Vision:    true,
		Streaming: true,
		JSON:      true,
		Thinking:  true, // Gemini supports thinking mode
		PDF:       true, // Gemini supports PDF input
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Send
// ═══════════════════════════════════════════════════════════════════════════

// Send executes a non-streaming request.
func (p *GoogleProvider) Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "GOOGLE_API_KEY or GEMINI_API_KEY not set",
		}
	}

	geminiReq := p.buildRequest(req)
	model := resolveModel(ProviderGoogle, Model(req.Model))

	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", p.config.BaseURL, model, p.config.APIKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST /models/%s:generateContent\n", colorDim("→"), p.Name(), model)
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
func (p *GoogleProvider) SendStream(ctx context.Context, req *ProviderRequest, callback StreamCallback) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "GOOGLE_API_KEY or GEMINI_API_KEY not set",
		}
	}

	geminiReq := p.buildRequest(req)
	model := resolveModel(ProviderGoogle, Model(req.Model))

	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", p.config.BaseURL, model, p.config.APIKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST /models/%s:streamGenerateContent (stream)\n", colorDim("→"), p.Name(), model)
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

		var chunk struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}

		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}

		if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
			text := chunk.Candidates[0].Content.Parts[0].Text
			fullContent.WriteString(text)
			callback(text)
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

// geminiRequest is Gemini's API format
type geminiRequest struct {
	Contents         []geminiContent       `json:"contents"`
	SystemInstruct   *geminiContent        `json:"systemInstruction,omitempty"`
	GenerationConfig *geminiGenerateConfig `json:"generationConfig,omitempty"`
	Tools            []geminiTool          `json:"tools,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text       string          `json:"text,omitempty"`
	InlineData *geminiInline   `json:"inlineData,omitempty"`
	FileData   *geminiFileData `json:"fileData,omitempty"`
}

type geminiInline struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type geminiGenerateConfig struct {
	Temperature      *float64              `json:"temperature,omitempty"`
	ResponseMimeType string                `json:"responseMimeType,omitempty"`
	ThinkingConfig   *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiThinkingConfig struct {
	// Gemini supports both a legacy "thinkingBudget" and the newer "thinkingLevel".
	// Do not send both in the same request (Gemini 3 returns a 400).
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
	ThinkingBudget int    `json:"thinkingBudget,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDecl `json:"functionDeclarations,omitempty"`
}

type geminiFunctionDecl struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

func (p *GoogleProvider) buildRequest(req *ProviderRequest) *geminiRequest {
	geminiReq := &geminiRequest{
		Contents: []geminiContent{},
	}

	// Process messages
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Gemini uses systemInstruction
			if content, ok := msg.Content.(string); ok {
				geminiReq.SystemInstruct = &geminiContent{
					Parts: []geminiPart{{Text: content}},
				}
			}
			continue
		}

		// Map roles: user -> user, assistant -> model
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}

		var parts []geminiPart

		switch c := msg.Content.(type) {
		case string:
			parts = append(parts, geminiPart{Text: c})
		case []ContentPart:
			for _, part := range c {
				switch part.Type {
				case "text":
					parts = append(parts, geminiPart{Text: part.Text})
				case "image_url":
					if part.ImageURL != nil {
						url := part.ImageURL.URL
						if strings.HasPrefix(url, "data:") {
							// Parse data URI
							dataParts := strings.SplitN(url, ",", 2)
							if len(dataParts) == 2 {
								mimeType := strings.TrimPrefix(strings.Split(dataParts[0], ";")[0], "data:")
								parts = append(parts, geminiPart{
									InlineData: &geminiInline{
										MimeType: mimeType,
										Data:     dataParts[1],
									},
								})
							}
						}
					}
				case "document":
					if part.Document != nil {
						doc := part.Document
						if doc.Data != "" {
							// Inline base64 document
							parts = append(parts, geminiPart{
								InlineData: &geminiInline{
									MimeType: doc.MimeType,
									Data:     doc.Data,
								},
							})
						} else if doc.URL != "" {
							// File URI (for Google Cloud Storage)
							parts = append(parts, geminiPart{
								FileData: &geminiFileData{
									MimeType: doc.MimeType,
									FileURI:  doc.URL,
								},
							})
						}
					}
				}
			}
		}

		if len(parts) > 0 {
			geminiReq.Contents = append(geminiReq.Contents, geminiContent{
				Role:  role,
				Parts: parts,
			})
		}
	}

	// Generation config
	geminiReq.GenerationConfig = &geminiGenerateConfig{}

	if req.Temperature != nil {
		geminiReq.GenerationConfig.Temperature = req.Temperature
	}

	if req.JSONMode {
		geminiReq.GenerationConfig.ResponseMimeType = "application/json"
	}

	// Thinking/reasoning config
	if req.Thinking != "" {
		geminiReq.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
			ThinkingLevel: string(req.Thinking),
		}
	}

	// Tools
	if len(req.Tools) > 0 {
		var funcs []geminiFunctionDecl
		for _, tool := range req.Tools {
			funcs = append(funcs, geminiFunctionDecl{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			})
		}
		geminiReq.Tools = []geminiTool{{FunctionDeclarations: funcs}}
	}

	return geminiReq
}

func (p *GoogleProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")

	for k, v := range p.config.Headers {
		req.Header.Set(k, v)
	}
}

func (p *GoogleProvider) parseResponse(body []byte) (*ProviderResponse, error) {
	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text         string `json:"text,omitempty"`
					FunctionCall *struct {
						Name string         `json:"name"`
						Args map[string]any `json:"args"`
					} `json:"functionCall,omitempty"`
				} `json:"parts"`
				Role string `json:"role"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
		Error *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
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
			Code:     result.Error.Status,
			Message:  result.Error.Message,
		}
	}

	if len(result.Candidates) == 0 {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "no response candidates",
		}
	}

	// Extract text content and tool calls
	var content strings.Builder
	var toolCalls []ToolCall

	candidate := result.Candidates[0]
	for i, part := range candidate.Content.Parts {
		if part.Text != "" {
			content.WriteString(part.Text)
		}
		if part.FunctionCall != nil {
			argsJSON, _ := json.Marshal(part.FunctionCall.Args)
			toolCalls = append(toolCalls, ToolCall{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	return &ProviderResponse{
		Content:          content.String(),
		ToolCalls:        toolCalls,
		PromptTokens:     result.UsageMetadata.PromptTokenCount,
		CompletionTokens: result.UsageMetadata.CandidatesTokenCount,
		TotalTokens:      result.UsageMetadata.TotalTokenCount,
		FinishReason:     candidate.FinishReason,
	}, nil
}
