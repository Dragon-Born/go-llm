package ai

import (
	"context"
	"fmt"
	"os"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Client - Multi-Provider Client
// ═══════════════════════════════════════════════════════════════════════════

// Client wraps a Provider and creates builders for it
type Client struct {
	provider     Provider
	providerType ProviderType
}

// NewClient creates a client for a specific provider
func NewClient(providerType ProviderType, opts ...ClientOption) *Client {
	config := ProviderConfig{}
	for _, opt := range opts {
		opt(&config)
	}

	var provider Provider
	switch providerType {
	case ProviderOpenRouter:
		provider = NewOpenRouterProvider(config)
	case ProviderOpenAI:
		provider = NewOpenAIProvider(config)
	case ProviderAnthropic:
		provider = NewAnthropicProvider(config)
	case ProviderGoogle:
		provider = NewGoogleProvider(config)
	case ProviderOllama:
		provider = NewOllamaProvider(config)
	case ProviderAzure:
		provider = NewAzureProvider(config)
	default:
		provider = NewOpenRouterProvider(config)
	}

	return &Client{
		provider:     provider,
		providerType: providerType,
	}
}

// NewClientWithProvider creates a client with a custom provider implementation
func NewClientWithProvider(provider Provider) *Client {
	return &Client{
		provider:     provider,
		providerType: ProviderType(provider.Name()),
	}
}

// Provider returns the underlying provider
func (c *Client) Provider() Provider {
	return c.provider
}

// ═══════════════════════════════════════════════════════════════════════════
// Client Options (functional options pattern)
// ═══════════════════════════════════════════════════════════════════════════

// ClientOption configures a client
type ClientOption func(*ProviderConfig)

// WithAPIKey sets the API key
func WithAPIKey(key string) ClientOption {
	return func(c *ProviderConfig) {
		c.APIKey = key
	}
}

// WithBaseURL sets a custom base URL
func WithBaseURL(url string) ClientOption {
	return func(c *ProviderConfig) {
		c.BaseURL = url
	}
}

// WithTimeout sets the request timeout
func WithTimeout(d time.Duration) ClientOption {
	return func(c *ProviderConfig) {
		c.Timeout = d
	}
}

// WithHeaders sets custom headers
func WithHeaders(headers map[string]string) ClientOption {
	return func(c *ProviderConfig) {
		c.Headers = headers
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Shortcuts on Client
// ═══════════════════════════════════════════════════════════════════════════

// New creates a builder for any model on this client
func (c *Client) New(model Model) *Builder {
	b := New(model)
	b.client = c
	return b
}

// Use creates a builder for a model by string ID (for Ollama, custom models)
func (c *Client) Use(modelID string) *Builder {
	return c.New(Model(modelID))
}

// ═══════════════════════════════════════════════════════════════════════════
// OpenAI Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

func (c *Client) GPT5() *Builder      { return c.New(ModelGPT5) }      // alias of GPT-5.2
func (c *Client) GPT5Codex() *Builder { return c.New(ModelGPT5Codex) } // alias of GPT-5.1 Codex Max
func (c *Client) GPT4o() *Builder     { return c.New(ModelGPT4o) }
func (c *Client) GPT4oMini() *Builder { return c.New(ModelGPT4oMini) }
func (c *Client) O1() *Builder        { return c.New(ModelO1) }

// GPT 5.x family
func (c *Client) GPT52() *Builder    { return c.New(ModelGPT52) }
func (c *Client) GPT52Pro() *Builder { return c.New(ModelGPT52Pro) }
func (c *Client) GPT51() *Builder    { return c.New(ModelGPT51) }
func (c *Client) GPT5Base() *Builder { return c.New(ModelGPT5Base) }
func (c *Client) GPT5Pro() *Builder  { return c.New(ModelGPT5Pro) }
func (c *Client) GPT5Mini() *Builder { return c.New(ModelGPT5Mini) }
func (c *Client) GPT5Nano() *Builder { return c.New(ModelGPT5Nano) }

// Codex
func (c *Client) GPT51Codex() *Builder      { return c.New(ModelGPT51Codex) }
func (c *Client) GPT51CodexMax() *Builder   { return c.New(ModelGPT51CodexMax) }
func (c *Client) GPT5CodexBase() *Builder   { return c.New(ModelGPT5CodexBase) }
func (c *Client) GPT51CodexMini() *Builder  { return c.New(ModelGPT51CodexMini) }
func (c *Client) CodexMiniLatest() *Builder { return c.New(ModelCodexMiniLatest) }

// Search + agent tools
func (c *Client) GPT5SearchAPI() *Builder      { return c.New(ModelGPT5SearchAPI) }
func (c *Client) ComputerUsePreview() *Builder { return c.New(ModelComputerUsePreview) }

// Aliases
func (c *Client) GPT5ChatLatest() *Builder  { return c.New(ModelGPT5ChatLatest) }
func (c *Client) GPT52ChatLatest() *Builder { return c.New(ModelGPT52ChatLatest) }
func (c *Client) GPT51ChatLatest() *Builder { return c.New(ModelGPT51ChatLatest) }
func (c *Client) ChatGPT4oLatest() *Builder { return c.New(ModelChatGPT4oLatest) }

// GPT-4.1
func (c *Client) GPT41() *Builder     { return c.New(ModelGPT41) }
func (c *Client) GPT41Mini() *Builder { return c.New(ModelGPT41Mini) }
func (c *Client) GPT41Nano() *Builder { return c.New(ModelGPT41Nano) }

// GPT-4o dated snapshot
func (c *Client) GPT4o20240513() *Builder { return c.New(ModelGPT4o20240513) }

// o-series
func (c *Client) O1Mini() *Builder    { return c.New(ModelO1Mini) }
func (c *Client) O1Pro() *Builder     { return c.New(ModelO1Pro) }
func (c *Client) O1Preview() *Builder { return c.New(ModelO1Preview) }
func (c *Client) O3() *Builder        { return c.New(ModelO3) }
func (c *Client) O3Mini() *Builder    { return c.New(ModelO3Mini) }
func (c *Client) O3Pro() *Builder     { return c.New(ModelO3Pro) }
func (c *Client) O3DeepResearch() *Builder {
	return c.New(ModelO3DeepResearch)
}
func (c *Client) O4Mini() *Builder { return c.New(ModelO4Mini) }
func (c *Client) O4MiniDeepResearch() *Builder {
	return c.New(ModelO4MiniDeepResearch)
}

// Realtime / audio
func (c *Client) GPTRealtime() *Builder              { return c.New(ModelGPTRealtime) }
func (c *Client) GPTRealtimeMini() *Builder          { return c.New(ModelGPTRealtimeMini) }
func (c *Client) GPT4oRealtimePreview() *Builder     { return c.New(ModelGPT4oRealtimePreview) }
func (c *Client) GPT4oMiniRealtimePreview() *Builder { return c.New(ModelGPT4oMiniRealtimePreview) }
func (c *Client) GPTAudio() *Builder                 { return c.New(ModelGPTAudio) }
func (c *Client) GPTAudioMini() *Builder             { return c.New(ModelGPTAudioMini) }
func (c *Client) GPT4oAudioPreview() *Builder        { return c.New(ModelGPT4oAudioPreview) }
func (c *Client) GPT4oMiniAudioPreview() *Builder    { return c.New(ModelGPT4oMiniAudioPreview) }

// Search previews
func (c *Client) GPT4oMiniSearchPreview() *Builder { return c.New(ModelGPT4oMiniSearchPreview) }
func (c *Client) GPT4oSearchPreview() *Builder     { return c.New(ModelGPT4oSearchPreview) }

// Speech / transcription
func (c *Client) GPT4oMiniTTS() *Builder           { return c.New(ModelGPT4oMiniTTS) }
func (c *Client) GPT4oTranscribe() *Builder        { return c.New(ModelGPT4oTranscribe) }
func (c *Client) GPT4oTranscribeDiarize() *Builder { return c.New(ModelGPT4oTranscribeDiarize) }
func (c *Client) GPT4oMiniTranscribe() *Builder    { return c.New(ModelGPT4oMiniTranscribe) }

// Image generation
func (c *Client) GPTImage15() *Builder         { return c.New(ModelGPTImage15) }
func (c *Client) GPTImage1() *Builder          { return c.New(ModelGPTImage1) }
func (c *Client) GPTImage1Mini() *Builder      { return c.New(ModelGPTImage1Mini) }
func (c *Client) ChatGPTImageLatest() *Builder { return c.New(ModelChatGPTImageLatest) }

// Open-weight models
func (c *Client) GPTOSS120B() *Builder { return c.New(ModelGPTOSS120B) }
func (c *Client) GPTOSS20B() *Builder  { return c.New(ModelGPTOSS20B) }

// Video generation
func (c *Client) Sora2() *Builder    { return c.New(ModelSora2) }
func (c *Client) Sora2Pro() *Builder { return c.New(ModelSora2Pro) }

// ═══════════════════════════════════════════════════════════════════════════
// Anthropic Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

func (c *Client) Claude() *Builder       { return c.New(ModelClaudeOpus) }
func (c *Client) ClaudeOpus() *Builder   { return c.New(ModelClaudeOpus) }
func (c *Client) ClaudeSonnet() *Builder { return c.New(ModelClaudeSonnet) }
func (c *Client) ClaudeHaiku() *Builder  { return c.New(ModelClaudeHaiku) }

// Additional Claude variants (OpenRouter slugs, normalized for native Anthropic)
func (c *Client) ClaudeOpus41() *Builder   { return c.New(ModelClaudeOpus41) }
func (c *Client) ClaudeOpus4() *Builder    { return c.New(ModelClaudeOpus4) }
func (c *Client) ClaudeSonnet4() *Builder  { return c.New(ModelClaudeSonnet4) }
func (c *Client) ClaudeSonnet37() *Builder { return c.New(ModelClaudeSonnet37) }
func (c *Client) ClaudeHaiku35() *Builder  { return c.New(ModelClaudeHaiku35) }
func (c *Client) ClaudeHaiku3() *Builder   { return c.New(ModelClaudeHaiku3) }
func (c *Client) ClaudeOpus3() *Builder    { return c.New(ModelClaudeOpus3) }
func (c *Client) ClaudeSonnet3() *Builder  { return c.New(ModelClaudeSonnet3) }

// ═══════════════════════════════════════════════════════════════════════════
// Google Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

func (c *Client) Gemini() *Builder      { return c.New(ModelGemini3Flash) }
func (c *Client) GeminiPro() *Builder   { return c.New(ModelGemini3Pro) }
func (c *Client) GeminiFlash() *Builder { return c.New(ModelGemini3Flash) }

// Explicit Gemini 3 names (aliases for clarity)
func (c *Client) Gemini3Pro() *Builder        { return c.New(ModelGemini3Pro) }
func (c *Client) Gemini3Flash() *Builder      { return c.New(ModelGemini3Flash) }
func (c *Client) Gemini25Pro() *Builder       { return c.New(ModelGemini25Pro) }
func (c *Client) Gemini25Flash() *Builder     { return c.New(ModelGemini25Flash) }
func (c *Client) Gemini25FlashLite() *Builder { return c.New(ModelGemini25FlashLite) }
func (c *Client) Gemini2Pro() *Builder        { return c.New(ModelGemini2Pro) } // compat (maps to Gemini 2.5 Pro)
func (c *Client) Gemini2Flash() *Builder      { return c.New(ModelGemini2Flash) }
func (c *Client) Gemini2FlashLite() *Builder  { return c.New(ModelGemini2FlashLite) }

// ═══════════════════════════════════════════════════════════════════════════
// Other Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

func (c *Client) Grok() *Builder     { return c.New(ModelGrok3) }
func (c *Client) GrokFast() *Builder { return c.New(ModelGrok41Fast) }
func (c *Client) GrokMini() *Builder { return c.New(ModelGrok3Mini) }

// Explicit xAI names (aliases for symmetry)
func (c *Client) Grok3() *Builder      { return c.New(ModelGrok3) }
func (c *Client) Grok3Mini() *Builder  { return c.New(ModelGrok3Mini) }
func (c *Client) Grok41Fast() *Builder { return c.New(ModelGrok41Fast) }

func (c *Client) Qwen() *Builder      { return c.New(ModelQwen3Next) }
func (c *Client) Qwen3Next() *Builder { return c.New(ModelQwen3Next) }
func (c *Client) Qwen3() *Builder     { return c.New(ModelQwen3) }

func (c *Client) Llama() *Builder        { return c.New(ModelLlama4) }
func (c *Client) Llama4() *Builder       { return c.New(ModelLlama4) }
func (c *Client) Mistral() *Builder      { return c.New(ModelMistralLarge) }
func (c *Client) MistralLarge() *Builder { return c.New(ModelMistralLarge) }

// ═══════════════════════════════════════════════════════════════════════════
// Azure Provider (special constructor)
// ═══════════════════════════════════════════════════════════════════════════

// AzureProvider is a specialized OpenAI provider for Azure deployments
type AzureProvider struct {
	*OpenAIProvider
	deploymentURL string
}

// NewAzureProvider creates an Azure OpenAI provider
func NewAzureProvider(config ProviderConfig) *AzureProvider {
	// Azure requires a custom endpoint
	if config.BaseURL == "" {
		// User must provide endpoint
		config.BaseURL = "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT"
	}
	if config.APIKey == "" {
		config.APIKey = getEnvWithFallback("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
	}

	return &AzureProvider{
		OpenAIProvider: NewOpenAIProvider(config),
		deploymentURL:  config.BaseURL,
	}
}

func (p *AzureProvider) Name() string {
	return "azure"
}

// Helper to get env with fallback
func getEnvWithFallback(primary, fallback string) string {
	if v := os.Getenv(primary); v != "" {
		return v
	}
	return os.Getenv(fallback)
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy compatibility - Send functions
// These maintain backward compatibility with the original client.go
// ═══════════════════════════════════════════════════════════════════════════

// Send makes a request using the default provider (backward compatible)
func Send(model Model, messages []Message, opts ...SendOptions) (string, *Response, error) {
	var opt SendOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	// Check cache first
	if cached, ok := getCached(model, messages, opt); ok {
		if Debug {
			printDebugCacheHit()
		}
		return cached, &Response{}, nil
	}

	client := getDefaultClient()

	req := &ProviderRequest{
		Model:       string(model),
		Messages:    messages,
		Temperature: opt.Temperature,
		Thinking:    opt.Thinking,
	}

	if Debug {
		printDebugRequest(model, messages)
	}

	waitForRateLimit()
	resp, err := client.provider.Send(context.Background(), req)
	if err != nil {
		return "", nil, err
	}

	// Cache the response
	setCached(model, messages, opt, resp.Content)

	if Debug {
		printDebugResponse(resp.Content, toResponse(resp))
	}

	return resp.Content, toResponse(resp), nil
}

// SendWithTools makes a request with tool calling support (backward compatible)
func SendWithTools(model Model, messages []Message, tools []Tool, opts ...SendOptions) (string, *Response, []ToolCall, error) {
	var opt SendOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	client := getDefaultClient()

	req := &ProviderRequest{
		Model:       string(model),
		Messages:    messages,
		Temperature: opt.Temperature,
		Thinking:    opt.Thinking,
		Tools:       tools,
	}

	if Debug {
		printDebugRequest(model, messages)
	}

	waitForRateLimit()
	resp, err := client.provider.Send(context.Background(), req)
	if err != nil {
		return "", nil, nil, err
	}

	return resp.Content, toResponse(resp), resp.ToolCalls, nil
}

// toResponse converts ProviderResponse to legacy Response
func toResponse(pr *ProviderResponse) *Response {
	return &Response{
		Choices: []struct {
			Message      ResponseMessage `json:"message"`
			FinishReason string          `json:"finish_reason"`
		}{
			{
				Message: ResponseMessage{
					Role:      "assistant",
					Content:   pr.Content,
					ToolCalls: pr.ToolCalls,
				},
				FinishReason: pr.FinishReason,
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     pr.PromptTokens,
			CompletionTokens: pr.CompletionTokens,
			TotalTokens:      pr.TotalTokens,
		},
	}
}

// printDebugCacheHit prints cache hit message
func printDebugCacheHit() {
	fmt.Println(colorYellow("⚡ Cache hit"))
}
