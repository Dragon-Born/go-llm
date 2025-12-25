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

// Client wraps a Provider and manages the creation of builders.
// It serves as the entry point for interacting with a specific AI provider.
type Client struct {
	provider     Provider
	providerType ProviderType
}

// NewClient creates a new Client for the specified provider type.
// It accepts optional configuration options like API key, base URL, etc.
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

// NewClientWithProvider creates a client using a custom provider implementation.
// Useful for testing or adding new providers without modifying the package.
func NewClientWithProvider(provider Provider) *Client {
	return &Client{
		provider:     provider,
		providerType: ProviderType(provider.Name()),
	}
}

// Provider returns the underlying provider interface.
func (c *Client) Provider() Provider {
	return c.provider
}

// ═══════════════════════════════════════════════════════════════════════════
// Client Options (functional options pattern)
// ═══════════════════════════════════════════════════════════════════════════

// ClientOption defines a function for configuring a provider.
type ClientOption func(*ProviderConfig)

// WithAPIKey sets the API key for the provider.
func WithAPIKey(key string) ClientOption {
	return func(c *ProviderConfig) {
		c.APIKey = key
	}
}

// WithBaseURL sets a custom base URL for the API.
// Useful for proxies or enterprise endpoints.
func WithBaseURL(url string) ClientOption {
	return func(c *ProviderConfig) {
		c.BaseURL = url
	}
}

// WithTimeout sets the request timeout duration.
func WithTimeout(d time.Duration) ClientOption {
	return func(c *ProviderConfig) {
		c.Timeout = d
	}
}

// WithHeaders sets custom HTTP headers for requests.
func WithHeaders(headers map[string]string) ClientOption {
	return func(c *ProviderConfig) {
		c.Headers = headers
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Shortcuts on Client
// ═══════════════════════════════════════════════════════════════════════════

// New creates a new Builder instance associated with this client.
// All requests made by this builder will use the client's provider.
func (c *Client) New(model Model) *Builder {
	b := New(model)
	b.client = c
	return b
}

// Use creates a builder for a model by its string ID.
// Useful for models not defined in the package constants (e.g., custom Ollama models).
func (c *Client) Use(modelID string) *Builder {
	return c.New(Model(modelID))
}

// ═══════════════════════════════════════════════════════════════════════════
// OpenAI Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

// GPT5 returns a Builder configured for ModelGPT5 using this client's provider.
func (c *Client) GPT5() *Builder { return c.New(ModelGPT5) }

// GPT5Codex returns a Builder configured for ModelGPT5Codex using this client's provider.
func (c *Client) GPT5Codex() *Builder { return c.New(ModelGPT5Codex) }

// GPT4o returns a Builder configured for ModelGPT4o using this client's provider.
func (c *Client) GPT4o() *Builder { return c.New(ModelGPT4o) }

// GPT4oMini returns a Builder configured for ModelGPT4oMini using this client's provider.
func (c *Client) GPT4oMini() *Builder { return c.New(ModelGPT4oMini) }

// O1 returns a Builder configured for ModelO1 using this client's provider.
func (c *Client) O1() *Builder { return c.New(ModelO1) }

// GPT 5.x family
// GPT52 returns a Builder configured for ModelGPT52 using this client's provider.
func (c *Client) GPT52() *Builder { return c.New(ModelGPT52) }

// GPT52Pro returns a Builder configured for ModelGPT52Pro using this client's provider.
func (c *Client) GPT52Pro() *Builder { return c.New(ModelGPT52Pro) }

// GPT51 returns a Builder configured for ModelGPT51 using this client's provider.
func (c *Client) GPT51() *Builder { return c.New(ModelGPT51) }

// GPT5Base returns a Builder configured for ModelGPT5Base using this client's provider.
func (c *Client) GPT5Base() *Builder { return c.New(ModelGPT5Base) }

// GPT5Pro returns a Builder configured for ModelGPT5Pro using this client's provider.
func (c *Client) GPT5Pro() *Builder { return c.New(ModelGPT5Pro) }

// GPT5Mini returns a Builder configured for ModelGPT5Mini using this client's provider.
func (c *Client) GPT5Mini() *Builder { return c.New(ModelGPT5Mini) }

// GPT5Nano returns a Builder configured for ModelGPT5Nano using this client's provider.
func (c *Client) GPT5Nano() *Builder { return c.New(ModelGPT5Nano) }

// Codex
// GPT51Codex returns a Builder configured for ModelGPT51Codex using this client's provider.
func (c *Client) GPT51Codex() *Builder { return c.New(ModelGPT51Codex) }

// GPT51CodexMax returns a Builder configured for ModelGPT51CodexMax using this client's provider.
func (c *Client) GPT51CodexMax() *Builder { return c.New(ModelGPT51CodexMax) }

// GPT5CodexBase returns a Builder configured for ModelGPT5CodexBase using this client's provider.
func (c *Client) GPT5CodexBase() *Builder { return c.New(ModelGPT5CodexBase) }

// GPT51CodexMini returns a Builder configured for ModelGPT51CodexMini using this client's provider.
func (c *Client) GPT51CodexMini() *Builder { return c.New(ModelGPT51CodexMini) }

// CodexMiniLatest returns a Builder configured for ModelCodexMiniLatest using this client's provider.
func (c *Client) CodexMiniLatest() *Builder { return c.New(ModelCodexMiniLatest) }

// Search + agent tools
// GPT5SearchAPI returns a Builder configured for ModelGPT5SearchAPI using this client's provider.
func (c *Client) GPT5SearchAPI() *Builder { return c.New(ModelGPT5SearchAPI) }

// ComputerUsePreview returns a Builder configured for ModelComputerUsePreview using this client's provider.
func (c *Client) ComputerUsePreview() *Builder { return c.New(ModelComputerUsePreview) }

// Aliases
// GPT5ChatLatest returns a Builder configured for ModelGPT5ChatLatest using this client's provider.
func (c *Client) GPT5ChatLatest() *Builder { return c.New(ModelGPT5ChatLatest) }

// GPT52ChatLatest returns a Builder configured for ModelGPT52ChatLatest using this client's provider.
func (c *Client) GPT52ChatLatest() *Builder { return c.New(ModelGPT52ChatLatest) }

// GPT51ChatLatest returns a Builder configured for ModelGPT51ChatLatest using this client's provider.
func (c *Client) GPT51ChatLatest() *Builder { return c.New(ModelGPT51ChatLatest) }

// ChatGPT4oLatest returns a Builder configured for ModelChatGPT4oLatest using this client's provider.
func (c *Client) ChatGPT4oLatest() *Builder { return c.New(ModelChatGPT4oLatest) }

// GPT-4.1
// GPT41 returns a Builder configured for ModelGPT41 using this client's provider.
func (c *Client) GPT41() *Builder { return c.New(ModelGPT41) }

// GPT41Mini returns a Builder configured for ModelGPT41Mini using this client's provider.
func (c *Client) GPT41Mini() *Builder { return c.New(ModelGPT41Mini) }

// GPT41Nano returns a Builder configured for ModelGPT41Nano using this client's provider.
func (c *Client) GPT41Nano() *Builder { return c.New(ModelGPT41Nano) }

// GPT-4o dated snapshot
// GPT4o20240513 returns a Builder configured for ModelGPT4o20240513 using this client's provider.
func (c *Client) GPT4o20240513() *Builder { return c.New(ModelGPT4o20240513) }

// o-series
// O1Mini returns a Builder configured for ModelO1Mini using this client's provider.
func (c *Client) O1Mini() *Builder { return c.New(ModelO1Mini) }

// O1Pro returns a Builder configured for ModelO1Pro using this client's provider.
func (c *Client) O1Pro() *Builder { return c.New(ModelO1Pro) }

// O1Preview returns a Builder configured for ModelO1Preview using this client's provider.
func (c *Client) O1Preview() *Builder { return c.New(ModelO1Preview) }

// O3 returns a Builder configured for ModelO3 using this client's provider.
func (c *Client) O3() *Builder { return c.New(ModelO3) }

// O3Mini returns a Builder configured for ModelO3Mini using this client's provider.
func (c *Client) O3Mini() *Builder { return c.New(ModelO3Mini) }

// O3Pro returns a Builder configured for ModelO3Pro using this client's provider.
func (c *Client) O3Pro() *Builder { return c.New(ModelO3Pro) }

// O3DeepResearch returns a Builder configured for ModelO3DeepResearch using this client's provider.
func (c *Client) O3DeepResearch() *Builder {
	return c.New(ModelO3DeepResearch)
}

// O4Mini returns a Builder configured for ModelO4Mini using this client's provider.
func (c *Client) O4Mini() *Builder { return c.New(ModelO4Mini) }

// O4MiniDeepResearch returns a Builder configured for ModelO4MiniDeepResearch using this client's provider.
func (c *Client) O4MiniDeepResearch() *Builder {
	return c.New(ModelO4MiniDeepResearch)
}

// Realtime / audio
// GPTRealtime returns a Builder configured for ModelGPTRealtime using this client's provider.
func (c *Client) GPTRealtime() *Builder { return c.New(ModelGPTRealtime) }

// GPTRealtimeMini returns a Builder configured for ModelGPTRealtimeMini using this client's provider.
func (c *Client) GPTRealtimeMini() *Builder { return c.New(ModelGPTRealtimeMini) }

// GPT4oRealtimePreview returns a Builder configured for ModelGPT4oRealtimePreview using this client's provider.
func (c *Client) GPT4oRealtimePreview() *Builder { return c.New(ModelGPT4oRealtimePreview) }

// GPT4oMiniRealtimePreview returns a Builder configured for ModelGPT4oMiniRealtimePreview using this client's provider.
func (c *Client) GPT4oMiniRealtimePreview() *Builder { return c.New(ModelGPT4oMiniRealtimePreview) }

// GPTAudio returns a Builder configured for ModelGPTAudio using this client's provider.
func (c *Client) GPTAudio() *Builder { return c.New(ModelGPTAudio) }

// GPTAudioMini returns a Builder configured for ModelGPTAudioMini using this client's provider.
func (c *Client) GPTAudioMini() *Builder { return c.New(ModelGPTAudioMini) }

// GPT4oAudioPreview returns a Builder configured for ModelGPT4oAudioPreview using this client's provider.
func (c *Client) GPT4oAudioPreview() *Builder { return c.New(ModelGPT4oAudioPreview) }

// GPT4oMiniAudioPreview returns a Builder configured for ModelGPT4oMiniAudioPreview using this client's provider.
func (c *Client) GPT4oMiniAudioPreview() *Builder { return c.New(ModelGPT4oMiniAudioPreview) }

// Search previews
// GPT4oMiniSearchPreview returns a Builder configured for ModelGPT4oMiniSearchPreview using this client's provider.
func (c *Client) GPT4oMiniSearchPreview() *Builder { return c.New(ModelGPT4oMiniSearchPreview) }

// GPT4oSearchPreview returns a Builder configured for ModelGPT4oSearchPreview using this client's provider.
func (c *Client) GPT4oSearchPreview() *Builder { return c.New(ModelGPT4oSearchPreview) }

// Speech / transcription
// GPT4oMiniTTS returns a Builder configured for ModelGPT4oMiniTTS using this client's provider.
func (c *Client) GPT4oMiniTTS() *Builder { return c.New(ModelGPT4oMiniTTS) }

// GPT4oTranscribe returns a Builder configured for ModelGPT4oTranscribe using this client's provider.
func (c *Client) GPT4oTranscribe() *Builder { return c.New(ModelGPT4oTranscribe) }

// GPT4oTranscribeDiarize returns a Builder configured for ModelGPT4oTranscribeDiarize using this client's provider.
func (c *Client) GPT4oTranscribeDiarize() *Builder { return c.New(ModelGPT4oTranscribeDiarize) }

// GPT4oMiniTranscribe returns a Builder configured for ModelGPT4oMiniTranscribe using this client's provider.
func (c *Client) GPT4oMiniTranscribe() *Builder { return c.New(ModelGPT4oMiniTranscribe) }

// Image generation
// GPTImage15 returns a Builder configured for ModelGPTImage15 using this client's provider.
func (c *Client) GPTImage15() *Builder { return c.New(ModelGPTImage15) }

// GPTImage1 returns a Builder configured for ModelGPTImage1 using this client's provider.
func (c *Client) GPTImage1() *Builder { return c.New(ModelGPTImage1) }

// GPTImage1Mini returns a Builder configured for ModelGPTImage1Mini using this client's provider.
func (c *Client) GPTImage1Mini() *Builder { return c.New(ModelGPTImage1Mini) }

// ChatGPTImageLatest returns a Builder configured for ModelChatGPTImageLatest using this client's provider.
func (c *Client) ChatGPTImageLatest() *Builder { return c.New(ModelChatGPTImageLatest) }

// Open-weight models
// GPTOSS120B returns a Builder configured for ModelGPTOSS120B using this client's provider.
func (c *Client) GPTOSS120B() *Builder { return c.New(ModelGPTOSS120B) }

// GPTOSS20B returns a Builder configured for ModelGPTOSS20B using this client's provider.
func (c *Client) GPTOSS20B() *Builder { return c.New(ModelGPTOSS20B) }

// Video generation
// Sora2 returns a Builder configured for ModelSora2 using this client's provider.
func (c *Client) Sora2() *Builder { return c.New(ModelSora2) }

// Sora2Pro returns a Builder configured for ModelSora2Pro using this client's provider.
func (c *Client) Sora2Pro() *Builder { return c.New(ModelSora2Pro) }

// ═══════════════════════════════════════════════════════════════════════════
// Anthropic Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

// Claude returns a Builder configured for ModelClaudeOpus using this client's provider.
func (c *Client) Claude() *Builder { return c.New(ModelClaudeOpus) }

// ClaudeOpus returns a Builder configured for ModelClaudeOpus using this client's provider.
func (c *Client) ClaudeOpus() *Builder { return c.New(ModelClaudeOpus) }

// ClaudeSonnet returns a Builder configured for ModelClaudeSonnet using this client's provider.
func (c *Client) ClaudeSonnet() *Builder { return c.New(ModelClaudeSonnet) }

// ClaudeHaiku returns a Builder configured for ModelClaudeHaiku using this client's provider.
func (c *Client) ClaudeHaiku() *Builder { return c.New(ModelClaudeHaiku) }

// Additional Claude variants (OpenRouter slugs, normalized for native Anthropic)
// ClaudeOpus41 returns a Builder configured for ModelClaudeOpus41 using this client's provider.
func (c *Client) ClaudeOpus41() *Builder { return c.New(ModelClaudeOpus41) }

// ClaudeOpus4 returns a Builder configured for ModelClaudeOpus4 using this client's provider.
func (c *Client) ClaudeOpus4() *Builder { return c.New(ModelClaudeOpus4) }

// ClaudeSonnet4 returns a Builder configured for ModelClaudeSonnet4 using this client's provider.
func (c *Client) ClaudeSonnet4() *Builder { return c.New(ModelClaudeSonnet4) }

// ClaudeSonnet37 returns a Builder configured for ModelClaudeSonnet37 using this client's provider.
func (c *Client) ClaudeSonnet37() *Builder { return c.New(ModelClaudeSonnet37) }

// ClaudeHaiku35 returns a Builder configured for ModelClaudeHaiku35 using this client's provider.
func (c *Client) ClaudeHaiku35() *Builder { return c.New(ModelClaudeHaiku35) }

// ClaudeHaiku3 returns a Builder configured for ModelClaudeHaiku3 using this client's provider.
func (c *Client) ClaudeHaiku3() *Builder { return c.New(ModelClaudeHaiku3) }

// ClaudeOpus3 returns a Builder configured for ModelClaudeOpus3 using this client's provider.
func (c *Client) ClaudeOpus3() *Builder { return c.New(ModelClaudeOpus3) }

// ClaudeSonnet3 returns a Builder configured for ModelClaudeSonnet3 using this client's provider.
func (c *Client) ClaudeSonnet3() *Builder { return c.New(ModelClaudeSonnet3) }

// ═══════════════════════════════════════════════════════════════════════════
// Google Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

// Gemini returns a Builder configured for ModelGemini3Flash using this client's provider.
func (c *Client) Gemini() *Builder { return c.New(ModelGemini3Flash) }

// GeminiPro returns a Builder configured for ModelGemini3Pro using this client's provider.
func (c *Client) GeminiPro() *Builder { return c.New(ModelGemini3Pro) }

// GeminiFlash returns a Builder configured for ModelGemini3Flash using this client's provider.
func (c *Client) GeminiFlash() *Builder { return c.New(ModelGemini3Flash) }

// Explicit Gemini 3 names (aliases for clarity)
// Gemini3Pro returns a Builder configured for ModelGemini3Pro using this client's provider.
func (c *Client) Gemini3Pro() *Builder { return c.New(ModelGemini3Pro) }

// Gemini3Flash returns a Builder configured for ModelGemini3Flash using this client's provider.
func (c *Client) Gemini3Flash() *Builder { return c.New(ModelGemini3Flash) }

// Gemini25Pro returns a Builder configured for ModelGemini25Pro using this client's provider.
func (c *Client) Gemini25Pro() *Builder { return c.New(ModelGemini25Pro) }

// Gemini25Flash returns a Builder configured for ModelGemini25Flash using this client's provider.
func (c *Client) Gemini25Flash() *Builder { return c.New(ModelGemini25Flash) }

// Gemini25FlashLite returns a Builder configured for ModelGemini25FlashLite using this client's provider.
func (c *Client) Gemini25FlashLite() *Builder { return c.New(ModelGemini25FlashLite) }

// Gemini2Pro returns a Builder configured for ModelGemini2Pro using this client's provider.
func (c *Client) Gemini2Pro() *Builder { return c.New(ModelGemini2Pro) }

// Gemini2Flash returns a Builder configured for ModelGemini2Flash using this client's provider.
func (c *Client) Gemini2Flash() *Builder { return c.New(ModelGemini2Flash) }

// Gemini2FlashLite returns a Builder configured for ModelGemini2FlashLite using this client's provider.
func (c *Client) Gemini2FlashLite() *Builder { return c.New(ModelGemini2FlashLite) }

// ═══════════════════════════════════════════════════════════════════════════
// Other Model Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

// Grok returns a Builder configured for ModelGrok3 using this client's provider.
func (c *Client) Grok() *Builder { return c.New(ModelGrok3) }

// GrokFast returns a Builder configured for ModelGrok41Fast using this client's provider.
func (c *Client) GrokFast() *Builder { return c.New(ModelGrok41Fast) }

// GrokMini returns a Builder configured for ModelGrok3Mini using this client's provider.
func (c *Client) GrokMini() *Builder { return c.New(ModelGrok3Mini) }

// Explicit xAI names (aliases for symmetry)
// Grok3 returns a Builder configured for ModelGrok3 using this client's provider.
func (c *Client) Grok3() *Builder { return c.New(ModelGrok3) }

// Grok3Mini returns a Builder configured for ModelGrok3Mini using this client's provider.
func (c *Client) Grok3Mini() *Builder { return c.New(ModelGrok3Mini) }

// Grok41Fast returns a Builder configured for ModelGrok41Fast using this client's provider.
func (c *Client) Grok41Fast() *Builder { return c.New(ModelGrok41Fast) }

// Qwen returns a Builder configured for ModelQwen3Next using this client's provider.
func (c *Client) Qwen() *Builder { return c.New(ModelQwen3Next) }

// Qwen3Next returns a Builder configured for ModelQwen3Next using this client's provider.
func (c *Client) Qwen3Next() *Builder { return c.New(ModelQwen3Next) }

// Qwen3 returns a Builder configured for ModelQwen3 using this client's provider.
func (c *Client) Qwen3() *Builder { return c.New(ModelQwen3) }

// Llama returns a Builder configured for ModelLlama4 using this client's provider.
func (c *Client) Llama() *Builder { return c.New(ModelLlama4) }

// Llama4 returns a Builder configured for ModelLlama4 using this client's provider.
func (c *Client) Llama4() *Builder { return c.New(ModelLlama4) }

// Mistral returns a Builder configured for ModelMistralLarge using this client's provider.
func (c *Client) Mistral() *Builder { return c.New(ModelMistralLarge) }

// MistralLarge returns a Builder configured for ModelMistralLarge using this client's provider.
func (c *Client) MistralLarge() *Builder { return c.New(ModelMistralLarge) }

// ═══════════════════════════════════════════════════════════════════════════
// Azure Provider (special constructor)
// ═══════════════════════════════════════════════════════════════════════════

// AzureProvider is a specialized OpenAI provider for Azure deployments.
// It handles Azure-specific authentication and endpoint formatting.
type AzureProvider struct {
	*OpenAIProvider
	deploymentURL string
}

// NewAzureProvider creates a new Azure OpenAI provider.
// It requires a deployment URL and API key.
// If the BaseURL is empty, it uses a placeholder that must be replaced.
// If the APIKey is empty, it attempts to read from AZURE_OPENAI_API_KEY or OPENAI_API_KEY env vars.
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

// Name returns the provider identifier ("azure").
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

// Send makes a request using the default provider.
// This function is kept for backward compatibility; prefer using Builder.
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

// SendWithTools makes a request with tool calling support using the default provider.
// This function is kept for backward compatibility; prefer using Builder.Tool().
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
