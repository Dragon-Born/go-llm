package ai

import (
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════
// Default Provider Configuration
// ═══════════════════════════════════════════════════════════════════════════

var (
	// DefaultProvider determines which provider is used by package-level shortcuts
	// Change this to switch all ai.Claude(), ai.GPT5() etc. to a different backend
	DefaultProvider = ProviderOpenRouter

	// Provider clients (lazy-initialized, thread-safe)
	clientsMu       sync.RWMutex
	defaultClient   *Client
	anthropicClient *Client
	openaiClient    *Client
	googleClient    *Client
	ollamaClient    *Client
)

// ═══════════════════════════════════════════════════════════════════════════
// Provider-First Shortcuts
// These let you explicitly choose which provider to use
// ═══════════════════════════════════════════════════════════════════════════

// Anthropic returns a client for Anthropic's API
// Usage: ai.Anthropic().Claude().Ask("...")
func Anthropic() *Client {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	if anthropicClient == nil {
		anthropicClient = NewClient(ProviderAnthropic)
	}
	return anthropicClient
}

// OpenAI returns a client for OpenAI's API
// Usage: ai.OpenAI().GPT4o().Ask("...")
func OpenAI() *Client {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	if openaiClient == nil {
		openaiClient = NewClient(ProviderOpenAI)
	}
	return openaiClient
}

// Google returns a client for Google's Gemini API
// Usage: ai.Google().Gemini().Ask("...")
func Google() *Client {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	if googleClient == nil {
		googleClient = NewClient(ProviderGoogle)
	}
	return googleClient
}

// Ollama returns a client for local Ollama (default: localhost:11434)
// Usage: ai.Ollama().Use("llama3:8b").Ask("...")
func Ollama() *Client {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	if ollamaClient == nil {
		ollamaClient = NewClient(ProviderOllama)
	}
	return ollamaClient
}

// OllamaAt returns a client for Ollama at a custom URL
// Usage: ai.OllamaAt("http://gpu-server:11434").Use("llama3:70b").Ask("...")
func OllamaAt(url string) *Client {
	return NewClient(ProviderOllama, WithBaseURL(url))
}

// Azure returns a client for Azure OpenAI
// Usage: ai.Azure("https://mycompany.openai.azure.com").GPT4o().Ask("...")
func Azure(endpoint string) *Client {
	return NewClient(ProviderAzure, WithBaseURL(endpoint))
}

// OpenRouter returns a client for OpenRouter (explicit)
// Usage: ai.OpenRouter().Claude().Ask("...")
func OpenRouter() *Client {
	return getDefaultClient() // OpenRouter is the default
}

// ═══════════════════════════════════════════════════════════════════════════
// Default Client Management
// ═══════════════════════════════════════════════════════════════════════════

// getDefaultClient returns the default client based on DefaultProvider
func getDefaultClient() *Client {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	if defaultClient == nil {
		defaultClient = NewClient(DefaultProvider)
	}
	return defaultClient
}

// SetDefaultProvider changes the default provider and resets the default client
func SetDefaultProvider(provider ProviderType) {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	DefaultProvider = provider
	defaultClient = nil // Will be recreated on next use
}

// SetDefaultClient sets a custom default client
func SetDefaultClient(client *Client) {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	defaultClient = client
}

// ResetClients clears all cached clients (useful for testing)
func ResetClients() {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	defaultClient = nil
	anthropicClient = nil
	openaiClient = nil
	googleClient = nil
	ollamaClient = nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: Create clients with custom config
// ═══════════════════════════════════════════════════════════════════════════

// AnthropicWith creates an Anthropic client with custom options
func AnthropicWith(opts ...ClientOption) *Client {
	return NewClient(ProviderAnthropic, opts...)
}

// OpenAIWith creates an OpenAI client with custom options
func OpenAIWith(opts ...ClientOption) *Client {
	return NewClient(ProviderOpenAI, opts...)
}

// GoogleWith creates a Google/Gemini client with custom options
func GoogleWith(opts ...ClientOption) *Client {
	return NewClient(ProviderGoogle, opts...)
}

// OllamaWith creates an Ollama client with custom options
func OllamaWith(opts ...ClientOption) *Client {
	return NewClient(ProviderOllama, opts...)
}
