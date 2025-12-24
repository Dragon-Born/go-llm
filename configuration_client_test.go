package ai

import (
	"testing"
	"time"
)

func TestSetDefaultProvider_ResetsDefaultClient(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	SetDefaultProvider(ProviderOpenRouter)
	c1 := getDefaultClient()
	if c1 == nil || c1.providerType != ProviderOpenRouter {
		t.Fatalf("expected default client providerType=%s, got %#v", ProviderOpenRouter, c1)
	}

	SetDefaultProvider(ProviderAnthropic)
	c2 := getDefaultClient()
	if c2 == nil || c2.providerType != ProviderAnthropic {
		t.Fatalf("expected default client providerType=%s, got %#v", ProviderAnthropic, c2)
	}
	if c1 == c2 {
		t.Fatal("expected default client to be recreated after provider switch")
	}
}

func TestNewClient_OptionsAreApplied(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	c := NewClient(ProviderOpenAI,
		WithAPIKey("test-key"),
		WithBaseURL("http://example.test"),
		WithTimeout(123*time.Millisecond),
		WithHeaders(map[string]string{"X-Test": "1"}),
	)

	p, ok := c.provider.(*OpenAIProvider)
	if !ok {
		t.Fatalf("expected *OpenAIProvider, got %T", c.provider)
	}
	if p.config.APIKey != "test-key" {
		t.Fatalf("expected APIKey to be set, got %q", p.config.APIKey)
	}
	if p.config.BaseURL != "http://example.test" {
		t.Fatalf("expected BaseURL to be set, got %q", p.config.BaseURL)
	}
	if p.httpClient == nil || p.httpClient.Timeout != 123*time.Millisecond {
		t.Fatalf("expected http client timeout=123ms, got %#v", p.httpClient)
	}
	if p.config.Headers["X-Test"] != "1" {
		t.Fatalf("expected custom header to be set")
	}
}

func TestNewClient_EnvKeyAutoDetection_GoogleFallback(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("GOOGLE_API_KEY", "")

	c := NewClient(ProviderGoogle)
	p := c.provider.(*GoogleProvider)
	if p.config.APIKey != "gemini-key" {
		t.Fatalf("expected GEMINI_API_KEY fallback, got %q", p.config.APIKey)
	}
}

func TestNewClient_AzureKeyFallsBackToOpenAI(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	t.Setenv("OPENAI_API_KEY", "openai-key")
	t.Setenv("AZURE_OPENAI_API_KEY", "")

	c := NewClient(ProviderAzure, WithBaseURL("https://azure.example/openai/deployments/test"))
	p := c.provider.(*AzureProvider)
	if p.OpenAIProvider.config.APIKey != "openai-key" {
		t.Fatalf("expected OPENAI_API_KEY fallback, got %q", p.OpenAIProvider.config.APIKey)
	}
	if p.deploymentURL == "" {
		t.Fatalf("expected deployment URL to be set")
	}
}
