package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestOpenAIProvider_Send_BuildsExpectedRequest(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	var gotAuth string
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("expected /chat/completions, got %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
		}`))
	}))
	defer srv.Close()

	p := NewOpenAIProvider(ProviderConfig{
		APIKey:  "k",
		BaseURL: srv.URL,
	})

	temp := 0.2
	resp, err := p.Send(context.Background(), &ProviderRequest{
		Model:       string(ModelGPT5),
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: &temp,
		Thinking:    ThinkingHigh,
		JSONMode:    true,
		Tools: []Tool{
			{
				Type: "function",
				Function: ToolFunction{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  Params().String("city", "City", true).Build(),
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "Bearer k" {
		t.Fatalf("expected Authorization header, got %q", gotAuth)
	}

	if gotBody["model"] != "gpt-5.2" {
		t.Fatalf("expected resolved model gpt-5.2, got %#v", gotBody["model"])
	}
	if gotBody["reasoning_effort"] != "high" {
		t.Fatalf("expected reasoning_effort=high, got %#v", gotBody["reasoning_effort"])
	}
	if gotBody["response_format"] == nil {
		t.Fatalf("expected response_format to be set for JSON mode")
	}
	if _, ok := gotBody["tools"]; !ok {
		t.Fatalf("expected tools to be present")
	}

	if resp.Content != "ok" || resp.TotalTokens != 3 || resp.PromptTokens != 1 || resp.CompletionTokens != 2 {
		t.Fatalf("unexpected response: %#v", resp)
	}
}

func TestOpenRouterProvider_Send_BuildsExpectedRequest(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	var gotAuth string
	var gotReferer string
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("expected /chat/completions, got %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		gotReferer = r.Header.Get("HTTP-Referer")

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
		}`))
	}))
	defer srv.Close()

	p := NewOpenRouterProvider(ProviderConfig{
		APIKey:  "k",
		BaseURL: srv.URL,
	})

	temp := 0.3
	resp, err := p.Send(context.Background(), &ProviderRequest{
		Model:       string(ModelClaudeOpus),
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: &temp,
		Thinking:    ThinkingLow,
		JSONMode:    true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotAuth != "Bearer k" {
		t.Fatalf("expected Authorization header, got %q", gotAuth)
	}
	if gotReferer == "" {
		t.Fatalf("expected HTTP-Referer header to be set")
	}
	if gotBody["model"] != "anthropic/claude-opus-4.5" {
		t.Fatalf("expected resolved model anthropic/claude-opus-4.5, got %#v", gotBody["model"])
	}
	if gotBody["reasoning"] != string(ThinkingLow) {
		t.Fatalf("expected reasoning=%q, got %#v", ThinkingLow, gotBody["reasoning"])
	}
	if gotBody["response_format"] == nil {
		t.Fatalf("expected response_format to be set for JSON mode")
	}

	if resp.Content != "ok" || resp.TotalTokens != 30 {
		t.Fatalf("unexpected response: %#v", resp)
	}
}

func TestOpenAIProvider_WithTimeout_IsRespected(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"late"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	p := NewOpenAIProvider(ProviderConfig{
		APIKey:  "k",
		BaseURL: srv.URL,
		Timeout: 50 * time.Millisecond,
		Headers: map[string]string{"X-Test": "1"},
	})

	_, err := p.Send(context.Background(), &ProviderRequest{
		Model:    string(ModelGPT4o),
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatalf("expected timeout error")
	}
	// Provider wraps the underlying timeout error; assert via Unwrap().
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context deadline exceeded via unwrap, got: %v", err)
	}
}
