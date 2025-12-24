package ai

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
)

func TestCompare_On_ReturnsResultsWithoutNetwork(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{Content: "resp:" + req.Model, TotalTokens: 123}, nil
		},
	}
	setDefaultClientForTest(t, p, ProviderOpenAI)

	// Capture stdout because Compare prints.
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	defer func() {
		_ = w.Close()
		os.Stdout = old
		_, _ = io.ReadAll(r)
		_ = r.Close()
	}()

	results := Compare("hello").On(ModelGPT5, ModelClaudeOpus)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for _, res := range results {
		if res.Error != nil {
			t.Fatalf("unexpected error: %v", res.Error)
		}
		if !strings.HasPrefix(res.Response, "resp:") {
			t.Fatalf("unexpected response: %q", res.Response)
		}
	}
}

