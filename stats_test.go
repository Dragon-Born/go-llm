package ai

import (
	"testing"
	"time"
)

func TestTrackRequest(t *testing.T) {
	// Reset stats
	ResetStats()
	defer ResetStats()

	meta := &ResponseMeta{
		Content:          "test response",
		Model:            ModelGPT5,
		Tokens:           100,
		PromptTokens:     30,
		CompletionTokens: 70,
		Latency:          time.Second,
		Retries:          1,
	}

	trackRequest(meta)

	stats := GetStats()

	if stats.Requests != 1 {
		t.Errorf("expected 1 request, got %d", stats.Requests)
	}
	if stats.TotalTokens != 100 {
		t.Errorf("expected 100 total tokens, got %d", stats.TotalTokens)
	}
	if stats.PromptTokens != 30 {
		t.Errorf("expected 30 prompt tokens, got %d", stats.PromptTokens)
	}
	if stats.CompletionTokens != 70 {
		t.Errorf("expected 70 completion tokens, got %d", stats.CompletionTokens)
	}
	if stats.TotalLatency != time.Second {
		t.Errorf("expected 1s latency, got %v", stats.TotalLatency)
	}
	if stats.Retries != 1 {
		t.Errorf("expected 1 retry, got %d", stats.Retries)
	}
	if stats.ModelUsage[ModelGPT5] != 1 {
		t.Error("model usage not tracked correctly")
	}
}

func TestMultipleRequests(t *testing.T) {
	ResetStats()
	defer ResetStats()

	trackRequest(&ResponseMeta{
		Model:            ModelGPT5,
		Tokens:           50,
		PromptTokens:     20,
		CompletionTokens: 30,
		Latency:          time.Second,
	})

	trackRequest(&ResponseMeta{
		Model:            ModelClaudeOpus,
		Tokens:           100,
		PromptTokens:     40,
		CompletionTokens: 60,
		Latency:          2 * time.Second,
	})

	trackRequest(&ResponseMeta{
		Model:            ModelGPT5,
		Tokens:           50,
		PromptTokens:     20,
		CompletionTokens: 30,
		Latency:          time.Second,
	})

	stats := GetStats()

	if stats.Requests != 3 {
		t.Errorf("expected 3 requests, got %d", stats.Requests)
	}
	if stats.TotalTokens != 200 {
		t.Errorf("expected 200 total tokens, got %d", stats.TotalTokens)
	}
	if stats.TotalLatency != 4*time.Second {
		t.Errorf("expected 4s total latency, got %v", stats.TotalLatency)
	}
	if stats.ModelUsage[ModelGPT5] != 2 {
		t.Errorf("expected 2 GPT5 uses, got %d", stats.ModelUsage[ModelGPT5])
	}
	if stats.ModelUsage[ModelClaudeOpus] != 1 {
		t.Errorf("expected 1 Claude use, got %d", stats.ModelUsage[ModelClaudeOpus])
	}
}

func TestResetStats(t *testing.T) {
	// Add some stats
	trackRequest(&ResponseMeta{
		Model:  ModelGPT5,
		Tokens: 100,
	})

	ResetStats()

	stats := GetStats()

	if stats.Requests != 0 {
		t.Error("stats should be reset")
	}
	if stats.TotalTokens != 0 {
		t.Error("tokens should be reset")
	}
}

func TestEstimateCost(t *testing.T) {
	ResetStats()
	defer ResetStats()

	trackRequest(&ResponseMeta{
		Model:            ModelGPT5,
		PromptTokens:     1000000, // 1M tokens
		CompletionTokens: 1000000, // 1M tokens
		Tokens:           2000000,
	})

	cost := EstimateCost()

	// Should be some positive cost
	if cost <= 0 {
		t.Error("cost should be positive")
	}
}

func TestGetStatsIsolation(t *testing.T) {
	ResetStats()
	defer ResetStats()

	trackRequest(&ResponseMeta{Model: ModelGPT5, Tokens: 100})

	stats1 := GetStats()
	stats1.Requests = 999 // Modify returned copy
	if stats1.Requests != 999 {
		t.Fatal("expected local copy to be mutable")
	}

	stats2 := GetStats()

	if stats2.Requests == 999 {
		t.Error("GetStats should return a copy, not the original")
	}
}
