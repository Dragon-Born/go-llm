package ai

import (
	"strings"
	"testing"
)

func TestCost_CalculateAndFormat(t *testing.T) {
	meta := &ResponseMeta{
		Model:            ModelGPT4oMini,
		PromptTokens:     1000,
		CompletionTokens: 500,
	}
	cost := meta.Cost()
	if cost <= 0 {
		t.Fatalf("expected positive cost, got %f", cost)
	}

	s := meta.CostString()
	if !strings.HasPrefix(s, "$") {
		t.Fatalf("expected formatted cost string, got %q", s)
	}
}

func TestCostTracker_TrackAndBudget(t *testing.T) {
	ct := NewCostTracker()
	ct.Track(&ResponseMeta{Model: ModelGPT4oMini, PromptTokens: 100, CompletionTokens: 100, Tokens: 200})
	ct.Track(&ResponseMeta{Model: ModelClaudeHaiku, PromptTokens: 200, CompletionTokens: 100, Tokens: 300})

	if ct.RequestCount != 2 {
		t.Fatalf("expected RequestCount=2, got %d", ct.RequestCount)
	}
	if ct.TokensUsed != 500 {
		t.Fatalf("expected TokensUsed=500, got %d", ct.TokensUsed)
	}
	if ct.TotalCost <= 0 {
		t.Fatalf("expected TotalCost > 0, got %f", ct.TotalCost)
	}
	if ct.CostByModel[ModelGPT4oMini] <= 0 {
		t.Fatalf("expected per-model cost to be tracked")
	}

	summary := ct.Summary()
	if !strings.Contains(summary, "Cost Summary:") {
		t.Fatalf("unexpected summary: %q", summary)
	}

	bt := WithBudget(0.0000001) // tiny
	bt.Track(&ResponseMeta{Model: ModelGPT4oMini, PromptTokens: 1000000, CompletionTokens: 0, Tokens: 1000000})
	if err := bt.CheckBudget(); err == nil {
		t.Fatalf("expected budget exceeded error")
	}
	if bt.Remaining() != 0 {
		t.Fatalf("expected remaining budget to clamp to 0, got %f", bt.Remaining())
	}
}

func TestCostAwareModelSelection(t *testing.T) {
	cheap := CheapestModel(ModelClaudeOpus, ModelClaudeHaiku, ModelGPT5)
	if cheap != ModelClaudeHaiku {
		t.Fatalf("expected cheapest to be ClaudeHaiku, got %s", cheap)
	}

	exp := MostExpensiveModel(ModelClaudeHaiku, ModelGPT4oMini, ModelClaudeOpus)
	if exp != ModelClaudeOpus {
		t.Fatalf("expected most expensive to be ClaudeOpus, got %s", exp)
	}
}

func TestOpenAIModelPricingCoverage(t *testing.T) {
	// Ensure all built-in OpenAI model constants have pricing entries.
	// (These are the models we expose as first-class constants.)
	models := []Model{
		ModelGPT52,
		ModelGPT51,
		ModelGPT5Base,
		ModelGPT5Pro,
		ModelGPT5Mini,
		ModelGPT5Nano,
		ModelGPT52Pro,
		ModelGPT51Codex,
		ModelGPT51CodexMax,
		ModelGPT5CodexBase,
		ModelGPT51CodexMini,
		ModelCodexMiniLatest,
		ModelGPT5SearchAPI,
		ModelComputerUsePreview,
		ModelGPT52ChatLatest,
		ModelGPT51ChatLatest,
		ModelGPT5ChatLatest,
		ModelGPT41,
		ModelGPT41Mini,
		ModelGPT41Nano,
		ModelGPT4o,
		ModelGPT4o20240513,
		ModelGPT4oMini,
		ModelGPTRealtime,
		ModelGPTRealtimeMini,
		ModelGPT4oRealtimePreview,
		ModelGPT4oMiniRealtimePreview,
		ModelGPTAudio,
		ModelGPTAudioMini,
		ModelGPT4oAudioPreview,
		ModelGPT4oMiniAudioPreview,
		ModelGPT4oMiniSearchPreview,
		ModelGPT4oSearchPreview,
		ModelChatGPT4oLatest,
		ModelO1,
		ModelO1Mini,
		ModelO1Pro,
		ModelO3,
		ModelO3Mini,
		ModelO3Pro,
		ModelO3DeepResearch,
		ModelO4Mini,
		ModelO4MiniDeepResearch,
		ModelGPTImage15,
		ModelChatGPTImageLatest,
		ModelGPTImage1,
		ModelGPTImage1Mini,
	}

	for _, m := range models {
		if _, ok := ModelPricingMap[m]; !ok {
			t.Fatalf("missing pricing for model %q", m)
		}
	}
}
