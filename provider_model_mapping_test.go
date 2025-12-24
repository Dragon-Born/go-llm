package ai

import "testing"

func TestResolveModel_Anthropic_ClaudeDefaults(t *testing.T) {
	tests := []struct {
		name string
		in   Model
		want string
	}{
		{"ClaudeSonnetDefault", ModelClaudeSonnet, "claude-sonnet-4-5-20250929"},
		{"ClaudeHaikuDefault", ModelClaudeHaiku, "claude-haiku-4-5-20251001"},
		{"ClaudeOpusDefault", ModelClaudeOpus, "claude-opus-4-5-20251101"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveModel(ProviderAnthropic, tt.in); got != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, got)
			}
		})
	}
}

func TestResolveModel_Anthropic_NormalizesOpenRouterClaudeIDs(t *testing.T) {
	tests := []struct {
		name string
		in   Model
		want string
	}{
		{"OpenRouterSonnet45", Model("anthropic/claude-sonnet-4.5"), "claude-sonnet-4-5-20250929"},
		{"OpenRouterHaiku45", Model("anthropic/claude-haiku-4.5"), "claude-haiku-4-5-20251001"},
		{"OpenRouterOpus45", Model("anthropic/claude-opus-4.5"), "claude-opus-4-5-20251101"},
		{"OpenRouterSonnet37ThinkingSuffix", Model("anthropic/claude-3.7-sonnet:thinking"), "claude-3-7-sonnet-20250219"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveModel(ProviderAnthropic, tt.in); got != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, got)
			}
		})
	}
}

func TestResolveModel_OpenRouter_ClaudeDefaults(t *testing.T) {
	if got := resolveModel(ProviderOpenRouter, ModelClaudeSonnet); got != "anthropic/claude-sonnet-4.5" {
		t.Fatalf("expected %q, got %q", "anthropic/claude-sonnet-4.5", got)
	}
	if got := resolveModel(ProviderOpenRouter, ModelClaudeHaiku); got != "anthropic/claude-haiku-4.5" {
		t.Fatalf("expected %q, got %q", "anthropic/claude-haiku-4.5", got)
	}
}

