package ai

import (
	"testing"
)

func TestModelString(t *testing.T) {
	tests := []struct {
		model    Model
		expected string
	}{
		{ModelGPT5, "openai/gpt-5.2"},
		{ModelGPT5Codex, "openai/gpt-5.1-codex-max"},
		{ModelClaudeOpus, "anthropic/claude-opus-4.5"},
		{ModelClaudeSonnet, "anthropic/claude-sonnet-4.5"},
		{ModelClaudeHaiku, "anthropic/claude-haiku-4.5"},
		{ModelGemini3Pro, "google/gemini-3-pro-preview"},
		{ModelGemini3Flash, "google/gemini-3-flash-preview"},
		{ModelGemini25Pro, "google/gemini-2.5-pro"},
		{ModelGemini25Flash, "google/gemini-2.5-flash"},
		{ModelGemini25FlashLite, "google/gemini-2.5-flash-lite"},
		{ModelGemini2Flash, "google/gemini-2.0-flash-001"},
		{ModelGemini2FlashLite, "google/gemini-2.0-flash-lite-001"},
		{ModelGrok3, "x-ai/grok-3"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if tt.model.String() != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, tt.model.String())
			}
		})
	}
}

func TestModelsRegistry(t *testing.T) {
	// Check that registry has expected models
	expectedModels := []Model{
		ModelGPT5,
		ModelGPT5Codex,
		ModelGPT4o,
		ModelClaudeOpus,
		ModelClaudeSonnet,
		ModelClaudeHaiku,
		ModelGemini3Pro,
		ModelGemini3Flash,
		ModelGemini25Pro,
		ModelGemini25Flash,
		ModelGemini25FlashLite,
		ModelGemini2Flash,
		ModelGemini2FlashLite,
		ModelGrok41Fast,
		ModelGrok3,
		ModelQwen3Next,
		ModelLlama4,
	}

	for _, model := range expectedModels {
		info, exists := Models[model]
		if !exists {
			t.Errorf("model %s should be in registry", model)
			continue
		}
		if info.ID != model {
			t.Errorf("model info ID mismatch: expected %s, got %s", model, info.ID)
		}
		if info.Name == "" {
			t.Errorf("model %s should have a name", model)
		}
		if info.Provider == "" {
			t.Errorf("model %s should have a provider", model)
		}
	}
}

func TestModelInfoFields(t *testing.T) {
	info := Models[ModelGPT5]

	if info.ID != ModelGPT5 {
		t.Error("ID mismatch")
	}
	if info.Name != "GPT-5.2" {
		t.Errorf("expected name 'GPT-5.2', got %q", info.Name)
	}
	if info.Provider != "OpenAI" {
		t.Errorf("expected provider 'OpenAI', got %q", info.Provider)
	}
	if info.Description == "" {
		t.Error("description should not be empty")
	}
}

func TestModelConstants(t *testing.T) {
	// Verify model constants are properly defined
	if ModelGPT5 == "" {
		t.Error("ModelGPT5 should not be empty")
	}
	if ModelClaudeOpus == "" {
		t.Error("ModelClaudeOpus should not be empty")
	}
	if ModelGemini3Flash == "" {
		t.Error("ModelGemini3Flash should not be empty")
	}
	if ModelGrok3 == "" {
		t.Error("ModelGrok3 should not be empty")
	}
}

func TestOpenAIModelConstants(t *testing.T) {
	tests := []struct {
		name string
		m    Model
		want string
	}{
		{"GPT52", ModelGPT52, "openai/gpt-5.2"},
		{"GPT52Pro", ModelGPT52Pro, "openai/gpt-5.2-pro"},
		{"GPT51", ModelGPT51, "openai/gpt-5.1"},
		{"GPT5Alias", ModelGPT5, "openai/gpt-5.2"},
		{"GPT5Mini", ModelGPT5Mini, "openai/gpt-5-mini"},
		{"GPT5Nano", ModelGPT5Nano, "openai/gpt-5-nano"},
		{"GPT51Codex", ModelGPT51Codex, "openai/gpt-5.1-codex"},
		{"GPT51CodexMax", ModelGPT51CodexMax, "openai/gpt-5.1-codex-max"},
		{"GPT5CodexAlias", ModelGPT5Codex, "openai/gpt-5.1-codex-max"},
		{"GPT41", ModelGPT41, "openai/gpt-4.1"},
		{"O3", ModelO3, "openai/o3"},
		{"O4Mini", ModelO4Mini, "openai/o4-mini"},
		{"Realtime", ModelGPTRealtime, "openai/gpt-realtime"},
		{"Audio", ModelGPTAudio, "openai/gpt-audio"},
		{"Image15", ModelGPTImage15, "openai/gpt-image-1.5"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.m.String() != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, tt.m.String())
			}
		})
	}
}

func TestCustomModel(t *testing.T) {
	custom := Model("my-provider/custom-model")

	if custom.String() != "my-provider/custom-model" {
		t.Error("custom model string conversion failed")
	}

	// Custom model won't be in registry
	_, exists := Models[custom]
	if exists {
		t.Error("custom model should not be in registry")
	}
}
