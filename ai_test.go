package ai

import (
	"testing"
)

func TestDefaultModel(t *testing.T) {
	// Save original
	original := DefaultModel
	defer func() { DefaultModel = original }()

	// Test default is GPT5
	if DefaultModel != ModelGPT5 {
		t.Errorf("expected default model to be ModelGPT5, got %s", DefaultModel)
	}

	// Test changing default
	DefaultModel = ModelClaudeOpus
	b := Default()
	if b.model != ModelClaudeOpus {
		t.Errorf("expected Default() to use ModelClaudeOpus, got %s", b.model)
	}
}

func TestModelShortcuts(t *testing.T) {
	tests := []struct {
		name     string
		builder  *Builder
		expected Model
	}{
		{"GPT5", GPT5(), ModelGPT5},
		{"GPT5Codex", GPT5Codex(), ModelGPT5Codex},
		{"GPT4o", GPT4o(), ModelGPT4o},
		{"GPT4oMini", GPT4oMini(), ModelGPT4oMini},
		{"O1", O1(), ModelO1},
		{"GPT52", GPT52(), ModelGPT52},
		{"GPT52Pro", GPT52Pro(), ModelGPT52Pro},
		{"GPT51", GPT51(), ModelGPT51},
		{"GPT5Base", GPT5Base(), ModelGPT5Base},
		{"GPT5Pro", GPT5Pro(), ModelGPT5Pro},
		{"GPT5Mini", GPT5Mini(), ModelGPT5Mini},
		{"GPT5Nano", GPT5Nano(), ModelGPT5Nano},
		{"GPT51Codex", GPT51Codex(), ModelGPT51Codex},
		{"GPT51CodexMax", GPT51CodexMax(), ModelGPT51CodexMax},
		{"GPT5CodexBase", GPT5CodexBase(), ModelGPT5CodexBase},
		{"GPT51CodexMini", GPT51CodexMini(), ModelGPT51CodexMini},
		{"CodexMiniLatest", CodexMiniLatest(), ModelCodexMiniLatest},
		{"GPT5SearchAPI", GPT5SearchAPI(), ModelGPT5SearchAPI},
		{"ComputerUsePreview", ComputerUsePreview(), ModelComputerUsePreview},
		{"GPT5ChatLatest", GPT5ChatLatest(), ModelGPT5ChatLatest},
		{"GPT52ChatLatest", GPT52ChatLatest(), ModelGPT52ChatLatest},
		{"GPT51ChatLatest", GPT51ChatLatest(), ModelGPT51ChatLatest},
		{"ChatGPT4oLatest", ChatGPT4oLatest(), ModelChatGPT4oLatest},
		{"GPT41", GPT41(), ModelGPT41},
		{"GPT41Mini", GPT41Mini(), ModelGPT41Mini},
		{"GPT41Nano", GPT41Nano(), ModelGPT41Nano},
		{"GPT4o20240513", GPT4o20240513(), ModelGPT4o20240513},
		{"O1Mini", O1Mini(), ModelO1Mini},
		{"O1Pro", O1Pro(), ModelO1Pro},
		{"O1Preview", O1Preview(), ModelO1Preview},
		{"O3", O3(), ModelO3},
		{"O3Mini", O3Mini(), ModelO3Mini},
		{"O3Pro", O3Pro(), ModelO3Pro},
		{"O3DeepResearch", O3DeepResearch(), ModelO3DeepResearch},
		{"O4Mini", O4Mini(), ModelO4Mini},
		{"O4MiniDeepResearch", O4MiniDeepResearch(), ModelO4MiniDeepResearch},
		{"GPTRealtime", GPTRealtime(), ModelGPTRealtime},
		{"GPTRealtimeMini", GPTRealtimeMini(), ModelGPTRealtimeMini},
		{"GPT4oRealtimePreview", GPT4oRealtimePreview(), ModelGPT4oRealtimePreview},
		{"GPT4oMiniRealtimePreview", GPT4oMiniRealtimePreview(), ModelGPT4oMiniRealtimePreview},
		{"GPTAudio", GPTAudio(), ModelGPTAudio},
		{"GPTAudioMini", GPTAudioMini(), ModelGPTAudioMini},
		{"GPT4oAudioPreview", GPT4oAudioPreview(), ModelGPT4oAudioPreview},
		{"GPT4oMiniAudioPreview", GPT4oMiniAudioPreview(), ModelGPT4oMiniAudioPreview},
		{"GPT4oMiniSearchPreview", GPT4oMiniSearchPreview(), ModelGPT4oMiniSearchPreview},
		{"GPT4oSearchPreview", GPT4oSearchPreview(), ModelGPT4oSearchPreview},
		{"GPT4oMiniTTS", GPT4oMiniTTS(), ModelGPT4oMiniTTS},
		{"GPT4oTranscribe", GPT4oTranscribe(), ModelGPT4oTranscribe},
		{"GPT4oTranscribeDiarize", GPT4oTranscribeDiarize(), ModelGPT4oTranscribeDiarize},
		{"GPT4oMiniTranscribe", GPT4oMiniTranscribe(), ModelGPT4oMiniTranscribe},
		{"GPTImage15", GPTImage15(), ModelGPTImage15},
		{"GPTImage1", GPTImage1(), ModelGPTImage1},
		{"GPTImage1Mini", GPTImage1Mini(), ModelGPTImage1Mini},
		{"ChatGPTImageLatest", ChatGPTImageLatest(), ModelChatGPTImageLatest},
		{"GPTOSS120B", GPTOSS120B(), ModelGPTOSS120B},
		{"GPTOSS20B", GPTOSS20B(), ModelGPTOSS20B},
		{"Sora2", Sora2(), ModelSora2},
		{"Sora2Pro", Sora2Pro(), ModelSora2Pro},
		{"Claude", Claude(), ModelClaudeOpus},
		{"ClaudeSonnet", ClaudeSonnet(), ModelClaudeSonnet},
		{"ClaudeHaiku", ClaudeHaiku(), ModelClaudeHaiku},
		{"Gemini", Gemini(), ModelGemini3Flash},
		{"GeminiPro", GeminiPro(), ModelGemini3Pro},
		{"GeminiFlash", GeminiFlash(), ModelGemini3Flash},
		{"Grok", Grok(), ModelGrok3},
		{"GrokFast", GrokFast(), ModelGrok41Fast},
		{"GrokMini", GrokMini(), ModelGrok3Mini},
		{"Qwen", Qwen(), ModelQwen3Next},
		{"Llama", Llama(), ModelLlama4},
		{"Mistral", Mistral(), ModelMistralLarge},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.builder.model != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, tt.builder.model)
			}
		})
	}
}

func TestUse(t *testing.T) {
	customModel := "custom/model-id"
	b := Use(customModel)
	if string(b.model) != customModel {
		t.Errorf("expected %s, got %s", customModel, b.model)
	}
}

func TestVars(t *testing.T) {
	vars := Vars{
		"key1": "value1",
		"key2": "value2",
	}

	if vars["key1"] != "value1" {
		t.Error("Vars map not working correctly")
	}
}

func TestApplyTemplate(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		vars     Vars
		expected string
	}{
		{
			name:     "single var",
			text:     "Hello {{name}}!",
			vars:     Vars{"name": "World"},
			expected: "Hello World!",
		},
		{
			name:     "multiple vars",
			text:     "{{greeting}} {{name}}!",
			vars:     Vars{"greeting": "Hello", "name": "World"},
			expected: "Hello World!",
		},
		{
			name:     "no vars",
			text:     "Hello World!",
			vars:     Vars{},
			expected: "Hello World!",
		},
		{
			name:     "missing var unchanged",
			text:     "Hello {{name}}!",
			vars:     Vars{},
			expected: "Hello {{name}}!",
		},
		{
			name:     "repeated var",
			text:     "{{x}} and {{x}} again",
			vars:     Vars{"x": "test"},
			expected: "test and test again",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := applyTemplate(tt.text, tt.vars)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestApplyVars(t *testing.T) {
	text := "Hello {{name}}!"
	vars := Vars{"name": "Test"}
	result := ApplyVars(text, vars)
	if result != "Hello Test!" {
		t.Errorf("expected 'Hello Test!', got %q", result)
	}
}

func TestLoadFile(t *testing.T) {
	// Test non-existent file
	_, err := LoadFile("nonexistent.txt")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestMustLoadFilePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-existent file")
		}
	}()
	MustLoadFile("nonexistent.txt")
}
