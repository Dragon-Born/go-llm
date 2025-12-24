package ai

import (
	"os"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Global Settings
// ═══════════════════════════════════════════════════════════════════════════

var (
	// DefaultModel is used by Ask(), Prompt(), and error fallbacks
	// Change this to use a different default model
	DefaultModel Model = ModelGPT5

	Debug  = false // Print requests/responses
	Pretty = true  // Colored output
	Cache  = false // Cache identical requests
)

// ═══════════════════════════════════════════════════════════════════════════
// Quick One-Liners (use DefaultModel)
// ═══════════════════════════════════════════════════════════════════════════

// Ask sends a quick message using DefaultModel
func Ask(prompt string) (string, error) {
	return New(DefaultModel).User(prompt).Send()
}

// AskWith sends a message with a system prompt using DefaultModel
func AskWith(system, prompt string) (string, error) {
	return New(DefaultModel).System(system).User(prompt).Send()
}

// AskModel sends a message to a specific model
func AskModel(model Model, prompt string) (string, error) {
	return New(model).User(prompt).Send()
}

// Default returns a builder with DefaultModel
func Default() *Builder {
	return New(DefaultModel)
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Shortcuts - Start a builder for each model
// These use DefaultProvider (OpenRouter by default)
// For direct provider access, use: ai.Anthropic().Claude(), ai.OpenAI().GPT4o(), etc.
// ═══════════════════════════════════════════════════════════════════════════

// OpenAI
func GPT5() *Builder      { return getDefaultClient().GPT5() }      // General purpose
func GPT5Codex() *Builder { return getDefaultClient().GPT5Codex() } // Code-specialized
func GPT4o() *Builder     { return getDefaultClient().GPT4o() }
func GPT4oMini() *Builder { return getDefaultClient().GPT4oMini() }
func O1() *Builder        { return getDefaultClient().O1() }

// GPT 5.x family
func GPT52() *Builder    { return getDefaultClient().GPT52() }
func GPT52Pro() *Builder { return getDefaultClient().GPT52Pro() }
func GPT51() *Builder    { return getDefaultClient().GPT51() }
func GPT5Base() *Builder { return getDefaultClient().GPT5Base() }
func GPT5Pro() *Builder  { return getDefaultClient().GPT5Pro() }
func GPT5Mini() *Builder { return getDefaultClient().GPT5Mini() }
func GPT5Nano() *Builder { return getDefaultClient().GPT5Nano() }

// Codex
func GPT51Codex() *Builder      { return getDefaultClient().GPT51Codex() }
func GPT51CodexMax() *Builder   { return getDefaultClient().GPT51CodexMax() }
func GPT5CodexBase() *Builder   { return getDefaultClient().GPT5CodexBase() }
func GPT51CodexMini() *Builder  { return getDefaultClient().GPT51CodexMini() }
func CodexMiniLatest() *Builder { return getDefaultClient().CodexMiniLatest() }

// Search + agent tools
func GPT5SearchAPI() *Builder      { return getDefaultClient().GPT5SearchAPI() }
func ComputerUsePreview() *Builder { return getDefaultClient().ComputerUsePreview() }

// Aliases
func GPT5ChatLatest() *Builder  { return getDefaultClient().GPT5ChatLatest() }
func GPT52ChatLatest() *Builder { return getDefaultClient().GPT52ChatLatest() }
func GPT51ChatLatest() *Builder { return getDefaultClient().GPT51ChatLatest() }
func ChatGPT4oLatest() *Builder { return getDefaultClient().ChatGPT4oLatest() }

// GPT-4.1
func GPT41() *Builder     { return getDefaultClient().GPT41() }
func GPT41Mini() *Builder { return getDefaultClient().GPT41Mini() }
func GPT41Nano() *Builder { return getDefaultClient().GPT41Nano() }

// GPT-4o dated snapshot
func GPT4o20240513() *Builder { return getDefaultClient().GPT4o20240513() }

// o-series
func O1Mini() *Builder         { return getDefaultClient().O1Mini() }
func O1Pro() *Builder          { return getDefaultClient().O1Pro() }
func O1Preview() *Builder      { return getDefaultClient().O1Preview() }
func O3() *Builder             { return getDefaultClient().O3() }
func O3Mini() *Builder         { return getDefaultClient().O3Mini() }
func O3Pro() *Builder          { return getDefaultClient().O3Pro() }
func O3DeepResearch() *Builder { return getDefaultClient().O3DeepResearch() }
func O4Mini() *Builder         { return getDefaultClient().O4Mini() }
func O4MiniDeepResearch() *Builder {
	return getDefaultClient().O4MiniDeepResearch()
}

// Realtime / audio
func GPTRealtime() *Builder              { return getDefaultClient().GPTRealtime() }
func GPTRealtimeMini() *Builder          { return getDefaultClient().GPTRealtimeMini() }
func GPT4oRealtimePreview() *Builder     { return getDefaultClient().GPT4oRealtimePreview() }
func GPT4oMiniRealtimePreview() *Builder { return getDefaultClient().GPT4oMiniRealtimePreview() }
func GPTAudio() *Builder                 { return getDefaultClient().GPTAudio() }
func GPTAudioMini() *Builder             { return getDefaultClient().GPTAudioMini() }
func GPT4oAudioPreview() *Builder        { return getDefaultClient().GPT4oAudioPreview() }
func GPT4oMiniAudioPreview() *Builder    { return getDefaultClient().GPT4oMiniAudioPreview() }

// Search previews
func GPT4oMiniSearchPreview() *Builder { return getDefaultClient().GPT4oMiniSearchPreview() }
func GPT4oSearchPreview() *Builder     { return getDefaultClient().GPT4oSearchPreview() }

// Speech / transcription
func GPT4oMiniTTS() *Builder           { return getDefaultClient().GPT4oMiniTTS() }
func GPT4oTranscribe() *Builder        { return getDefaultClient().GPT4oTranscribe() }
func GPT4oTranscribeDiarize() *Builder { return getDefaultClient().GPT4oTranscribeDiarize() }
func GPT4oMiniTranscribe() *Builder    { return getDefaultClient().GPT4oMiniTranscribe() }

// Image generation
func GPTImage15() *Builder         { return getDefaultClient().GPTImage15() }
func GPTImage1() *Builder          { return getDefaultClient().GPTImage1() }
func GPTImage1Mini() *Builder      { return getDefaultClient().GPTImage1Mini() }
func ChatGPTImageLatest() *Builder { return getDefaultClient().ChatGPTImageLatest() }

// Open-weight models
func GPTOSS120B() *Builder { return getDefaultClient().GPTOSS120B() }
func GPTOSS20B() *Builder  { return getDefaultClient().GPTOSS20B() }

// Video generation
func Sora2() *Builder    { return getDefaultClient().Sora2() }
func Sora2Pro() *Builder { return getDefaultClient().Sora2Pro() }

// Anthropic
func Claude() *Builder       { return getDefaultClient().Claude() }
func ClaudeSonnet() *Builder { return getDefaultClient().ClaudeSonnet() }
func ClaudeHaiku() *Builder  { return getDefaultClient().ClaudeHaiku() }
func ClaudeOpus41() *Builder { return getDefaultClient().ClaudeOpus41() }
func ClaudeOpus4() *Builder  { return getDefaultClient().ClaudeOpus4() }
func ClaudeSonnet4() *Builder {
	return getDefaultClient().ClaudeSonnet4()
}
func ClaudeSonnet37() *Builder {
	return getDefaultClient().ClaudeSonnet37()
}
func ClaudeHaiku35() *Builder { return getDefaultClient().ClaudeHaiku35() }
func ClaudeHaiku3() *Builder  { return getDefaultClient().ClaudeHaiku3() }
func ClaudeOpus3() *Builder   { return getDefaultClient().ClaudeOpus3() }
func ClaudeSonnet3() *Builder { return getDefaultClient().ClaudeSonnet3() }

// Google
func Gemini() *Builder      { return getDefaultClient().Gemini() }    // Fast default
func GeminiPro() *Builder   { return getDefaultClient().GeminiPro() } // High-precision, 1M ctx
func GeminiFlash() *Builder { return getDefaultClient().GeminiFlash() }

// xAI
func Grok() *Builder     { return getDefaultClient().Grok() }
func GrokFast() *Builder { return getDefaultClient().GrokFast() } // 2M context, speed optimized
func GrokMini() *Builder { return getDefaultClient().GrokMini() }

// Alibaba
func Qwen() *Builder { return getDefaultClient().Qwen() }

// Meta
func Llama() *Builder { return getDefaultClient().Llama() }

// Mistral
func Mistral() *Builder { return getDefaultClient().Mistral() }

// ═══════════════════════════════════════════════════════════════════════════
// Use any model by string (for dynamic model selection)
// ═══════════════════════════════════════════════════════════════════════════

// Use creates a builder for any model by string ID
func Use(modelID string) *Builder {
	return getDefaultClient().Use(modelID)
}

// ═══════════════════════════════════════════════════════════════════════════
// Template Processing
// ═══════════════════════════════════════════════════════════════════════════

// Vars is a shorthand for template variables
type Vars map[string]string

// applyTemplate replaces {{key}} with values
func applyTemplate(text string, vars Vars) string {
	for k, v := range vars {
		text = strings.ReplaceAll(text, "{{"+k+"}}", v)
	}
	return text
}

// ═══════════════════════════════════════════════════════════════════════════
// File Helpers
// ═══════════════════════════════════════════════════════════════════════════

// LoadFile reads a file and returns its content
func LoadFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// MustLoadFile reads a file or panics
func MustLoadFile(path string) string {
	data, err := LoadFile(path)
	if err != nil {
		panic(err)
	}
	return data
}
