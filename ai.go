package ai

import (
	"os"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Global Settings
// ═══════════════════════════════════════════════════════════════════════════

var (
	// DefaultModel is the model used by top-level functions like Ask().
	// Change this to use a different default model across the application.
	DefaultModel Model = ModelGPT5

	// Debug enables printing of raw requests and responses to stdout.
	Debug = false

	// Pretty enables colored output for debug logs.
	Pretty = true

	// Cache enables caching of identical requests (experimental).
	Cache = false
)

// ═══════════════════════════════════════════════════════════════════════════
// Quick One-Liners (use DefaultModel)
// ═══════════════════════════════════════════════════════════════════════════

// Ask sends a quick message using the DefaultModel and returns the response.
// This is the simplest way to get an answer from the AI.
func Ask(prompt string) (string, error) {
	return New(DefaultModel).User(prompt).Send()
}

// AskWith sends a message with a specific system prompt using the DefaultModel.
// Useful for one-off tasks requiring specific instructions.
func AskWith(system, prompt string) (string, error) {
	return New(DefaultModel).System(system).User(prompt).Send()
}

// AskModel sends a message to a specific model.
// Shortcut for ai.New(model).User(prompt).Send().
func AskModel(model Model, prompt string) (string, error) {
	return New(model).User(prompt).Send()
}

// Default returns a new Builder initialized with the DefaultModel.
func Default() *Builder {
	return New(DefaultModel)
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Shortcuts - Start a builder for each model
// These use DefaultProvider (OpenRouter by default)
// For direct provider access, use: ai.Anthropic().Claude(), ai.OpenAI().GPT4o(), etc.
// ═══════════════════════════════════════════════════════════════════════════

// OpenAI
// GPT5 returns a Builder configured for ModelGPT5 using the current DefaultProvider.
func GPT5() *Builder { return getDefaultClient().GPT5() }

// GPT5Codex returns a Builder configured for ModelGPT5Codex using the current DefaultProvider.
func GPT5Codex() *Builder { return getDefaultClient().GPT5Codex() }

// GPT4o returns a Builder configured for ModelGPT4o using the current DefaultProvider.
func GPT4o() *Builder { return getDefaultClient().GPT4o() }

// GPT4oMini returns a Builder configured for ModelGPT4oMini using the current DefaultProvider.
func GPT4oMini() *Builder { return getDefaultClient().GPT4oMini() }

// O1 returns a Builder configured for ModelO1 using the current DefaultProvider.
func O1() *Builder { return getDefaultClient().O1() }

// GPT 5.x family
// GPT52 returns a Builder configured for ModelGPT52 using the current DefaultProvider.
func GPT52() *Builder { return getDefaultClient().GPT52() }

// GPT52Pro returns a Builder configured for ModelGPT52Pro using the current DefaultProvider.
func GPT52Pro() *Builder { return getDefaultClient().GPT52Pro() }

// GPT51 returns a Builder configured for ModelGPT51 using the current DefaultProvider.
func GPT51() *Builder { return getDefaultClient().GPT51() }

// GPT5Base returns a Builder configured for ModelGPT5Base using the current DefaultProvider.
func GPT5Base() *Builder { return getDefaultClient().GPT5Base() }

// GPT5Pro returns a Builder configured for ModelGPT5Pro using the current DefaultProvider.
func GPT5Pro() *Builder { return getDefaultClient().GPT5Pro() }

// GPT5Mini returns a Builder configured for ModelGPT5Mini using the current DefaultProvider.
func GPT5Mini() *Builder { return getDefaultClient().GPT5Mini() }

// GPT5Nano returns a Builder configured for ModelGPT5Nano using the current DefaultProvider.
func GPT5Nano() *Builder { return getDefaultClient().GPT5Nano() }

// Codex
// GPT51Codex returns a Builder configured for ModelGPT51Codex using the current DefaultProvider.
func GPT51Codex() *Builder { return getDefaultClient().GPT51Codex() }

// GPT51CodexMax returns a Builder configured for ModelGPT51CodexMax using the current DefaultProvider.
func GPT51CodexMax() *Builder { return getDefaultClient().GPT51CodexMax() }

// GPT5CodexBase returns a Builder configured for ModelGPT5CodexBase using the current DefaultProvider.
func GPT5CodexBase() *Builder { return getDefaultClient().GPT5CodexBase() }

// GPT51CodexMini returns a Builder configured for ModelGPT51CodexMini using the current DefaultProvider.
func GPT51CodexMini() *Builder { return getDefaultClient().GPT51CodexMini() }

// CodexMiniLatest returns a Builder configured for ModelCodexMiniLatest using the current DefaultProvider.
func CodexMiniLatest() *Builder { return getDefaultClient().CodexMiniLatest() }

// Search + agent tools
// GPT5SearchAPI returns a Builder configured for ModelGPT5SearchAPI using the current DefaultProvider.
func GPT5SearchAPI() *Builder { return getDefaultClient().GPT5SearchAPI() }

// ComputerUsePreview returns a Builder configured for ModelComputerUsePreview using the current DefaultProvider.
func ComputerUsePreview() *Builder { return getDefaultClient().ComputerUsePreview() }

// Aliases
// GPT5ChatLatest returns a Builder configured for ModelGPT5ChatLatest using the current DefaultProvider.
func GPT5ChatLatest() *Builder { return getDefaultClient().GPT5ChatLatest() }

// GPT52ChatLatest returns a Builder configured for ModelGPT52ChatLatest using the current DefaultProvider.
func GPT52ChatLatest() *Builder { return getDefaultClient().GPT52ChatLatest() }

// GPT51ChatLatest returns a Builder configured for ModelGPT51ChatLatest using the current DefaultProvider.
func GPT51ChatLatest() *Builder { return getDefaultClient().GPT51ChatLatest() }

// ChatGPT4oLatest returns a Builder configured for ModelChatGPT4oLatest using the current DefaultProvider.
func ChatGPT4oLatest() *Builder { return getDefaultClient().ChatGPT4oLatest() }

// GPT-4.1
// GPT41 returns a Builder configured for ModelGPT41 using the current DefaultProvider.
func GPT41() *Builder { return getDefaultClient().GPT41() }

// GPT41Mini returns a Builder configured for ModelGPT41Mini using the current DefaultProvider.
func GPT41Mini() *Builder { return getDefaultClient().GPT41Mini() }

// GPT41Nano returns a Builder configured for ModelGPT41Nano using the current DefaultProvider.
func GPT41Nano() *Builder { return getDefaultClient().GPT41Nano() }

// GPT-4o dated snapshot
// GPT4o20240513 returns a Builder configured for ModelGPT4o20240513 using the current DefaultProvider.
func GPT4o20240513() *Builder { return getDefaultClient().GPT4o20240513() }

// o-series
// O1Mini returns a Builder configured for ModelO1Mini using the current DefaultProvider.
func O1Mini() *Builder { return getDefaultClient().O1Mini() }

// O1Pro returns a Builder configured for ModelO1Pro using the current DefaultProvider.
func O1Pro() *Builder { return getDefaultClient().O1Pro() }

// O1Preview returns a Builder configured for ModelO1Preview using the current DefaultProvider.
func O1Preview() *Builder { return getDefaultClient().O1Preview() }

// O3 returns a Builder configured for ModelO3 using the current DefaultProvider.
func O3() *Builder { return getDefaultClient().O3() }

// O3Mini returns a Builder configured for ModelO3Mini using the current DefaultProvider.
func O3Mini() *Builder { return getDefaultClient().O3Mini() }

// O3Pro returns a Builder configured for ModelO3Pro using the current DefaultProvider.
func O3Pro() *Builder { return getDefaultClient().O3Pro() }

// O3DeepResearch returns a Builder configured for ModelO3DeepResearch using the current DefaultProvider.
func O3DeepResearch() *Builder { return getDefaultClient().O3DeepResearch() }

// O4Mini returns a Builder configured for ModelO4Mini using the current DefaultProvider.
func O4Mini() *Builder { return getDefaultClient().O4Mini() }

// O4MiniDeepResearch returns a Builder configured for ModelO4MiniDeepResearch using the current DefaultProvider.
func O4MiniDeepResearch() *Builder {
	return getDefaultClient().O4MiniDeepResearch()
}

// Realtime / audio
// GPTRealtime returns a Builder configured for ModelGPTRealtime using the current DefaultProvider.
func GPTRealtime() *Builder { return getDefaultClient().GPTRealtime() }

// GPTRealtimeMini returns a Builder configured for ModelGPTRealtimeMini using the current DefaultProvider.
func GPTRealtimeMini() *Builder { return getDefaultClient().GPTRealtimeMini() }

// GPT4oRealtimePreview returns a Builder configured for ModelGPT4oRealtimePreview using the current DefaultProvider.
func GPT4oRealtimePreview() *Builder { return getDefaultClient().GPT4oRealtimePreview() }

// GPT4oMiniRealtimePreview returns a Builder configured for ModelGPT4oMiniRealtimePreview using the current DefaultProvider.
func GPT4oMiniRealtimePreview() *Builder { return getDefaultClient().GPT4oMiniRealtimePreview() }

// GPTAudio returns a Builder configured for ModelGPTAudio using the current DefaultProvider.
func GPTAudio() *Builder { return getDefaultClient().GPTAudio() }

// GPTAudioMini returns a Builder configured for ModelGPTAudioMini using the current DefaultProvider.
func GPTAudioMini() *Builder { return getDefaultClient().GPTAudioMini() }

// GPT4oAudioPreview returns a Builder configured for ModelGPT4oAudioPreview using the current DefaultProvider.
func GPT4oAudioPreview() *Builder { return getDefaultClient().GPT4oAudioPreview() }

// GPT4oMiniAudioPreview returns a Builder configured for ModelGPT4oMiniAudioPreview using the current DefaultProvider.
func GPT4oMiniAudioPreview() *Builder { return getDefaultClient().GPT4oMiniAudioPreview() }

// Search previews
// GPT4oMiniSearchPreview returns a Builder configured for ModelGPT4oMiniSearchPreview using the current DefaultProvider.
func GPT4oMiniSearchPreview() *Builder { return getDefaultClient().GPT4oMiniSearchPreview() }

// GPT4oSearchPreview returns a Builder configured for ModelGPT4oSearchPreview using the current DefaultProvider.
func GPT4oSearchPreview() *Builder { return getDefaultClient().GPT4oSearchPreview() }

// Speech / transcription
// GPT4oMiniTTS returns a Builder configured for ModelGPT4oMiniTTS using the current DefaultProvider.
func GPT4oMiniTTS() *Builder { return getDefaultClient().GPT4oMiniTTS() }

// GPT4oTranscribe returns a Builder configured for ModelGPT4oTranscribe using the current DefaultProvider.
func GPT4oTranscribe() *Builder { return getDefaultClient().GPT4oTranscribe() }

// GPT4oTranscribeDiarize returns a Builder configured for ModelGPT4oTranscribeDiarize using the current DefaultProvider.
func GPT4oTranscribeDiarize() *Builder { return getDefaultClient().GPT4oTranscribeDiarize() }

// GPT4oMiniTranscribe returns a Builder configured for ModelGPT4oMiniTranscribe using the current DefaultProvider.
func GPT4oMiniTranscribe() *Builder { return getDefaultClient().GPT4oMiniTranscribe() }

// Image generation
// GPTImage15 returns a Builder configured for ModelGPTImage15 using the current DefaultProvider.
func GPTImage15() *Builder { return getDefaultClient().GPTImage15() }

// GPTImage1 returns a Builder configured for ModelGPTImage1 using the current DefaultProvider.
func GPTImage1() *Builder { return getDefaultClient().GPTImage1() }

// GPTImage1Mini returns a Builder configured for ModelGPTImage1Mini using the current DefaultProvider.
func GPTImage1Mini() *Builder { return getDefaultClient().GPTImage1Mini() }

// ChatGPTImageLatest returns a Builder configured for ModelChatGPTImageLatest using the current DefaultProvider.
func ChatGPTImageLatest() *Builder { return getDefaultClient().ChatGPTImageLatest() }

// Open-weight models
// GPTOSS120B returns a Builder configured for ModelGPTOSS120B using the current DefaultProvider.
func GPTOSS120B() *Builder { return getDefaultClient().GPTOSS120B() }

// GPTOSS20B returns a Builder configured for ModelGPTOSS20B using the current DefaultProvider.
func GPTOSS20B() *Builder { return getDefaultClient().GPTOSS20B() }

// Video generation
// Sora2 returns a Builder configured for ModelSora2 using the current DefaultProvider.
func Sora2() *Builder { return getDefaultClient().Sora2() }

// Sora2Pro returns a Builder configured for ModelSora2Pro using the current DefaultProvider.
func Sora2Pro() *Builder { return getDefaultClient().Sora2Pro() }

// Anthropic
// Claude returns a Builder configured for the default Claude model (currently ModelClaudeOpus) using the current DefaultProvider.
func Claude() *Builder { return getDefaultClient().Claude() }

// ClaudeSonnet returns a Builder configured for ModelClaudeSonnet using the current DefaultProvider.
func ClaudeSonnet() *Builder { return getDefaultClient().ClaudeSonnet() }

// ClaudeHaiku returns a Builder configured for ModelClaudeHaiku using the current DefaultProvider.
func ClaudeHaiku() *Builder { return getDefaultClient().ClaudeHaiku() }

// ClaudeOpus41 returns a Builder configured for ModelClaudeOpus41 using the current DefaultProvider.
func ClaudeOpus41() *Builder { return getDefaultClient().ClaudeOpus41() }

// ClaudeOpus4 returns a Builder configured for ModelClaudeOpus4 using the current DefaultProvider.
func ClaudeOpus4() *Builder { return getDefaultClient().ClaudeOpus4() }

// ClaudeSonnet4 returns a Builder configured for ModelClaudeSonnet4 using the current DefaultProvider.
func ClaudeSonnet4() *Builder {
	return getDefaultClient().ClaudeSonnet4()
}

// ClaudeSonnet37 returns a Builder configured for ModelClaudeSonnet37 using the current DefaultProvider.
func ClaudeSonnet37() *Builder {
	return getDefaultClient().ClaudeSonnet37()
}

// ClaudeHaiku35 returns a Builder configured for ModelClaudeHaiku35 using the current DefaultProvider.
func ClaudeHaiku35() *Builder { return getDefaultClient().ClaudeHaiku35() }

// ClaudeHaiku3 returns a Builder configured for ModelClaudeHaiku3 using the current DefaultProvider.
func ClaudeHaiku3() *Builder { return getDefaultClient().ClaudeHaiku3() }

// ClaudeOpus3 returns a Builder configured for ModelClaudeOpus3 using the current DefaultProvider.
func ClaudeOpus3() *Builder { return getDefaultClient().ClaudeOpus3() }

// ClaudeSonnet3 returns a Builder configured for ModelClaudeSonnet3 using the current DefaultProvider.
func ClaudeSonnet3() *Builder { return getDefaultClient().ClaudeSonnet3() }

// Google
// Gemini returns a Builder configured for the default Gemini model using the current DefaultProvider.
func Gemini() *Builder { return getDefaultClient().Gemini() }

// GeminiPro returns a Builder configured for the Gemini Pro model using the current DefaultProvider.
func GeminiPro() *Builder { return getDefaultClient().GeminiPro() }

// GeminiFlash returns a Builder configured for the Gemini Flash model using the current DefaultProvider.
func GeminiFlash() *Builder { return getDefaultClient().GeminiFlash() }

// xAI
// Grok returns a Builder configured for the default Grok model using the current DefaultProvider.
func Grok() *Builder { return getDefaultClient().Grok() }

// GrokFast returns a Builder configured for ModelGrok41Fast using the current DefaultProvider.
func GrokFast() *Builder { return getDefaultClient().GrokFast() }

// GrokMini returns a Builder configured for ModelGrok3Mini using the current DefaultProvider.
func GrokMini() *Builder { return getDefaultClient().GrokMini() }

// Alibaba
// Qwen returns a Builder configured for the default Qwen model using the current DefaultProvider.
func Qwen() *Builder { return getDefaultClient().Qwen() }

// Meta
// Llama returns a Builder configured for the default Llama model using the current DefaultProvider.
func Llama() *Builder { return getDefaultClient().Llama() }

// Mistral
// Mistral returns a Builder configured for the default Mistral model using the current DefaultProvider.
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
