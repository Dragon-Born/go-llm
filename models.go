package ai

// Model represents a unique identifier for an AI model.
// Models are typically namespaced by provider (e.g., "openai/gpt-5").
type Model string

// Latest models as of December 2025
const (
	// OpenAI
	// Frontier / GPT family
	ModelGPT52    Model = "openai/gpt-5.2"
	ModelGPT52Pro Model = "openai/gpt-5.2-pro"
	ModelGPT51    Model = "openai/gpt-5.1"
	ModelGPT5Base Model = "openai/gpt-5" // pricing table includes "gpt-5"
	ModelGPT5Pro  Model = "openai/gpt-5-pro"

	// Back-compat aliases
	ModelGPT5 Model = ModelGPT52 // "GPT5" means current recommended GPT-5.x (5.2)

	ModelGPT5Mini Model = "openai/gpt-5-mini"
	ModelGPT5Nano Model = "openai/gpt-5-nano"

	// Codex (agentic coding)
	ModelGPT51Codex      Model = "openai/gpt-5.1-codex"
	ModelGPT51CodexMax   Model = "openai/gpt-5.1-codex-max"
	ModelGPT5Codex       Model = ModelGPT51CodexMax // Back-compat: "Codex" shortcut targets the most capable Codex
	ModelGPT5CodexBase   Model = "openai/gpt-5-codex"
	ModelGPT51CodexMini  Model = "openai/gpt-5.1-codex-mini"
	ModelCodexMiniLatest Model = "openai/codex-mini-latest"

	// Search + agent tools
	ModelGPT5SearchAPI      Model = "openai/gpt-5-search-api"
	ModelComputerUsePreview Model = "openai/computer-use-preview"

	// ChatGPT aliases
	ModelGPT52ChatLatest Model = "openai/gpt-5.2-chat-latest"
	ModelGPT51ChatLatest Model = "openai/gpt-5.1-chat-latest"
	ModelGPT5ChatLatest  Model = "openai/gpt-5-chat-latest"
	ModelChatGPT4oLatest Model = "openai/chatgpt-4o-latest"

	// Previous GPT family still in wide use
	ModelGPT41         Model = "openai/gpt-4.1"
	ModelGPT41Mini     Model = "openai/gpt-4.1-mini"
	ModelGPT41Nano     Model = "openai/gpt-4.1-nano"
	ModelGPT4o         Model = "openai/gpt-4o"
	ModelGPT4o20240513 Model = "openai/gpt-4o-2024-05-13"
	ModelGPT4oMini     Model = "openai/gpt-4o-mini"

	// o-series reasoning
	ModelO1        Model = "openai/o1"
	ModelO1Mini    Model = "openai/o1-mini"
	ModelO1Pro     Model = "openai/o1-pro"
	ModelO1Preview Model = "openai/o1-preview"

	ModelO3                 Model = "openai/o3"
	ModelO3Mini             Model = "openai/o3-mini"
	ModelO3Pro              Model = "openai/o3-pro"
	ModelO3DeepResearch     Model = "openai/o3-deep-research"
	ModelO4Mini             Model = "openai/o4-mini"
	ModelO4MiniDeepResearch Model = "openai/o4-mini-deep-research"

	// Realtime / audio models
	ModelGPTRealtime              Model = "openai/gpt-realtime"
	ModelGPTRealtimeMini          Model = "openai/gpt-realtime-mini"
	ModelGPT4oRealtimePreview     Model = "openai/gpt-4o-realtime-preview"
	ModelGPT4oMiniRealtimePreview Model = "openai/gpt-4o-mini-realtime-preview"
	ModelGPTAudio                 Model = "openai/gpt-audio"
	ModelGPTAudioMini             Model = "openai/gpt-audio-mini"
	ModelGPT4oAudioPreview        Model = "openai/gpt-4o-audio-preview"
	ModelGPT4oMiniAudioPreview    Model = "openai/gpt-4o-mini-audio-preview"

	// Search previews
	ModelGPT4oMiniSearchPreview Model = "openai/gpt-4o-mini-search-preview"
	ModelGPT4oSearchPreview     Model = "openai/gpt-4o-search-preview"

	// Speech / transcription
	ModelGPT4oMiniTTS           Model = "openai/gpt-4o-mini-tts"
	ModelGPT4oTranscribe        Model = "openai/gpt-4o-transcribe"
	ModelGPT4oTranscribeDiarize Model = "openai/gpt-4o-transcribe-diarize"
	ModelGPT4oMiniTranscribe    Model = "openai/gpt-4o-mini-transcribe"

	// Image generation
	ModelGPTImage15         Model = "openai/gpt-image-1.5"
	ModelGPTImage1          Model = "openai/gpt-image-1"
	ModelGPTImage1Mini      Model = "openai/gpt-image-1-mini"
	ModelChatGPTImageLatest Model = "openai/chatgpt-image-latest"

	// Open-weight models
	ModelGPTOSS120B Model = "openai/gpt-oss-120b"
	ModelGPTOSS20B  Model = "openai/gpt-oss-20b"

	// Video generation
	ModelSora2    Model = "openai/sora-2"
	ModelSora2Pro Model = "openai/sora-2-pro"

	// Anthropic
	// Defaults: current recommended Claude lineup (4.5)
	ModelClaudeOpus   Model = "anthropic/claude-opus-4.5"
	ModelClaudeSonnet Model = "anthropic/claude-sonnet-4.5"
	ModelClaudeHaiku  Model = "anthropic/claude-haiku-4.5"

	// Additional Claude models (OpenRouter slugs)
	ModelClaudeOpus41   Model = "anthropic/claude-opus-4.1"
	ModelClaudeOpus4    Model = "anthropic/claude-opus-4"
	ModelClaudeSonnet4  Model = "anthropic/claude-sonnet-4"
	ModelClaudeSonnet37 Model = "anthropic/claude-3.7-sonnet"
	ModelClaudeHaiku35  Model = "anthropic/claude-3.5-haiku"
	ModelClaudeHaiku3   Model = "anthropic/claude-3-haiku"
	ModelClaudeOpus3    Model = "anthropic/claude-3-opus"
	ModelClaudeSonnet3  Model = "anthropic/claude-3-sonnet"

	// Google
	// Gemini 3 (Preview)
	ModelGemini3Pro   Model = "google/gemini-3-pro-preview"   // Flagship reasoning (preview)
	ModelGemini3Flash Model = "google/gemini-3-flash-preview" // Fast + strong (preview)

	// Gemini 2.5 (Stable)
	ModelGemini25Pro       Model = "google/gemini-2.5-pro"
	ModelGemini25Flash     Model = "google/gemini-2.5-flash"
	ModelGemini25FlashLite Model = "google/gemini-2.5-flash-lite"

	// Gemini 2.0 (OpenRouter uses -001 model IDs)
	ModelGemini2Flash     Model = "google/gemini-2.0-flash-001"
	ModelGemini2FlashLite Model = "google/gemini-2.0-flash-lite-001"

	// Backwards-compatible aliases
	// Historically these were used as "Gemini 2 Pro" / "Gemini 2 Flash" shortcuts.
	// Gemini 2.0 Pro is not listed on OpenRouter; map the "2 Pro" slot to Gemini 2.5 Pro.
	ModelGemini2Pro Model = ModelGemini25Pro

	// xAI
	ModelGrok41Fast Model = "x-ai/grok-4.1-fast" // 2M context, speed optimized
	ModelGrok3      Model = "x-ai/grok-3"
	ModelGrok3Mini  Model = "x-ai/grok-3-mini"

	// Alibaba
	ModelQwen3Next Model = "qwen/qwen3-next"
	ModelQwen3     Model = "qwen/qwen-3-235b"

	// Meta
	ModelLlama4 Model = "meta-llama/llama-4-maverick"

	// Mistral
	ModelMistralLarge Model = "mistralai/mistral-large"
)

// ModelInfo contains descriptive metadata about a model.
type ModelInfo struct {
	ID          Model
	Name        string
	Provider    string
	Description string
}

// Models is a registry of known models with their metadata.
// Useful for UI lists or validation.
var Models = map[Model]ModelInfo{
	ModelGPT5:              {ModelGPT5, "GPT-5.2", "OpenAI", "General purpose, long context"},
	ModelGPT5Mini:          {ModelGPT5Mini, "GPT-5 mini", "OpenAI", "Faster, cost-efficient version of GPT-5"},
	ModelGPT5Nano:          {ModelGPT5Nano, "GPT-5 nano", "OpenAI", "Fastest, most cost-efficient version of GPT-5"},
	ModelGPT5Codex:         {ModelGPT5Codex, "GPT-5.1 Codex Max", "OpenAI", "Most intelligent agentic coding model (Codex)"},
	ModelGPT4o:             {ModelGPT4o, "GPT-4o", "OpenAI", "Multimodal flagship"},
	ModelClaudeOpus:        {ModelClaudeOpus, "Claude Opus 4.5", "Anthropic", "Premium model combining maximum intelligence with practical performance"},
	ModelClaudeSonnet:      {ModelClaudeSonnet, "Claude Sonnet 4.5", "Anthropic", "Best balance of intelligence, speed, and cost for most use cases"},
	ModelClaudeHaiku:       {ModelClaudeHaiku, "Claude Haiku 4.5", "Anthropic", "Fastest Claude with near-frontier intelligence"},
	ModelClaudeOpus41:      {ModelClaudeOpus41, "Claude Opus 4.1", "Anthropic", "Legacy (still available): premium Opus snapshot"},
	ModelClaudeOpus4:       {ModelClaudeOpus4, "Claude Opus 4", "Anthropic", "Legacy (still available): Opus 4 snapshot"},
	ModelClaudeSonnet4:     {ModelClaudeSonnet4, "Claude Sonnet 4", "Anthropic", "Legacy (still available): Sonnet 4 snapshot"},
	ModelClaudeSonnet37:    {ModelClaudeSonnet37, "Claude Sonnet 3.7", "Anthropic", "Legacy/deprecated: Sonnet 3.7"},
	ModelClaudeHaiku35:     {ModelClaudeHaiku35, "Claude Haiku 3.5", "Anthropic", "Legacy/deprecated: Haiku 3.5"},
	ModelClaudeHaiku3:      {ModelClaudeHaiku3, "Claude Haiku 3", "Anthropic", "Legacy: Haiku 3"},
	ModelClaudeOpus3:       {ModelClaudeOpus3, "Claude Opus 3", "Anthropic", "Legacy/deprecated: Opus 3"},
	ModelClaudeSonnet3:     {ModelClaudeSonnet3, "Claude Sonnet 3", "Anthropic", "Legacy/retired: Sonnet 3"},
	ModelGemini3Pro:        {ModelGemini3Pro, "Gemini 3 Pro (Preview)", "Google", "Flagship reasoning (preview), 1M context"},
	ModelGemini3Flash:      {ModelGemini3Flash, "Gemini 3 Flash (Preview)", "Google", "Fast + strong reasoning (preview), 1M context"},
	ModelGemini25Pro:       {ModelGemini25Pro, "Gemini 2.5 Pro", "Google", "Advanced reasoning, long context"},
	ModelGemini25Flash:     {ModelGemini25Flash, "Gemini 2.5 Flash", "Google", "Best price-performance workhorse"},
	ModelGemini25FlashLite: {ModelGemini25FlashLite, "Gemini 2.5 Flash-Lite", "Google", "Ultra-fast + cost efficient"},
	ModelGemini2Flash:      {ModelGemini2Flash, "Gemini 2.0 Flash", "Google", "2.0 workhorse (OpenRouter -001)"},
	ModelGemini2FlashLite:  {ModelGemini2FlashLite, "Gemini 2.0 Flash-Lite", "Google", "2.0 lightweight (OpenRouter -001)"},
	ModelGrok41Fast:        {ModelGrok41Fast, "Grok 4.1 Fast", "xAI", "2M context, speed optimized"},
	ModelGrok3:             {ModelGrok3, "Grok 3", "xAI", "Real-time knowledge"},
	ModelQwen3Next:         {ModelQwen3Next, "Qwen3-Next", "Alibaba", "Long context specialist"},
	ModelLlama4:            {ModelLlama4, "Llama 4 Maverick", "Meta", "Open weights leader"},
}

// String returns the provider-qualified model ID string.
func (m Model) String() string {
	return string(m)
}
