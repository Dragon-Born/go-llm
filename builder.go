package ai

import (
	gocontext "context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Builder provides a fluent API for constructing AI requests.
// It allows chaining methods to configure the model, system prompt, messages,
// tools, and other parameters before sending the request.
type Builder struct {
	// model is the AI model to use for the request.
	model Model

	// system is the system prompt (instructions for the AI).
	system string

	// messages is the history of the conversation.
	messages []Message

	// vars holds template variables for prompt substitution.
	vars Vars

	// fileContext contains file contents injected into the context.
	fileContext []string // file contents to inject (renamed from context)

	// debug enables verbose logging for this request.
	debug bool

	// maxRetries specifies the number of retry attempts on failure.
	maxRetries int

	// fallbacks is a list of models to try if the primary model fails.
	fallbacks []Model

	// jsonMode forces the model to output valid JSON.
	jsonMode bool

	// temperature controls randomness (0.0 to 2.0).
	temperature *float64

	// thinking controls the reasoning effort level.
	thinking ThinkingLevel

	// Tool calling (function tools)
	tools        []Tool
	toolHandlers map[string]ToolHandler

	// Built-in tools (Responses API: web_search, file_search, code_interpreter, mcp)
	builtinTools []BuiltinTool

	// Vision
	images []ImageInput

	// Documents (PDF)
	documents []DocumentInput

	// Schema enforcement
	schema any

	// Context for cancellation/timeout
	ctx gocontext.Context

	// Provider client (nil = use default)
	client *Client

	// Smart retry with backoff
	retryConfig *RetryConfig

	// Validation / Guardrails
	validators []Validator
}

// New creates a new Builder instance for the specified model.
// It initializes the builder with empty messages and default settings.
func New(model Model) *Builder {
	return &Builder{
		model:       model,
		messages:    []Message{},
		vars:        Vars{},
		fileContext: []string{},
		maxRetries:  0,
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// System Prompt Methods
// ═══════════════════════════════════════════════════════════════════════════

// System sets the system prompt for the request.
// The system prompt defines the behavior, persona, and constraints of the AI.
func (b *Builder) System(prompt string) *Builder {
	b.system = prompt
	return b
}

// SystemFile loads the system prompt from a file at the given path.
// It reads the file content and sets it as the system prompt.
// If the file cannot be read, it logs an error and leaves the system prompt unchanged.
func (b *Builder) SystemFile(path string) *Builder {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("%s Error loading prompt from %s: %v\n", colorRed("✗"), path, err)
		return b
	}
	b.system = string(data)
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Message Methods
// ═══════════════════════════════════════════════════════════════════════════

// User adds a user message to the conversation history.
// This represents the input from the human user.
func (b *Builder) User(content string) *Builder {
	b.messages = append(b.messages, Message{Role: "user", Content: content})
	return b
}

// Assistant adds an assistant message to the conversation history.
// This is used to provide context from previous turns or to pre-fill the assistant's response.
func (b *Builder) Assistant(content string) *Builder {
	b.messages = append(b.messages, Message{Role: "assistant", Content: content})
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Context Injection - Add files as context
// ═══════════════════════════════════════════════════════════════════════════

// Context adds the content of a file (or files matching a glob pattern) to the context.
// The file content is formatted and appended to the system prompt context area.
func (b *Builder) Context(path string) *Builder {
	// Check for glob pattern
	if strings.Contains(path, "*") {
		matches, err := filepath.Glob(path)
		if err != nil {
			fmt.Printf("%s Error with glob pattern %s: %v\n", colorRed("✗"), path, err)
			return b
		}
		for _, match := range matches {
			b.addFileContext(match)
		}
	} else {
		b.addFileContext(path)
	}
	return b
}

func (b *Builder) addFileContext(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("%s Error loading context from %s: %v\n", colorRed("✗"), path, err)
		return
	}
	b.fileContext = append(b.fileContext, fmt.Sprintf("--- %s ---\n%s", path, string(data)))
}

// ContextString adds a raw string as context with a given name.
// This is useful for adding in-memory data or snippets as context.
func (b *Builder) ContextString(name, content string) *Builder {
	b.fileContext = append(b.fileContext, fmt.Sprintf("--- %s ---\n%s", name, content))
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Template Variables
// ═══════════════════════════════════════════════════════════════════════════

// With adds multiple template variables to replace {{key}} in prompts.
// It merges the provided map with existing variables.
func (b *Builder) With(vars Vars) *Builder {
	for k, v := range vars {
		b.vars[k] = v
	}
	return b
}

// Var adds a single template variable to replace {{key}} with the value.
func (b *Builder) Var(key, value string) *Builder {
	b.vars[key] = value
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry & Fallback
// ═══════════════════════════════════════════════════════════════════════════

// Retry sets the maximum number of retry attempts for failed requests.
// This uses a simple retry mechanism. For advanced backoff, use WithRetryConfig (if available) or check retry.go.
func (b *Builder) Retry(times int) *Builder {
	b.maxRetries = times
	return b
}

// Fallback sets a list of fallback models to try if the primary model fails.
// The builder will attempt each model in order until one succeeds or all fail.
func (b *Builder) Fallback(models ...Model) *Builder {
	b.fallbacks = models
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON Mode
// ═══════════════════════════════════════════════════════════════════════════

// JSON enables JSON mode for the request.
// It instructs the model to return valid JSON output and enforces it if supported by the provider.
func (b *Builder) JSON() *Builder {
	b.jsonMode = true
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Temperature
// ═══════════════════════════════════════════════════════════════════════════

// Temperature sets the sampling temperature (0.0 to 2.0).
// Higher values (e.g., 0.8) make output more random/creative.
// Lower values (e.g., 0.2) make output more focused/deterministic.
func (b *Builder) Temperature(temp float64) *Builder {
	b.temperature = &temp
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Thinking Level (Reasoning Effort)
// ═══════════════════════════════════════════════════════════════════════════

// Thinking sets the reasoning/thinking effort level (minimal, low, medium, high).
// This is supported by reasoning models like Gemini and others.
func (b *Builder) Thinking(level ThinkingLevel) *Builder {
	b.thinking = level
	return b
}

// ThinkMinimal sets thinking to minimal effort.
// Currently optimized for Gemini Flash (fastest reasoning).
func (b *Builder) ThinkMinimal() *Builder { return b.Thinking(ThinkingMinimal) }

// ThinkLow sets thinking to low effort.
func (b *Builder) ThinkLow() *Builder { return b.Thinking(ThinkingLow) }

// ThinkMedium sets thinking to medium effort.
func (b *Builder) ThinkMedium() *Builder { return b.Thinking(ThinkingMedium) }

// ThinkHigh sets thinking to high effort.
func (b *Builder) ThinkHigh() *Builder { return b.Thinking(ThinkingHigh) }

// ═══════════════════════════════════════════════════════════════════════════
// Debug Mode
// ═══════════════════════════════════════════════════════════════════════════

// Debug enables debug output for this specific request.
// It prints the request payload and response details to the console.
func (b *Builder) Debug() *Builder {
	b.debug = true
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution
// ═══════════════════════════════════════════════════════════════════════════

// buildMessages constructs the final message list by processing templates and context.
func (b *Builder) buildMessages() []Message {
	var msgs []Message

	// Build system message
	system := b.system
	if len(b.vars) > 0 {
		system = applyTemplate(system, b.vars)
	}

	// Add JSON instruction if enabled
	if b.jsonMode && system != "" {
		system += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation."
	} else if b.jsonMode {
		system = "Respond with valid JSON only. No markdown, no explanation."
	}

	// Add context to system if present
	if len(b.fileContext) > 0 {
		contextStr := "\n\n# Context\n" + strings.Join(b.fileContext, "\n\n")
		system += contextStr
	}

	if system != "" {
		msgs = append(msgs, Message{Role: "system", Content: system})
	}

	// Add user/assistant messages with template vars applied
	for _, m := range b.messages {
		content := m.Content
		if str, ok := content.(string); ok && len(b.vars) > 0 {
			content = applyTemplate(str, b.vars)
		}
		msgs = append(msgs, Message{Role: m.Role, Content: content, ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID})
	}

	// If we have images or documents, convert the last user message to multimodal
	if (len(b.images) > 0 || len(b.documents) > 0) && len(msgs) > 0 {
		// Find last user message
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == "user" {
				// Convert to multimodal content
				var parts []ContentPart

				// Add text content first
				if text, ok := msgs[i].Content.(string); ok && text != "" {
					parts = append(parts, ContentPart{Type: "text", Text: text})
				}

				// Add images
				for _, img := range b.images {
					parts = append(parts, ContentPart{
						Type: "image_url",
						ImageURL: &ImageURL{
							URL:    img.URL,
							Detail: img.Detail,
						},
					})
				}

				// Add documents (PDFs)
				for _, doc := range b.documents {
					parts = append(parts, ContentPart{
						Type: "document",
						Document: &DocumentRef{
							Data:     doc.Data,
							URL:      doc.URL,
							MimeType: doc.MimeType,
							Name:     doc.Name,
						},
					})
				}

				msgs[i].Content = parts
				break
			}
		}
	}

	return msgs
}

// Send executes the request and returns the response content as a string.
// It handles retries, fallbacks, and error handling as configured.
func (b *Builder) Send() (string, error) {
	meta := b.SendWithMeta()
	return meta.Content, meta.Error
}

// ResponseMeta contains metadata about the AI response.
type ResponseMeta struct {
	// Content is the actual text response from the model.
	Content string

	// Error indicates if the request failed.
	Error error

	// Model is the model that actually fulfilled the request (could be a fallback).
	Model Model

	// Tokens is the total number of tokens used.
	Tokens int

	// PromptTokens is the number of tokens in the prompt.
	PromptTokens int

	// CompletionTokens is the number of tokens in the completion.
	CompletionTokens int

	// Latency is the duration of the request.
	Latency time.Duration

	// Retries is the number of retry attempts made.
	Retries int

	// Responses API output (populated when using built-in tools)
	// Contains citations, sources, and tool call details
	ResponsesOutput *ResponsesOutput
}

// SendWithMeta executes the request and returns the response with full metadata.
// This includes token usage, latency, and the specific model used.
func (b *Builder) SendWithMeta() *ResponseMeta {
	msgs := b.buildMessages()
	start := time.Now()

	// Enable debug for this request if set
	oldDebug := Debug
	if b.debug {
		Debug = true
	}
	defer func() { Debug = oldDebug }()

	// Get the client to use
	client := b.client
	if client == nil {
		client = getDefaultClient()
	}

	// Get context
	ctx := b.getContext()

	// Try primary model with fallbacks
	models := append([]Model{b.model}, b.fallbacks...)
	var lastErr error
	var totalRetries int

	for _, model := range models {
		// Build provider request
		req := &ProviderRequest{
			Model:        string(model),
			Messages:     msgs,
			Temperature:  b.temperature,
			Thinking:     b.thinking,
			Tools:        b.tools,
			BuiltinTools: b.builtinTools,
			JSONMode:     b.jsonMode,
		}

		// Check capability warnings
		if len(b.tools) > 0 {
			checkCapability(client.provider, "tools", client.provider.Capabilities().Tools)
		}
		if b.thinking != "" {
			checkCapability(client.provider, "thinking/reasoning", client.provider.Capabilities().Thinking)
		}
		// Check built-in tool capabilities
		for _, bt := range b.builtinTools {
			switch bt.Type {
			case "web_search":
				checkCapability(client.provider, "web_search", client.provider.Capabilities().WebSearch)
			case "file_search":
				checkCapability(client.provider, "file_search", client.provider.Capabilities().FileSearch)
			case "code_interpreter":
				checkCapability(client.provider, "code_interpreter", client.provider.Capabilities().CodeInterpreter)
			case "mcp":
				checkCapability(client.provider, "mcp", client.provider.Capabilities().MCP)
			case "image_generation":
				checkCapability(client.provider, "image_generation", client.provider.Capabilities().ImageGeneration)
			case "computer_use_preview":
				checkCapability(client.provider, "computer_use", client.provider.Capabilities().ComputerUse)
			case "shell":
				checkCapability(client.provider, "shell", client.provider.Capabilities().Shell)
			case "apply_patch":
				checkCapability(client.provider, "apply_patch", client.provider.Capabilities().ApplyPatch)
			}
		}

		if Debug {
			printDebugRequest(model, msgs)
		}

		// Use smart retry if configured
		var resp *ProviderResponse
		var err error

		if b.retryConfig != nil {
			// Smart retry with exponential backoff + jitter
			var retries int
			resp, err = WithRetry(ctx, b.retryConfig, func() (*ProviderResponse, error) {
				retries++
				if retries > 1 {
					totalRetries++
				}
				invokeBeforeRequest(model, msgs)
				waitForRateLimit()
				r, e := client.provider.Send(ctx, req)
				if e != nil {
					invokeOnError(model, e)
				}
				return r, e
			})
		} else if b.maxRetries > 0 {
			// Legacy retry (simple)
			for attempt := 0; attempt <= b.maxRetries; attempt++ {
				if attempt > 0 {
					totalRetries++
					time.Sleep(time.Duration(attempt*attempt) * 100 * time.Millisecond)
				}
				invokeBeforeRequest(model, msgs)
				waitForRateLimit()
				resp, err = client.provider.Send(ctx, req)
				if err == nil {
					break
				}
				invokeOnError(model, err)
			}
		} else {
			// No retry
			invokeBeforeRequest(model, msgs)
			waitForRateLimit()
			resp, err = client.provider.Send(ctx, req)
			if err != nil {
				invokeOnError(model, err)
			}
		}

		if err == nil {
			// Validate response if validators configured (and apply any content filters)
			content := resp.Content
			if len(b.validators) > 0 {
				validated, validationErr := b.runValidators(content)
				if validationErr != nil {
					invokeOnError(model, validationErr)
					return &ResponseMeta{
						Error:   validationErr,
						Model:   model,
						Latency: time.Since(start),
						Retries: totalRetries,
					}
				}
				content = validated
			}

			meta := &ResponseMeta{
				Content:          content,
				Model:            model,
				Latency:          time.Since(start),
				Retries:          totalRetries,
				Tokens:           resp.TotalTokens,
				PromptTokens:     resp.PromptTokens,
				CompletionTokens: resp.CompletionTokens,
				ResponsesOutput:  resp.ResponsesOutput,
			}

			if Pretty {
				printPrettyResponse(model, content)
			}

			// Track stats
			trackRequest(meta)
			invokeOnTokens(model, meta.PromptTokens, meta.CompletionTokens)
			invokeAfterResponse(model, meta.Content, meta.Latency)

			return meta
		}
		lastErr = err
	}

	return &ResponseMeta{Error: lastErr, Model: b.model, Latency: time.Since(start), Retries: totalRetries}
}

// Ask is a convenience alias for User(prompt).Send().
// It quickly sends a user message and returns the response.
func (b *Builder) Ask(prompt string) (string, error) {
	return b.User(prompt).Send()
}

// AskJSON sends a request and attempts to unmarshal the JSON response into the target struct.
// It automatically enables JSON mode and strips code blocks from the response.
func (b *Builder) AskJSON(prompt string, target any) error {
	resp, err := b.JSON().User(prompt).Send()
	if err != nil {
		return err
	}

	// Clean response (remove markdown code blocks if present)
	resp = strings.TrimPrefix(resp, "```json")
	resp = strings.TrimPrefix(resp, "```")
	resp = strings.TrimSuffix(resp, "```")
	resp = strings.TrimSpace(resp)

	return json.Unmarshal([]byte(resp), target)
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversation Mode
// ═══════════════════════════════════════════════════════════════════════════

// Chat initiates a conversation mode using this builder's configuration.
// It returns a Conversation struct that manages message history.
func (b *Builder) Chat() *Conversation {
	return &Conversation{
		builder: b,
		history: []Message{},
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Switching
// ═══════════════════════════════════════════════════════════════════════════

// Model changes the model for this builder.
func (b *Builder) Model(model Model) *Builder {
	b.model = model
	return b
}

// UseModel changes the model using a string ID (useful for models not defined in constants).
func (b *Builder) UseModel(modelID string) *Builder {
	b.model = Model(modelID)
	return b
}

// GetModel returns the currently configured model.
func (b *Builder) GetModel() Model {
	return b.model
}

// GetSystem returns the current system prompt.
func (b *Builder) GetSystem() string {
	return b.system
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider Switching
// ═══════════════════════════════════════════════════════════════════════════

// WithClient sets a specific client for this builder.
// This overrides the default client logic.
func (b *Builder) WithClient(client *Client) *Builder {
	b.client = client
	return b
}

// Provider switches to a different provider for this builder.
// It creates a new client for the specified provider type.
// Usage: ai.Claude().Provider(ai.ProviderAnthropic).Ask("...")
func (b *Builder) Provider(providerType ProviderType) *Builder {
	b.client = NewClient(providerType)
	return b
}

// GetClient returns the current client associated with the builder.
// It returns nil if the default client is being used.
func (b *Builder) GetClient() *Client {
	return b.client
}

// Clone creates a deep copy of the builder.
// This is useful when you want to branch off a configuration without affecting the original.
func (b *Builder) Clone() *Builder {
	var tempCopy *float64
	if b.temperature != nil {
		v := *b.temperature
		tempCopy = &v
	}
	newB := &Builder{
		model:        b.model,
		system:       b.system,
		messages:     make([]Message, len(b.messages)),
		vars:         make(Vars),
		fileContext:  make([]string, len(b.fileContext)),
		debug:        b.debug,
		maxRetries:   b.maxRetries,
		fallbacks:    make([]Model, len(b.fallbacks)),
		jsonMode:     b.jsonMode,
		temperature:  tempCopy,
		thinking:     b.thinking,
		tools:        make([]Tool, len(b.tools)),
		builtinTools: make([]BuiltinTool, len(b.builtinTools)),
		images:       make([]ImageInput, len(b.images)),
		documents:    make([]DocumentInput, len(b.documents)),
		client:       b.client,
		ctx:          b.ctx,
		retryConfig:  b.retryConfig,
		validators:   make([]Validator, len(b.validators)),
	}
	copy(newB.messages, b.messages)
	copy(newB.fileContext, b.fileContext)
	copy(newB.fallbacks, b.fallbacks)
	copy(newB.tools, b.tools)
	copy(newB.builtinTools, b.builtinTools)
	copy(newB.images, b.images)
	copy(newB.documents, b.documents)
	copy(newB.validators, b.validators)
	maps.Copy(newB.vars, b.vars)
	if b.toolHandlers != nil {
		newB.toolHandlers = make(map[string]ToolHandler)
		for k, v := range b.toolHandlers {
			newB.toolHandlers[k] = v
		}
	}
	return newB
}
