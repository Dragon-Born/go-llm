package ai

import (
	"context"
	"fmt"
	"time"
)

// StreamCallback is a function called for each chunk of a streamed response.
type StreamCallback func(chunk string)

// Stream sends a request and prints the response chunks to stdout in real-time.
// It is a convenience method for simple streaming to the console.
func (b *Builder) Stream(prompt string) (string, error) {
	return b.User(prompt).StreamResponse(func(chunk string) {
		fmt.Print(chunk)
	})
}

// StreamResponse sends a request and calls the provided callback for each chunk of the response.
// It handles rate limiting, error checking, and optional debug output.
// Returns the full concatenated response string upon completion.
func (b *Builder) StreamResponse(callback StreamCallback) (string, error) {
	msgs := b.buildMessages()
	start := time.Now()

	// Get the client to use
	client := b.client
	if client == nil {
		client = getDefaultClient()
	}

	// Build provider request
	req := &ProviderRequest{
		Model:       string(b.model),
		Messages:    msgs,
		Temperature: b.temperature,
		Thinking:    b.thinking,
		Tools:       b.tools,
		JSONMode:    b.jsonMode,
		Stream:      true,
	}

	// Get context
	ctx := b.ctx
	if ctx == nil {
		ctx = context.Background()
	}

	if Debug {
		printDebugRequest(b.model, msgs)
	}

	// Check streaming capability
	if !client.provider.Capabilities().Streaming {
		if Debug {
			fmt.Printf("%s Warning: %s does not support streaming, falling back to regular request\n",
				colorYellow("⚠"), client.provider.Name())
		}
		// Fallback to non-streaming
		waitForRateLimit()
		resp, err := client.provider.Send(ctx, req)
		if err != nil {
			return "", err
		}
		callback(resp.Content)
		return resp.Content, nil
	}

	if Pretty {
		fmt.Printf("\n%s %s\n", colorCyan("▸"), colorDim(string(b.model)))
		fmt.Println(colorDim("─────────────────────────────────────────────────────────────"))
	}

	waitForRateLimit()
	resp, err := client.provider.SendStream(ctx, req, callback)
	if err != nil {
		return "", err
	}

	if Pretty {
		fmt.Println()
		fmt.Println()
	}

	// Track stats
	trackRequest(&ResponseMeta{
		Content:          resp.Content,
		Model:            b.model,
		Latency:          time.Since(start),
		Tokens:           resp.TotalTokens,
		PromptTokens:     resp.PromptTokens,
		CompletionTokens: resp.CompletionTokens,
	})

	return resp.Content, nil
}

// StreamWithMeta sends a request, streams the response via callback, and returns full metadata.
// This is useful when you need token usage stats or latency information along with the streamed content.
func (b *Builder) StreamWithMeta(callback StreamCallback) (*ResponseMeta, error) {
	msgs := b.buildMessages()
	start := time.Now()

	client := b.client
	if client == nil {
		client = getDefaultClient()
	}

	req := &ProviderRequest{
		Model:       string(b.model),
		Messages:    msgs,
		Temperature: b.temperature,
		Thinking:    b.thinking,
		Tools:       b.tools,
		JSONMode:    b.jsonMode,
		Stream:      true,
	}

	ctx := b.ctx
	if ctx == nil {
		ctx = context.Background()
	}

	if Debug {
		printDebugRequest(b.model, msgs)
	}

	waitForRateLimit()
	resp, err := client.provider.SendStream(ctx, req, callback)
	if err != nil {
		return &ResponseMeta{Error: err, Model: b.model, Latency: time.Since(start)}, err
	}

	meta := &ResponseMeta{
		Content:          resp.Content,
		Model:            b.model,
		Latency:          time.Since(start),
		Tokens:           resp.TotalTokens,
		PromptTokens:     resp.PromptTokens,
		CompletionTokens: resp.CompletionTokens,
	}

	trackRequest(meta)
	return meta, nil
}
