package ai

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Batch Processing
// ═══════════════════════════════════════════════════════════════════════════

// BatchResult holds a single result from batch processing
type BatchResult struct {
	Index   int           // original index in batch
	Content string        // response content
	Error   error         // error if any
	Model   Model         // model used
	Tokens  int           // tokens used
	Latency time.Duration // request latency
}

// BatchConfig configures batch processing
type BatchConfig struct {
	MaxConcurrency int           // max parallel requests (default: 5)
	Timeout        time.Duration // per-request timeout (0 = no timeout)
	StopOnError    bool          // stop all on first error
	RetryConfig    *RetryConfig  // retry config for failed requests
}

// DefaultBatchConfig returns sensible defaults
func DefaultBatchConfig() *BatchConfig {
	return &BatchConfig{
		MaxConcurrency: 5,
		Timeout:        0,
		StopOnError:    false,
		RetryConfig:    nil,
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Builder - Fluent API
// ═══════════════════════════════════════════════════════════════════════════

// BatchBuilder provides a fluent API for batch processing
type BatchBuilder struct {
	builders []*Builder
	config   *BatchConfig
	ctx      context.Context
}

// Batch creates a new batch builder
func Batch(builders ...*Builder) *BatchBuilder {
	return &BatchBuilder{
		builders: builders,
		config:   DefaultBatchConfig(),
		ctx:      context.Background(),
	}
}

// Add adds more builders to the batch
func (b *BatchBuilder) Add(builders ...*Builder) *BatchBuilder {
	b.builders = append(b.builders, builders...)
	return b
}

// Concurrency sets max parallel requests
func (b *BatchBuilder) Concurrency(n int) *BatchBuilder {
	if n < 1 {
		n = 1
	}
	b.config.MaxConcurrency = n
	return b
}

// Timeout sets per-request timeout
func (b *BatchBuilder) Timeout(d time.Duration) *BatchBuilder {
	b.config.Timeout = d
	return b
}

// StopOnError stops all processing on first error
func (b *BatchBuilder) StopOnError() *BatchBuilder {
	b.config.StopOnError = true
	return b
}

// WithRetry sets retry config for failed requests
func (b *BatchBuilder) WithRetry(config *RetryConfig) *BatchBuilder {
	b.config.RetryConfig = config
	return b
}

// WithContext sets the context for all requests
func (b *BatchBuilder) WithContext(ctx context.Context) *BatchBuilder {
	b.ctx = ctx
	return b
}

// Config sets a custom batch config
func (b *BatchBuilder) Config(config *BatchConfig) *BatchBuilder {
	b.config = config
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution
// ═══════════════════════════════════════════════════════════════════════════

// Do executes all requests and returns results
func (b *BatchBuilder) Do() []BatchResult {
	if len(b.builders) == 0 {
		return nil
	}

	results := make([]BatchResult, len(b.builders))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var stopped bool

	// Create semaphore for concurrency control
	sem := make(chan struct{}, b.config.MaxConcurrency)

	// Create cancelable context if StopOnError is set
	ctx := b.ctx
	var cancel context.CancelFunc
	if b.config.StopOnError {
		ctx, cancel = context.WithCancel(b.ctx)
		defer cancel()
	}

	if Debug {
		fmt.Printf("%s Batch: processing %d requests (concurrency=%d)\n",
			colorCyan("→"), len(b.builders), b.config.MaxConcurrency)
	}

	start := time.Now()

	for i, builder := range b.builders {
		wg.Add(1)
		go func(idx int, bldr *Builder) {
			defer wg.Done()

			// Check if stopped
			mu.Lock()
			if stopped {
				mu.Unlock()
				return
			}
			mu.Unlock()

			// Acquire semaphore
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				results[idx] = BatchResult{
					Index: idx,
					Error: ctx.Err(),
					Model: bldr.model,
				}
				return
			}

			// Apply timeout if configured
			reqCtx := ctx
			if b.config.Timeout > 0 {
				var reqCancel context.CancelFunc
				reqCtx, reqCancel = context.WithTimeout(ctx, b.config.Timeout)
				defer reqCancel()
			}

			// Apply retry config if set
			if b.config.RetryConfig != nil && bldr.retryConfig == nil {
				bldr = bldr.Clone()
				bldr.retryConfig = b.config.RetryConfig
			}

			// Execute request
			reqStart := time.Now()
			bldr.ctx = reqCtx
			meta := bldr.SendWithMeta()

			results[idx] = BatchResult{
				Index:   idx,
				Content: meta.Content,
				Error:   meta.Error,
				Model:   meta.Model,
				Tokens:  meta.Tokens,
				Latency: time.Since(reqStart),
			}

			// Stop on error if configured
			if meta.Error != nil && b.config.StopOnError {
				mu.Lock()
				stopped = true
				mu.Unlock()
				if cancel != nil {
					cancel()
				}
			}
		}(i, builder)
	}

	wg.Wait()

	if Debug {
		successCount := 0
		for _, r := range results {
			if r.Error == nil {
				successCount++
			}
		}
		fmt.Printf("%s Batch complete: %d/%d succeeded in %v\n",
			colorGreen("✓"), successCount, len(results), time.Since(start).Round(time.Millisecond))
	}

	return results
}

// DoStrings executes and returns just the content strings
func (b *BatchBuilder) DoStrings() ([]string, error) {
	results := b.Do()

	var firstErr error
	strings := make([]string, len(results))
	for i, r := range results {
		strings[i] = r.Content
		if r.Error != nil && firstErr == nil {
			firstErr = r.Error
		}
	}

	return strings, firstErr
}

// ═══════════════════════════════════════════════════════════════════════════
// Quick Batch Helpers
// ═══════════════════════════════════════════════════════════════════════════

// BatchPrompts creates a batch from multiple prompts using the same model
func BatchPrompts(model Model, prompts ...string) *BatchBuilder {
	builders := make([]*Builder, len(prompts))
	for i, prompt := range prompts {
		builders[i] = New(model).User(prompt)
	}
	return Batch(builders...)
}

// BatchPromptsWithSystem creates a batch with shared system prompt
func BatchPromptsWithSystem(model Model, system string, prompts ...string) *BatchBuilder {
	builders := make([]*Builder, len(prompts))
	for i, prompt := range prompts {
		builders[i] = New(model).System(system).User(prompt)
	}
	return Batch(builders...)
}

// BatchModels sends the same prompt to multiple models
func BatchModels(prompt string, models ...Model) *BatchBuilder {
	builders := make([]*Builder, len(models))
	for i, model := range models {
		builders[i] = New(model).User(prompt)
	}
	return Batch(builders...)
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Result Helpers
// ═══════════════════════════════════════════════════════════════════════════

// BatchResults wraps []BatchResult with helper methods
type BatchResults []BatchResult

// Successful returns only successful results
func (r BatchResults) Successful() BatchResults {
	var results BatchResults
	for _, res := range r {
		if res.Error == nil {
			results = append(results, res)
		}
	}
	return results
}

// Failed returns only failed results
func (r BatchResults) Failed() BatchResults {
	var results BatchResults
	for _, res := range r {
		if res.Error != nil {
			results = append(results, res)
		}
	}
	return results
}

// Errors returns all errors
func (r BatchResults) Errors() []error {
	var errors []error
	for _, res := range r {
		if res.Error != nil {
			errors = append(errors, res.Error)
		}
	}
	return errors
}

// Contents returns all content strings (empty for errors)
func (r BatchResults) Contents() []string {
	contents := make([]string, len(r))
	for i, res := range r {
		contents[i] = res.Content
	}
	return contents
}

// TotalTokens returns sum of all tokens used
func (r BatchResults) TotalTokens() int {
	var total int
	for _, res := range r {
		total += res.Tokens
	}
	return total
}

// TotalLatency returns sum of all latencies
func (r BatchResults) TotalLatency() time.Duration {
	var total time.Duration
	for _, res := range r {
		total += res.Latency
	}
	return total
}

// SuccessRate returns the success rate (0-1)
func (r BatchResults) SuccessRate() float64 {
	if len(r) == 0 {
		return 0
	}
	return float64(len(r.Successful())) / float64(len(r))
}

// ═══════════════════════════════════════════════════════════════════════════
// Fan-Out / Fan-In Patterns
// ═══════════════════════════════════════════════════════════════════════════

// FanOut sends a prompt to multiple models and returns the first success
func FanOut(prompt string, models ...Model) (string, Model, error) {
	if len(models) == 0 {
		return "", "", fmt.Errorf("no models provided")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	type result struct {
		content string
		model   Model
		err     error
	}

	resultChan := make(chan result, len(models))

	for _, model := range models {
		go func(m Model) {
			builder := New(m).User(prompt).WithContext(ctx)
			meta := builder.SendWithMeta()
			select {
			case resultChan <- result{meta.Content, m, meta.Error}:
			case <-ctx.Done():
			}
		}(model)
	}

	// Wait for first success or all failures
	var errors []error
	for i := 0; i < len(models); i++ {
		res := <-resultChan
		if res.err == nil {
			cancel() // Cancel remaining requests
			return res.content, res.model, nil
		}
		errors = append(errors, res.err)
	}

	return "", "", fmt.Errorf("all %d models failed: %v", len(models), errors)
}

// Race is an alias for FanOut - sends prompt to multiple models, returns first success
func Race(prompt string, models ...Model) (string, Model, error) {
	return FanOut(prompt, models...)
}
