package ai

import (
	"fmt"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Hooks / Middleware / Callbacks
// ═══════════════════════════════════════════════════════════════════════════

// Hook types for different events
type (
	BeforeRequestHook func(model Model, messages []Message)
	AfterResponseHook func(model Model, content string, duration time.Duration)
	OnErrorHook       func(model Model, err error)
	OnTokensHook      func(model Model, prompt, completion int)
)

// Global hooks (can be set by users for observability)
var (
	hooksLock sync.RWMutex

	beforeRequestHooks []BeforeRequestHook
	afterResponseHooks []AfterResponseHook
	onErrorHooks       []OnErrorHook
	onTokensHooks      []OnTokensHook
)

// ═══════════════════════════════════════════════════════════════════════════
// Hook Registration
// ═══════════════════════════════════════════════════════════════════════════

// OnBeforeRequest registers a hook called before each request
func OnBeforeRequest(hook BeforeRequestHook) {
	hooksLock.Lock()
	defer hooksLock.Unlock()
	beforeRequestHooks = append(beforeRequestHooks, hook)
}

// OnAfterResponse registers a hook called after each successful response
func OnAfterResponse(hook AfterResponseHook) {
	hooksLock.Lock()
	defer hooksLock.Unlock()
	afterResponseHooks = append(afterResponseHooks, hook)
}

// OnError registers a hook called on errors
func OnError(hook OnErrorHook) {
	hooksLock.Lock()
	defer hooksLock.Unlock()
	onErrorHooks = append(onErrorHooks, hook)
}

// OnTokens registers a hook called with token counts
func OnTokens(hook OnTokensHook) {
	hooksLock.Lock()
	defer hooksLock.Unlock()
	onTokensHooks = append(onTokensHooks, hook)
}

// ClearHooks removes all registered hooks
func ClearHooks() {
	hooksLock.Lock()
	defer hooksLock.Unlock()
	beforeRequestHooks = nil
	afterResponseHooks = nil
	onErrorHooks = nil
	onTokensHooks = nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook Invocation (called internally)
// ═══════════════════════════════════════════════════════════════════════════

func invokeBeforeRequest(model Model, messages []Message) {
	hooksLock.RLock()
	hooks := beforeRequestHooks
	hooksLock.RUnlock()

	for _, hook := range hooks {
		hook(model, messages)
	}
}

func invokeAfterResponse(model Model, content string, duration time.Duration) {
	hooksLock.RLock()
	hooks := afterResponseHooks
	hooksLock.RUnlock()

	for _, hook := range hooks {
		hook(model, content, duration)
	}
}

func invokeOnError(model Model, err error) {
	hooksLock.RLock()
	hooks := onErrorHooks
	hooksLock.RUnlock()

	for _, hook := range hooks {
		hook(model, err)
	}
}

func invokeOnTokens(model Model, prompt, completion int) {
	hooksLock.RLock()
	hooks := onTokensHooks
	hooksLock.RUnlock()

	for _, hook := range hooks {
		hook(model, prompt, completion)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Common Hook Examples (optional utilities)
// ═══════════════════════════════════════════════════════════════════════════

// LoggingHooks returns common logging hooks
func LoggingHooks() (BeforeRequestHook, AfterResponseHook, OnErrorHook) {
	before := func(model Model, messages []Message) {
		printDebugRequest(model, messages)
	}

	after := func(model Model, content string, duration time.Duration) {
		fmt.Printf("%s Response from %s in %v\n", colorGreen("✓"), model, duration)
	}

	onErr := func(model Model, err error) {
		fmt.Printf("%s Error from %s: %v\n", colorRed("✗"), model, err)
	}

	return before, after, onErr
}

// MetricsCollector is an example hook for collecting metrics
type MetricsCollector struct {
	mu            sync.Mutex
	RequestCount  int
	TotalTokens   int
	TotalDuration time.Duration
	ErrorCount    int
	ModelCounts   map[Model]int
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		ModelCounts: make(map[Model]int),
	}
}

// Hook returns the after-response hook for this collector
func (m *MetricsCollector) Hook() AfterResponseHook {
	return func(model Model, content string, duration time.Duration) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.RequestCount++
		m.TotalDuration += duration
		m.ModelCounts[model]++
	}
}

// TokenHook returns the tokens hook for this collector
func (m *MetricsCollector) TokenHook() OnTokensHook {
	return func(model Model, prompt, completion int) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.TotalTokens += prompt + completion
	}
}

// ErrorHook returns the error hook for this collector
func (m *MetricsCollector) ErrorHook() OnErrorHook {
	return func(model Model, err error) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.ErrorCount++
	}
}

// Register registers all hooks for this collector
func (m *MetricsCollector) Register() {
	OnAfterResponse(m.Hook())
	OnTokens(m.TokenHook())
	OnError(m.ErrorHook())
}
