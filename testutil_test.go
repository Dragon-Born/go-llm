package ai

import (
	"context"
	"sync"
	"testing"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

func withTestGlobals(t *testing.T) func() {
	t.Helper()

	oldPretty := Pretty
	oldDebug := Debug
	oldCache := Cache
	oldDefaultProvider := DefaultProvider

	// Keep tests quiet and deterministic.
	Pretty = false
	Debug = false
	Cache = false

	// Avoid cross-test leakage from cached clients.
	ResetClients()

	return func() {
		Pretty = oldPretty
		Debug = oldDebug
		Cache = oldCache
		SetDefaultProvider(oldDefaultProvider)
		ResetClients()
	}
}

func setDefaultClientForTest(t *testing.T, provider Provider, providerType ProviderType) {
	t.Helper()
	SetDefaultClient(&Client{
		provider:     provider,
		providerType: providerType,
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// Stub providers (no network)
// ─────────────────────────────────────────────────────────────────────────────

type stubProvider struct {
	name string
	caps ProviderCapabilities

	mu        sync.Mutex
	sendFn    func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error)
	streamFn  func(ctx context.Context, req *ProviderRequest, cb StreamCallback) (*ProviderResponse, error)
	sendCalls int
	reqs      []*ProviderRequest
}

func (p *stubProvider) Name() string { return p.name }
func (p *stubProvider) Capabilities() ProviderCapabilities {
	return p.caps
}

func (p *stubProvider) Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	p.mu.Lock()
	p.sendCalls++
	p.reqs = append(p.reqs, req)
	fn := p.sendFn
	p.mu.Unlock()
	if fn == nil {
		return &ProviderResponse{Content: ""}, nil
	}
	return fn(ctx, req)
}

func (p *stubProvider) SendStream(ctx context.Context, req *ProviderRequest, cb StreamCallback) (*ProviderResponse, error) {
	p.mu.Lock()
	fn := p.streamFn
	p.mu.Unlock()
	if fn == nil {
		// Default: emulate streaming by returning full Send content once.
		resp, err := p.Send(ctx, req)
		if err != nil {
			return nil, err
		}
		cb(resp.Content)
		return resp, nil
	}
	return fn(ctx, req, cb)
}

func (p *stubProvider) Calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sendCalls
}

func (p *stubProvider) Requests() []*ProviderRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*ProviderRequest, len(p.reqs))
	copy(out, p.reqs)
	return out
}

type stubEmbedderProvider struct {
	*stubProvider

	mu       sync.Mutex
	embedFn  func(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
	embedReq []*EmbeddingRequest
}

func (p *stubEmbedderProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	p.mu.Lock()
	p.embedReq = append(p.embedReq, req)
	fn := p.embedFn
	p.mu.Unlock()
	if fn == nil {
		return &EmbeddingResponse{}, nil
	}
	return fn(ctx, req)
}

func (p *stubEmbedderProvider) EmbedRequests() []*EmbeddingRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*EmbeddingRequest, len(p.embedReq))
	copy(out, p.embedReq)
	return out
}

type stubAudioProvider struct {
	*stubProvider

	mu      sync.Mutex
	ttsFn   func(ctx context.Context, req *TTSRequest) (*TTSResponse, error)
	sttFn   func(ctx context.Context, req *STTRequest) (*STTResponse, error)
	ttsReqs []*TTSRequest
	sttReqs []*STTRequest
}

func (p *stubAudioProvider) TextToSpeech(ctx context.Context, req *TTSRequest) (*TTSResponse, error) {
	p.mu.Lock()
	p.ttsReqs = append(p.ttsReqs, req)
	fn := p.ttsFn
	p.mu.Unlock()
	if fn == nil {
		return &TTSResponse{}, nil
	}
	return fn(ctx, req)
}

func (p *stubAudioProvider) SpeechToText(ctx context.Context, req *STTRequest) (*STTResponse, error) {
	p.mu.Lock()
	p.sttReqs = append(p.sttReqs, req)
	fn := p.sttFn
	p.mu.Unlock()
	if fn == nil {
		return &STTResponse{}, nil
	}
	return fn(ctx, req)
}

func (p *stubAudioProvider) TTSRequests() []*TTSRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*TTSRequest, len(p.ttsReqs))
	copy(out, p.ttsReqs)
	return out
}

func (p *stubAudioProvider) STTRequests() []*STTRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*STTRequest, len(p.sttReqs))
	copy(out, p.sttReqs)
	return out
}

// Helps keep retry tests fast (no sleep).
func noSleepRetryConfig(maxRetries int) *RetryConfig {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = maxRetries
	cfg.InitialDelay = 0
	cfg.MaxDelay = 0
	cfg.Jitter = 0
	return cfg
}

// Tiny helper for timeouts without flakiness.
func mustWithin(t *testing.T, d time.Duration, fn func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		fn()
	}()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("timed out after %v", d)
	}
}

