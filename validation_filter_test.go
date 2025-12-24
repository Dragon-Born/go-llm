package ai

import (
	"context"
	"strings"
	"testing"
)

func TestBuilder_Validators_BlockBadContent(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{Content: "this is too long"}, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI}).MaxLength(5)
	meta := b.User("hi").SendWithMeta()
	if meta.Error == nil {
		t.Fatalf("expected validation error")
	}
	if _, ok := meta.Error.(*ValidationError); !ok {
		t.Fatalf("expected *ValidationError, got %T (%v)", meta.Error, meta.Error)
	}
}

func TestBuilder_WithFilter_TransformsContent(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{Content: "  hello  "}, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI}).
		WithFilter(TrimFilter())

	meta := b.User("hi").SendWithMeta()
	if meta.Error != nil {
		t.Fatalf("unexpected error: %v", meta.Error)
	}
	if meta.Content != "hello" {
		t.Fatalf("expected trimmed content, got %q", meta.Content)
	}
}

func TestBuilder_StrictJSON_ValidatesJSONEvenWithFences(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{JSON: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{Content: "```json\n{\"a\":1}\n```"}, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI}).StrictJSON()
	meta := b.User("hi").SendWithMeta()
	if meta.Error != nil {
		t.Fatalf("unexpected error: %v", meta.Error)
	}
	if !strings.Contains(meta.Content, `"a"`) {
		t.Fatalf("unexpected content: %q", meta.Content)
	}
}

