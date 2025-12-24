package ai

import (
	"testing"
)

func TestCompareBuilder(t *testing.T) {
	cb := Compare("Test prompt")

	if cb.prompt != "Test prompt" {
		t.Errorf("expected prompt 'Test prompt', got %q", cb.prompt)
	}
	if cb.system != "" {
		t.Error("system should be empty initially")
	}
}

func TestCompareBuilderSystem(t *testing.T) {
	cb := Compare("Test").System("You are helpful")

	if cb.system != "You are helpful" {
		t.Errorf("expected system 'You are helpful', got %q", cb.system)
	}
}

func TestCompareBuilderWith(t *testing.T) {
	cb := Compare("Test").With(Vars{"key": "value"})

	if cb.vars["key"] != "value" {
		t.Error("vars should be set")
	}
}

func TestCompareBuilderChaining(t *testing.T) {
	cb := Compare("Test").
		System("System prompt").
		With(Vars{"k1": "v1"}).
		With(Vars{"k2": "v2"})

	if cb.prompt != "Test" {
		t.Error("prompt should be preserved")
	}
	if cb.system != "System prompt" {
		t.Error("system should be set")
	}
	if len(cb.vars) != 2 {
		t.Errorf("expected 2 vars, got %d", len(cb.vars))
	}
}

func TestCompareResultStruct(t *testing.T) {
	result := CompareResult{
		Model:    ModelGPT5,
		Response: "Test response",
		Tokens:   100,
	}

	if result.Model != ModelGPT5 {
		t.Error("model mismatch")
	}
	if result.Response != "Test response" {
		t.Error("response mismatch")
	}
	if result.Tokens != 100 {
		t.Error("tokens mismatch")
	}
}
