package ai

import (
	"strings"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	b := New(ModelGPT5)

	if b.model != ModelGPT5 {
		t.Errorf("expected model %s, got %s", ModelGPT5, b.model)
	}
	if b.system != "" {
		t.Error("expected empty system prompt")
	}
	if len(b.messages) != 0 {
		t.Error("expected empty messages")
	}
	if len(b.vars) != 0 {
		t.Error("expected empty vars")
	}
	if b.debug != false {
		t.Error("expected debug to be false")
	}
	if b.maxRetries != 0 {
		t.Error("expected maxRetries to be 0")
	}
}

func TestBuilderSystem(t *testing.T) {
	b := New(ModelGPT5).System("You are helpful")

	if b.system != "You are helpful" {
		t.Errorf("expected 'You are helpful', got %q", b.system)
	}
}

func TestBuilderSystemChaining(t *testing.T) {
	b := New(ModelGPT5).
		System("First").
		System("Second")

	if b.system != "Second" {
		t.Errorf("expected 'Second', got %q", b.system)
	}
}

func TestBuilderUser(t *testing.T) {
	b := New(ModelGPT5).User("Hello")

	if len(b.messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(b.messages))
	}
	if b.messages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", b.messages[0].Role)
	}
	if b.messages[0].Content != "Hello" {
		t.Errorf("expected content 'Hello', got %q", b.messages[0].Content)
	}
}

func TestBuilderAssistant(t *testing.T) {
	b := New(ModelGPT5).Assistant("Hi there")

	if len(b.messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(b.messages))
	}
	if b.messages[0].Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", b.messages[0].Role)
	}
}

func TestBuilderMessageChaining(t *testing.T) {
	b := New(ModelGPT5).
		User("Hello").
		Assistant("Hi!").
		User("How are you?")

	if len(b.messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(b.messages))
	}

	expected := []struct {
		role    string
		content string
	}{
		{"user", "Hello"},
		{"assistant", "Hi!"},
		{"user", "How are you?"},
	}

	for i, exp := range expected {
		if b.messages[i].Role != exp.role {
			t.Errorf("message %d: expected role %q, got %q", i, exp.role, b.messages[i].Role)
		}
		if b.messages[i].Content != exp.content {
			t.Errorf("message %d: expected content %q, got %q", i, exp.content, b.messages[i].Content)
		}
	}
}

func TestBuilderWith(t *testing.T) {
	b := New(ModelGPT5).With(Vars{
		"key1": "value1",
		"key2": "value2",
	})

	if len(b.vars) != 2 {
		t.Fatalf("expected 2 vars, got %d", len(b.vars))
	}
	if b.vars["key1"] != "value1" {
		t.Errorf("expected key1=value1, got %q", b.vars["key1"])
	}
}

func TestBuilderVar(t *testing.T) {
	b := New(ModelGPT5).
		Var("key1", "value1").
		Var("key2", "value2")

	if len(b.vars) != 2 {
		t.Fatalf("expected 2 vars, got %d", len(b.vars))
	}
}

func TestBuilderVarOverwrite(t *testing.T) {
	b := New(ModelGPT5).
		Var("key", "first").
		Var("key", "second")

	if b.vars["key"] != "second" {
		t.Errorf("expected 'second', got %q", b.vars["key"])
	}
}

func TestBuilderContext(t *testing.T) {
	b := New(ModelGPT5).ContextString("test", "content here")

	if len(b.fileContext) != 1 {
		t.Fatalf("expected 1 context, got %d", len(b.fileContext))
	}
	if !strings.Contains(b.fileContext[0], "content here") {
		t.Error("context should contain the content")
	}
	if !strings.Contains(b.fileContext[0], "test") {
		t.Error("context should contain the name")
	}
}

func TestBuilderContextMultiple(t *testing.T) {
	b := New(ModelGPT5).
		ContextString("file1", "content1").
		ContextString("file2", "content2")

	if len(b.fileContext) != 2 {
		t.Fatalf("expected 2 contexts, got %d", len(b.fileContext))
	}
}

func TestBuilderRetry(t *testing.T) {
	b := New(ModelGPT5).Retry(3)

	if b.maxRetries != 3 {
		t.Errorf("expected maxRetries=3, got %d", b.maxRetries)
	}
}

func TestBuilderFallback(t *testing.T) {
	b := New(ModelGPT5).Fallback(ModelClaudeOpus, ModelGemini3Flash)

	if len(b.fallbacks) != 2 {
		t.Fatalf("expected 2 fallbacks, got %d", len(b.fallbacks))
	}
	if b.fallbacks[0] != ModelClaudeOpus {
		t.Errorf("expected first fallback to be Claude")
	}
	if b.fallbacks[1] != ModelGemini3Flash {
		t.Errorf("expected second fallback to be Gemini")
	}
}

func TestBuilderJSON(t *testing.T) {
	b := New(ModelGPT5).JSON()

	if !b.jsonMode {
		t.Error("expected jsonMode to be true")
	}
}

func TestBuilderDebug(t *testing.T) {
	b := New(ModelGPT5).Debug()

	if !b.debug {
		t.Error("expected debug to be true")
	}
}

func TestBuilderModel(t *testing.T) {
	b := New(ModelGPT5).Model(ModelClaudeOpus)

	if b.model != ModelClaudeOpus {
		t.Errorf("expected ModelClaudeOpus, got %s", b.model)
	}
}

func TestBuilderUseModel(t *testing.T) {
	b := New(ModelGPT5).UseModel("custom/model")

	if string(b.model) != "custom/model" {
		t.Errorf("expected 'custom/model', got %s", b.model)
	}
}

func TestBuilderGetModel(t *testing.T) {
	b := New(ModelGPT5)

	if b.GetModel() != ModelGPT5 {
		t.Errorf("expected ModelGPT5, got %s", b.GetModel())
	}
}

func TestBuilderGetSystem(t *testing.T) {
	b := New(ModelGPT5).System("test prompt")

	if b.GetSystem() != "test prompt" {
		t.Errorf("expected 'test prompt', got %q", b.GetSystem())
	}
}

func TestBuilderClone(t *testing.T) {
	original := New(ModelGPT5).
		System("You are helpful").
		User("Hello").
		With(Vars{"key": "value"}).
		ContextString("ctx", "content").
		Retry(3).
		Fallback(ModelClaudeOpus).
		Debug()

	clone := original.Clone()

	// Verify clone has same values
	if clone.model != original.model {
		t.Error("clone model mismatch")
	}
	if clone.system != original.system {
		t.Error("clone system mismatch")
	}
	if len(clone.messages) != len(original.messages) {
		t.Error("clone messages length mismatch")
	}
	if len(clone.vars) != len(original.vars) {
		t.Error("clone vars length mismatch")
	}
	if len(clone.fileContext) != len(original.fileContext) {
		t.Error("clone fileContext length mismatch")
	}
	if clone.maxRetries != original.maxRetries {
		t.Error("clone maxRetries mismatch")
	}
	if len(clone.fallbacks) != len(original.fallbacks) {
		t.Error("clone fallbacks length mismatch")
	}
	if clone.debug != original.debug {
		t.Error("clone debug mismatch")
	}

	// Verify clone is independent
	clone.System("Different")
	if original.system == clone.system {
		t.Error("clone should be independent from original")
	}

	clone.vars["newkey"] = "newvalue"
	if _, exists := original.vars["newkey"]; exists {
		t.Error("modifying clone vars should not affect original")
	}
}

func TestBuildMessages(t *testing.T) {
	b := New(ModelGPT5).
		System("You are {{role}}").
		With(Vars{"role": "helpful"}).
		User("Hello {{name}}").
		Var("name", "World")

	msgs := b.buildMessages()

	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}

	// System message with var applied
	if msgs[0].Role != "system" {
		t.Errorf("expected first message role 'system', got %q", msgs[0].Role)
	}
	if msgs[0].Content != "You are helpful" {
		t.Errorf("expected system content 'You are helpful', got %q", msgs[0].Content)
	}

	// User message with var applied
	if msgs[1].Role != "user" {
		t.Errorf("expected second message role 'user', got %q", msgs[1].Role)
	}
	if msgs[1].Content != "Hello World" {
		t.Errorf("expected user content 'Hello World', got %q", msgs[1].Content)
	}
}

func TestBuildMessagesWithContext(t *testing.T) {
	b := New(ModelGPT5).
		System("You are helpful").
		ContextString("data", "some content").
		User("Analyze")

	msgs := b.buildMessages()

	// System should contain context
	sysContent := msgs[0].Content.(string)
	if !strings.Contains(sysContent, "# Context") {
		t.Error("system message should contain context header")
	}
	if !strings.Contains(sysContent, "some content") {
		t.Error("system message should contain context content")
	}
}

func TestBuildMessagesWithJSON(t *testing.T) {
	b := New(ModelGPT5).
		System("You are helpful").
		JSON().
		User("Give me data")

	msgs := b.buildMessages()

	if !strings.Contains(msgs[0].Content.(string), "JSON") {
		t.Error("JSON mode should add JSON instruction to system message")
	}
}

func TestBuildMessagesJSONNoSystem(t *testing.T) {
	b := New(ModelGPT5).
		JSON().
		User("Give me data")

	msgs := b.buildMessages()

	// Should create a system message for JSON
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Error("JSON mode should create system message")
	}
	if !strings.Contains(msgs[0].Content.(string), "JSON") {
		t.Error("JSON system message should contain JSON instruction")
	}
}

func TestBuildMessagesNoSystem(t *testing.T) {
	b := New(ModelGPT5).User("Hello")

	msgs := b.buildMessages()

	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Error("should only have user message")
	}
}

func TestBuilderChat(t *testing.T) {
	b := New(ModelGPT5).System("You are helpful")
	chat := b.Chat()

	if chat == nil {
		t.Fatal("Chat() should return a Conversation")
	}
	if chat.builder != b {
		t.Error("Conversation should reference the builder")
	}
	if len(chat.history) != 0 {
		t.Error("Conversation should start with empty history")
	}
}

func TestBuilderFluentChaining(t *testing.T) {
	// Test that all methods return *Builder for chaining
	b := New(ModelGPT5).
		System("test").
		User("test").
		Assistant("test").
		With(Vars{"k": "v"}).
		Var("k2", "v2").
		Context("nonexistent"). // Will print error but still chain
		ContextString("name", "content").
		Retry(1).
		Fallback(ModelClaudeOpus).
		JSON().
		Debug().
		Model(ModelGPT5).
		UseModel("test")

	if b == nil {
		t.Error("Fluent chaining should always return a builder")
	}
}
