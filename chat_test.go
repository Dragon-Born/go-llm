package ai

import (
	"testing"
)

func TestConversationCreation(t *testing.T) {
	b := New(ModelGPT5).System("You are helpful")
	chat := b.Chat()

	if chat.builder != b {
		t.Error("conversation should reference builder")
	}
	if len(chat.history) != 0 {
		t.Error("conversation should start with empty history")
	}
}

func TestConversationHistory(t *testing.T) {
	chat := &Conversation{
		builder: New(ModelGPT5),
		history: []Message{},
	}

	// Manually add to history for testing
	chat.history = append(chat.history, Message{Role: "user", Content: "Hello"})
	chat.history = append(chat.history, Message{Role: "assistant", Content: "Hi!"})

	history := chat.History()

	if len(history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(history))
	}
	if history[0].Role != "user" {
		t.Error("first message should be user")
	}
	if history[1].Role != "assistant" {
		t.Error("second message should be assistant")
	}
}

func TestConversationClear(t *testing.T) {
	// Disable pretty for test
	oldPretty := Pretty
	Pretty = false
	defer func() { Pretty = oldPretty }()

	chat := &Conversation{
		builder: New(ModelGPT5),
		history: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
		},
	}

	chat.Clear()

	if len(chat.history) != 0 {
		t.Error("history should be empty after clear")
	}
}

func TestConversationLastResponse(t *testing.T) {
	chat := &Conversation{
		builder: New(ModelGPT5),
		history: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "First response"},
			{Role: "user", Content: "Follow up"},
			{Role: "assistant", Content: "Second response"},
		},
	}

	last := chat.LastResponse()

	if last != "Second response" {
		t.Errorf("expected 'Second response', got %q", last)
	}
}

func TestConversationLastResponseEmpty(t *testing.T) {
	chat := &Conversation{
		builder: New(ModelGPT5),
		history: []Message{},
	}

	last := chat.LastResponse()

	if last != "" {
		t.Error("should return empty string when no responses")
	}
}

func TestConversationLastResponseNoAssistant(t *testing.T) {
	chat := &Conversation{
		builder: New(ModelGPT5),
		history: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	last := chat.LastResponse()

	if last != "" {
		t.Error("should return empty string when no assistant messages")
	}
}

func TestConversationSummarize(t *testing.T) {
	chat := &Conversation{
		builder: New(ModelGPT5),
		history: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
			{Role: "user", Content: "How are you?"},
			{Role: "assistant", Content: "Good!"},
		},
	}

	summary := chat.Summarize()

	if summary == "" {
		t.Error("summary should not be empty")
	}
}

func TestConversationBuildMessages(t *testing.T) {
	b := New(ModelGPT5).
		System("You are {{role}}").
		With(Vars{"role": "helpful"})

	chat := &Conversation{
		builder: b,
		history: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
		},
	}

	msgs := chat.buildMessages()

	// Should have system + history
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}

	// System should have vars applied
	if msgs[0].Content != "You are helpful" {
		t.Errorf("expected 'You are helpful', got %q", msgs[0].Content)
	}

	// History should be intact
	if msgs[1].Content != "Hello" {
		t.Errorf("expected 'Hello', got %q", msgs[1].Content)
	}
}

func TestConversationBuildMessagesNoSystem(t *testing.T) {
	b := New(ModelGPT5)

	chat := &Conversation{
		builder: b,
		history: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	msgs := chat.buildMessages()

	// Should only have history (no system)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Error("should only have user message")
	}
}
