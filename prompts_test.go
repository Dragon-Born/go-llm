package ai

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPromptsDir(t *testing.T) {
	if PromptsDir != "prompts" {
		t.Errorf("expected default PromptsDir to be 'prompts', got %q", PromptsDir)
	}
}

func TestPromptsDirConfigurable(t *testing.T) {
	original := PromptsDir
	defer func() { PromptsDir = original }()

	PromptsDir = "custom-prompts"
	if PromptsDir != "custom-prompts" {
		t.Error("PromptsDir should be configurable")
	}
}

func TestLoadPromptNotFound(t *testing.T) {
	_, err := LoadPrompt("nonexistent-prompt")
	if err == nil {
		t.Error("expected error for non-existent prompt")
	}
}

func TestMustLoadPromptPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-existent prompt")
		}
	}()
	MustLoadPrompt("nonexistent-prompt")
}

func TestLoadPromptWithExtension(t *testing.T) {
	// Create temp directory and file
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	// Create test prompt
	content := "Test prompt content"
	err := os.WriteFile(filepath.Join(tmpDir, "test.md"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Load without extension
	loaded, err := LoadPrompt("test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != content {
		t.Errorf("expected %q, got %q", content, loaded)
	}

	// Load with extension
	loaded, err = LoadPrompt("test.md")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != content {
		t.Errorf("expected %q, got %q", content, loaded)
	}
}

func TestLoadPromptTxt(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	content := "Text prompt content"
	err := os.WriteFile(filepath.Join(tmpDir, "test.txt"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadPrompt("test.txt")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != content {
		t.Errorf("expected %q, got %q", content, loaded)
	}
}

func TestPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	originalDefault := DefaultModel
	PromptsDir = tmpDir
	DefaultModel = ModelClaudeOpus
	defer func() {
		PromptsDir = original
		DefaultModel = originalDefault
	}()

	content := "You are helpful"
	err := os.WriteFile(filepath.Join(tmpDir, "test.md"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Disable pretty for test
	oldPretty := Pretty
	Pretty = false
	defer func() { Pretty = oldPretty }()

	b := Prompt("test")

	if b.model != ModelClaudeOpus {
		t.Errorf("expected DefaultModel (Claude), got %s", b.model)
	}
	if b.system != content {
		t.Errorf("expected system %q, got %q", content, b.system)
	}
}

func TestPromptWith(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	content := "You are a {{role}} assistant"
	err := os.WriteFile(filepath.Join(tmpDir, "test.md"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	Pretty = false
	defer func() { Pretty = true }()

	b := PromptWith("test", Vars{"role": "helpful"})

	if b.vars["role"] != "helpful" {
		t.Error("vars should be set")
	}
}

func TestPromptFor(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	content := "Test content"
	err := os.WriteFile(filepath.Join(tmpDir, "test.md"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	Pretty = false
	defer func() { Pretty = true }()

	b := PromptFor("test", ModelGrok3)

	if b.model != ModelGrok3 {
		t.Errorf("expected Grok3, got %s", b.model)
	}
	if b.system != content {
		t.Errorf("expected system %q, got %q", content, b.system)
	}
}

func TestPromptForWith(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	content := "You are {{role}}"
	err := os.WriteFile(filepath.Join(tmpDir, "test.md"), []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	Pretty = false
	defer func() { Pretty = true }()

	b := PromptForWith("test", ModelGemini3Flash, Vars{"role": "creative"})

	if b.model != ModelGemini3Flash {
		t.Errorf("expected Gemini, got %s", b.model)
	}
	if b.vars["role"] != "creative" {
		t.Error("vars should be set")
	}
}

func TestListPrompts(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	// Create test files
	os.WriteFile(filepath.Join(tmpDir, "prompt1.md"), []byte(""), 0644)
	os.WriteFile(filepath.Join(tmpDir, "prompt2.md"), []byte(""), 0644)
	os.WriteFile(filepath.Join(tmpDir, "prompt3.txt"), []byte(""), 0644)
	os.WriteFile(filepath.Join(tmpDir, "other.json"), []byte(""), 0644)

	names, err := ListPrompts()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(names) != 3 {
		t.Errorf("expected 3 prompts, got %d: %v", len(names), names)
	}

	// Check that .json file is not included
	for _, name := range names {
		if name == "other" {
			t.Error("should not include .json files")
		}
	}
}

func TestListPromptsEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	original := PromptsDir
	PromptsDir = tmpDir
	defer func() { PromptsDir = original }()

	names, err := ListPrompts()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(names) != 0 {
		t.Errorf("expected 0 prompts, got %d", len(names))
	}
}

func TestListPromptsNotFound(t *testing.T) {
	original := PromptsDir
	PromptsDir = "nonexistent-directory"
	defer func() { PromptsDir = original }()

	_, err := ListPrompts()
	if err == nil {
		t.Error("expected error for non-existent directory")
	}
}



