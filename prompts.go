package ai

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

var (
	// PromptsDir is where prompt files are stored.
	// Change this to use a different directory.
	PromptsDir = "prompts"
)

// ═══════════════════════════════════════════════════════════════════════════
// Prompt Loading
// ═══════════════════════════════════════════════════════════════════════════

// LoadPrompt loads a prompt file by name (with or without extension).
// It looks in PromptsDir (default: "prompts/").
func LoadPrompt(name string) (string, error) {
	// Add .md extension if not present
	if !strings.HasSuffix(name, ".md") && !strings.HasSuffix(name, ".txt") {
		name = name + ".md"
	}

	path := filepath.Join(PromptsDir, name)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("prompt %q not found: %w", name, err)
	}

	return string(data), nil
}

// MustLoadPrompt loads a prompt or panics.
func MustLoadPrompt(name string) string {
	p, err := LoadPrompt(name)
	if err != nil {
		panic(err)
	}
	return p
}

// ═══════════════════════════════════════════════════════════════════════════
// Prompt Builders (use DefaultModel)
// ═══════════════════════════════════════════════════════════════════════════

// Prompt loads a prompt and returns a Builder with DefaultModel.
func Prompt(name string) *Builder {
	content, err := LoadPrompt(name)
	if err != nil {
		fmt.Printf("%s Error loading prompt %s: %v\n", colorRed("✗"), name, err)
		return New(DefaultModel)
	}
	return New(DefaultModel).System(content)
}

// PromptWith loads a prompt, applies variables, and returns a Builder with DefaultModel.
func PromptWith(name string, vars Vars) *Builder {
	content, err := LoadPrompt(name)
	if err != nil {
		fmt.Printf("%s Error loading prompt %s: %v\n", colorRed("✗"), name, err)
		return New(DefaultModel)
	}
	return New(DefaultModel).System(content).With(vars)
}

// ═══════════════════════════════════════════════════════════════════════════
// Prompt Builders with Model
// ═══════════════════════════════════════════════════════════════════════════

// PromptFor loads a prompt and returns a Builder for the specified model.
func PromptFor(name string, model Model) *Builder {
	content, err := LoadPrompt(name)
	if err != nil {
		fmt.Printf("%s Error loading prompt %s: %v\n", colorRed("✗"), name, err)
		return New(model)
	}
	return New(model).System(content)
}

// PromptForWith loads a prompt, applies variables, and returns a Builder for the specified model.
func PromptForWith(name string, model Model, vars Vars) *Builder {
	content, err := LoadPrompt(name)
	if err != nil {
		fmt.Printf("%s Error loading prompt %s: %v\n", colorRed("✗"), name, err)
		return New(model)
	}
	return New(model).System(content).With(vars)
}

// ═══════════════════════════════════════════════════════════════════════════
// Listing
// ═══════════════════════════════════════════════════════════════════════════

// ListPrompts returns all available prompt names in PromptsDir.
func ListPrompts() ([]string, error) {
	entries, err := os.ReadDir(PromptsDir)
	if err != nil {
		return nil, err
	}

	var names []string
	for _, e := range entries {
		if !e.IsDir() && (strings.HasSuffix(e.Name(), ".md") || strings.HasSuffix(e.Name(), ".txt")) {
			name := e.Name()
			name = strings.TrimSuffix(name, ".md")
			name = strings.TrimSuffix(name, ".txt")
			names = append(names, name)
		}
	}
	return names, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Template Utilities
// ═══════════════════════════════════════════════════════════════════════════

// ApplyVars replaces {{key}} placeholders in text.
func ApplyVars(text string, vars Vars) string {
	return applyTemplate(text, vars)
}
