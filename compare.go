package ai

import (
	"fmt"
	"sync"
)

// CompareBuilder helps compare responses across multiple models.
type CompareBuilder struct {
	system string
	prompt string
	vars   Vars
}

// Compare starts a comparison builder.
func Compare(prompt string) *CompareBuilder {
	return &CompareBuilder{
		prompt: prompt,
		vars:   Vars{},
	}
}

// System sets a system prompt for comparison.
func (c *CompareBuilder) System(system string) *CompareBuilder {
	c.system = system
	return c
}

// SystemFile loads a system prompt from a file.
func (c *CompareBuilder) SystemFile(path string) *CompareBuilder {
	b := New(ModelGPT5).SystemFile(path)
	c.system = b.system
	return c
}

// With adds template variables.
func (c *CompareBuilder) With(vars Vars) *CompareBuilder {
	for k, v := range vars {
		c.vars[k] = v
	}
	return c
}

// CompareResult holds a single model's response.
type CompareResult struct {
	Model    Model
	Response string
	Error    error
	Tokens   int
}

// On runs the comparison across specified models.
func (c *CompareBuilder) On(models ...Model) []CompareResult {
	results := make([]CompareResult, len(models))
	var wg sync.WaitGroup

	fmt.Println()
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))
	fmt.Printf("%s Comparing %d models...\n", colorCyan("⚡"), len(models))
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))

	// Run all models in parallel
	for i, model := range models {
		wg.Add(1)
		go func(idx int, m Model) {
			defer wg.Done()

			b := New(m)
			if c.system != "" {
				b = b.System(c.system)
			}
			if len(c.vars) > 0 {
				b = b.With(c.vars)
			}

			// Temporarily disable pretty for parallel calls
			oldPretty := Pretty
			Pretty = false

			msgs := b.User(c.prompt).buildMessages()
			content, resp, err := Send(m, msgs)

			Pretty = oldPretty

			results[idx] = CompareResult{
				Model:    m,
				Response: content,
				Error:    err,
			}
			if resp != nil {
				results[idx].Tokens = resp.Usage.TotalTokens
			}
		}(i, model)
	}

	wg.Wait()

	// Print results
	for _, r := range results {
		fmt.Println()
		fmt.Printf("%s %s\n", colorYellow("▸"), colorCyan(string(r.Model)))
		fmt.Println(colorDim("─────────────────────────────────────────────────────────────"))

		if r.Error != nil {
			fmt.Printf("%s %v\n", colorRed("✗"), r.Error)
		} else {
			fmt.Println(r.Response)
			if r.Tokens > 0 {
				fmt.Printf("\n%s\n", colorDim(fmt.Sprintf("(%d tokens)", r.Tokens)))
			}
		}
	}

	fmt.Println()
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))

	return results
}

// AllGPT compares a small set of common GPT models.
func (c *CompareBuilder) AllGPT() []CompareResult {
	return c.On(ModelGPT5, ModelGPT4o, ModelO1)
}

// AllClaude compares the default Claude lineup.
func (c *CompareBuilder) AllClaude() []CompareResult {
	return c.On(ModelClaudeOpus, ModelClaudeSonnet, ModelClaudeHaiku)
}

// TopModels compares a few popular high-quality models across providers.
func (c *CompareBuilder) TopModels() []CompareResult {
	return c.On(ModelGPT5, ModelClaudeOpus, ModelGemini3Flash, ModelGrok3)
}

// FastModels compares a few low-latency / cost-efficient models across providers.
func (c *CompareBuilder) FastModels() []CompareResult {
	return c.On(ModelGPT4oMini, ModelClaudeHaiku, ModelGemini2Flash, ModelGrok3Mini)
}
