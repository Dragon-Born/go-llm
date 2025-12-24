package ai

import (
	"fmt"
	"strings"
)

// Conversation maintains chat history for multi-turn conversations
type Conversation struct {
	builder *Builder
	history []Message
}

// Say sends a message and continues the conversation
func (c *Conversation) Say(message string) (string, error) {
	// Add user message to history
	c.history = append(c.history, Message{Role: "user", Content: message})

	// Build full message list
	msgs := c.buildMessages()

	content, _, err := Send(c.builder.model, msgs, SendOptions{
		Temperature: c.builder.temperature,
		Thinking:    c.builder.thinking,
	})
	if err != nil {
		return "", err
	}

	// Add assistant response to history
	c.history = append(c.history, Message{Role: "assistant", Content: content})

	if Pretty {
		printPrettyConversation(c.builder.model, message, content)
	}

	return content, nil
}

// buildMessages combines system + history
func (c *Conversation) buildMessages() []Message {
	var msgs []Message

	// Add system message if present
	if c.builder.system != "" {
		system := c.builder.system
		if len(c.builder.vars) > 0 {
			system = applyTemplate(system, c.builder.vars)
		}

		// Add JSON instruction if enabled
		if c.builder.jsonMode && system != "" {
			system += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation."
		} else if c.builder.jsonMode {
			system = "Respond with valid JSON only. No markdown, no explanation."
		}

		// Add context to system if present
		if len(c.builder.fileContext) > 0 {
			contextStr := "\n\n# Context\n" + strings.Join(c.builder.fileContext, "\n\n")
			system += contextStr
		}

		msgs = append(msgs, Message{Role: "system", Content: system})
	}

	// Add conversation history
	msgs = append(msgs, c.history...)

	return msgs
}

// History returns the conversation history
func (c *Conversation) History() []Message {
	return c.history
}

// Clear resets the conversation history
func (c *Conversation) Clear() {
	c.history = []Message{}
	if Pretty {
		fmt.Println(colorYellow("↻ Conversation cleared"))
	}
}

// Dump prints the full conversation history
func (c *Conversation) Dump() {
	fmt.Println()
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))
	fmt.Printf("%s Conversation History (%s)\n", colorCyan("💬"), c.builder.model)
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))

	if c.builder.system != "" {
		fmt.Printf("\n%s\n", colorMagenta("[SYSTEM]"))
		fmt.Println(c.builder.system)
	}

	for _, m := range c.history {
		fmt.Println()
		if m.Role == "user" {
			fmt.Printf("%s\n", colorGreen("[YOU]"))
		} else {
			fmt.Printf("%s\n", colorBlue("[ASSISTANT]"))
		}
		fmt.Println(m.Content)
	}

	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))
	fmt.Printf("Total messages: %d\n\n", len(c.history))
}

// LastResponse returns the last assistant response
func (c *Conversation) LastResponse() string {
	for i := len(c.history) - 1; i >= 0; i-- {
		if c.history[i].Role == "assistant" {
			if content, ok := c.history[i].Content.(string); ok {
				return content
			}
		}
	}
	return ""
}

// Summarize returns a brief summary of the conversation
func (c *Conversation) Summarize() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Model: %s\n", c.builder.model))
	sb.WriteString(fmt.Sprintf("Messages: %d\n", len(c.history)))

	userCount := 0
	assistantCount := 0
	for _, m := range c.history {
		if m.Role == "user" {
			userCount++
		} else {
			assistantCount++
		}
	}
	sb.WriteString(fmt.Sprintf("User: %d, Assistant: %d\n", userCount, assistantCount))

	return sb.String()
}
