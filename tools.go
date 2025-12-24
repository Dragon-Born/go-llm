package ai

import (
	"encoding/json"
	"fmt"
)

// ═══════════════════════════════════════════════════════════════════════════
// Tool/Function Calling
// ═══════════════════════════════════════════════════════════════════════════

// Tool represents a function the model can call
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes the function schema
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolCall represents a tool call from the model
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ToolResult is returned after executing a tool
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
}

// ToolHandler is a function that handles tool calls
type ToolHandler func(args map[string]any) (string, error)

// ToolDef is a convenient way to define tools
type ToolDef struct {
	Name        string
	Description string
	Parameters  map[string]any
	Handler     ToolHandler
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Tools
// ═══════════════════════════════════════════════════════════════════════════

// Tool adds a tool/function the model can call
func (b *Builder) Tool(name, description string, params map[string]any) *Builder {
	b.tools = append(b.tools, Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        name,
			Description: description,
			Parameters:  params,
		},
	})
	return b
}

// ToolDef adds a tool from a ToolDef struct (includes handler)
func (b *Builder) ToolDef(def ToolDef) *Builder {
	b.tools = append(b.tools, Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        def.Name,
			Description: def.Description,
			Parameters:  def.Parameters,
		},
	})
	if b.toolHandlers == nil {
		b.toolHandlers = make(map[string]ToolHandler)
	}
	b.toolHandlers[def.Name] = def.Handler
	return b
}

// Tools adds multiple tools at once
func (b *Builder) Tools(tools ...Tool) *Builder {
	b.tools = append(b.tools, tools...)
	return b
}

// OnToolCall registers a handler for a specific tool
func (b *Builder) OnToolCall(name string, handler ToolHandler) *Builder {
	if b.toolHandlers == nil {
		b.toolHandlers = make(map[string]ToolHandler)
	}
	b.toolHandlers[name] = handler
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool Schema Helpers - DX-friendly parameter builders
// ═══════════════════════════════════════════════════════════════════════════

// Params creates a JSON Schema object for function parameters
func Params() *ParamBuilder {
	return &ParamBuilder{
		schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
			"required":   []string{},
		},
	}
}

// ParamBuilder helps construct parameter schemas
type ParamBuilder struct {
	schema map[string]any
}

// String adds a string parameter
func (p *ParamBuilder) String(name, desc string, required bool) *ParamBuilder {
	props := p.schema["properties"].(map[string]any)
	props[name] = map[string]any{
		"type":        "string",
		"description": desc,
	}
	if required {
		req := p.schema["required"].([]string)
		p.schema["required"] = append(req, name)
	}
	return p
}

// Number adds a number parameter
func (p *ParamBuilder) Number(name, desc string, required bool) *ParamBuilder {
	props := p.schema["properties"].(map[string]any)
	props[name] = map[string]any{
		"type":        "number",
		"description": desc,
	}
	if required {
		req := p.schema["required"].([]string)
		p.schema["required"] = append(req, name)
	}
	return p
}

// Int adds an integer parameter
func (p *ParamBuilder) Int(name, desc string, required bool) *ParamBuilder {
	props := p.schema["properties"].(map[string]any)
	props[name] = map[string]any{
		"type":        "integer",
		"description": desc,
	}
	if required {
		req := p.schema["required"].([]string)
		p.schema["required"] = append(req, name)
	}
	return p
}

// Bool adds a boolean parameter
func (p *ParamBuilder) Bool(name, desc string, required bool) *ParamBuilder {
	props := p.schema["properties"].(map[string]any)
	props[name] = map[string]any{
		"type":        "boolean",
		"description": desc,
	}
	if required {
		req := p.schema["required"].([]string)
		p.schema["required"] = append(req, name)
	}
	return p
}

// Enum adds a string enum parameter
func (p *ParamBuilder) Enum(name, desc string, values []string, required bool) *ParamBuilder {
	props := p.schema["properties"].(map[string]any)
	props[name] = map[string]any{
		"type":        "string",
		"description": desc,
		"enum":        values,
	}
	if required {
		req := p.schema["required"].([]string)
		p.schema["required"] = append(req, name)
	}
	return p
}

// Array adds an array parameter
func (p *ParamBuilder) Array(name, desc, itemType string, required bool) *ParamBuilder {
	props := p.schema["properties"].(map[string]any)
	props[name] = map[string]any{
		"type":        "array",
		"description": desc,
		"items":       map[string]any{"type": itemType},
	}
	if required {
		req := p.schema["required"].([]string)
		p.schema["required"] = append(req, name)
	}
	return p
}

// Build returns the final schema
func (p *ParamBuilder) Build() map[string]any {
	return p.schema
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool Response Handling
// ═══════════════════════════════════════════════════════════════════════════

// ToolResponse contains the model's response with potential tool calls
type ToolResponse struct {
	Content   string     // Text response (may be empty if tool calls)
	ToolCalls []ToolCall // Tool calls the model wants to make
	Model     Model
	Tokens    int
}

// HasToolCalls returns true if the response contains tool calls
func (r *ToolResponse) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// SendWithTools executes the request and returns tool calls if any
func (b *Builder) SendWithTools() (*ToolResponse, error) {
	msgs := b.buildMessages()

	content, resp, toolCalls, err := SendWithTools(b.model, msgs, b.tools, SendOptions{
		Temperature: b.temperature,
		Thinking:    b.thinking,
	})
	if err != nil {
		return nil, err
	}

	result := &ToolResponse{
		Content:   content,
		ToolCalls: toolCalls,
		Model:     b.model,
	}
	if resp != nil {
		result.Tokens = resp.Usage.TotalTokens
	}

	return result, nil
}

// RunTools executes tool calls and continues the conversation automatically
// This is the "agentic" mode - it loops until the model stops calling tools
func (b *Builder) RunTools(maxIterations int) (string, error) {
	if maxIterations <= 0 {
		maxIterations = 10 // sensible default
	}

	for i := 0; i < maxIterations; i++ {
		resp, err := b.SendWithTools()
		if err != nil {
			return "", err
		}

		// No tool calls = we're done
		if !resp.HasToolCalls() {
			if Pretty {
				printPrettyResponse(b.model, resp.Content)
			}
			return resp.Content, nil
		}

		// Execute each tool call
		for _, tc := range resp.ToolCalls {
			handler, ok := b.toolHandlers[tc.Function.Name]
			if !ok {
				return "", fmt.Errorf("no handler for tool: %s", tc.Function.Name)
			}

			// Parse arguments
			var args map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				return "", fmt.Errorf("invalid tool arguments: %w", err)
			}

			if Debug {
				fmt.Printf("%s Calling tool: %s(%v)\n", colorYellow("🔧"), tc.Function.Name, args)
			}

			// Execute handler
			result, err := handler(args)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			if Debug {
				fmt.Printf("%s Tool result: %s\n", colorGreen("✓"), truncate(result, 100))
			}

			// Add tool result to conversation
			b.messages = append(b.messages, Message{
				Role:       "assistant",
				Content:    "",
				ToolCalls:  resp.ToolCalls,
				ToolCallID: "",
			})
			b.messages = append(b.messages, Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}

	return "", fmt.Errorf("max tool iterations (%d) reached", maxIterations)
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
