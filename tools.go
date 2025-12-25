package ai

import (
	"encoding/json"
	"fmt"
)

// ═══════════════════════════════════════════════════════════════════════════
// Tool/Function Calling
// ═══════════════════════════════════════════════════════════════════════════

// Tool represents a function the model can call.
// It follows the OpenAI tool calling schema.
type Tool struct {
	Type     string       `json:"type"`     // "function" is currently the only supported type
	Function ToolFunction `json:"function"` // The function definition
}

// ToolFunction describes the function schema, including name, description, and parameters.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"` // JSON Schema for parameters
}

// ToolCall represents a specific invocation of a tool by the model.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // JSON string arguments
	} `json:"function"`
}

// ToolResult is the output returned to the model after executing a tool.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
}

// ToolHandler is a callback function that handles tool execution.
// It takes a map of arguments and returns a string result or error.
type ToolHandler func(args map[string]any) (string, error)

// ToolDef simplifies defining tools by bundling the schema and handler together.
type ToolDef struct {
	Name        string
	Description string
	Parameters  map[string]any
	Handler     ToolHandler
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Tools
// ═══════════════════════════════════════════════════════════════════════════

// Tool adds a function tool to the request configuration.
// It registers the tool definition but not a handler. Use OnToolCall to register a handler separately.
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

// ToolDef adds a tool using a ToolDef struct.
// This registers both the tool definition and the execution handler.
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

// Tools adds multiple tool definitions at once.
func (b *Builder) Tools(tools ...Tool) *Builder {
	b.tools = append(b.tools, tools...)
	return b
}

// OnToolCall registers a handler function for a specific tool name.
// This is used when the tool was defined without a handler (e.g., via Tool() or raw Tool struct).
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

// Params creates a new ParamBuilder for constructing JSON Schemas.
func Params() *ParamBuilder {
	return &ParamBuilder{
		schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
			"required":   []string{},
		},
	}
}

// ParamBuilder helps construct JSON Schema objects for tool parameters.
// It provides a fluent API for defining strings, numbers, booleans, and arrays.
type ParamBuilder struct {
	schema map[string]any
}

// String adds a string parameter to the schema.
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

// Number adds a number (float) parameter to the schema.
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

// Int adds an integer parameter to the schema.
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

// Bool adds a boolean parameter to the schema.
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

// Enum adds a string parameter restricted to a set of values.
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

// Array adds an array parameter with a specific item type.
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

// Build finalizes and returns the map representing the JSON Schema.
func (p *ParamBuilder) Build() map[string]any {
	return p.schema
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool Response Handling
// ═══════════════════════════════════════════════════════════════════════════

// ToolResponse encapsulates the response from a model that may contain tool calls.
type ToolResponse struct {
	Content   string     // Text response (may be empty if tool calls are present)
	ToolCalls []ToolCall // Tool calls the model wants to make
	Model     Model
	Tokens    int
}

// HasToolCalls reports whether the response contains any tool calls.
func (r *ToolResponse) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// SendWithTools executes the request and returns a ToolResponse.
// This is used for manual handling of tool calls. For automatic execution, use RunTools.
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

// RunTools executes the request in an "agentic" loop.
// It automatically executes tool calls and feeds the results back to the model.
// It continues until the model provides a final text response or maxIterations is reached.
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
