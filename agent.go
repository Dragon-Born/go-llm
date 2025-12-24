package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Agent - Agentic Loops with ReAct Pattern
// ═══════════════════════════════════════════════════════════════════════════

// Agent provides an agentic interface for complex reasoning tasks.
// It uses the ReAct (Reasoning + Acting) pattern with observations.
type Agent struct {
	builder       *Builder
	maxSteps      int
	tools         map[string]ToolHandler
	toolDefs      []Tool
	onStep        func(AgentStep)
	onThought     func(string)
	onAction      func(string, map[string]any)
	onObservation func(string, string)
	onComplete    func(AgentResult)
	humanApproval func(AgentStep) bool // return false to abort
	state         map[string]any
	ctx           context.Context
	timeout       time.Duration
}

// AgentStep represents a single step in the agent's execution
type AgentStep struct {
	Number      int            `json:"number"`
	Thought     string         `json:"thought,omitempty"`
	Action      string         `json:"action,omitempty"`
	ActionInput map[string]any `json:"action_input,omitempty"`
	Observation string         `json:"observation,omitempty"`
	Duration    time.Duration  `json:"duration,omitempty"`
	Tokens      int            `json:"tokens,omitempty"`
}

// AgentResult is the final result of agent execution
type AgentResult struct {
	Answer      string         `json:"answer"`
	Steps       []AgentStep    `json:"steps"`
	TotalSteps  int            `json:"total_steps"`
	Duration    time.Duration  `json:"duration"`
	TotalTokens int            `json:"total_tokens"`
	State       map[string]any `json:"state,omitempty"`
	Error       error          `json:"-"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Agent Constructor
// ═══════════════════════════════════════════════════════════════════════════

// NewAgent creates a new agent with the given builder as the base
func NewAgent(builder *Builder) *Agent {
	return &Agent{
		builder:  builder.Clone(),
		maxSteps: 10,
		tools:    make(map[string]ToolHandler),
		toolDefs: []Tool{},
		state:    make(map[string]any),
	}
}

// Agent creates an agent from this builder
func (b *Builder) Agent() *Agent {
	return NewAgent(b)
}

// ═══════════════════════════════════════════════════════════════════════════
// Agent Configuration - Fluent API
// ═══════════════════════════════════════════════════════════════════════════

// MaxSteps sets the maximum number of reasoning steps
func (a *Agent) MaxSteps(n int) *Agent {
	a.maxSteps = n
	return a
}

// Tool adds a tool the agent can use
func (a *Agent) Tool(name, description string, params map[string]any, handler ToolHandler) *Agent {
	a.tools[name] = handler
	a.toolDefs = append(a.toolDefs, Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        name,
			Description: description,
			Parameters:  params,
		},
	})
	return a
}

// ToolDef adds a tool from a ToolDef struct
func (a *Agent) ToolDef(def ToolDef) *Agent {
	return a.Tool(def.Name, def.Description, def.Parameters, def.Handler)
}

// WithContext sets a context for cancellation
func (a *Agent) WithContext(ctx context.Context) *Agent {
	a.ctx = ctx
	return a
}

// Timeout sets a timeout for the entire agent run
func (a *Agent) Timeout(d time.Duration) *Agent {
	a.timeout = d
	return a
}

// State sets initial state for the agent
func (a *Agent) State(state map[string]any) *Agent {
	a.state = state
	return a
}

// Set sets a single state value
func (a *Agent) Set(key string, value any) *Agent {
	a.state[key] = value
	return a
}

// ═══════════════════════════════════════════════════════════════════════════
// Agent Callbacks - Observability
// ═══════════════════════════════════════════════════════════════════════════

// OnStep registers a callback for each step
func (a *Agent) OnStep(fn func(AgentStep)) *Agent {
	a.onStep = fn
	return a
}

// OnThought registers a callback for agent thoughts
func (a *Agent) OnThought(fn func(string)) *Agent {
	a.onThought = fn
	return a
}

// OnAction registers a callback before each action
func (a *Agent) OnAction(fn func(action string, input map[string]any)) *Agent {
	a.onAction = fn
	return a
}

// OnObservation registers a callback after each action
func (a *Agent) OnObservation(fn func(action, result string)) *Agent {
	a.onObservation = fn
	return a
}

// OnComplete registers a callback when agent finishes
func (a *Agent) OnComplete(fn func(AgentResult)) *Agent {
	a.onComplete = fn
	return a
}

// RequireApproval requires human approval before each action
func (a *Agent) RequireApproval(fn func(AgentStep) bool) *Agent {
	a.humanApproval = fn
	return a
}

// ═══════════════════════════════════════════════════════════════════════════
// Agent Execution
// ═══════════════════════════════════════════════════════════════════════════

// Run executes the agent with the given task
func (a *Agent) Run(task string) AgentResult {
	start := time.Now()
	result := AgentResult{
		Steps: []AgentStep{},
		State: a.state,
	}

	// Setup context with timeout
	ctx := a.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if a.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, a.timeout)
		defer cancel()
	}

	// Build the ReAct system prompt
	systemPrompt := a.buildReActPrompt()

	// Create conversation with tools
	builder := a.builder.Clone().
		System(systemPrompt).
		Tools(a.toolDefs...).
		WithContext(ctx)

	// Copy tool handlers
	for name, handler := range a.tools {
		builder.OnToolCall(name, handler)
	}

	// Build initial user message
	messages := []Message{
		{Role: "user", Content: fmt.Sprintf("Task: %s\n\nBegin!", task)},
	}

	// Agent loop
	for step := 1; step <= a.maxSteps; step++ {
		// Check context
		select {
		case <-ctx.Done():
			result.Error = ctx.Err()
			result.Duration = time.Since(start)
			return result
		default:
		}

		stepStart := time.Now()
		currentStep := AgentStep{Number: step}

		// Send request with tools
		builder.messages = messages
		resp, err := builder.SendWithTools()
		if err != nil {
			result.Error = err
			result.Duration = time.Since(start)
			return result
		}

		currentStep.Duration = time.Since(stepStart)
		currentStep.Tokens = resp.Tokens
		result.TotalTokens += resp.Tokens

		// Check if we got a final answer (no tool calls)
		if !resp.HasToolCalls() {
			currentStep.Thought = resp.Content
			result.Answer = a.extractFinalAnswer(resp.Content)
			result.Steps = append(result.Steps, currentStep)
			break
		}

		// Process tool calls
		for _, tc := range resp.ToolCalls {
			// Parse action input
			var actionInput map[string]any
			json.Unmarshal([]byte(tc.Function.Arguments), &actionInput)

			currentStep.Action = tc.Function.Name
			currentStep.ActionInput = actionInput

			// Callback: OnAction
			if a.onAction != nil {
				a.onAction(tc.Function.Name, actionInput)
			}

			// Human approval if required
			if a.humanApproval != nil {
				if !a.humanApproval(currentStep) {
					result.Error = fmt.Errorf("action rejected by human: %s", tc.Function.Name)
					result.Duration = time.Since(start)
					return result
				}
			}

			// Execute tool
			handler, ok := a.tools[tc.Function.Name]
			if !ok {
				currentStep.Observation = fmt.Sprintf("Error: unknown tool %q", tc.Function.Name)
			} else {
				observation, err := handler(actionInput)
				if err != nil {
					currentStep.Observation = fmt.Sprintf("Error: %v", err)
				} else {
					currentStep.Observation = observation
				}
			}

			// Callback: OnObservation
			if a.onObservation != nil {
				a.onObservation(tc.Function.Name, currentStep.Observation)
			}

			// Add assistant message with tool call
			messages = append(messages, Message{
				Role:      "assistant",
				Content:   resp.Content,
				ToolCalls: resp.ToolCalls,
			})

			// Add tool result
			messages = append(messages, Message{
				Role:       "tool",
				Content:    currentStep.Observation,
				ToolCallID: tc.ID,
			})
		}

		// Callback: OnStep
		if a.onStep != nil {
			a.onStep(currentStep)
		}

		result.Steps = append(result.Steps, currentStep)

		if Debug {
			a.printStep(currentStep)
		}
	}

	result.TotalSteps = len(result.Steps)
	result.Duration = time.Since(start)

	// Callback: OnComplete
	if a.onComplete != nil {
		a.onComplete(result)
	}

	if Pretty && result.Answer != "" {
		printPrettyResponse(a.builder.model, result.Answer)
	}

	return result
}

// ═══════════════════════════════════════════════════════════════════════════
// ReAct Prompt Builder
// ═══════════════════════════════════════════════════════════════════════════

func (a *Agent) buildReActPrompt() string {
	var sb strings.Builder

	// Base system prompt if set
	if a.builder.system != "" {
		sb.WriteString(a.builder.system)
		sb.WriteString("\n\n")
	}

	sb.WriteString(`You are an AI agent that solves tasks step by step using available tools.

For each step:
1. Think about what to do next
2. If you need information, use a tool
3. Observe the tool's result
4. Continue until you have the final answer

When you have the final answer, respond WITHOUT calling any tools, and include:
FINAL ANSWER: [your answer here]

Be concise and efficient. Avoid unnecessary tool calls.`)

	// Add state context if available
	if len(a.state) > 0 {
		sb.WriteString("\n\nCurrent State:\n")
		stateJSON, _ := json.MarshalIndent(a.state, "", "  ")
		sb.WriteString(string(stateJSON))
	}

	return sb.String()
}

func (a *Agent) extractFinalAnswer(content string) string {
	// Look for explicit final answer marker
	if idx := strings.Index(strings.ToUpper(content), "FINAL ANSWER:"); idx != -1 {
		return strings.TrimSpace(content[idx+13:])
	}
	// Otherwise return the full content
	return content
}

func (a *Agent) printStep(step AgentStep) {
	fmt.Printf("\n%s Step %d\n", colorCyan("═══"), step.Number)
	if step.Thought != "" {
		fmt.Printf("  %s %s\n", colorYellow("💭"), step.Thought)
	}
	if step.Action != "" {
		fmt.Printf("  %s %s(%v)\n", colorBlue("🔧"), step.Action, step.ActionInput)
	}
	if step.Observation != "" {
		fmt.Printf("  %s %s\n", colorGreen("👁"), truncate(step.Observation, 200))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Methods
// ═══════════════════════════════════════════════════════════════════════════

// Success returns true if agent completed without error
func (r AgentResult) Success() bool {
	return r.Error == nil && r.Answer != ""
}

// String returns the final answer
func (r AgentResult) String() string {
	return r.Answer
}

// GetState retrieves a value from agent state
func (a *Agent) GetState(key string) (any, bool) {
	v, ok := a.state[key]
	return v, ok
}

// ═══════════════════════════════════════════════════════════════════════════
// Quick Agent Helpers
// ═══════════════════════════════════════════════════════════════════════════

// QuickAgent creates an agent with common defaults
func QuickAgent(model Model) *Agent {
	return NewAgent(New(model)).MaxSteps(15)
}

// ResearchAgent creates an agent configured for research tasks
func ResearchAgent(model Model) *Agent {
	return NewAgent(New(model).ThinkHigh()).MaxSteps(20)
}

// CodeAgent creates an agent configured for coding tasks
func CodeAgent() *Agent {
	return NewAgent(GPT5Codex().ThinkHigh()).MaxSteps(15)
}
