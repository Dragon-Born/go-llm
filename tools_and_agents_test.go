package ai

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func TestRunTools_LoopsUntilNoToolCalls(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	call := 0
	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{Tools: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			call++
			if call == 1 {
				return &ProviderResponse{
					Content: "",
					ToolCalls: []ToolCall{
						{
							ID:   "tc_1",
							Type: "function",
							Function: struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							}{
								Name:      "get_weather",
								Arguments: `{"city":"Paris"}`,
							},
						},
					},
					TotalTokens: 10,
				}, nil
			}
			return &ProviderResponse{
				Content:      "It is 22°C.",
				ToolCalls:    nil,
				TotalTokens:  5,
				FinishReason: "stop",
			}, nil
		},
	}
	setDefaultClientForTest(t, p, ProviderOpenAI)

	called := 0
	out, err := New(ModelGPT5).
		Tool("get_weather", "Get weather", Params().String("city", "City", true).Build()).
		OnToolCall("get_weather", func(args map[string]any) (string, error) {
			called++
			if args["city"] != "Paris" {
				t.Fatalf("unexpected args: %#v", args)
			}
			return `{"temp":22,"unit":"C"}`, nil
		}).
		User("What's the weather?").
		RunTools(5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "It is 22°C." {
		t.Fatalf("unexpected final output: %q", out)
	}
	if called != 1 {
		t.Fatalf("expected tool to be called once, got %d", called)
	}
}

func TestAgent_Run_UsesToolsAndExtractsFinalAnswer(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	call := 0
	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{Tools: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			call++
			if call == 1 {
				return &ProviderResponse{
					Content: "",
					ToolCalls: []ToolCall{
						{
							ID:   "tc_1",
							Type: "function",
							Function: struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							}{
								Name:      "calculate",
								Arguments: `{"expression":"6*7"}`,
							},
						},
					},
					TotalTokens: 7,
				}, nil
			}
			return &ProviderResponse{
				Content:      "FINAL ANSWER: 42",
				ToolCalls:    nil,
				TotalTokens:  3,
				FinishReason: "stop",
			}, nil
		},
	}
	setDefaultClientForTest(t, p, ProviderOpenAI)

	var onStepCalled int
	var onActionCalled int
	var onObsCalled int

	agent := New(ModelGPT5).
		Agent().
		Tool("calculate", "Math", Params().String("expression", "expr", true).Build(), func(args map[string]any) (string, error) {
			// Minimal "calculator": just return 42 for this test.
			_ = args
			return "42", nil
		}).
		MaxSteps(3).
		OnStep(func(step AgentStep) { onStepCalled++ }).
		OnAction(func(action string, input map[string]any) { onActionCalled++ }).
		OnObservation(func(action, result string) { onObsCalled++ }).
		Timeout(2 * time.Second)

	res := agent.Run("what is 6*7?")
	if res.Error != nil {
		t.Fatalf("unexpected error: %v", res.Error)
	}
	if res.Answer != "42" {
		t.Fatalf("expected final answer 42, got %q", res.Answer)
	}
	if !res.Success() {
		t.Fatalf("expected Success()")
	}
	if onStepCalled == 0 || onActionCalled == 0 || onObsCalled == 0 {
		t.Fatalf("expected callbacks to fire, got onStep=%d onAction=%d onObs=%d", onStepCalled, onActionCalled, onObsCalled)
	}
}

func TestToolArguments_JSONUnmarshalErrorsAreHandled(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	// Use RunTools (builder path) which json.Unmarshal()s tool args.
	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{Tools: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{
				ToolCalls: []ToolCall{
					{
						ID:   "tc_bad",
						Type: "function",
						Function: struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						}{
							Name:      "bad",
							Arguments: "{not json}",
						},
					},
				},
			}, nil
		},
	}
	setDefaultClientForTest(t, p, ProviderOpenAI)

	_, err := New(ModelGPT5).
		Tool("bad", "bad", Params().Build()).
		OnToolCall("bad", func(args map[string]any) (string, error) { return "x", nil }).
		User("hi").
		RunTools(1)
	if err == nil {
		t.Fatalf("expected error for invalid tool args")
	}
	if _, ok := err.(*json.SyntaxError); ok {
		// acceptable, but the error is wrapped; just ensure it exists
	}
}
