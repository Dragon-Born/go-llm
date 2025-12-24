package ai

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

type testOrderWithValidate struct {
	ID       string  `json:"id"`
	Amount   float64 `json:"amount"`
	Currency string  `json:"currency"`
}

func (o *testOrderWithValidate) Validate() error {
	if err := ValidateRequired(o, "ID", "Currency"); err != nil {
		return err
	}
	if o.Amount <= 0 {
		return fmt.Errorf("amount must be positive")
	}
	if err := ValidateOneOf(o.Currency, "USD", "EUR"); err != nil {
		return err
	}
	return nil
}

func TestStructToSchema_RequiredOptionalAndDescriptions(t *testing.T) {
	type Address struct {
		City string `json:"city" desc:"city name"`
	}
	type Example struct {
		Name    string   `json:"name"`
		Age     int      `json:"age"`
		Tags    []string `json:"tags"`
		Notes   string   `json:"notes,omitempty"`
		Address Address  `json:"address"`
		skipMe  string
	}

	schema := structToSchema(&Example{})
	props := schema["properties"].(map[string]any)
	if _, ok := props["name"]; !ok {
		t.Fatalf("expected name property")
	}
	if props["address"].(map[string]any)["type"] != "object" {
		t.Fatalf("expected nested struct to be object schema")
	}

	// desc should appear in nested schema too
	addrProps := props["address"].(map[string]any)["properties"].(map[string]any)
	if addrProps["city"].(map[string]any)["description"] != "city name" {
		t.Fatalf("expected desc tag in schema")
	}

	req := schema["required"].([]string)
	if contains(req, "notes") {
		t.Fatalf("expected notes to be optional (omitempty), required=%v", req)
	}
	if !contains(req, "name") || !contains(req, "age") || !contains(req, "tags") || !contains(req, "address") {
		t.Fatalf("expected required fields to be present, required=%v", req)
	}
}

func TestInto_CleansMarkdownAndUnmarshals(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{JSON: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{
				Content: "```json\n{\"sentiment\":\"positive\",\"score\":0.9}\n```",
			}, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI})

	var out struct {
		Sentiment string  `json:"sentiment"`
		Score     float64 `json:"score"`
	}

	if err := b.Into("analyze", &out); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Sentiment != "positive" || out.Score != 0.9 {
		t.Fatalf("unexpected parsed result: %+v", out)
	}
}

func TestIntoWithRetry_RetriesOnParseErrorAndAddsCorrectionMessages(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	call := 0
	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{JSON: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			call++
			if call == 1 {
				return &ProviderResponse{Content: `{"name":"alice"`}, nil // invalid JSON
			}
			return &ProviderResponse{Content: `{"name":"alice","age":30}`}, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI})

	var person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	err := b.IntoWithRetry("extract", &person, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if person.Name != "alice" || person.Age != 30 {
		t.Fatalf("unexpected person: %+v", person)
	}
	if p.Calls() != 2 {
		t.Fatalf("expected 2 calls, got %d", p.Calls())
	}

	reqs := p.Requests()
	if len(reqs) != 2 {
		t.Fatalf("expected 2 captured requests, got %d", len(reqs))
	}
	// Second request should include correction prompt.
	found := false
	for _, m := range reqs[1].Messages {
		if m.Role == "user" {
			if s, ok := m.Content.(string); ok && strings.Contains(s, "Please fix the JSON") {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("expected correction prompt in second request messages")
	}
}

func TestIntoWithRetry_RetriesOnStructValidation(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	call := 0
	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{JSON: true},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			call++
			if call == 1 {
				// Parses but fails Validate (amount <= 0)
				return &ProviderResponse{Content: `{"id":"123","amount":0,"currency":"USD"}`}, nil
			}
			return &ProviderResponse{Content: `{"id":"123","amount":10,"currency":"USD"}`}, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI})
	var out testOrderWithValidate

	err := b.IntoWithRetry("extract order", &out, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Amount != 10 {
		t.Fatalf("expected amount to be fixed on retry, got %+v", out)
	}
}

func TestExtractors_CodeBlocksAndJSON(t *testing.T) {
	resp := "text\n```python\nprint('hi')\n```\nmore\n```go\nfmt.Println(\"x\")\n```"
	if got := ExtractCodeBlock(resp, "python"); !strings.Contains(got, "print('hi')") {
		t.Fatalf("unexpected python block: %q", got)
	}
	blocks := ExtractAllCodeBlocks(resp)
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d: %#v", len(blocks), blocks)
	}

	type P struct {
		Name string `json:"name"`
	}
	p, err := Parse[P]("```json\n{\"name\":\"a\"}\n```")
	if err != nil || p.Name != "a" {
		t.Fatalf("unexpected parse: %+v err=%v", p, err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected MustParse to panic on invalid json")
		}
	}()
	_ = MustParse[P]("{not json}")
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

func TestCleanJSONResponse_TrimsFences(t *testing.T) {
	got := cleanJSONResponse("```json\n{\"a\":1}\n```")
	if got != `{"a":1}` {
		t.Fatalf("unexpected clean: %q", got)
	}
}

func TestSchema_FunctionSetsJSONModeAndSchema(t *testing.T) {
	b := New(ModelGPT5)
	type X struct {
		A string `json:"a"`
	}
	b.Schema(&X{})
	if !b.jsonMode {
		t.Fatalf("expected jsonMode to be enabled by Schema()")
	}
	if b.schema == nil || reflect.ValueOf(b.schema).IsZero() {
		t.Fatalf("expected schema to be set")
	}
}
