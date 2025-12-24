package ai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Structured Output / Schema Enforcement
// ═══════════════════════════════════════════════════════════════════════════

// Schema sets a struct type for guaranteed structured output
// The model will be forced to return JSON matching this exact schema
func (b *Builder) Schema(v any) *Builder {
	b.schema = v
	b.jsonMode = true
	return b
}

// Into sends request and unmarshals response into target struct
// This is the main DX-friendly way to get structured output
func (b *Builder) Into(prompt string, target any) error {
	// Generate schema from target type
	schema := structToSchema(target)
	b.schema = schema

	resp, err := b.JSON().User(prompt).Send()
	if err != nil {
		return err
	}

	// Clean response (remove markdown if present)
	resp = cleanJSONResponse(resp)

	return json.Unmarshal([]byte(resp), target)
}

// AskInto is an alias for Into (for consistency with Ask pattern)
func (b *Builder) AskInto(prompt string, target any) error {
	return b.Into(prompt, target)
}

// ═══════════════════════════════════════════════════════════════════════════
// Schema Generation from Struct
// ═══════════════════════════════════════════════════════════════════════════

// structToSchema generates a JSON Schema from a Go struct
func structToSchema(v any) map[string]any {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	return typeToSchema(t)
}

func typeToSchema(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.Struct:
		return structTypeToSchema(t)
	case reflect.Slice, reflect.Array:
		return map[string]any{
			"type":  "array",
			"items": typeToSchema(t.Elem()),
		}
	case reflect.Map:
		return map[string]any{
			"type":                 "object",
			"additionalProperties": typeToSchema(t.Elem()),
		}
	case reflect.Ptr:
		return typeToSchema(t.Elem())
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	default:
		return map[string]any{"type": "string"}
	}
}

func structTypeToSchema(t reflect.Type) map[string]any {
	properties := make(map[string]any)
	required := []string{}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get JSON tag
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		name := field.Name
		isRequired := true

		if jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" {
				name = parts[0]
			}
			for _, part := range parts[1:] {
				if part == "omitempty" {
					isRequired = false
				}
			}
		}

		// Get description from `desc` tag
		desc := field.Tag.Get("desc")

		fieldSchema := typeToSchema(field.Type)
		if desc != "" {
			fieldSchema["description"] = desc
		}

		properties[name] = fieldSchema

		if isRequired {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

// cleanJSONResponse removes markdown code blocks from response
func cleanJSONResponse(resp string) string {
	resp = strings.TrimSpace(resp)
	resp = strings.TrimPrefix(resp, "```json")
	resp = strings.TrimPrefix(resp, "```")
	resp = strings.TrimSuffix(resp, "```")
	resp = strings.TrimSpace(resp)
	return resp
}

// ═══════════════════════════════════════════════════════════════════════════
// Quick Extraction Helpers
// ═══════════════════════════════════════════════════════════════════════════

// ExtractJSON extracts and parses JSON from a response string
func ExtractJSON[T any](response string) (T, error) {
	var result T
	cleaned := cleanJSONResponse(response)
	err := json.Unmarshal([]byte(cleaned), &result)
	return result, err
}

// ExtractCodeBlock extracts a code block of a specific language from response
func ExtractCodeBlock(response, language string) string {
	prefix := "```" + language
	start := strings.Index(response, prefix)
	if start == -1 {
		return ""
	}

	start += len(prefix)
	// Skip newline after prefix
	if start < len(response) && response[start] == '\n' {
		start++
	}

	end := strings.Index(response[start:], "```")
	if end == -1 {
		return response[start:]
	}

	return strings.TrimSpace(response[start : start+end])
}

// ExtractAllCodeBlocks extracts all code blocks from response
func ExtractAllCodeBlocks(response string) []string {
	var blocks []string
	remaining := response

	for {
		start := strings.Index(remaining, "```")
		if start == -1 {
			break
		}

		start += 3
		// Skip language identifier
		newline := strings.Index(remaining[start:], "\n")
		if newline != -1 {
			start += newline + 1
		}

		end := strings.Index(remaining[start:], "```")
		if end == -1 {
			break
		}

		blocks = append(blocks, strings.TrimSpace(remaining[start:start+end]))
		remaining = remaining[start+end+3:]
	}

	return blocks
}

// ═══════════════════════════════════════════════════════════════════════════
// Type-Safe Extraction with Generics
// ═══════════════════════════════════════════════════════════════════════════

// Parse parses a response into a typed struct
func Parse[T any](response string) (T, error) {
	return ExtractJSON[T](response)
}

// MustParse parses a response or panics
func MustParse[T any](response string) T {
	result, err := ExtractJSON[T](response)
	if err != nil {
		panic(fmt.Sprintf("failed to parse response: %v", err))
	}
	return result
}
