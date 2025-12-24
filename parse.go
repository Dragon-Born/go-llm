package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Structured Output with Retry-on-Parse-Error (Instructor-style)
// ═══════════════════════════════════════════════════════════════════════════

// ParseConfig configures structured output parsing with retries
type ParseConfig struct {
	MaxRetries     int           // max parse retry attempts (default: 3)
	ValidateOutput bool          // run struct validators (default: true)
	IncludeSchema  bool          // include schema hint in prompt (default: true)
	StrictMode     bool          // require exact field matches (default: false)
	Timeout        time.Duration // timeout per attempt
}

// DefaultParseConfig returns sensible defaults
func DefaultParseConfig() *ParseConfig {
	return &ParseConfig{
		MaxRetries:     3,
		ValidateOutput: true,
		IncludeSchema:  true,
		StrictMode:     false,
	}
}

// ParseError contains details about a parse failure
type ParseError struct {
	Attempt       int
	RawResponse   string
	ParseErr      error
	ValidationErr error
}

func (e *ParseError) Error() string {
	if e.ValidationErr != nil {
		return fmt.Sprintf("validation failed on attempt %d: %v", e.Attempt, e.ValidationErr)
	}
	return fmt.Sprintf("parse failed on attempt %d: %v", e.Attempt, e.ParseErr)
}

// ParseResult contains the parsed output and metadata
type ParseResult[T any] struct {
	Value       T
	RawResponse string
	Attempts    int
	Duration    time.Duration
	Tokens      int
	Error       error
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Parse with Retry
// ═══════════════════════════════════════════════════════════════════════════

// IntoWithRetry parses response into target struct, retrying on parse errors.
// On failure, it tells the model what went wrong and asks for correction.
func (b *Builder) IntoWithRetry(prompt string, target any, maxRetries int) error {
	return parseIntoAny(b, prompt, target, &ParseConfig{
		MaxRetries:     maxRetries,
		ValidateOutput: true,
		IncludeSchema:  true,
	})
}

// ParseIntoWithConfig parses with full configuration control
func (b *Builder) ParseIntoWithConfig(prompt string, target any, config *ParseConfig) error {
	return parseIntoAny(b, prompt, target, config)
}

// MustInto parses into target or panics (useful for scripts/tests)
func (b *Builder) MustInto(prompt string, target any) {
	if err := b.IntoWithRetry(prompt, target, 3); err != nil {
		panic(fmt.Sprintf("parse failed: %v", err))
	}
}

// parseIntoAny is the non-generic version for builder methods
func parseIntoAny(b *Builder, prompt string, target any, config *ParseConfig) error {
	if config == nil {
		config = DefaultParseConfig()
	}

	// Generate schema from target type
	schema := structToSchema(target)
	schemaJSON, _ := json.MarshalIndent(schema, "", "  ")

	// Build the initial prompt with schema hint
	fullPrompt := prompt
	if config.IncludeSchema {
		fullPrompt = fmt.Sprintf(`%s

Respond with valid JSON matching this exact schema:
%s

Important:
- Output ONLY valid JSON, no markdown, no explanation
- Include ALL required fields
- Use correct data types`, prompt, string(schemaJSON))
	}

	// Clone builder and enable JSON mode
	builder := b.Clone().JSON()

	// Setup context
	ctx := b.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, config.Timeout*time.Duration(config.MaxRetries+1))
		defer cancel()
	}
	builder = builder.WithContext(ctx)

	var lastError error
	messages := []Message{}

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		// Check context
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Build messages for this attempt
		if attempt == 0 {
			messages = append(messages, Message{Role: "user", Content: fullPrompt})
		}

		// Clone and set messages
		attemptBuilder := builder.Clone()
		attemptBuilder.messages = messages

		// Send request
		meta := attemptBuilder.SendWithMeta()
		if meta.Error != nil {
			lastError = meta.Error
			continue
		}

		// Clean and parse response
		cleaned := cleanJSONResponse(meta.Content)
		parseErr := json.Unmarshal([]byte(cleaned), target)

		if parseErr == nil {
			// Parse successful! Now validate if configured
			if config.ValidateOutput {
				if validErr := validateStruct(target); validErr != nil {
					lastError = &ParseError{
						Attempt:       attempt + 1,
						RawResponse:   meta.Content,
						ValidationErr: validErr,
					}
					messages = appendCorrectionMessages(messages, meta.Content, validErr)
					continue
				}
			}
			// Success!
			return nil
		}

		// Parse failed - prepare retry with error feedback
		lastError = &ParseError{
			Attempt:     attempt + 1,
			RawResponse: meta.Content,
			ParseErr:    parseErr,
		}

		if attempt < config.MaxRetries {
			messages = appendCorrectionMessages(messages, meta.Content, parseErr)
			if Debug {
				fmt.Printf("%s Parse attempt %d/%d failed: %v\n",
					colorYellow("↻"), attempt+1, config.MaxRetries+1, parseErr)
			}
		}
	}

	return fmt.Errorf("failed after %d attempts: %w", config.MaxRetries+1, lastError)
}

// ═══════════════════════════════════════════════════════════════════════════
// Core Parse Function with Retry Logic
// ═══════════════════════════════════════════════════════════════════════════

// ParseInto attempts to parse LLM response into target struct with retries
func ParseInto[T any](b *Builder, prompt string, target *T, config *ParseConfig) ParseResult[T] {
	if config == nil {
		config = DefaultParseConfig()
	}

	start := time.Now()
	result := ParseResult[T]{
		Attempts: 0,
	}

	// Generate schema from target type
	schema := structToSchema(target)
	schemaJSON, _ := json.MarshalIndent(schema, "", "  ")

	// Build the initial prompt with schema hint
	fullPrompt := prompt
	if config.IncludeSchema {
		fullPrompt = fmt.Sprintf(`%s

Respond with valid JSON matching this exact schema:
%s

Important:
- Output ONLY valid JSON, no markdown, no explanation
- Include ALL required fields
- Use correct data types`, prompt, string(schemaJSON))
	}

	// Clone builder and enable JSON mode
	builder := b.Clone().JSON()

	// Setup context
	ctx := b.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, config.Timeout*time.Duration(config.MaxRetries+1))
		defer cancel()
	}
	builder = builder.WithContext(ctx)

	var lastError error
	messages := []Message{}

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		result.Attempts = attempt + 1

		// Check context
		select {
		case <-ctx.Done():
			result.Error = ctx.Err()
			result.Duration = time.Since(start)
			return result
		default:
		}

		// Build messages for this attempt
		if attempt == 0 {
			messages = append(messages, Message{Role: "user", Content: fullPrompt})
		}

		// Clone and set messages
		attemptBuilder := builder.Clone()
		attemptBuilder.messages = messages

		// Send request
		meta := attemptBuilder.SendWithMeta()
		if meta.Error != nil {
			lastError = meta.Error
			continue
		}

		result.RawResponse = meta.Content
		result.Tokens += meta.Tokens

		// Clean and parse response
		cleaned := cleanJSONResponse(meta.Content)
		parseErr := json.Unmarshal([]byte(cleaned), target)

		if parseErr == nil {
			// Parse successful! Now validate if configured
			if config.ValidateOutput {
				if validErr := validateStruct(target); validErr != nil {
					lastError = &ParseError{
						Attempt:       attempt + 1,
						RawResponse:   meta.Content,
						ValidationErr: validErr,
					}
					// Add correction message for retry
					messages = appendCorrectionMessages(messages, meta.Content, validErr)
					continue
				}
			}

			// Success!
			result.Value = *target
			result.Duration = time.Since(start)
			return result
		}

		// Parse failed - prepare retry with error feedback
		lastError = &ParseError{
			Attempt:     attempt + 1,
			RawResponse: meta.Content,
			ParseErr:    parseErr,
		}

		if attempt < config.MaxRetries {
			// Add the failed response and correction request
			messages = appendCorrectionMessages(messages, meta.Content, parseErr)

			if Debug {
				fmt.Printf("%s Parse attempt %d/%d failed: %v\n",
					colorYellow("↻"), attempt+1, config.MaxRetries+1, parseErr)
			}
		}
	}

	result.Error = fmt.Errorf("failed after %d attempts: %w", result.Attempts, lastError)
	result.Duration = time.Since(start)
	return result
}

// appendCorrectionMessages adds error feedback for the model to correct
func appendCorrectionMessages(messages []Message, response string, err error) []Message {
	// Add the failed response as assistant message
	messages = append(messages, Message{
		Role:    "assistant",
		Content: response,
	})

	// Add correction request
	correctionPrompt := fmt.Sprintf(`Your previous response had an error:
%v

Please fix the JSON and try again. Remember:
- Output ONLY valid JSON
- No markdown code blocks
- Include all required fields with correct types`, err)

	messages = append(messages, Message{
		Role:    "user",
		Content: correctionPrompt,
	})

	return messages
}

// ═══════════════════════════════════════════════════════════════════════════
// Struct Validation
// ═══════════════════════════════════════════════════════════════════════════

// StructValidator can be implemented by structs for custom validation
type StructValidator interface {
	Validate() error
}

// validateStruct runs validation on parsed struct
func validateStruct(v any) error {
	// Check if struct implements Validate()
	if validator, ok := v.(StructValidator); ok {
		return validator.Validate()
	}

	// Check pointer to struct
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Ptr && !rv.IsNil() {
		if validator, ok := rv.Interface().(StructValidator); ok {
			return validator.Validate()
		}
	}

	return nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Extractors with Retry
// ═══════════════════════════════════════════════════════════════════════════

// Extract parses response into a new struct of type T with retries
func Extract[T any](b *Builder, prompt string) (T, error) {
	var result T
	parseResult := ParseInto(b, prompt, &result, DefaultParseConfig())
	return parseResult.Value, parseResult.Error
}

// ExtractWithRetry parses with specified retry count
func ExtractWithRetry[T any](b *Builder, prompt string, maxRetries int) (T, error) {
	var result T
	config := DefaultParseConfig()
	config.MaxRetries = maxRetries
	parseResult := ParseInto(b, prompt, &result, config)
	return parseResult.Value, parseResult.Error
}

// ExtractList parses a list of items with retries
func ExtractList[T any](b *Builder, prompt string) ([]T, error) {
	config := DefaultParseConfig()
	// Wrap in a container for proper schema generation
	type listWrapper struct {
		Items []T `json:"items"`
	}
	var wrapper listWrapper

	fullPrompt := fmt.Sprintf(`%s

Respond with JSON in this format:
{"items": [...]}`, prompt)

	parseResult := ParseInto(b, fullPrompt, &wrapper, config)
	if parseResult.Error != nil {
		return nil, parseResult.Error
	}
	return parseResult.Value.Items, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Field Validators (via struct tags)
// ═══════════════════════════════════════════════════════════════════════════

// Common validation errors
var (
	ErrFieldRequired = fmt.Errorf("field is required")
	ErrFieldTooShort = fmt.Errorf("field value too short")
	ErrFieldTooLong  = fmt.Errorf("field value too long")
	ErrFieldInvalid  = fmt.Errorf("field value invalid")
)

// ValidateRequired checks that required fields are not empty
func ValidateRequired(v any, fields ...string) error {
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Ptr {
		rv = rv.Elem()
	}
	if rv.Kind() != reflect.Struct {
		return nil
	}

	for _, fieldName := range fields {
		field := rv.FieldByName(fieldName)
		if !field.IsValid() {
			continue
		}
		if isZero(field) {
			return fmt.Errorf("field %q is required but empty", fieldName)
		}
	}
	return nil
}

// ValidateStringLength checks string field lengths
func ValidateStringLength(s string, min, max int) error {
	length := len(s)
	if min > 0 && length < min {
		return fmt.Errorf("string length %d is below minimum %d", length, min)
	}
	if max > 0 && length > max {
		return fmt.Errorf("string length %d exceeds maximum %d", length, max)
	}
	return nil
}

// ValidateOneOf checks if value is in allowed list
func ValidateOneOf[T comparable](value T, allowed ...T) error {
	for _, a := range allowed {
		if value == a {
			return nil
		}
	}
	return fmt.Errorf("value must be one of %v", allowed)
}

func isZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.String:
		return v.String() == ""
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Slice, reflect.Map:
		return v.Len() == 0
	case reflect.Ptr, reflect.Interface:
		return v.IsNil()
	default:
		return false
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Quick Parse Helpers
// ═══════════════════════════════════════════════════════════════════════════

// QuickParse is a shorthand for common extraction patterns
func QuickParse[T any](model Model, prompt string) (T, error) {
	return Extract[T](New(model).JSON(), prompt)
}

// Classify extracts a classification with confidence
type Classification struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	Reasoning  string  `json:"reasoning,omitempty"`
}

// Classify classifies text into one of the given labels
func Classify(b *Builder, text string, labels []string) (Classification, error) {
	prompt := fmt.Sprintf(`Classify the following text into exactly one of these labels: %s

Text: %s

Respond with JSON containing:
- label: the chosen label (must be one of the given options)
- confidence: your confidence from 0.0 to 1.0
- reasoning: brief explanation`, strings.Join(labels, ", "), text)

	return Extract[Classification](b, prompt)
}

// Entity represents an extracted entity
type Entity struct {
	Name  string `json:"name"`
	Type  string `json:"type"`
	Value string `json:"value,omitempty"`
}

// ExtractEntities extracts named entities from text
func ExtractEntities(b *Builder, text string, entityTypes []string) ([]Entity, error) {
	prompt := fmt.Sprintf(`Extract entities of these types from the text: %s

Text: %s

Return a JSON array of entities with name, type, and optional value.`,
		strings.Join(entityTypes, ", "), text)

	return ExtractList[Entity](b, prompt)
}

// Sentiment represents sentiment analysis result
type Sentiment struct {
	Label    string   `json:"label"` // positive, negative, neutral
	Score    float64  `json:"score"` // -1.0 to 1.0
	Emotions []string `json:"emotions,omitempty"`
}

// AnalyzeSentiment performs sentiment analysis on text
func AnalyzeSentiment(b *Builder, text string) (Sentiment, error) {
	prompt := fmt.Sprintf(`Analyze the sentiment of this text:

"%s"

Respond with:
- label: "positive", "negative", or "neutral"
- score: from -1.0 (very negative) to 1.0 (very positive)
- emotions: array of detected emotions (optional)`, text)

	return Extract[Sentiment](b, prompt)
}
