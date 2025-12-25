package ai

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// ═══════════════════════════════════════════════════════════════════════════
// Validator Interface
// ═══════════════════════════════════════════════════════════════════════════

// Validator validates a response and returns an error if validation fails.
type Validator interface {
	Validate(content string) error
	Name() string
}

// ValidatorFunc is an adapter to allow using a function as a Validator.
type ValidatorFunc struct {
	name string
	fn   func(string) error
}

// Validate validates content using the underlying function.
func (v ValidatorFunc) Validate(content string) error { return v.fn(content) }

// Name returns the validator name.
func (v ValidatorFunc) Name() string { return v.name }

// ═══════════════════════════════════════════════════════════════════════════
// Validation Error
// ═══════════════════════════════════════════════════════════════════════════

// ValidationError is returned when validation fails.
type ValidationError struct {
	Validator string
	Message   string
	Content   string // the content that failed validation
}

// Error implements the error interface.
func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation failed [%s]: %s", e.Validator, e.Message)
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Validation
// ═══════════════════════════════════════════════════════════════════════════

// Validate adds a custom validator.
func (b *Builder) Validate(v Validator) *Builder {
	b.validators = append(b.validators, v)
	return b
}

// ValidateWith adds a custom validation function.
func (b *Builder) ValidateWith(name string, fn func(string) error) *Builder {
	b.validators = append(b.validators, ValidatorFunc{name: name, fn: fn})
	return b
}

// MaxLength validates that response length doesn't exceed chars.
func (b *Builder) MaxLength(chars int) *Builder {
	return b.Validate(&maxLengthValidator{maxChars: chars})
}

// MinLength validates that response length meets the minimum chars.
func (b *Builder) MinLength(chars int) *Builder {
	return b.Validate(&minLengthValidator{minChars: chars})
}

// MustContain validates that response contains substr.
func (b *Builder) MustContain(substr string) *Builder {
	return b.Validate(&containsValidator{substr: substr, required: true})
}

// MustNotContain validates that response does not contain substr.
func (b *Builder) MustNotContain(substr string) *Builder {
	return b.Validate(&containsValidator{substr: substr, required: false})
}

// MustMatch validates that response matches the regex pattern.
func (b *Builder) MustMatch(pattern string) *Builder {
	re, err := regexp.Compile(pattern)
	if err != nil {
		fmt.Printf("%s Invalid regex pattern: %v\n", colorRed("✗"), err)
		return b
	}
	return b.Validate(&regexValidator{pattern: re, mustMatch: true})
}

// MustNotMatch validates that response does not match the regex pattern.
func (b *Builder) MustNotMatch(pattern string) *Builder {
	re, err := regexp.Compile(pattern)
	if err != nil {
		fmt.Printf("%s Invalid regex pattern: %v\n", colorRed("✗"), err)
		return b
	}
	return b.Validate(&regexValidator{pattern: re, mustMatch: false})
}

// MustBeJSON validates that response is valid JSON.
func (b *Builder) MustBeJSON() *Builder {
	return b.Validate(&jsonValidator{})
}

// MustBeJSONSchema validates that response can be unmarshaled into schema.
func (b *Builder) MustBeJSONSchema(schema any) *Builder {
	return b.Validate(&jsonSchemaValidator{schema: schema})
}

// NoEmptyResponse validates that response is not empty.
func (b *Builder) NoEmptyResponse() *Builder {
	return b.Validate(&nonEmptyValidator{})
}

// WordCount validates that response word count is within [min, max].
func (b *Builder) WordCount(min, max int) *Builder {
	return b.Validate(&wordCountValidator{minWords: min, maxWords: max})
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Validators
// ═══════════════════════════════════════════════════════════════════════════

// maxLengthValidator ensures content doesn't exceed length
type maxLengthValidator struct {
	maxChars int
}

func (v *maxLengthValidator) Name() string { return "MaxLength" }
func (v *maxLengthValidator) Validate(content string) error {
	length := utf8.RuneCountInString(content)
	if length > v.maxChars {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response length %d exceeds maximum %d", length, v.maxChars),
			Content:   content,
		}
	}
	return nil
}

// minLengthValidator ensures content meets minimum length
type minLengthValidator struct {
	minChars int
}

func (v *minLengthValidator) Name() string { return "MinLength" }
func (v *minLengthValidator) Validate(content string) error {
	length := utf8.RuneCountInString(content)
	if length < v.minChars {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response length %d below minimum %d", length, v.minChars),
			Content:   content,
		}
	}
	return nil
}

// containsValidator checks for substring presence
type containsValidator struct {
	substr   string
	required bool
}

func (v *containsValidator) Name() string {
	if v.required {
		return "MustContain"
	}
	return "MustNotContain"
}

func (v *containsValidator) Validate(content string) error {
	contains := strings.Contains(strings.ToLower(content), strings.ToLower(v.substr))
	if v.required && !contains {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response must contain %q", v.substr),
			Content:   content,
		}
	}
	if !v.required && contains {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response must not contain %q", v.substr),
			Content:   content,
		}
	}
	return nil
}

// regexValidator validates against regex pattern
type regexValidator struct {
	pattern   *regexp.Regexp
	mustMatch bool
}

func (v *regexValidator) Name() string {
	if v.mustMatch {
		return "MustMatch"
	}
	return "MustNotMatch"
}

func (v *regexValidator) Validate(content string) error {
	matches := v.pattern.MatchString(content)
	if v.mustMatch && !matches {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response must match pattern %q", v.pattern.String()),
			Content:   content,
		}
	}
	if !v.mustMatch && matches {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response must not match pattern %q", v.pattern.String()),
			Content:   content,
		}
	}
	return nil
}

// jsonValidator ensures content is valid JSON
type jsonValidator struct{}

func (v *jsonValidator) Name() string { return "MustBeJSON" }
func (v *jsonValidator) Validate(content string) error {
	// Clean markdown if present
	cleaned := cleanJSONResponse(content)
	if !json.Valid([]byte(cleaned)) {
		return &ValidationError{
			Validator: v.Name(),
			Message:   "response is not valid JSON",
			Content:   content,
		}
	}
	return nil
}

// jsonSchemaValidator validates JSON against a schema (struct type)
type jsonSchemaValidator struct {
	schema any
}

func (v *jsonSchemaValidator) Name() string { return "MustBeJSONSchema" }
func (v *jsonSchemaValidator) Validate(content string) error {
	cleaned := cleanJSONResponse(content)
	if err := json.Unmarshal([]byte(cleaned), v.schema); err != nil {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("response doesn't match schema: %v", err),
			Content:   content,
		}
	}
	return nil
}

// nonEmptyValidator ensures content is not empty
type nonEmptyValidator struct{}

func (v *nonEmptyValidator) Name() string { return "NoEmptyResponse" }
func (v *nonEmptyValidator) Validate(content string) error {
	if strings.TrimSpace(content) == "" {
		return &ValidationError{
			Validator: v.Name(),
			Message:   "response is empty",
			Content:   content,
		}
	}
	return nil
}

// wordCountValidator validates word count range
type wordCountValidator struct {
	minWords int
	maxWords int
}

func (v *wordCountValidator) Name() string { return "WordCount" }
func (v *wordCountValidator) Validate(content string) error {
	words := len(strings.Fields(content))
	if words < v.minWords {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("word count %d below minimum %d", words, v.minWords),
			Content:   content,
		}
	}
	if v.maxWords > 0 && words > v.maxWords {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("word count %d exceeds maximum %d", words, v.maxWords),
			Content:   content,
		}
	}
	return nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Validator Execution
// ═══════════════════════════════════════════════════════════════════════════

// runValidators runs all validators on the content.
// It also applies content filters (WithFilter) in-place, so filters can transform output.
func (b *Builder) runValidators(content string) (string, error) {
	for _, v := range b.validators {
		// Content filters can transform the content
		if fv, ok := v.(*filterValidator); ok {
			newContent, err := fv.filter(content)
			if err != nil {
				if Debug {
					fmt.Printf("%s Validation failed [%s]: %v\n", colorRed("✗"), v.Name(), err)
				}
				return content, err
			}
			content = newContent
			if Debug {
				fmt.Printf("%s Validation passed [%s]\n", colorGreen("✓"), v.Name())
			}
			continue
		}

		if err := v.Validate(content); err != nil {
			if Debug {
				fmt.Printf("%s Validation failed [%s]: %v\n", colorRed("✗"), v.Name(), err)
			}
			return content, err
		}
		if Debug {
			fmt.Printf("%s Validation passed [%s]\n", colorGreen("✓"), v.Name())
		}
	}
	return content, nil
}

// ClearValidators removes all validators
func (b *Builder) ClearValidators() *Builder {
	b.validators = nil
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Guardrail Presets
// ═══════════════════════════════════════════════════════════════════════════

// SafeContent adds common content safety validators
func (b *Builder) SafeContent() *Builder {
	return b.NoEmptyResponse().
		MustNotContain("<script").
		MustNotContain("javascript:").
		MustNotContain("data:text/html")
}

// ConciseResponse limits response to reasonable length
func (b *Builder) ConciseResponse(maxWords int) *Builder {
	if maxWords <= 0 {
		maxWords = 500
	}
	return b.NoEmptyResponse().WordCount(1, maxWords)
}

// StrictJSON ensures valid JSON output
func (b *Builder) StrictJSON() *Builder {
	return b.JSON().MustBeJSON()
}

// ═══════════════════════════════════════════════════════════════════════════
// Composite Validators
// ═══════════════════════════════════════════════════════════════════════════

// CompositeValidator combines multiple validators
type CompositeValidator struct {
	name       string
	validators []Validator
	mode       string // "all" or "any"
}

// AllOf creates a validator that passes if all sub-validators pass
func AllOf(name string, validators ...Validator) *CompositeValidator {
	return &CompositeValidator{name: name, validators: validators, mode: "all"}
}

// AnyOf creates a validator that passes if any sub-validator passes
func AnyOf(name string, validators ...Validator) *CompositeValidator {
	return &CompositeValidator{name: name, validators: validators, mode: "any"}
}

func (v *CompositeValidator) Name() string { return v.name }

func (v *CompositeValidator) Validate(content string) error {
	var errors []error

	for _, sub := range v.validators {
		err := sub.Validate(content)
		if v.mode == "all" && err != nil {
			return err
		}
		if v.mode == "any" && err == nil {
			return nil
		}
		if err != nil {
			errors = append(errors, err)
		}
	}

	if v.mode == "any" && len(errors) == len(v.validators) {
		return &ValidationError{
			Validator: v.Name(),
			Message:   fmt.Sprintf("none of %d validators passed", len(v.validators)),
			Content:   content,
		}
	}

	return nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Content Filters
// ═══════════════════════════════════════════════════════════════════════════

// ContentFilter filters or transforms content after validation
type ContentFilter func(content string) (string, error)

// WithFilter applies a content filter after successful response
func (b *Builder) WithFilter(filter ContentFilter) *Builder {
	b.validators = append(b.validators, &filterValidator{filter: filter})
	return b
}

type filterValidator struct {
	filter ContentFilter
}

func (v *filterValidator) Name() string { return "ContentFilter" }
func (v *filterValidator) Validate(content string) error {
	_, err := v.filter(content)
	return err
}

// Common filters

// TrimFilter trims whitespace
func TrimFilter() ContentFilter {
	return func(content string) (string, error) {
		return strings.TrimSpace(content), nil
	}
}

// MaxLengthFilter truncates content to max length
func MaxLengthFilter(maxChars int) ContentFilter {
	return func(content string) (string, error) {
		runes := []rune(content)
		if len(runes) > maxChars {
			return string(runes[:maxChars]) + "...", nil
		}
		return content, nil
	}
}
