package ai

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Retry Configuration
// ═══════════════════════════════════════════════════════════════════════════

// RetryConfig defines the strategy for retrying failed requests.
// It supports exponential backoff, jitter, and selective retries based on error types.
type RetryConfig struct {
	MaxRetries    int           // Maximum number of retry attempts (default: 3)
	InitialDelay  time.Duration // Delay before the first retry (default: 1s)
	MaxDelay      time.Duration // Maximum delay between retries (default: 60s)
	Multiplier    float64       // Exponential backoff multiplier (default: 2.0)
	Jitter        float64       // Random jitter factor (0.0 - 1.0) to avoid thundering herd (default: 0.1)
	RetryOnStatus []int         // List of HTTP status codes that trigger a retry
	RetryOnErrors []string      // List of error substrings that trigger a retry
}

// DefaultRetryConfig returns a sensible default configuration for most use cases.
// It retries on common transient errors (rate limits, timeouts, server errors).
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   2.0,
		Jitter:       0.1, // 10% jitter
		RetryOnStatus: []int{
			http.StatusTooManyRequests,     // 429
			http.StatusInternalServerError, // 500
			http.StatusBadGateway,          // 502
			http.StatusServiceUnavailable,  // 503
			http.StatusGatewayTimeout,      // 504
		},
		RetryOnErrors: []string{
			"connection reset",
			"connection refused",
			"timeout",
			"temporary failure",
			"rate limit",
			"overloaded",
		},
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry Builder Methods
// ═══════════════════════════════════════════════════════════════════════════

// RetryConfig sets a custom retry configuration for the request.
// This overrides any previous retry settings.
func (b *Builder) RetryConfig(config *RetryConfig) *Builder {
	b.retryConfig = config
	return b
}

// RetryWithBackoff enables smart retry with exponential backoff using default settings.
// It allows specifying the maximum number of retries.
func (b *Builder) RetryWithBackoff(maxRetries int) *Builder {
	b.retryConfig = DefaultRetryConfig()
	b.retryConfig.MaxRetries = maxRetries
	return b
}

// NoRetry disables all retry logic for the request.
// The request will fail immediately upon any error.
func (b *Builder) NoRetry() *Builder {
	b.retryConfig = nil
	b.maxRetries = 0
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry Logic
// ═══════════════════════════════════════════════════════════════════════════

// RetryInfo contains metadata about the current state of a retry loop.
type RetryInfo struct {
	Attempt    int
	MaxRetries int
	LastError  error
	NextDelay  time.Duration
	TotalTime  time.Duration
}

// calculateBackoff computes the delay for the next retry attempt.
// It applies exponential backoff and random jitter.
func calculateBackoff(config *RetryConfig, attempt int) time.Duration {
	if config == nil {
		return time.Second
	}

	// Exponential backoff: delay = initial * (multiplier ^ attempt)
	delay := float64(config.InitialDelay) * math.Pow(config.Multiplier, float64(attempt))

	// Cap at max delay
	if delay > float64(config.MaxDelay) {
		delay = float64(config.MaxDelay)
	}

	// Add jitter: delay * (1 ± jitter)
	if config.Jitter > 0 {
		jitterRange := delay * config.Jitter
		jitter := (rand.Float64() * 2 * jitterRange) - jitterRange
		delay += jitter
	}

	// Ensure non-negative
	if delay < 0 {
		delay = float64(config.InitialDelay)
	}

	return time.Duration(delay)
}

// shouldRetry determines if an error is transient and should trigger a retry.
// It checks against configured status codes and error messages.
func shouldRetry(config *RetryConfig, err error) bool {
	if config == nil || err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Check for status code matches
	for _, status := range config.RetryOnStatus {
		if strings.Contains(errStr, fmt.Sprintf("[%d]", status)) ||
			strings.Contains(errStr, fmt.Sprintf("status %d", status)) ||
			strings.Contains(errStr, strconv.Itoa(status)) {
			return true
		}
	}

	// Check for error substring matches
	for _, substr := range config.RetryOnErrors {
		if strings.Contains(errStr, strings.ToLower(substr)) {
			return true
		}
	}

	// Check for ProviderError with retryable status
	if pe, ok := err.(*ProviderError); ok {
		if code, err := strconv.Atoi(pe.Code); err == nil {
			for _, status := range config.RetryOnStatus {
				if code == status {
					return true
				}
			}
		}
	}

	return false
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry Executor
// ═══════════════════════════════════════════════════════════════════════════

// RetryFunc is the signature for a function that can be retried.
type RetryFunc[T any] func() (T, error)

// WithRetry executes a function with automatic retries based on the configuration.
// It handles context cancellation, backoff delays, and error checking.
func WithRetry[T any](ctx context.Context, config *RetryConfig, fn RetryFunc[T]) (T, error) {
	var zero T
	if config == nil {
		config = DefaultRetryConfig()
	}

	var lastErr error
	start := time.Now()

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		// Check context before attempting
		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		default:
		}

		result, err := fn()
		if err == nil {
			return result, nil
		}

		lastErr = err

		// Check if we should retry
		if !shouldRetry(config, err) {
			if Debug {
				fmt.Printf("%s Not retrying: error not retryable\n", colorRed("✗"))
			}
			return zero, err
		}

		// Don't retry if we've exhausted attempts
		if attempt >= config.MaxRetries {
			break
		}

		// Calculate delay
		delay := calculateBackoff(config, attempt)

		if Debug {
			fmt.Printf("%s Retry %d/%d after %v (error: %v)\n",
				colorYellow("↻"), attempt+1, config.MaxRetries, delay.Round(time.Millisecond), err)
		}

		// Wait with context awareness
		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-time.After(delay):
		}
	}

	if Debug {
		fmt.Printf("%s All %d retries exhausted after %v\n",
			colorRed("✗"), config.MaxRetries, time.Since(start).Round(time.Millisecond))
	}

	return zero, fmt.Errorf("max retries (%d) exceeded: %w", config.MaxRetries, lastErr)
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenient Retry Wrappers
// ═══════════════════════════════════════════════════════════════════════════

// DoWithRetry executes a function that returns only an error, with retry logic.
func DoWithRetry(ctx context.Context, config *RetryConfig, fn func() error) error {
	_, err := WithRetry(ctx, config, func() (struct{}, error) {
		return struct{}{}, fn()
	})
	return err
}

// ═══════════════════════════════════════════════════════════════════════════
// RetryConfig Builder Pattern
// ═══════════════════════════════════════════════════════════════════════════

// NewRetryConfig creates a new retry configuration initialized with defaults.
func NewRetryConfig() *RetryConfig {
	return DefaultRetryConfig()
}

// WithMaxRetries sets the maximum number of retry attempts.
func (c *RetryConfig) WithMaxRetries(n int) *RetryConfig {
	c.MaxRetries = n
	return c
}

// WithInitialDelay sets the delay before the first retry.
func (c *RetryConfig) WithInitialDelay(d time.Duration) *RetryConfig {
	c.InitialDelay = d
	return c
}

// WithMaxDelay sets the maximum delay cap between retries.
func (c *RetryConfig) WithMaxDelay(d time.Duration) *RetryConfig {
	c.MaxDelay = d
	return c
}

// WithMultiplier sets the exponential backoff multiplier.
func (c *RetryConfig) WithMultiplier(m float64) *RetryConfig {
	c.Multiplier = m
	return c
}

// WithJitter sets the jitter factor (0.0 to 1.0) to randomize delays.
func (c *RetryConfig) WithJitter(j float64) *RetryConfig {
	if j < 0 {
		j = 0
	}
	if j > 1 {
		j = 1
	}
	c.Jitter = j
	return c
}

// WithRetryOnStatus adds HTTP status codes that should trigger a retry.
func (c *RetryConfig) WithRetryOnStatus(codes ...int) *RetryConfig {
	c.RetryOnStatus = append(c.RetryOnStatus, codes...)
	return c
}

// WithRetryOnError adds error message substrings that should trigger a retry.
func (c *RetryConfig) WithRetryOnError(errors ...string) *RetryConfig {
	c.RetryOnErrors = append(c.RetryOnErrors, errors...)
	return c
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry Presets
// ═══════════════════════════════════════════════════════════════════════════

// AggressiveRetryConfig returns a configuration for aggressive retrying.
// Short delays, high multiplier, suitable for high-throughput scenarios where speed matters.
func AggressiveRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:   5,
		InitialDelay: 500 * time.Millisecond,
		MaxDelay:     30 * time.Second,
		Multiplier:   1.5,
		Jitter:       0.2,
		RetryOnStatus: []int{
			http.StatusTooManyRequests,
			http.StatusInternalServerError,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout,
		},
		RetryOnErrors: []string{
			"connection reset",
			"connection refused",
			"timeout",
			"temporary failure",
			"rate limit",
			"overloaded",
			"ECONNRESET",
		},
	}
}

// GentleRetryConfig returns a configuration for gentle retrying.
// Longer delays, suitable for background tasks or strictly rate-limited APIs.
func GentleRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:   3,
		InitialDelay: 2 * time.Second,
		MaxDelay:     120 * time.Second,
		Multiplier:   3.0,
		Jitter:       0.15,
		RetryOnStatus: []int{
			http.StatusTooManyRequests,
			http.StatusServiceUnavailable,
		},
		RetryOnErrors: []string{
			"rate limit",
			"overloaded",
		},
	}
}

// NoRetryConfig returns a configuration that disables retries.
func NoRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries: 0,
	}
}
