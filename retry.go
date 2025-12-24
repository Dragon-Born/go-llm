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

// RetryConfig configures retry behavior
type RetryConfig struct {
	MaxRetries    int           // max number of retries (default: 3)
	InitialDelay  time.Duration // initial delay before first retry (default: 1s)
	MaxDelay      time.Duration // maximum delay cap (default: 60s)
	Multiplier    float64       // backoff multiplier (default: 2.0)
	Jitter        float64       // jitter factor 0-1 (default: 0.1)
	RetryOnStatus []int         // HTTP status codes to retry on
	RetryOnErrors []string      // error substrings to retry on
}

// DefaultRetryConfig returns sensible defaults
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

// RetryConfig sets a custom retry configuration on the builder
func (b *Builder) RetryConfig(config *RetryConfig) *Builder {
	b.retryConfig = config
	return b
}

// RetryWithBackoff enables smart retry with exponential backoff
func (b *Builder) RetryWithBackoff(maxRetries int) *Builder {
	b.retryConfig = DefaultRetryConfig()
	b.retryConfig.MaxRetries = maxRetries
	return b
}

// NoRetry disables retries
func (b *Builder) NoRetry() *Builder {
	b.retryConfig = nil
	b.maxRetries = 0
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry Logic
// ═══════════════════════════════════════════════════════════════════════════

// RetryInfo contains information about a retry attempt
type RetryInfo struct {
	Attempt    int
	MaxRetries int
	LastError  error
	NextDelay  time.Duration
	TotalTime  time.Duration
}

// calculateBackoff calculates the next delay with exponential backoff and jitter
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

// shouldRetry determines if we should retry based on the error
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

// RetryFunc is a function that can be retried
type RetryFunc[T any] func() (T, error)

// WithRetry executes a function with retry logic
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

// DoWithRetry executes a no-return function with retries
func DoWithRetry(ctx context.Context, config *RetryConfig, fn func() error) error {
	_, err := WithRetry(ctx, config, func() (struct{}, error) {
		return struct{}{}, fn()
	})
	return err
}

// ═══════════════════════════════════════════════════════════════════════════
// RetryConfig Builder Pattern
// ═══════════════════════════════════════════════════════════════════════════

// NewRetryConfig creates a new retry configuration with defaults
func NewRetryConfig() *RetryConfig {
	return DefaultRetryConfig()
}

// WithMaxRetries sets max retries
func (c *RetryConfig) WithMaxRetries(n int) *RetryConfig {
	c.MaxRetries = n
	return c
}

// WithInitialDelay sets initial delay
func (c *RetryConfig) WithInitialDelay(d time.Duration) *RetryConfig {
	c.InitialDelay = d
	return c
}

// WithMaxDelay sets max delay cap
func (c *RetryConfig) WithMaxDelay(d time.Duration) *RetryConfig {
	c.MaxDelay = d
	return c
}

// WithMultiplier sets backoff multiplier
func (c *RetryConfig) WithMultiplier(m float64) *RetryConfig {
	c.Multiplier = m
	return c
}

// WithJitter sets jitter factor (0-1)
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

// WithRetryOnStatus adds status codes to retry on
func (c *RetryConfig) WithRetryOnStatus(codes ...int) *RetryConfig {
	c.RetryOnStatus = append(c.RetryOnStatus, codes...)
	return c
}

// WithRetryOnError adds error substrings to retry on
func (c *RetryConfig) WithRetryOnError(errors ...string) *RetryConfig {
	c.RetryOnErrors = append(c.RetryOnErrors, errors...)
	return c
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry Presets
// ═══════════════════════════════════════════════════════════════════════════

// AggressiveRetryConfig returns a config for aggressive retrying
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

// GentleRetryConfig returns a config for gentle retrying (longer delays)
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

// NoRetryConfig returns a config that never retries
func NoRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries: 0,
	}
}
