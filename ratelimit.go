package ai

import (
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Rate Limiting
// ═══════════════════════════════════════════════════════════════════════════

// RateLimiter optionally limits request throughput for the whole package.
// Set it to a Limiter implementation (for example, NewLimiter(60, time.Minute)).
var RateLimiter Limiter

// Limiter is the interface implemented by rate limiters.
type Limiter interface {
	Wait()       // Block until request is allowed
	Allow() bool // Check if request is allowed without blocking
}

// ═══════════════════════════════════════════════════════════════════════════
// Token Bucket Rate Limiter
// ═══════════════════════════════════════════════════════════════════════════

// TokenBucket implements a token bucket rate limiter.
type TokenBucket struct {
	rate       float64   // tokens per second
	capacity   float64   // max tokens
	tokens     float64   // current tokens
	lastRefill time.Time // last refill time
	mu         sync.Mutex
}

// NewLimiter creates a rate limiter with specified requests per interval.
// Example: NewLimiter(60, time.Minute) = 60 requests per minute
func NewLimiter(requests int, interval time.Duration) *TokenBucket {
	rate := float64(requests) / interval.Seconds()
	return &TokenBucket{
		rate:       rate,
		capacity:   float64(requests),
		tokens:     float64(requests),
		lastRefill: time.Now(),
	}
}

// NewLimiterPerSecond creates a rate limiter with requests per second.
func NewLimiterPerSecond(rps float64) *TokenBucket {
	return &TokenBucket{
		rate:       rps,
		capacity:   rps * 2, // Allow small bursts
		tokens:     rps * 2,
		lastRefill: time.Now(),
	}
}

// Wait blocks until a request is allowed.
func (tb *TokenBucket) Wait() {
	for {
		if tb.Allow() {
			return
		}
		// Calculate time until next token
		tb.mu.Lock()
		waitTime := time.Duration((1.0 / tb.rate) * float64(time.Second))
		tb.mu.Unlock()
		time.Sleep(waitTime)
	}
}

// Allow checks if a request is allowed (non-blocking).
func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	// Refill tokens based on time elapsed
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens += elapsed * tb.rate
	if tb.tokens > tb.capacity {
		tb.tokens = tb.capacity
	}
	tb.lastRefill = now

	// Try to consume a token
	if tb.tokens >= 1.0 {
		tb.tokens -= 1.0
		return true
	}
	return false
}

// ═══════════════════════════════════════════════════════════════════════════
// Concurrency Limiter
// ═══════════════════════════════════════════════════════════════════════════

// ConcurrencyLimiter limits concurrent requests.
type ConcurrencyLimiter struct {
	sem chan struct{}
}

// NewConcurrencyLimiter creates a limiter for max concurrent requests.
func NewConcurrencyLimiter(maxConcurrent int) *ConcurrencyLimiter {
	return &ConcurrencyLimiter{
		sem: make(chan struct{}, maxConcurrent),
	}
}

// Acquire blocks until a slot is available.
func (cl *ConcurrencyLimiter) Acquire() {
	cl.sem <- struct{}{}
}

// Release returns a slot.
func (cl *ConcurrencyLimiter) Release() {
	<-cl.sem
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Integration
// ═══════════════════════════════════════════════════════════════════════════

// waitForRateLimit waits if a global rate limiter is set
func waitForRateLimit() {
	if RateLimiter != nil {
		RateLimiter.Wait()
	}
}
