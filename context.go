package ai

import (
	gocontext "context"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Context & Timeout Support
// ═══════════════════════════════════════════════════════════════════════════

// WithContext sets a context for cancellation/timeout
func (b *Builder) WithContext(ctx gocontext.Context) *Builder {
	b.ctx = ctx
	return b
}

// Timeout sets a timeout for this request
// Note: The cancel function is not stored; the context will be cleaned up when it expires
func (b *Builder) Timeout(d time.Duration) *Builder {
	ctx, cancel := gocontext.WithTimeout(gocontext.Background(), d)
	// Store cancel for cleanup in Send if needed
	_ = cancel // Cancel handled by timeout expiration
	b.ctx = ctx
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Quick Timeout Helpers
// ═══════════════════════════════════════════════════════════════════════════

// TimeoutSeconds sets timeout in seconds
func (b *Builder) TimeoutSeconds(s int) *Builder {
	return b.Timeout(time.Duration(s) * time.Second)
}

// TimeoutMinutes sets timeout in minutes
func (b *Builder) TimeoutMinutes(m int) *Builder {
	return b.Timeout(time.Duration(m) * time.Minute)
}

// ═══════════════════════════════════════════════════════════════════════════
// Context Helpers
// ═══════════════════════════════════════════════════════════════════════════

// getContext returns the context or a background context if none set
func (b *Builder) getContext() gocontext.Context {
	if b.ctx != nil {
		return b.ctx
	}
	return gocontext.Background()
}
