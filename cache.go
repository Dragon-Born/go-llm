package ai

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════
// Response Caching
// ═══════════════════════════════════════════════════════════════════════════

var (
	cache     = make(map[string]string)
	cacheLock sync.RWMutex
)

// CacheKey generates a unique key for a request
func cacheKey(model Model, messages []Message, opts SendOptions) string {
	data, _ := json.Marshal(struct {
		Model    string    `json:"m"`
		Messages []Message `json:"msgs"`
		Temp     *float64  `json:"t,omitempty"`
		Thinking string    `json:"r,omitempty"`
	}{
		Model:    string(model),
		Messages: messages,
		Temp:     opts.Temperature,
		Thinking: string(opts.Thinking),
	})

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// GetCached returns cached response if available
func getCached(model Model, messages []Message, opts SendOptions) (string, bool) {
	if !Cache {
		return "", false
	}

	cacheLock.RLock()
	defer cacheLock.RUnlock()

	key := cacheKey(model, messages, opts)
	resp, ok := cache[key]
	return resp, ok
}

// SetCached stores a response in cache
func setCached(model Model, messages []Message, opts SendOptions, response string) {
	if !Cache {
		return
	}

	cacheLock.Lock()
	defer cacheLock.Unlock()

	key := cacheKey(model, messages, opts)
	cache[key] = response
}

// ClearCache empties the cache
func ClearCache() {
	cacheLock.Lock()
	defer cacheLock.Unlock()
	cache = make(map[string]string)
}

// CacheSize returns number of cached responses
func CacheSize() int {
	cacheLock.RLock()
	defer cacheLock.RUnlock()
	return len(cache)
}
