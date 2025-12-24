package ai

import (
	"testing"
)

func TestCacheKey(t *testing.T) {
	msgs1 := []Message{{Role: "user", Content: "Hello"}}
	msgs2 := []Message{{Role: "user", Content: "Hello"}}
	msgs3 := []Message{{Role: "user", Content: "Goodbye"}}

	key1 := cacheKey(ModelGPT5, msgs1, SendOptions{})
	key2 := cacheKey(ModelGPT5, msgs2, SendOptions{})
	key3 := cacheKey(ModelGPT5, msgs3, SendOptions{})
	key4 := cacheKey(ModelClaudeOpus, msgs1, SendOptions{})

	// Same messages should produce same key
	if key1 != key2 {
		t.Error("identical messages should produce same cache key")
	}

	// Different messages should produce different key
	if key1 == key3 {
		t.Error("different messages should produce different cache key")
	}

	// Different model should produce different key
	if key1 == key4 {
		t.Error("different model should produce different cache key")
	}

	// Different options should produce different key
	temp := 0.7
	key5 := cacheKey(ModelGPT5, msgs1, SendOptions{Temperature: &temp})
	if key1 == key5 {
		t.Error("different options should produce different cache key")
	}
}

func TestGetSetCached(t *testing.T) {
	// Save original and enable cache
	originalCache := Cache
	Cache = true
	defer func() {
		Cache = originalCache
		ClearCache()
	}()

	msgs := []Message{{Role: "user", Content: "test"}}

	// Initially not cached
	_, ok := getCached(ModelGPT5, msgs, SendOptions{})
	if ok {
		t.Error("should not be cached initially")
	}

	// Set cache
	setCached(ModelGPT5, msgs, SendOptions{}, "cached response")

	// Now should be cached
	response, ok := getCached(ModelGPT5, msgs, SendOptions{})
	if !ok {
		t.Error("should be cached after setCached")
	}
	if response != "cached response" {
		t.Errorf("expected 'cached response', got %q", response)
	}
}

func TestCacheDisabled(t *testing.T) {
	// Save original
	originalCache := Cache
	Cache = false
	defer func() { Cache = originalCache }()

	msgs := []Message{{Role: "user", Content: "test"}}

	// Set should be no-op when disabled
	setCached(ModelGPT5, msgs, SendOptions{}, "response")

	// Get should return false when disabled
	_, ok := getCached(ModelGPT5, msgs, SendOptions{})
	if ok {
		t.Error("cache should not work when disabled")
	}
}

func TestClearCache(t *testing.T) {
	// Save original and enable cache
	originalCache := Cache
	Cache = true
	defer func() {
		Cache = originalCache
		ClearCache()
	}()

	msgs := []Message{{Role: "user", Content: "test"}}
	setCached(ModelGPT5, msgs, SendOptions{}, "response")

	if CacheSize() == 0 {
		t.Error("cache should have entries")
	}

	ClearCache()

	if CacheSize() != 0 {
		t.Error("cache should be empty after clear")
	}

	_, ok := getCached(ModelGPT5, msgs, SendOptions{})
	if ok {
		t.Error("cache should be empty after clear")
	}
}

func TestCacheSize(t *testing.T) {
	// Save original and enable cache
	originalCache := Cache
	Cache = true
	defer func() {
		Cache = originalCache
		ClearCache()
	}()

	ClearCache()

	if CacheSize() != 0 {
		t.Error("initial cache size should be 0")
	}

	msgs1 := []Message{{Role: "user", Content: "test1"}}
	msgs2 := []Message{{Role: "user", Content: "test2"}}

	setCached(ModelGPT5, msgs1, SendOptions{}, "response1")
	if CacheSize() != 1 {
		t.Errorf("expected cache size 1, got %d", CacheSize())
	}

	setCached(ModelGPT5, msgs2, SendOptions{}, "response2")
	if CacheSize() != 2 {
		t.Errorf("expected cache size 2, got %d", CacheSize())
	}

	// Same key shouldn't increase size
	setCached(ModelGPT5, msgs1, SendOptions{}, "updated response")
	if CacheSize() != 2 {
		t.Errorf("expected cache size 2, got %d", CacheSize())
	}
}
