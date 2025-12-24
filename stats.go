package ai

import (
	"fmt"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Session Stats Tracking
// ═══════════════════════════════════════════════════════════════════════════

var (
	stats     = &Stats{}
	statsLock sync.Mutex
)

// Stats holds session statistics
type Stats struct {
	Requests         int
	TotalTokens      int
	PromptTokens     int
	CompletionTokens int
	TotalLatency     time.Duration
	Errors           int
	Retries          int
	ModelUsage       map[Model]int
}

// trackRequest records a request for stats
func trackRequest(meta *ResponseMeta) {
	statsLock.Lock()
	defer statsLock.Unlock()

	stats.Requests++
	stats.TotalTokens += meta.Tokens
	stats.PromptTokens += meta.PromptTokens
	stats.CompletionTokens += meta.CompletionTokens
	stats.TotalLatency += meta.Latency
	stats.Retries += meta.Retries

	if stats.ModelUsage == nil {
		stats.ModelUsage = make(map[Model]int)
	}
	stats.ModelUsage[meta.Model]++
}

// GetStats returns current session statistics
func GetStats() Stats {
	statsLock.Lock()
	defer statsLock.Unlock()
	return *stats
}

// ResetStats clears all statistics
func ResetStats() {
	statsLock.Lock()
	defer statsLock.Unlock()
	stats = &Stats{}
}

// PrintStats displays session statistics
func PrintStats() {
	s := GetStats()

	fmt.Println()
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))
	fmt.Printf("%s Session Statistics\n", colorCyan("📊"))
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))
	fmt.Printf("  Requests:     %d\n", s.Requests)
	fmt.Printf("  Total Tokens: %d (prompt: %d, completion: %d)\n",
		s.TotalTokens, s.PromptTokens, s.CompletionTokens)
	if s.Requests > 0 {
		fmt.Printf("  Avg Latency:  %v\n", s.TotalLatency/time.Duration(s.Requests))
	}
	if s.Retries > 0 {
		fmt.Printf("  Retries:      %d\n", s.Retries)
	}
	if len(s.ModelUsage) > 0 {
		fmt.Println("  Models Used:")
		for model, count := range s.ModelUsage {
			fmt.Printf("    - %s: %d\n", model, count)
		}
	}
	fmt.Println(colorCyan("═══════════════════════════════════════════════════════════════"))
	fmt.Println()
}

// EstimateCost returns estimated cost in USD based on token usage
func EstimateCost() float64 {
	s := GetStats()
	// Use average pricing if we can't determine per-model
	inputCost := float64(s.PromptTokens) / 1_000_000 * 3.0       // $3/1M avg
	outputCost := float64(s.CompletionTokens) / 1_000_000 * 12.0 // $12/1M avg
	return inputCost + outputCost
}

// PrintCost displays estimated cost
func PrintCost() {
	cost := EstimateCost()
	s := GetStats()
	fmt.Printf("\n%s Estimated Cost: $%.4f (%d tokens)\n\n",
		colorYellow("💰"), cost, s.TotalTokens)
}
