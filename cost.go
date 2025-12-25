package ai

import (
	"fmt"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════
// Cost Tracking
// ═══════════════════════════════════════════════════════════════════════════

// ModelPricing contains pricing per 1M tokens for a model.
type ModelPricing struct {
	InputPerMillion  float64 // USD per 1M input tokens
	OutputPerMillion float64 // USD per 1M output tokens
}

// ModelPricingMap maps models to their pricing.
var ModelPricingMap = map[Model]ModelPricing{
	// OpenAI Models
	// Prices per 1M tokens (input/output) from OpenAI pricing table provided by user.
	// Note: OpenAI has separate pricing for cached input tokens, audio tokens, image tokens,
	// and per-image / per-second endpoints. This library currently tracks only standard
	// text token usage from model responses, so costs are "text token" estimates.
	ModelGPT52:    {1.75, 14.00},
	ModelGPT51:    {1.25, 10.00},
	ModelGPT5Base: {1.25, 10.00},
	ModelGPT5Mini: {0.25, 2.00},
	ModelGPT5Nano: {0.05, 0.40},

	ModelGPT52Pro: {21.00, 168.00},
	ModelGPT5Pro:  {15.00, 120.00},

	// Codex (priced like gpt-5.1 in the table)
	ModelGPT51CodexMax:   {1.25, 10.00},
	ModelGPT51Codex:      {1.25, 10.00},
	ModelGPT5CodexBase:   {1.25, 10.00},
	ModelGPT51CodexMini:  {0.25, 2.00},
	ModelCodexMiniLatest: {1.50, 6.00},

	// Search + agent tools
	ModelGPT5SearchAPI:      {1.25, 10.00},
	ModelComputerUsePreview: {3.00, 12.00},

	// GPT-5 chat-latest variants
	ModelGPT52ChatLatest: {1.75, 14.00},
	ModelGPT51ChatLatest: {1.25, 10.00},
	ModelGPT5ChatLatest:  {1.25, 10.00},

	// GPT-4.1 family
	ModelGPT41:     {2.00, 8.00},
	ModelGPT41Mini: {0.40, 1.60},
	ModelGPT41Nano: {0.10, 0.40},

	// GPT-4o family
	ModelGPT4o:         {2.50, 10.00},
	ModelGPT4o20240513: {5.00, 15.00},
	ModelGPT4oMini:     {0.15, 0.60},

	// Realtime / audio models (text token pricing)
	ModelGPTRealtime:              {4.00, 16.00},
	ModelGPTRealtimeMini:          {0.60, 2.40},
	ModelGPT4oRealtimePreview:     {5.00, 20.00},
	ModelGPT4oMiniRealtimePreview: {0.60, 2.40},
	ModelGPTAudio:                 {2.50, 10.00},
	ModelGPTAudioMini:             {0.60, 2.40},
	ModelGPT4oAudioPreview:        {2.50, 10.00},
	ModelGPT4oMiniAudioPreview:    {0.15, 0.60},

	// Search previews
	ModelGPT4oMiniSearchPreview: {0.15, 0.60},
	ModelGPT4oSearchPreview:     {2.50, 10.00},

	// ChatGPT aliases / special models
	ModelChatGPT4oLatest: {5.00, 15.00}, // legacy table

	// o-series reasoning
	ModelO1:                 {15.00, 60.00},
	ModelO1Mini:             {1.10, 4.40},
	ModelO1Pro:              {150.00, 600.00},
	ModelO3:                 {2.00, 8.00},
	ModelO3Mini:             {1.10, 4.40},
	ModelO3Pro:              {20.00, 80.00},
	ModelO3DeepResearch:     {10.00, 40.00},
	ModelO4Mini:             {1.10, 4.40},
	ModelO4MiniDeepResearch: {2.00, 8.00},

	// Image generation models (text token pricing table)
	ModelGPTImage15:         {5.00, 10.00},
	ModelChatGPTImageLatest: {5.00, 10.00},
	ModelGPTImage1:          {5.00, 0.00}, // output "-" in table (image outputs are priced differently)
	ModelGPTImage1Mini:      {2.00, 0.00}, // output "-" in table

	// Anthropic Models
	// Latest (Claude 4.5)
	ModelClaudeOpus:   {5.00, 25.00},
	ModelClaudeSonnet: {3.00, 15.00},
	ModelClaudeHaiku:  {1.00, 5.00},

	// Legacy / still available
	ModelClaudeOpus41:  {15.00, 75.00},
	ModelClaudeOpus4:   {15.00, 75.00},
	ModelClaudeSonnet4: {3.00, 15.00},

	// Deprecated / legacy snapshots
	ModelClaudeSonnet37: {3.00, 15.00},
	ModelClaudeHaiku35:  {0.80, 4.00},
	ModelClaudeHaiku3:   {0.25, 1.25},
	ModelClaudeOpus3:    {15.00, 75.00},
	ModelClaudeSonnet3:  {3.00, 15.00},

	// Google Models
	ModelGemini3Pro:        {2.00, 12.00}, // Gemini 3 Pro (preview)
	ModelGemini3Flash:      {0.50, 3.00},  // Gemini 3 Flash (preview)
	ModelGemini25Pro:       {1.25, 10.00},
	ModelGemini25Flash:     {0.30, 2.50},
	ModelGemini25FlashLite: {0.10, 0.40},
	ModelGemini2Flash:      {0.10, 0.40},  // Gemini 2.0 Flash (-001 on OpenRouter)
	ModelGemini2FlashLite:  {0.075, 0.30}, // Gemini 2.0 Flash Lite (-001 on OpenRouter)

	// xAI Models
	ModelGrok3:      {3.00, 12.00},
	ModelGrok3Mini:  {0.50, 2.00},
	ModelGrok41Fast: {2.00, 8.00},

	// Other Models
	ModelQwen3Next:    {1.00, 4.00},
	ModelQwen3:        {2.00, 8.00},
	ModelLlama4:       {1.00, 4.00},
	ModelMistralLarge: {2.00, 6.00},
}

// EmbeddingPricingMap maps embedding models to their pricing (per 1M tokens).
var EmbeddingPricingMap = map[EmbeddingModel]float64{
	EmbedTextSmall3:  0.02,
	EmbedTextLarge3:  0.13,
	EmbedTextAda002:  0.10,
	EmbedGecko:       0.0001, // Google pricing is very low
	EmbedGeckoLatest: 0.0001,
}

// AudioPricingMap maps audio models to their pricing (per minute or per 1M chars).
var AudioPricingMap = map[string]float64{
	// TTS pricing per 1M characters
	string(TTSTTS1):   15.00,
	string(TTSTTS1HD): 30.00,
	// STT pricing per minute
	string(STTWhisper1): 0.006,
}

// ═══════════════════════════════════════════════════════════════════════════
// Cost Calculation
// ═══════════════════════════════════════════════════════════════════════════

// CalculateCost calculates the estimated cost for a request in USD.
func CalculateCost(model Model, promptTokens, completionTokens int) float64 {
	pricing, ok := ModelPricingMap[model]
	if !ok {
		// Use default pricing if model not found
		pricing = ModelPricing{2.50, 10.00}
	}

	inputCost := float64(promptTokens) / 1_000_000 * pricing.InputPerMillion
	outputCost := float64(completionTokens) / 1_000_000 * pricing.OutputPerMillion

	return inputCost + outputCost
}

// CalculateEmbeddingCost calculates estimated embedding cost in USD.
func CalculateEmbeddingCost(model EmbeddingModel, tokens int) float64 {
	pricing, ok := EmbeddingPricingMap[model]
	if !ok {
		pricing = 0.02 // default
	}
	return float64(tokens) / 1_000_000 * pricing
}

// CalculateTTSCost calculates estimated text-to-speech cost in USD (per character).
func CalculateTTSCost(model TTSModel, characters int) float64 {
	pricing, ok := AudioPricingMap[string(model)]
	if !ok {
		pricing = 15.00 // default
	}
	return float64(characters) / 1_000_000 * pricing
}

// CalculateSTTCost calculates estimated speech-to-text cost in USD (per minute).
func CalculateSTTCost(model STTModel, durationSeconds float64) float64 {
	pricing, ok := AudioPricingMap[string(model)]
	if !ok {
		pricing = 0.006 // default
	}
	return (durationSeconds / 60) * pricing
}

// ═══════════════════════════════════════════════════════════════════════════
// Cost in ResponseMeta
// ═══════════════════════════════════════════════════════════════════════════

// Cost returns the estimated cost for this response in USD.
func (m *ResponseMeta) Cost() float64 {
	return CalculateCost(m.Model, m.PromptTokens, m.CompletionTokens)
}

// CostString returns a formatted USD cost string.
func (m *ResponseMeta) CostString() string {
	cost := m.Cost()
	if cost < 0.01 {
		return fmt.Sprintf("$%.6f", cost)
	}
	return fmt.Sprintf("$%.4f", cost)
}

// ═══════════════════════════════════════════════════════════════════════════
// Cost Tracker - Accumulated Costs
// ═══════════════════════════════════════════════════════════════════════════

// CostTracker tracks accumulated costs across many responses.
type CostTracker struct {
	mu           sync.Mutex
	TotalCost    float64
	RequestCount int
	TokensUsed   int
	CostByModel  map[Model]float64
}

// NewCostTracker creates a new CostTracker.
func NewCostTracker() *CostTracker {
	return &CostTracker{
		CostByModel: make(map[Model]float64),
	}
}

// Track records a response's cost.
func (ct *CostTracker) Track(meta *ResponseMeta) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	cost := meta.Cost()
	ct.TotalCost += cost
	ct.RequestCount++
	ct.TokensUsed += meta.Tokens
	ct.CostByModel[meta.Model] += cost
}

// Reset clears all tracked costs.
func (ct *CostTracker) Reset() {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.TotalCost = 0
	ct.RequestCount = 0
	ct.TokensUsed = 0
	ct.CostByModel = make(map[Model]float64)
}

// Summary returns a formatted cost summary.
func (ct *CostTracker) Summary() string {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if ct.RequestCount == 0 {
		return "No requests tracked"
	}

	avgCost := ct.TotalCost / float64(ct.RequestCount)

	summary := fmt.Sprintf(`
Cost Summary:
  Total Cost:     $%.4f
  Request Count:  %d
  Tokens Used:    %d
  Avg Cost/Req:   $%.6f

Cost by Model:`,
		ct.TotalCost, ct.RequestCount, ct.TokensUsed, avgCost)

	for model, cost := range ct.CostByModel {
		summary += fmt.Sprintf("\n  - %s: $%.4f", model, cost)
	}

	return summary
}

// Print prints the cost summary to stdout.
func (ct *CostTracker) Print() {
	fmt.Println(ct.Summary())
}

// ═══════════════════════════════════════════════════════════════════════════
// Global Cost Tracking (opt-in)
// ═══════════════════════════════════════════════════════════════════════════

var (
	globalCostTracker     *CostTracker
	globalCostTrackerLock sync.Mutex
)

// EnableCostTracking enables package-level cost tracking (opt-in).
func EnableCostTracking() {
	globalCostTrackerLock.Lock()
	defer globalCostTrackerLock.Unlock()

	if globalCostTracker == nil {
		globalCostTracker = NewCostTracker()
	}
}

// GetCostTracker returns the package-level cost tracker (creating one if needed).
func GetCostTracker() *CostTracker {
	globalCostTrackerLock.Lock()
	defer globalCostTrackerLock.Unlock()

	if globalCostTracker == nil {
		globalCostTracker = NewCostTracker()
	}
	return globalCostTracker
}

// TotalCost returns the total cost tracked by the package-level tracker.
func TotalCost() float64 {
	return GetCostTracker().TotalCost
}

// PrintCostSummary prints the package-level cost summary.
func PrintCostSummary() {
	GetCostTracker().Print()
}

// ═══════════════════════════════════════════════════════════════════════════
// Cost-Aware Helpers
// ═══════════════════════════════════════════════════════════════════════════

// EstimatePromptCost estimates prompt cost in USD before sending, using a rough token heuristic.
func EstimatePromptCost(model Model, promptChars int) float64 {
	// Rough token estimate: 1 token ≈ 4 characters
	estimatedTokens := promptChars / 4
	inputPerMillion := 2.50
	if pricing, ok := ModelPricingMap[model]; ok {
		inputPerMillion = pricing.InputPerMillion
	}
	return float64(estimatedTokens) / 1_000_000 * inputPerMillion
}

// CheapestModel returns the cheapest model (by average in/out pricing) from a list.
func CheapestModel(models ...Model) Model {
	if len(models) == 0 {
		return ModelGPT4oMini // default cheap model
	}

	cheapest := models[0]
	cheapestCost := float64(999999)

	for _, m := range models {
		pricing, ok := ModelPricingMap[m]
		if !ok {
			continue
		}
		avgCost := (pricing.InputPerMillion + pricing.OutputPerMillion) / 2
		if avgCost < cheapestCost {
			cheapestCost = avgCost
			cheapest = m
		}
	}

	return cheapest
}

// MostExpensiveModel returns the most expensive model (by average in/out pricing) from a list.
func MostExpensiveModel(models ...Model) Model {
	if len(models) == 0 {
		return ModelClaudeOpus // default high-end model
	}

	expensive := models[0]
	highestCost := float64(0)

	for _, m := range models {
		pricing, ok := ModelPricingMap[m]
		if !ok {
			continue
		}
		avgCost := (pricing.InputPerMillion + pricing.OutputPerMillion) / 2
		if avgCost > highestCost {
			highestCost = avgCost
			expensive = m
		}
	}

	return expensive
}

// ═══════════════════════════════════════════════════════════════════════════
// Budget Controls
// ═══════════════════════════════════════════════════════════════════════════

// BudgetExceededError is returned when budget is exceeded.
type BudgetExceededError struct {
	Budget  float64
	Current float64
}

// Error implements the error interface.
func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf("budget exceeded: current $%.4f >= budget $%.4f", e.Current, e.Budget)
}

// WithBudget creates a BudgetTracker with a budget limit.
func WithBudget(maxCost float64) *BudgetTracker {
	return &BudgetTracker{
		CostTracker: NewCostTracker(),
		Budget:      maxCost,
	}
}

// BudgetTracker is a CostTracker with a budget limit.
type BudgetTracker struct {
	*CostTracker
	Budget float64
}

// CheckBudget returns an error if the budget is exceeded.
func (bt *BudgetTracker) CheckBudget() error {
	if bt.TotalCost >= bt.Budget {
		return &BudgetExceededError{Budget: bt.Budget, Current: bt.TotalCost}
	}
	return nil
}

// Remaining returns remaining budget in USD.
func (bt *BudgetTracker) Remaining() float64 {
	remaining := bt.Budget - bt.TotalCost
	if remaining < 0 {
		return 0
	}
	return remaining
}

// RemainingString returns a formatted remaining budget string.
func (bt *BudgetTracker) RemainingString() string {
	return fmt.Sprintf("$%.4f remaining of $%.4f budget", bt.Remaining(), bt.Budget)
}
