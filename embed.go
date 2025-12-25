package ai

import (
	"context"
	"fmt"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════
// Embedding Models
// ═══════════════════════════════════════════════════════════════════════════

// EmbeddingModel represents an embedding model identifier.
type EmbeddingModel string

const (
	// OpenAI Embedding Models
	EmbedTextSmall3 EmbeddingModel = "text-embedding-3-small"
	EmbedTextLarge3 EmbeddingModel = "text-embedding-3-large"
	EmbedTextAda002 EmbeddingModel = "text-embedding-ada-002"

	// Google Embedding Models
	EmbedGecko       EmbeddingModel = "text-embedding-004"
	EmbedGeckoLatest EmbeddingModel = "text-embedding-005"

	// Ollama (local) - common embedding models
	EmbedNomic     EmbeddingModel = "nomic-embed-text"
	EmbedMxbai     EmbeddingModel = "mxbai-embed-large"
	EmbedAllMiniLM EmbeddingModel = "all-minilm"
	EmbedSnowflake EmbeddingModel = "snowflake-arctic-embed"
)

// DefaultEmbeddingModel is the default embedding model used by Embed and EmbedMany.
var DefaultEmbeddingModel = EmbedTextSmall3

// ═══════════════════════════════════════════════════════════════════════════
// Embedding Request/Response
// ═══════════════════════════════════════════════════════════════════════════

// EmbeddingRequest is a provider-agnostic request format for embeddings.
type EmbeddingRequest struct {
	Model      string
	Input      []string // texts to embed
	Dimensions int      // optional: for models that support dimension reduction
}

// EmbeddingResponse is a provider-agnostic response format for embeddings.
type EmbeddingResponse struct {
	Embeddings  [][]float64 // embedding vectors
	Model       string
	TotalTokens int
	Dimensions  int
}

// ═══════════════════════════════════════════════════════════════════════════
// Embedding Builder - Fluent API
// ═══════════════════════════════════════════════════════════════════════════

// EmbedBuilder provides a fluent API for creating embeddings.
type EmbedBuilder struct {
	model      EmbeddingModel
	texts      []string
	dimensions int
	client     *Client
	ctx        context.Context
}

// Embed creates a new EmbedBuilder for a single text.
func Embed(text string) *EmbedBuilder {
	return &EmbedBuilder{
		model: DefaultEmbeddingModel,
		texts: []string{text},
	}
}

// EmbedMany creates a new EmbedBuilder for multiple texts.
func EmbedMany(texts ...string) *EmbedBuilder {
	return &EmbedBuilder{
		model: DefaultEmbeddingModel,
		texts: texts,
	}
}

// Model sets the embedding model.
func (e *EmbedBuilder) Model(model EmbeddingModel) *EmbedBuilder {
	e.model = model
	return e
}

// Dimensions sets the output dimensions (for models that support it).
func (e *EmbedBuilder) Dimensions(d int) *EmbedBuilder {
	e.dimensions = d
	return e
}

// WithClient sets a specific client/provider to execute the request with.
func (e *EmbedBuilder) WithClient(client *Client) *EmbedBuilder {
	e.client = client
	return e
}

// WithContext sets a context for cancellation.
func (e *EmbedBuilder) WithContext(ctx context.Context) *EmbedBuilder {
	e.ctx = ctx
	return e
}

// Add appends more texts to embed.
func (e *EmbedBuilder) Add(texts ...string) *EmbedBuilder {
	e.texts = append(e.texts, texts...)
	return e
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution
// ═══════════════════════════════════════════════════════════════════════════

// Do executes the embedding request and returns all vectors.
func (e *EmbedBuilder) Do() ([][]float64, error) {
	resp, err := e.DoWithMeta()
	if err != nil {
		return nil, err
	}
	return resp.Embeddings, nil
}

// DoWithMeta executes the request and returns the full response with metadata.
func (e *EmbedBuilder) DoWithMeta() (*EmbeddingResponse, error) {
	client := e.client
	if client == nil {
		client = getDefaultClient()
	}

	ctx := e.ctx
	if ctx == nil {
		ctx = context.Background()
	}

	// Check if provider supports embeddings
	embedder, ok := client.provider.(Embedder)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support embeddings", client.provider.Name())
	}

	req := &EmbeddingRequest{
		Model:      string(e.model),
		Input:      e.texts,
		Dimensions: e.dimensions,
	}

	if Debug {
		fmt.Printf("%s Embedding %d text(s) with %s\n", colorCyan("→"), len(e.texts), e.model)
	}

	waitForRateLimit()
	resp, err := embedder.Embed(ctx, req)
	if err != nil {
		return nil, err
	}

	if Debug {
		fmt.Printf("%s Got %d embedding(s), dim=%d, tokens=%d\n",
			colorGreen("✓"), len(resp.Embeddings), resp.Dimensions, resp.TotalTokens)
	}

	return resp, nil
}

// First returns the first embedding vector (convenience for single-text usage).
func (e *EmbedBuilder) First() ([]float64, error) {
	results, err := e.Do()
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return results[0], nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider-Specific Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

// Embed creates an EmbedBuilder bound to this client.
func (c *Client) Embed(text string) *EmbedBuilder {
	return Embed(text).WithClient(c)
}

// EmbedMany creates an EmbedBuilder bound to this client for multiple texts.
func (c *Client) EmbedMany(texts ...string) *EmbedBuilder {
	return EmbedMany(texts...).WithClient(c)
}

// ═══════════════════════════════════════════════════════════════════════════
// Similarity Functions
// ═══════════════════════════════════════════════════════════════════════════

// CosineSimilarity calculates cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// DotProduct calculates the dot product between two vectors.
func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var result float64
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

// EuclideanDistance calculates Euclidean distance between two vectors.
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sqrt(sum)
}

// sqrt is a simple square root implementation
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// ═══════════════════════════════════════════════════════════════════════════
// Semantic Search Helper
// ═══════════════════════════════════════════════════════════════════════════

// SearchResult represents a search result with similarity score
type SearchResult struct {
	Index     int
	Text      string
	Score     float64
	Embedding []float64
}

// SemanticSearch searches for the most similar texts to a query
func SemanticSearch(query string, corpus []string, topK int) ([]SearchResult, error) {
	// Embed query and corpus together for efficiency
	allTexts := append([]string{query}, corpus...)
	embeddings, err := EmbedMany(allTexts...).Do()
	if err != nil {
		return nil, err
	}

	queryEmbed := embeddings[0]
	corpusEmbeds := embeddings[1:]

	// Calculate similarities
	results := make([]SearchResult, len(corpus))
	for i, embed := range corpusEmbeds {
		results[i] = SearchResult{
			Index:     i,
			Text:      corpus[i],
			Score:     CosineSimilarity(queryEmbed, embed),
			Embedding: embed,
		}
	}

	// Sort by score (descending)
	sortResults(results)

	// Return top K
	if topK > len(results) {
		topK = len(results)
	}
	return results[:topK], nil
}

// sortResults sorts results by score descending (simple bubble sort)
func sortResults(results []SearchResult) {
	for i := 0; i < len(results)-1; i++ {
		for j := 0; j < len(results)-i-1; j++ {
			if results[j].Score < results[j+1].Score {
				results[j], results[j+1] = results[j+1], results[j]
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Embedding with Concurrency Control
// ═══════════════════════════════════════════════════════════════════════════

// EmbedBatch embeds a large number of texts with automatic batching
func EmbedBatch(texts []string, batchSize int) ([][]float64, error) {
	if batchSize <= 0 {
		batchSize = 100 // default batch size
	}

	var allEmbeddings [][]float64
	var mu sync.Mutex

	// Process in batches
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		embeddings, err := EmbedMany(batch...).Do()
		if err != nil {
			return nil, fmt.Errorf("batch %d-%d failed: %w", i, end, err)
		}

		mu.Lock()
		allEmbeddings = append(allEmbeddings, embeddings...)
		mu.Unlock()
	}

	return allEmbeddings, nil
}
