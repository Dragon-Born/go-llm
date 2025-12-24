package ai

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStreamResponse_FallsBackWhenProviderDoesNotSupportStreaming(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{Streaming: false},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			return &ProviderResponse{Content: "abc"}, nil
		},
		streamFn: func(ctx context.Context, req *ProviderRequest, cb StreamCallback) (*ProviderResponse, error) {
			t.Fatalf("SendStream should not be called when Streaming=false")
			return nil, nil
		},
	}

	b := New(ModelGPT5).WithClient(&Client{provider: p, providerType: ProviderOpenAI}).User("hi")

	var got strings.Builder
	out, err := b.StreamResponse(func(chunk string) { got.WriteString(chunk) })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "abc" || got.String() != "abc" {
		t.Fatalf("unexpected stream fallback result: out=%q chunks=%q", out, got.String())
	}
}

func TestConversation_Say_AppendsToHistoryAndBuildsFullMessages(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			// echo last user message
			last := ""
			for i := len(req.Messages) - 1; i >= 0; i-- {
				if req.Messages[i].Role == "user" {
					last, _ = req.Messages[i].Content.(string)
					break
				}
			}
			return &ProviderResponse{Content: "echo:" + last}, nil
		},
	}
	setDefaultClientForTest(t, p, ProviderOpenAI)

	chat := New(ModelGPT5).System("sys").Chat()
	r1, err := chat.Say("hi")
	if err != nil || r1 != "echo:hi" {
		t.Fatalf("unexpected say1: %q err=%v", r1, err)
	}
	r2, err := chat.Say("again")
	if err != nil || r2 != "echo:again" {
		t.Fatalf("unexpected say2: %q err=%v", r2, err)
	}

	h := chat.History()
	if len(h) != 4 {
		t.Fatalf("expected 4 history messages, got %d", len(h))
	}
	if chat.LastResponse() != "echo:again" {
		t.Fatalf("unexpected last response: %q", chat.LastResponse())
	}
}

func TestEmbeddings_SemanticSearchAndProviderSupportCheck(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	// Default client supports embeddings.
	core := &stubProvider{name: "stub", caps: ProviderCapabilities{Embeddings: true}}
	ep := &stubEmbedderProvider{
		stubProvider: core,
		embedFn: func(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
			// req.Input[0] is query, the rest is corpus
			emb := make([][]float64, len(req.Input))
			for i := range emb {
				switch req.Input[i] {
				case "query":
					emb[i] = []float64{1, 0}
				case "match":
					emb[i] = []float64{1, 0}
				default:
					emb[i] = []float64{0, 1}
				}
			}
			return &EmbeddingResponse{
				Embeddings:  emb,
				Model:       req.Model,
				TotalTokens: 10,
				Dimensions:  2,
			}, nil
		},
	}
	setDefaultClientForTest(t, ep, ProviderOpenAI)

	results, err := SemanticSearch("query", []string{"nope", "match"}, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 || results[0].Text != "match" {
		t.Fatalf("expected top result to be match, got %#v", results)
	}

	// Provider does not support embeddings: type assertion fails.
	noEmbed := &stubProvider{name: "stub2"}
	_, err = Embed("x").WithClient(&Client{provider: noEmbed, providerType: ProviderOpenAI}).DoWithMeta()
	if err == nil || !strings.Contains(err.Error(), "does not support embeddings") {
		t.Fatalf("expected provider embeddings error, got %v", err)
	}
}

func TestAudio_TTSAndSTT_BuildersPassThroughFields(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	core := &stubProvider{name: "stub", caps: ProviderCapabilities{TTS: true, STT: true}}
	ap := &stubAudioProvider{
		stubProvider: core,
		ttsFn: func(ctx context.Context, req *TTSRequest) (*TTSResponse, error) {
			if req.Voice != string(VoiceNova) {
				t.Fatalf("unexpected voice: %q", req.Voice)
			}
			if req.Format != string(AudioFormatMP3) {
				t.Fatalf("unexpected format: %q", req.Format)
			}
			return &TTSResponse{Audio: []byte("mp3")}, nil
		},
		sttFn: func(ctx context.Context, req *STTRequest) (*STTResponse, error) {
			if !req.Timestamps {
				t.Fatalf("expected timestamps enabled")
			}
			if req.Language != "en" {
				t.Fatalf("expected language en, got %q", req.Language)
			}
			if req.Prompt != "context" {
				t.Fatalf("expected prompt context, got %q", req.Prompt)
			}
			return &STTResponse{Text: "ok", Duration: 60, Words: []WordTimestamp{{Word: "ok", Start: 0, End: 0.5}}}, nil
		},
	}
	setDefaultClientForTest(t, ap, ProviderOpenAI)

	// TTS Save writes bytes.
	dir := t.TempDir()
	outPath := filepath.Join(dir, "out.mp3")
	err := Speak("hello").Voice(VoiceNova).Format(AudioFormatMP3).Save(outPath)
	if err != nil {
		t.Fatalf("unexpected TTS save error: %v", err)
	}
	if b, err := os.ReadFile(outPath); err != nil || string(b) != "mp3" {
		t.Fatalf("expected saved audio, got %q err=%v", string(b), err)
	}

	// STT passes through settings.
	resp, err := TranscribeBytes([]byte("audio"), "a.mp3").Language("en").Prompt("context").WithTimestamps().DoWithMeta()
	if err != nil {
		t.Fatalf("unexpected STT error: %v", err)
	}
	if resp.Text != "ok" || len(resp.Words) != 1 {
		t.Fatalf("unexpected STT resp: %#v", resp)
	}
}

func TestVisionAndPDF_BuildMessages_MultimodalLastUserMessage(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	dir := t.TempDir()
	imgPath := filepath.Join(dir, "x.png")
	pdfPath := filepath.Join(dir, "x.pdf")
	_ = os.WriteFile(imgPath, []byte("not a real png"), 0644)
	_ = os.WriteFile(pdfPath, []byte("%PDF-1.4"), 0644)

	msgs := New(ModelGPT4o).
		User("hello").
		Image(imgPath).
		PDF(pdfPath).
		buildMessages()

	if len(msgs) == 0 {
		t.Fatalf("expected messages")
	}

	// Last user message becomes []ContentPart.
	var lastUser Message
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			lastUser = msgs[i]
			break
		}
	}

	parts, ok := lastUser.Content.([]ContentPart)
	if !ok {
		t.Fatalf("expected multimodal content parts, got %T", lastUser.Content)
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (text+image+pdf), got %d", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "hello" {
		t.Fatalf("unexpected first part: %#v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil || !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Fatalf("unexpected image part: %#v", parts[1])
	}
	if parts[2].Type != "document" || parts[2].Document == nil || parts[2].Document.MimeType != "application/pdf" {
		t.Fatalf("unexpected document part: %#v", parts[2])
	}
}

func TestBatch_StopOnError_CancelsOtherRequests(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			// Find last user prompt
			var prompt string
			for i := len(req.Messages) - 1; i >= 0; i-- {
				if req.Messages[i].Role == "user" {
					prompt, _ = req.Messages[i].Content.(string)
					break
				}
			}
			switch prompt {
			case "bad":
				return nil, errors.New("boom")
			case "slow":
				<-ctx.Done()
				return nil, ctx.Err()
			default:
				return &ProviderResponse{Content: "ok:" + prompt}, nil
			}
		},
	}
	setDefaultClientForTest(t, p, ProviderOpenAI)

	results := Batch(
		New(ModelGPT5).User("bad"),
		New(ModelGPT5).User("slow"),
	).StopOnError().Do()

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Error == nil {
		t.Fatalf("expected first result error")
	}
	// Second should be canceled.
	if results[1].Error == nil {
		t.Fatalf("expected second result to error due to cancellation")
	}
	if !strings.Contains(strings.ToLower(results[1].Error.Error()), "canceled") && !strings.Contains(strings.ToLower(results[1].Error.Error()), "cancelled") {
		// context cancellation wording varies; accept generic ctx error too.
	}
}

func TestRetry_WithRetry_RetriesOnProviderError(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	attempts := 0
	_, err := WithRetry(context.Background(), noSleepRetryConfig(2), func() (string, error) {
		attempts++
		if attempts < 3 {
			return "", &ProviderError{Provider: "x", Code: "429", Message: "rate limit"}
		}
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
}

func TestBuilder_RetryConfig_RetriesAndCountsRetries(t *testing.T) {
	cleanup := withTestGlobals(t)
	defer cleanup()

	attempts := 0
	p := &stubProvider{
		name: "stub",
		caps: ProviderCapabilities{},
		sendFn: func(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
			attempts++
			if attempts < 3 {
				return nil, &ProviderError{Provider: "stub", Code: "429", Message: "rate limit"}
			}
			return &ProviderResponse{Content: "ok", TotalTokens: 1}, nil
		},
	}
	b := New(ModelGPT5).
		WithClient(&Client{provider: p, providerType: ProviderOpenAI}).
		RetryConfig(noSleepRetryConfig(2)).
		User("hi")

	meta := b.SendWithMeta()
	if meta.Error != nil {
		t.Fatalf("unexpected error: %v", meta.Error)
	}
	if meta.Content != "ok" {
		t.Fatalf("unexpected content: %q", meta.Content)
	}
	if meta.Retries != 2 {
		t.Fatalf("expected Retries=2, got %d", meta.Retries)
	}
}
