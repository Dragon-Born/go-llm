package ai

import (
	"context"
	"fmt"
	"os"
)

// ═══════════════════════════════════════════════════════════════════════════
// Audio Models
// ═══════════════════════════════════════════════════════════════════════════

// TTSModel represents a text-to-speech model
type TTSModel string

const (
	// OpenAI TTS Models
	TTSTTS1       TTSModel = "tts-1"           // Standard quality, faster
	TTSTTS1HD     TTSModel = "tts-1-hd"        // High definition, slower
	TTSGpt4oAudio TTSModel = "gpt-4o-mini-tts" // GPT-4o audio

	// Google TTS (via Gemini)
	TTSGemini TTSModel = "gemini-2.5-flash-preview-tts"
)

// STTModel represents a speech-to-text model
type STTModel string

const (
	// OpenAI STT Models
	STTWhisper1   STTModel = "whisper-1"
	STTGpt4oAudio STTModel = "gpt-4o-transcribe"

	// Google STT (via Gemini)
	STTGemini STTModel = "gemini-2.5-flash-preview-stt"
)

// Voice represents a TTS voice
type Voice string

const (
	// OpenAI Voices
	VoiceAlloy   Voice = "alloy"
	VoiceAsh     Voice = "ash"
	VoiceBallad  Voice = "ballad"
	VoiceCoral   Voice = "coral"
	VoiceEcho    Voice = "echo"
	VoiceFable   Voice = "fable"
	VoiceOnyx    Voice = "onyx"
	VoiceNova    Voice = "nova"
	VoiceSage    Voice = "sage"
	VoiceShimmer Voice = "shimmer"
	VoiceVerse   Voice = "verse"
)

// AudioFormat represents the output audio format
type AudioFormat string

const (
	AudioFormatMP3  AudioFormat = "mp3"
	AudioFormatOpus AudioFormat = "opus"
	AudioFormatAAC  AudioFormat = "aac"
	AudioFormatFLAC AudioFormat = "flac"
	AudioFormatWAV  AudioFormat = "wav"
	AudioFormatPCM  AudioFormat = "pcm"
)

// Default audio settings
var (
	DefaultTTSModel    = TTSTTS1
	DefaultSTTModel    = STTWhisper1
	DefaultVoice       = VoiceAlloy
	DefaultAudioFormat = AudioFormatMP3
)

// ═══════════════════════════════════════════════════════════════════════════
// TTS Request/Response
// ═══════════════════════════════════════════════════════════════════════════

// TTSRequest is the request for text-to-speech
type TTSRequest struct {
	Model  string
	Input  string  // text to speak
	Voice  string  // voice ID
	Format string  // output format
	Speed  float64 // 0.25 to 4.0 (1.0 is default)
}

// TTSResponse is the response from text-to-speech
type TTSResponse struct {
	Audio       []byte // raw audio data
	Format      string
	ContentType string
}

// ═══════════════════════════════════════════════════════════════════════════
// STT Request/Response
// ═══════════════════════════════════════════════════════════════════════════

// STTRequest is the request for speech-to-text
type STTRequest struct {
	Model       string
	Audio       []byte // audio data
	AudioURL    string // or audio URL
	Filename    string // filename for format detection
	Language    string // optional: language hint (ISO 639-1)
	Prompt      string // optional: context/prompt to guide transcription
	Temperature float64
	Timestamps  bool // include word-level timestamps
}

// STTResponse is the response from speech-to-text
type STTResponse struct {
	Text     string
	Language string
	Duration float64 // audio duration in seconds
	Words    []WordTimestamp
}

// WordTimestamp represents a word with timing info
type WordTimestamp struct {
	Word  string
	Start float64 // seconds
	End   float64 // seconds
}

// ═══════════════════════════════════════════════════════════════════════════
// TTS Builder - Fluent API
// ═══════════════════════════════════════════════════════════════════════════

// TTSBuilder provides a fluent API for text-to-speech
type TTSBuilder struct {
	model  TTSModel
	text   string
	voice  Voice
	format AudioFormat
	speed  float64
	client *Client
	ctx    context.Context
}

// Speak creates a new TTS builder
func Speak(text string) *TTSBuilder {
	return &TTSBuilder{
		model:  DefaultTTSModel,
		text:   text,
		voice:  DefaultVoice,
		format: DefaultAudioFormat,
		speed:  1.0,
	}
}

// Model sets the TTS model
func (t *TTSBuilder) Model(model TTSModel) *TTSBuilder {
	t.model = model
	return t
}

// Voice sets the voice
func (t *TTSBuilder) Voice(voice Voice) *TTSBuilder {
	t.voice = voice
	return t
}

// Format sets the output audio format
func (t *TTSBuilder) Format(format AudioFormat) *TTSBuilder {
	t.format = format
	return t
}

// Speed sets the speech speed (0.25 to 4.0)
func (t *TTSBuilder) Speed(speed float64) *TTSBuilder {
	if speed < 0.25 {
		speed = 0.25
	}
	if speed > 4.0 {
		speed = 4.0
	}
	t.speed = speed
	return t
}

// HD uses the high-definition TTS model
func (t *TTSBuilder) HD() *TTSBuilder {
	t.model = TTSTTS1HD
	return t
}

// WithClient sets a specific client/provider
func (t *TTSBuilder) WithClient(client *Client) *TTSBuilder {
	t.client = client
	return t
}

// WithContext sets a context for cancellation
func (t *TTSBuilder) WithContext(ctx context.Context) *TTSBuilder {
	t.ctx = ctx
	return t
}

// ═══════════════════════════════════════════════════════════════════════════
// TTS Execution
// ═══════════════════════════════════════════════════════════════════════════

// Do generates the audio and returns raw bytes
func (t *TTSBuilder) Do() ([]byte, error) {
	resp, err := t.DoWithMeta()
	if err != nil {
		return nil, err
	}
	return resp.Audio, nil
}

// DoWithMeta generates audio and returns full response
func (t *TTSBuilder) DoWithMeta() (*TTSResponse, error) {
	client := t.client
	if client == nil {
		client = getDefaultClient()
	}

	ctx := t.ctx
	if ctx == nil {
		ctx = context.Background()
	}

	// Check if provider supports TTS
	audioProvider, ok := client.provider.(AudioProvider)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support text-to-speech", client.provider.Name())
	}

	req := &TTSRequest{
		Model:  string(t.model),
		Input:  t.text,
		Voice:  string(t.voice),
		Format: string(t.format),
		Speed:  t.speed,
	}

	if Debug {
		fmt.Printf("%s TTS: %d chars → %s voice, %s format\n",
			colorCyan("→"), len(t.text), t.voice, t.format)
	}

	waitForRateLimit()
	resp, err := audioProvider.TextToSpeech(ctx, req)
	if err != nil {
		return nil, err
	}

	if Debug {
		fmt.Printf("%s Generated %d bytes of audio\n", colorGreen("✓"), len(resp.Audio))
	}

	return resp, nil
}

// Save generates audio and saves to file
func (t *TTSBuilder) Save(path string) error {
	audio, err := t.Do()
	if err != nil {
		return err
	}

	if err := os.WriteFile(path, audio, 0644); err != nil {
		return fmt.Errorf("failed to save audio: %w", err)
	}

	if Debug {
		fmt.Printf("%s Saved audio to %s\n", colorGreen("✓"), path)
	}

	return nil
}

// ═══════════════════════════════════════════════════════════════════════════
// STT Builder - Fluent API
// ═══════════════════════════════════════════════════════════════════════════

// STTBuilder provides a fluent API for speech-to-text
type STTBuilder struct {
	model       STTModel
	audio       []byte
	audioURL    string
	filename    string
	language    string
	prompt      string
	temperature float64
	timestamps  bool
	client      *Client
	ctx         context.Context
}

// Transcribe creates a new STT builder from a file path
func Transcribe(path string) *STTBuilder {
	audio, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("%s Error loading audio %s: %v\n", colorRed("✗"), path, err)
		return &STTBuilder{model: DefaultSTTModel}
	}

	return &STTBuilder{
		model:    DefaultSTTModel,
		audio:    audio,
		filename: path,
	}
}

// TranscribeBytes creates a new STT builder from audio bytes
func TranscribeBytes(audio []byte, filename string) *STTBuilder {
	return &STTBuilder{
		model:    DefaultSTTModel,
		audio:    audio,
		filename: filename,
	}
}

// TranscribeURL creates a new STT builder from an audio URL
func TranscribeURL(url string) *STTBuilder {
	return &STTBuilder{
		model:    DefaultSTTModel,
		audioURL: url,
	}
}

// Model sets the STT model
func (s *STTBuilder) Model(model STTModel) *STTBuilder {
	s.model = model
	return s
}

// Language sets the language hint (ISO 639-1 code, e.g., "en", "es", "fr")
func (s *STTBuilder) Language(lang string) *STTBuilder {
	s.language = lang
	return s
}

// Prompt sets context to guide transcription
func (s *STTBuilder) Prompt(prompt string) *STTBuilder {
	s.prompt = prompt
	return s
}

// Temperature sets the sampling temperature (0 to 1)
func (s *STTBuilder) Temperature(temp float64) *STTBuilder {
	s.temperature = temp
	return s
}

// WithTimestamps enables word-level timestamps
func (s *STTBuilder) WithTimestamps() *STTBuilder {
	s.timestamps = true
	return s
}

// WithClient sets a specific client/provider
func (s *STTBuilder) WithClient(client *Client) *STTBuilder {
	s.client = client
	return s
}

// WithContext sets a context for cancellation
func (s *STTBuilder) WithContext(ctx context.Context) *STTBuilder {
	s.ctx = ctx
	return s
}

// ═══════════════════════════════════════════════════════════════════════════
// STT Execution
// ═══════════════════════════════════════════════════════════════════════════

// Do transcribes the audio and returns text
func (s *STTBuilder) Do() (string, error) {
	resp, err := s.DoWithMeta()
	if err != nil {
		return "", err
	}
	return resp.Text, nil
}

// DoWithMeta transcribes audio and returns full response
func (s *STTBuilder) DoWithMeta() (*STTResponse, error) {
	client := s.client
	if client == nil {
		client = getDefaultClient()
	}

	ctx := s.ctx
	if ctx == nil {
		ctx = context.Background()
	}

	// Check if provider supports STT
	audioProvider, ok := client.provider.(AudioProvider)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support speech-to-text", client.provider.Name())
	}

	req := &STTRequest{
		Model:       string(s.model),
		Audio:       s.audio,
		AudioURL:    s.audioURL,
		Filename:    s.filename,
		Language:    s.language,
		Prompt:      s.prompt,
		Temperature: s.temperature,
		Timestamps:  s.timestamps,
	}

	if Debug {
		fmt.Printf("%s STT: %d bytes audio, model=%s\n",
			colorCyan("→"), len(s.audio), s.model)
	}

	waitForRateLimit()
	resp, err := audioProvider.SpeechToText(ctx, req)
	if err != nil {
		return nil, err
	}

	if Debug {
		fmt.Printf("%s Transcribed: %d chars, duration=%.1fs\n",
			colorGreen("✓"), len(resp.Text), resp.Duration)
	}

	return resp, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider-Specific Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

// Speak creates a TTS builder using this client
func (c *Client) Speak(text string) *TTSBuilder {
	return Speak(text).WithClient(c)
}

// Transcribe creates an STT builder using this client
func (c *Client) Transcribe(path string) *STTBuilder {
	return Transcribe(path).WithClient(c)
}

// TranscribeBytes creates an STT builder from bytes using this client
func (c *Client) TranscribeBytes(audio []byte, filename string) *STTBuilder {
	return TranscribeBytes(audio, filename).WithClient(c)
}
