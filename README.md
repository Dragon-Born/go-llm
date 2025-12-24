# go-llm

**The most developer-friendly Go SDK for Large Language Models**

[![Go Reference](https://pkg.go.dev/badge/gopkg.in/dragon-born/go-llm.v1.svg)](https://pkg.go.dev/gopkg.in/dragon-born/go-llm.v1)
[![Go Report Card](https://goreportcard.com/badge/gopkg.in/dragon-born/go-llm.v1)](https://goreportcard.com/report/gopkg.in/dragon-born/go-llm.v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified Go client for **OpenAI**, **Anthropic Claude**, **Google Gemini**, **Ollama**, **Azure OpenAI**, and 200+ models via OpenRouter. Build AI-powered applications with a beautiful fluent API, structured output parsing, function calling, AI agents, embeddings, vision, audio, and more.

---

## Why go-llm?

| Pain Point | go-llm Solution |
|------------|-----------------|
| Different SDKs for each provider | **One unified API** for all providers |
| Verbose boilerplate code | **Fluent builder pattern** — chain methods naturally |
| No structured output | **Type-safe parsing** into Go structs with auto-retry |
| Complex agent loops | **Built-in ReAct agents** with tools and callbacks |
| Rate limits crash your app | **Smart retry** with exponential backoff |
| No cost visibility | **Automatic cost tracking** per request |
| Need web search / code execution | **Built-in tools** — web search, file search, code interpreter, MCP |

---

## Installation

```bash
go get gopkg.in/dragon-born/go-llm.v1
```

## Quick Start

```go
package main

import (
    "fmt"
    ai "gopkg.in/dragon-born/go-llm.v1"
)

func main() {
    // Set your API key
    // export OPENROUTER_API_KEY="sk-or-..."

    response, err := ai.Claude().
        System("You are a helpful assistant.").
        Ask("What is the capital of France?")
    
    if err != nil {
        panic(err)
    }
    fmt.Println(response)
}
```

---

## Supported Providers & Models

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| **OpenRouter** (default) | 200+ models | `OPENROUTER_API_KEY` |
| **OpenAI** | GPT-5, GPT-4o, o1, o3 | `OPENAI_API_KEY` |
| **Anthropic** | Claude Opus/Sonnet/Haiku | `ANTHROPIC_API_KEY` |
| **Google** | Gemini 3/2.5/2 Pro/Flash | `GOOGLE_API_KEY` |
| **xAI** | Grok 4.1, Grok 3 | via OpenRouter |
| **Meta** | Llama 4 | via OpenRouter |
| **Mistral** | Mistral Large | via OpenRouter |
| **Ollama** | Any local model | None (local) |
| **Azure OpenAI** | OpenAI models | `AZURE_OPENAI_API_KEY` |

```go
// Via OpenRouter (default gateway to all models)
ai.Claude().Ask("Hello")
ai.GPT5().Ask("Hello")
ai.Gemini().Ask("Hello")

// Direct to provider APIs
ai.Anthropic().Claude().Ask("Hello")
ai.OpenAI().GPT5().Ask("Hello")  
ai.Google().GeminiPro().Ask("Hello")

// Local with Ollama (no API key needed)
ai.Ollama().Use("llama3:8b").Ask("Hello")

// Azure OpenAI
ai.Azure("https://mycompany.openai.azure.com").GPT4o().Ask("Hello")
```

---

## Features

### 🔗 Fluent Builder API

Chain methods naturally for clean, readable code:

```go
response, _ := ai.Claude().
    System("You are a senior Go developer").
    Temperature(0.2).
    ThinkHigh().                    // Enable extended thinking
    MaxTokens(1000).
    Ask("Review this code for bugs")
```

### 📐 Structured Output (Instructor-style)

Parse LLM responses directly into Go structs with automatic retry on parse errors:

```go
type Person struct {
    Name  string `json:"name"`
    Age   int    `json:"age"`
    Email string `json:"email"`
}

var person Person
err := ai.Claude().IntoWithRetry(
    "Extract: John Smith, 32 years old, john@example.com",
    &person,
    3, // retry up to 3 times on parse failure
)
// person = {Name: "John Smith", Age: 32, Email: "john@example.com"}
```

### 🔧 Function Calling / Tools

Let AI call your Go functions:

```go
ai.GPT5().
    Tool("get_weather", "Get weather for a city", ai.Params().
        String("city", "City name", true).
        Build()).
    OnToolCall("get_weather", func(args map[string]any) (string, error) {
        city := args["city"].(string)
        return fmt.Sprintf(`{"temp": 22, "city": "%s"}`, city), nil
    }).
    User("What's the weather in Paris?").
    RunTools(5) // Auto-loop until complete
```

### 🌐 Built-in Tools (OpenAI Responses API)

Access powerful OpenAI-hosted tools with a simple fluent API:

```go
// Web Search - get real-time information from the internet
resp, _ := ai.GPT5().
    WebSearch().
    User("What's the latest news about Go 1.24?").
    Send()

// With location & domain filtering
meta := ai.GPT5().
    WebSearchWith(ai.WebSearchOptions{
        Country:        "US",
        City:           "San Francisco",
        AllowedDomains: []string{"golang.org", "go.dev"},
    }).
    User("What's new in Go?").
    SendWithMeta()

// Access citations from the response
for _, cite := range meta.ResponsesOutput.Citations {
    fmt.Printf("Source: %s - %s\n", cite.Title, cite.URL)
}
```

```go
// File Search - search your vector stores
ai.GPT5().
    FileSearch("vs_abc123").
    User("What's our refund policy?").
    Send()

// With options
ai.GPT5().
    FileSearchWith(ai.FileSearchOptions{
        VectorStoreIDs: []string{"vs_abc123", "vs_def456"},
        MaxNumResults:  5,
    }).
    User("Find pricing documents").
    Send()
```

```go
// Code Interpreter - execute Python in a sandbox
ai.GPT5().
    CodeInterpreter().
    User("Calculate factorial of 100 and plot fibonacci numbers").
    Send()

// With more memory for large datasets
ai.GPT5().
    CodeInterpreterWith(ai.CodeInterpreterOptions{
        MemoryLimit: "4g", // 1g, 4g, 16g, or 64g
    }).
    User("Analyze this CSV data...").
    Send()
```

```go
// MCP - connect to remote MCP servers
ai.GPT5().
    MCP("dice", "https://dmcp-server.deno.dev/sse").
    User("Roll 2d6+3").
    Send()

// Use built-in connectors (Dropbox, Gmail, Google Calendar, etc.)
ai.GPT5().
    MCPConnector("calendar", ai.ConnectorGoogleCalendar, os.Getenv("GOOGLE_TOKEN")).
    User("What's on my calendar today?").
    Send()
```

```go
// Combine multiple tools
ai.GPT5().
    WebSearch().
    CodeInterpreter().
    User("Search for AAPL stock price and create a chart").
    Send()
```

### 🤖 AI Agents (ReAct Pattern)

Build autonomous agents that reason and act:

```go
result := ai.Claude().Agent().
    Tool("search", "Search the web", searchParams, searchHandler).
    Tool("calculate", "Do math", calcParams, calcHandler).
    MaxSteps(15).
    OnStep(func(s ai.AgentStep) {
        fmt.Printf("Step %d: %s\n", s.Number, s.Action)
    }).
    Run("What is 15% of the current US population?")

fmt.Println(result.Answer)
```

### 🖼️ Vision (Image Analysis)

Analyze images with multimodal models:

```go
ai.GPT4o().
    Image("screenshot.png").
    Ask("What's shown in this image?")

ai.Claude().
    Images("before.png", "after.png").
    Ask("What changed between these images?")
```

### 📄 PDF / Document Analysis

Process documents directly (Claude & Gemini):

```go
ai.Anthropic().Claude().
    PDF("report.pdf").
    Ask("Summarize the key findings")

ai.Google().GeminiPro().
    PDF("document.pdf").
    Image("chart.png").
    Ask("Explain the chart in context of the document")
```

### 🎤 Audio (Text-to-Speech & Transcription)

```go
// Text-to-Speech
ai.Speak("Hello world").Voice(ai.VoiceNova).HD().Save("hello.mp3")

// Speech-to-Text (Whisper)
text, _ := ai.Transcribe("meeting.mp3").Do()
```

### 🔢 Embeddings & Semantic Search

```go
// Create embeddings
embedding, _ := ai.Embed("Hello world").First()

// Semantic search
results, _ := ai.SemanticSearch("What is AI?", corpus, 5)
```

### 💬 Conversations with Memory

```go
chat := ai.Claude().
    System("You are a helpful tutor").
    Chat()

chat.Say("What is recursion?")
chat.Say("Can you give me an example?")  // Remembers context
chat.Say("How does that relate to stacks?")
```

### ⚡ Streaming Responses

```go
ai.Claude().
    System("You are a storyteller").
    Stream("Tell me a story about a robot")
```

### 🔄 Smart Retry with Exponential Backoff

```go
response, _ := ai.Claude().
    RetryWithBackoff(3).
    Ask("Complex analysis...")

// Retries automatically on rate limits, timeouts, 5xx errors
// with exponential backoff + jitter
```

### ✅ Response Validation

```go
ai.Claude().
    NoEmptyResponse().
    MinLength(100).
    MaxLength(1000).
    MustContain("conclusion").
    Ask("Write a summary with a conclusion")
```

### 💰 Cost Tracking

```go
meta := ai.Claude().User("Hello").SendWithMeta()
fmt.Printf("Cost: %s\n", meta.CostString()) // "$0.0012"
fmt.Printf("Tokens: %d\n", meta.Tokens)

// Track costs across your application
ai.EnableCostTracking()
// ... make requests ...
ai.PrintCostSummary()
```

### 📦 Batch Processing

```go
results := ai.Batch(
    ai.Claude().User("Question 1"),
    ai.GPT5().User("Question 2"),
    ai.Gemini().User("Question 3"),
).Concurrency(5).Do()

// Or race models for fastest response
answer, winner, _ := ai.Race("What is 2+2?",
    ai.ModelClaudeOpus, ai.ModelGPT5, ai.ModelGemini3Pro)
```

### 🚦 Rate Limiting

```go
// Global rate limiting
ai.RateLimiter = ai.NewLimiter(60, time.Minute) // 60 requests/min
```

---

## Configuration

```go
// Global settings
ai.DefaultModel = ai.ModelClaudeOpus
ai.SetDefaultProvider(ai.ProviderAnthropic)
ai.PromptsDir = "prompts"
ai.Debug = true
ai.Cache = true

// Custom client with options
client := ai.NewClient(ai.ProviderAnthropic,
    ai.WithAPIKey("sk-ant-..."),
    ai.WithTimeout(60 * time.Second),
)
```

---

## Feature Comparison by Provider

| Feature | OpenRouter | OpenAI | Anthropic | Google | Ollama |
|---------|:----------:|:------:|:---------:|:------:|:------:|
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ |
| Function Calling | ✅ | ✅ | ✅ | ✅ | ⚡ |
| Vision | ✅ | ✅ | ✅ | ✅ | ⚡ |
| JSON Mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| Extended Thinking | ✅ | ✅ | ✅ | ✅ | ❌ |
| PDF Input | ❌ | ❌ | ✅ | ✅ | ❌ |
| Web Search | ❌ | ✅ | ❌ | ❌ | ❌ |
| File Search | ❌ | ✅ | ❌ | ❌ | ❌ |
| Code Interpreter | ❌ | ✅ | ❌ | ❌ | ❌ |
| MCP Servers | ❌ | ✅ | ❌ | ❌ | ❌ |

⚡ = Partial support (model-dependent)

---

## Documentation

| Topic | Description |
|-------|-------------|
| [Quick Start](docs/quick-start.md) | Get up and running in 5 minutes |
| [Providers](docs/providers.md) | Multi-provider setup & switching |
| [Models](docs/models.md) | All supported models |
| [Builder API](docs/builder.md) | Fluent API reference |
| [Streaming](docs/streaming.md) | Real-time responses |
| [Conversations](docs/conversations.md) | Multi-turn chat |
| [Tools](docs/tools.md) | Function calling |
| [Built-in Tools](docs/builtin-tools.md) | Web search, file search, code interpreter, MCP |
| [Agents](docs/agents.md) | ReAct pattern agents |
| [Parsing](docs/parsing.md) | Structured output extraction |
| [Vision](docs/vision.md) | Image & PDF analysis |
| [Embeddings](docs/embeddings.md) | Vector embeddings & search |
| [Audio](docs/audio.md) | TTS & transcription |
| [Batch](docs/batch.md) | Parallel processing |
| [Retry](docs/retry.md) | Smart retry strategies |
| [Validation](docs/validation.md) | Response guardrails |
| [Cost](docs/cost.md) | Cost tracking & budgets |
| [Configuration](docs/configuration.md) | Global settings |

---

## Examples

### Build a Research Agent

```go
agent := ai.Claude().
    System("You are a research assistant").
    ThinkHigh().
    Agent().
    Tool("search", "Search the web", searchParams, searchFunc).
    Tool("summarize", "Summarize a URL", summarizeParams, summarizeFunc).
    MaxSteps(20).
    OnStep(func(s ai.AgentStep) {
        log.Printf("Step %d: %s", s.Number, s.Action)
    })

result := agent.Run("Research the latest advances in quantum computing")
fmt.Println(result.Answer)
```

### Extract Structured Data

```go
type Invoice struct {
    Vendor    string  `json:"vendor"`
    Amount    float64 `json:"amount"`
    Currency  string  `json:"currency"`
    DueDate   string  `json:"due_date"`
    LineItems []struct {
        Description string  `json:"description"`
        Quantity    int     `json:"quantity"`
        UnitPrice   float64 `json:"unit_price"`
    } `json:"line_items"`
}

var invoice Invoice
ai.Claude().
    PDF("invoice.pdf").
    IntoWithRetry("Extract all invoice details", &invoice, 3)
```

### Compare Model Responses

```go
results := ai.BatchModels(
    "Explain quantum entanglement in one paragraph",
    ai.ModelClaudeOpus,
    ai.ModelGPT5,
    ai.ModelGemini3Pro,
).Do()

for _, r := range results {
    fmt.Printf("=== %s ===\n%s\n\n", r.Model, r.Content)
}
```

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](license) for details.

---

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b><br>
  Built with ❤️ for the Go community
</p>
