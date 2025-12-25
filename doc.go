// Package ai provides a fluent Go client for multiple LLM providers (OpenAI, Anthropic,
// Google Gemini, Ollama, Azure OpenAI), either via OpenRouter (default) or direct provider clients.
//
// Quick start:
//
//	resp, err := ai.Claude().
//		System("You are a helpful assistant.").
//		Ask("What is the capital of France?")
//
// API keys:
//
// Most hosted providers read API keys from environment variables:
//
//	OPENROUTER_API_KEY   // OpenRouter (default)
//	OPENAI_API_KEY       // OpenAI
//	ANTHROPIC_API_KEY    // Anthropic
//	GOOGLE_API_KEY       // Google Gemini (or GEMINI_API_KEY)
//	AZURE_OPENAI_API_KEY // Azure OpenAI (falls back to OPENAI_API_KEY)
//
// You can also pass keys explicitly when creating clients:
//
//	openai := ai.OpenAIWith(ai.WithAPIKey("sk-..."))
//	resp, err := openai.GPT4o().Ask("Hello")
//
// Provider selection:
//
//	// Uses DefaultProvider (OpenRouter by default).
//	ai.GPT5().Ask("Hello")
//
//	// Explicit provider clients.
//	ai.OpenAI().GPT4o().Ask("Hello")
//	ai.Anthropic().Claude().Ask("Hello")
//	ai.Google().Gemini().Ask("Hello")
//	ai.Ollama().Use("llama3:8b").Ask("Hello")
//	ai.Azure("https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT").GPT4o().Ask("Hello")
//
// Prompt variables:
//
//	resp, err := ai.Claude().
//		System("Hello {{name}}").
//		With(ai.Vars{"name": "Arian"}).
//		Ask("Say hi")
//
// Streaming:
//
//	_, _ = ai.Claude().
//		System("You are a storyteller.").
//		User("Tell me a short story.").
//		StreamResponse(func(chunk string) { fmt.Print(chunk) })
//
// Tools / function calling:
//
//	b := ai.GPT5().
//		Tool("get_weather", "Get weather for a city", ai.Params().
//			String("city", "City name", true).
//			Build()).
//		OnToolCall("get_weather", func(args map[string]any) (string, error) {
//			return `{"temp_c": 22}`, nil
//		})
//
//	resp, err := b.User("What's the weather in Paris?").RunTools(5)
//
// Built-in Responses API tools (OpenAI-only):
//
//	ai.GPT5().WebSearch().User("What's new in Go?").Send()
//	ai.GPT5().FileSearch("vs_abc123").User("Find the refund policy").Send()
//	ai.GPT5().CodeInterpreter().User("Plot a sine wave").Send()
//	ai.GPT5().MCP("dice", "https://example.com/mcp").User("Roll 2d6+3").Send()
//
// Structured output:
//
//	type Person struct {
//		Name  string `json:"name"`
//		Age   int    `json:"age"`
//		Email string `json:"email"`
//	}
//
//	var p Person
//	if err := ai.Claude().Into("Extract name/age/email from: John, 32, john@example.com", &p); err != nil {
//		// handle error
//	}
//
// For parsing that retries on JSON/schema errors, use Builder.IntoWithRetry.
//
// Vision / documents:
//
//	ai.GPT4o().Image("screenshot.png").Ask("What is shown?")
//	ai.Claude().PDF("report.pdf").Ask("Summarize the key findings")
//
// Embeddings / audio:
//
//	vec, _ := ai.Embed("hello").First()
//	_ = vec
//	_ = ai.Speak("Hello world").Voice(ai.VoiceNova).HD().Save("hello.mp3")
//
// Retries, guardrails, and rate limiting:
//
//	ai.RateLimiter = ai.NewLimiter(60, time.Minute)
//	resp, err := ai.Claude().
//		RetryWithBackoff(3).
//		NoEmptyResponse().
//		MinLength(50).
//		Ask("Write a short summary")
//
// Costs and stats:
//
//	meta := ai.Claude().User("Hello").SendWithMeta()
//	fmt.Println(meta.CostString())
//
//	ai.EnableCostTracking()
//	// ... make requests ...
//	ai.PrintCostSummary()
//	ai.PrintStats()
//
// Batch / race:
//
//	results := ai.Batch(
//		ai.GPT5().User("Q1"),
//		ai.Claude().User("Q2"),
//	).Concurrency(5).Do()
//	_ = results
//
//	ans, winner, _ := ai.Race("What is 2+2?", ai.ModelGPT5, ai.ModelClaudeOpus)
//	_, _ = ans, winner
//
// Caching:
//
//	ai.Cache = true // enable in-memory caching of identical requests (experimental)
package ai
