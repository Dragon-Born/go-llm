package ai

import (
	"fmt"
	"strings"
)

// ANSI color codes
const (
	reset   = "\033[0m"
	dim     = "\033[2m"
	red     = "\033[31m"
	green   = "\033[32m"
	yellow  = "\033[33m"
	blue    = "\033[34m"
	magenta = "\033[35m"
	cyan    = "\033[36m"
)

func colorRed(s string) string     { return red + s + reset }
func colorGreen(s string) string   { return green + s + reset }
func colorYellow(s string) string  { return yellow + s + reset }
func colorBlue(s string) string    { return blue + s + reset }
func colorMagenta(s string) string { return magenta + s + reset }
func colorCyan(s string) string    { return cyan + s + reset }
func colorDim(s string) string     { return dim + s + reset }

// printDebugRequest prints the outgoing request
func printDebugRequest(model Model, messages []Message) {
	fmt.Println()
	fmt.Println(colorYellow("┌─────────────────────────────────────────────────────────────"))
	fmt.Printf("%s DEBUG REQUEST → %s\n", colorYellow("│"), colorCyan(string(model)))
	fmt.Println(colorYellow("├─────────────────────────────────────────────────────────────"))

	for _, m := range messages {
		var role string
		switch m.Role {
		case "system":
			role = colorMagenta(m.Role)
		case "user":
			role = colorGreen(m.Role)
		default:
			role = colorDim(m.Role)
		}

		fmt.Printf("%s [%s]\n", colorYellow("│"), role)

		// Indent content
		var contentStr string
		if str, ok := m.Content.(string); ok {
			contentStr = str
		}
		lines := strings.Split(contentStr, "\n")
		for _, line := range lines {
			if len(line) > 80 {
				fmt.Printf("%s   %s...\n", colorYellow("│"), line[:77])
			} else {
				fmt.Printf("%s   %s\n", colorYellow("│"), line)
			}
		}
	}

	fmt.Println(colorYellow("└─────────────────────────────────────────────────────────────"))
}

// printDebugResponse prints the incoming response
func printDebugResponse(content string, resp *Response) {
	fmt.Println()
	fmt.Println(colorGreen("┌─────────────────────────────────────────────────────────────"))
	fmt.Printf("%s DEBUG RESPONSE\n", colorGreen("│"))
	fmt.Println(colorGreen("├─────────────────────────────────────────────────────────────"))

	lines := strings.Split(content, "\n")
	for _, line := range lines {
		fmt.Printf("%s %s\n", colorGreen("│"), line)
	}

	fmt.Println(colorGreen("├─────────────────────────────────────────────────────────────"))
	fmt.Printf("%s Tokens: prompt=%d, completion=%d, total=%d\n",
		colorGreen("│"),
		resp.Usage.PromptTokens,
		resp.Usage.CompletionTokens,
		resp.Usage.TotalTokens,
	)
	fmt.Println(colorGreen("└─────────────────────────────────────────────────────────────"))
}

// printPrettyResponse prints a formatted response
func printPrettyResponse(model Model, content string) {
	fmt.Println()
	fmt.Printf("%s %s\n", colorCyan("▸"), colorDim(string(model)))
	fmt.Println(colorDim("─────────────────────────────────────────────────────────────"))
	fmt.Println(content)
	fmt.Println()
}

// printPrettyConversation prints a conversation exchange
func printPrettyConversation(model Model, userMsg, assistantMsg string) {
	fmt.Println()
	fmt.Printf("%s %s\n", colorGreen("You:"), userMsg)
	fmt.Println()
	fmt.Printf("%s %s\n", colorBlue(string(model)+":"), "")
	fmt.Println(assistantMsg)
	fmt.Println()
}
