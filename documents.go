package ai

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Document / PDF Input
// ═══════════════════════════════════════════════════════════════════════════

// DocumentInput represents a document (PDF) included in a request.
type DocumentInput struct {
	Data     string // base64-encoded data
	URL      string // or URL (mutually exclusive with Data)
	MimeType string // "application/pdf"
	Name     string // optional filename for context
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Documents
// ═══════════════════════════════════════════════════════════════════════════

// PDF adds a PDF document from a local file path.
func (b *Builder) PDF(path string) *Builder {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("%s Error loading PDF %s: %v\n", colorRed("✗"), path, err)
		return b
	}

	encoded := base64.StdEncoding.EncodeToString(data)
	b.documents = append(b.documents, DocumentInput{
		Data:     encoded,
		MimeType: detectDocMimeType(path),
		Name:     filepath.Base(path),
	})
	return b
}

// PDFURL adds a PDF document from a URL.
func (b *Builder) PDFURL(url string) *Builder {
	b.documents = append(b.documents, DocumentInput{
		URL:      url,
		MimeType: "application/pdf",
	})
	return b
}

// PDFBase64 adds a PDF document from base64-encoded data.
func (b *Builder) PDFBase64(data string) *Builder {
	b.documents = append(b.documents, DocumentInput{
		Data:     data,
		MimeType: "application/pdf",
	})
	return b
}

// PDFs adds multiple PDF documents at once.
func (b *Builder) PDFs(paths ...string) *Builder {
	for _, path := range paths {
		b.PDF(path)
	}
	return b
}

// Document adds a document with auto-detected type (PDF, etc.).
func (b *Builder) Document(path string) *Builder {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		return b.PDF(path)
	default:
		fmt.Printf("%s Unsupported document type: %s (only PDF supported)\n", colorYellow("⚠"), ext)
		return b
	}
}

// DocumentURL adds a document from URL with auto-detected type.
func (b *Builder) DocumentURL(url string) *Builder {
	// Check URL extension
	lower := strings.ToLower(url)
	if strings.HasSuffix(lower, ".pdf") || strings.Contains(lower, ".pdf?") {
		return b.PDFURL(url)
	}
	// Default to PDF
	return b.PDFURL(url)
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

// detectDocMimeType returns the MIME type based on file extension.
func detectDocMimeType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		return "application/pdf"
	default:
		return "application/pdf" // fallback
	}
}

// HasDocuments reports whether the builder has documents attached.
func (b *Builder) HasDocuments() bool {
	return len(b.documents) > 0
}

// GetDocuments returns the attached documents.
func (b *Builder) GetDocuments() []DocumentInput {
	return b.documents
}
