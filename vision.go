package ai

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// Vision / Image Input
// ═══════════════════════════════════════════════════════════════════════════

// ImageInput represents an image to include in the request
type ImageInput struct {
	URL    string // URL or base64 data URI
	Detail string // "auto", "low", "high"
}

// ImageDetail controls image processing quality
type ImageDetail string

const (
	ImageDetailAuto ImageDetail = "auto" // Let the model decide
	ImageDetailLow  ImageDetail = "low"  // Faster, less tokens
	ImageDetailHigh ImageDetail = "high" // More detail, more tokens
)

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Vision
// ═══════════════════════════════════════════════════════════════════════════

// Image adds an image from a local file path
func (b *Builder) Image(path string) *Builder {
	return b.ImageWithDetail(path, ImageDetailAuto)
}

// ImageWithDetail adds an image with specific detail level
func (b *Builder) ImageWithDetail(path string, detail ImageDetail) *Builder {
	dataURI, err := fileToDataURI(path)
	if err != nil {
		fmt.Printf("%s Error loading image %s: %v\n", colorRed("✗"), path, err)
		return b
	}

	b.images = append(b.images, ImageInput{
		URL:    dataURI,
		Detail: string(detail),
	})
	return b
}

// ImageURL adds an image from a URL
func (b *Builder) ImageURL(url string) *Builder {
	return b.ImageURLWithDetail(url, ImageDetailAuto)
}

// ImageURLWithDetail adds an image URL with specific detail level
func (b *Builder) ImageURLWithDetail(url string, detail ImageDetail) *Builder {
	b.images = append(b.images, ImageInput{
		URL:    url,
		Detail: string(detail),
	})
	return b
}

// ImageBase64 adds an image from base64 data
func (b *Builder) ImageBase64(data, mimeType string) *Builder {
	dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, data)
	b.images = append(b.images, ImageInput{
		URL:    dataURI,
		Detail: string(ImageDetailAuto),
	})
	return b
}

// Images adds multiple images at once
func (b *Builder) Images(paths ...string) *Builder {
	for _, path := range paths {
		b.Image(path)
	}
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

// fileToDataURI converts a local image file to a data URI
func fileToDataURI(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	mimeType := detectMimeType(path)
	encoded := base64.StdEncoding.EncodeToString(data)
	return fmt.Sprintf("data:%s;base64,%s", mimeType, encoded), nil
}

// detectMimeType returns the MIME type based on file extension
func detectMimeType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	default:
		return "image/png" // fallback
	}
}
