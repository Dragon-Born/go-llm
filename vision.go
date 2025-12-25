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

// ImageInput represents an image included in a multimodal request.
// It can contain either a URL or a base64-encoded data URI.
type ImageInput struct {
	URL    string // URL or base64 data URI
	Detail string // "auto", "low", "high"
}

// ImageDetail controls how the model processes the image.
type ImageDetail string

const (
	ImageDetailAuto ImageDetail = "auto" // Let the model decide resolution
	ImageDetailLow  ImageDetail = "low"  // Lower resolution, faster, fewer tokens
	ImageDetailHigh ImageDetail = "high" // Higher resolution, more detail, more tokens
)

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods for Vision
// ═══════════════════════════════════════════════════════════════════════════

// Image adds a local image file to the request.
// It automatically converts the file to a base64 data URI.
// Uses "auto" detail level by default.
func (b *Builder) Image(path string) *Builder {
	return b.ImageWithDetail(path, ImageDetailAuto)
}

// ImageWithDetail adds a local image file with a specific detail level.
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

// ImageURL adds a remote image URL to the request.
// Uses "auto" detail level by default.
func (b *Builder) ImageURL(url string) *Builder {
	return b.ImageURLWithDetail(url, ImageDetailAuto)
}

// ImageURLWithDetail adds a remote image URL with a specific detail level.
func (b *Builder) ImageURLWithDetail(url string, detail ImageDetail) *Builder {
	b.images = append(b.images, ImageInput{
		URL:    url,
		Detail: string(detail),
	})
	return b
}

// ImageBase64 adds an image from a raw base64 string and mime type.
// Example: b.ImageBase64("iVBORw0...", "image/png")
func (b *Builder) ImageBase64(data, mimeType string) *Builder {
	dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, data)
	b.images = append(b.images, ImageInput{
		URL:    dataURI,
		Detail: string(ImageDetailAuto),
	})
	return b
}

// Images adds multiple local image files to the request.
func (b *Builder) Images(paths ...string) *Builder {
	for _, path := range paths {
		b.Image(path)
	}
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

// fileToDataURI reads a file and converts it to a base64 data URI.
func fileToDataURI(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	mimeType := detectMimeType(path)
	encoded := base64.StdEncoding.EncodeToString(data)
	return fmt.Sprintf("data:%s;base64,%s", mimeType, encoded), nil
}

// detectMimeType guesses the MIME type from the file extension.
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
