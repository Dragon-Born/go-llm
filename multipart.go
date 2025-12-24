package ai

import (
	"io"
	"mime/multipart"
)

// multipartWriter wraps multipart.Writer for internal use
type multipartWriter struct {
	*multipart.Writer
}

// newMultipartWriter creates a new multipart writer
func newMultipartWriter(w io.Writer) *multipartWriter {
	return &multipartWriter{multipart.NewWriter(w)}
}
