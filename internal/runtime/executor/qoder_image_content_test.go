package executor

import (
	"reflect"
	"testing"
)

// TestBuildQoderContent_TextOnlyCollapses verifies plain-text multipart content
// collapses to a single string (Qoder's default for text-only turns).
func TestBuildQoderContent_TextOnlyCollapses(t *testing.T) {
	in := []interface{}{
		map[string]interface{}{"type": "text", "text": "hello"},
		map[string]interface{}{"type": "text", "text": "world"},
	}
	got := buildQoderContent(in)
	if s, ok := got.(string); !ok || s != "hello\nworld" {
		t.Fatalf("text-only content = %#v, want string %q", got, "hello\nworld")
	}
}

// TestBuildQoderContent_StringPassthrough verifies a bare string stays a string.
func TestBuildQoderContent_StringPassthrough(t *testing.T) {
	if got := buildQoderContent("just text"); got != "just text" {
		t.Fatalf("string content = %#v, want %q", got, "just text")
	}
}

// TestBuildQoderContent_PreservesImageOrder verifies image_url parts survive
// (both the Claude Code and Codex CLI paths land here as OpenAI image_url with
// an inline data: URL) and that text/image ordering is preserved.
func TestBuildQoderContent_PreservesImageOrder(t *testing.T) {
	dataURL := "data:image/png;base64,aGVsbG8="
	in := []interface{}{
		map[string]interface{}{"type": "text", "text": "look:"},
		map[string]interface{}{
			"type":      "image_url",
			"image_url": map[string]interface{}{"url": dataURL, "detail": "high"},
		},
	}
	got, ok := buildQoderContent(in).([]interface{})
	if !ok {
		t.Fatalf("content with image = %#v, want []interface{}", buildQoderContent(in))
	}
	if len(got) != 2 {
		t.Fatalf("part count = %d, want 2 (text then image)", len(got))
	}
	text, _ := got[0].(map[string]interface{})
	if text["type"] != "text" || text["text"] != "look:" {
		t.Fatalf("part[0] = %#v, want text 'look:'", got[0])
	}
	want := map[string]interface{}{
		"type":      "image_url",
		"image_url": map[string]interface{}{"url": dataURL, "detail": "high"},
	}
	if !reflect.DeepEqual(got[1], want) {
		t.Fatalf("part[1] = %#v, want %#v", got[1], want)
	}
}

// TestBuildQoderContent_BareStringImageURL verifies the tolerant path where
// image_url is a bare string (some translators emit that) is normalized to the
// canonical object form.
func TestBuildQoderContent_BareStringImageURL(t *testing.T) {
	in := []interface{}{
		map[string]interface{}{"type": "image_url", "image_url": "https://example.com/a.png"},
	}
	got, ok := buildQoderContent(in).([]interface{})
	if !ok || len(got) != 1 {
		t.Fatalf("content = %#v, want single image part", buildQoderContent(in))
	}
	want := map[string]interface{}{
		"type":      "image_url",
		"image_url": map[string]interface{}{"url": "https://example.com/a.png"},
	}
	if !reflect.DeepEqual(got[0], want) {
		t.Fatalf("part[0] = %#v, want %#v", got[0], want)
	}
}

// TestParseImageDataURL covers data-URL parsing used by the upload path.
func TestParseImageDataURL(t *testing.T) {
	mt, data, ok := parseImageDataURL("data:image/jpeg;base64,/9j/4AAQ")
	if !ok || mt != "image/jpeg" || data != "/9j/4AAQ" {
		t.Fatalf("parse base64 data URL = (%q,%q,%v)", mt, data, ok)
	}
	if _, _, ok := parseImageDataURL("https://example.com/a.png"); ok {
		t.Fatalf("remote URL should not parse as data URL")
	}
	if _, _, ok := parseImageDataURL("data:image/png,notbase64"); ok {
		t.Fatalf("non-base64 data URL should not parse")
	}
}
