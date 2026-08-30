package executor

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	qoderauth "github.com/router-for-me/CLIProxyAPI/v7/internal/auth/qoder"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

// qoderImageUploadPath is Qoder's image upload endpoint (COSY-signed PUT with a
// multipart/form-data body, field name "file"). Mirrors qodercli's x_i.
const qoderImageUploadPath = "/api/v2/image/upload"

// qoderImageUploadEnabled reports whether inline base64 images should be
// pre-uploaded to Qoder (and rewritten to URL references) before a chat call.
//
// This is opt-in because Qoder already accepts inline data: base64 URLs in
// image_url.url (see chat.proto ContentPart.image_url), so the upload is a
// payload-size / WAF optimization for large images rather than a requirement.
// Enable with QODER_IMAGE_UPLOAD=1 (or true/yes/on).
func qoderImageUploadEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("QODER_IMAGE_UPLOAD"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// uploadQoderImagesInMessages walks the normalized messages and, for every
// image_url part carrying an inline data:<mime>;base64,... URL, uploads the
// bytes to Qoder and rewrites the part's url to the returned reference —
// exactly what the official qodercli does before a chat call (its dfA/j3a
// path). The inline base64 is kept as a fallback when an upload fails, which
// is also qodercli's documented behavior ("keeping base64 image").
//
// Parts are the OpenAI chat shape produced by buildQoderContent:
// {"type":"image_url","image_url":{"url":"data:...;base64,..."}}. Remote http(s)
// image_url values are left untouched.
func (e *QoderExecutor) uploadQoderImagesInMessages(ctx context.Context, authRecord *cliproxyauth.Auth, storage *qoderauth.QoderTokenStorage, messages []interface{}) {
	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := msg["content"].([]interface{})
		if !ok {
			continue
		}
		for _, item := range content {
			part, ok := item.(map[string]interface{})
			if !ok || part["type"] != "image_url" {
				continue
			}
			imgURL, ok := part["image_url"].(map[string]interface{})
			if !ok {
				continue
			}
			url, _ := imgURL["url"].(string)
			mediaType, data, ok := parseImageDataURL(url)
			if !ok || data == "" {
				// Not an inline base64 image (e.g. a remote https URL) — skip.
				continue
			}
			uploaded, err := e.uploadQoderImage(ctx, authRecord, storage, mediaType, data)
			if err != nil || uploaded == "" {
				log.Warnf("qoder image-upload: upload failed, keeping inline base64: %v", err)
				continue
			}
			imgURL["url"] = uploaded
		}
	}
}

// uploadQoderImage PUTs a single base64 image to Qoder's image upload endpoint
// and returns the URL the server assigns. The multipart body is signed with the
// same COSY scheme used for chat.
func (e *QoderExecutor) uploadQoderImage(ctx context.Context, authRecord *cliproxyauth.Auth, storage *qoderauth.QoderTokenStorage, mediaType, b64 string) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", fmt.Errorf("decode base64: %w", err)
	}
	if mediaType == "" {
		mediaType = "image/png"
	}

	boundary := "----qodercli-" + strings.ReplaceAll(uuid.New().String(), "-", "")
	var buf bytes.Buffer
	buf.WriteString("--" + boundary + "\r\n")
	buf.WriteString(`Content-Disposition: form-data; name="file"; filename="image.` + qoderImageExt(mediaType) + `"` + "\r\n")
	buf.WriteString("Content-Type: " + mediaType + "\r\n\r\n")
	buf.Write(raw)
	buf.WriteString("\r\n--" + boundary + "--\r\n")
	body := buf.Bytes()

	reqID := strings.ReplaceAll(uuid.New().String(), "-", "")
	url := qoderauth.QoderCenterBase + qoderImageUploadPath + "?request_id=" + reqID

	headers, err := qoderauth.BuildAuthHeaders(body, url, qoderauth.CosyCredentials{
		UserID:    storage.UserID,
		AuthToken: storage.Token,
		Name:      storage.Name,
		Email:     storage.Email,
		MachineID: storage.MachineID,
	})
	if err != nil {
		return "", fmt.Errorf("build COSY auth: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("Accept-Encoding", "identity")
	httpReq.Header.Set("AI-CLIENT-TIMESTAMP", strconv.FormatInt(time.Now().Unix(), 10))
	headers.Apply(httpReq)

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, authRecord, 60*time.Second)
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("upload request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("upload HTTP %d: %s", resp.StatusCode, truncate(string(respBody), 300))
	}

	// Match qodercli's U_i URL extraction order.
	for _, path := range []string{"url", "result.url", "result.oss_url", "data.url", "data.oss_url"} {
		if v := gjson.GetBytes(respBody, path).String(); v != "" {
			return v, nil
		}
	}
	return "", fmt.Errorf("upload response missing url: %s", truncate(string(respBody), 300))
}

// qoderImageExt maps a media type to the file extension qodercli uses for the
// uploaded part's filename (image.<ext>).
func qoderImageExt(mediaType string) string {
	switch strings.ToLower(strings.TrimSpace(mediaType)) {
	case "image/png":
		return "png"
	case "image/jpeg", "image/jpg":
		return "jpg"
	case "image/gif":
		return "gif"
	case "image/webp":
		return "webp"
	case "image/bmp":
		return "bmp"
	default:
		return "png"
	}
}
