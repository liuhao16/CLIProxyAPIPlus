package executor

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
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

// qoderImageURLCache dedupes uploads of identical image bytes across requests
// and conversation turns — the same picture (keyed by user + content hash) is
// uploaded once and its resulting URL reused, mirroring qodercli's HzA/p9e
// content cache. The uploaded qoder OSS URLs are unsigned/permanent object
// links, so caching them is safe. A crude size cap keeps memory bounded.
var (
	qoderImageURLCacheMu sync.Mutex
	qoderImageURLCache   = map[string]string{}
)

const qoderImageURLCacheMax = 1024

func qoderImageCacheGet(key string) (string, bool) {
	qoderImageURLCacheMu.Lock()
	defer qoderImageURLCacheMu.Unlock()
	v, ok := qoderImageURLCache[key]
	return v, ok
}

func qoderImageCachePut(key, url string) {
	qoderImageURLCacheMu.Lock()
	defer qoderImageURLCacheMu.Unlock()
	if len(qoderImageURLCache) >= qoderImageURLCacheMax {
		qoderImageURLCache = map[string]string{}
	}
	qoderImageURLCache[key] = url
}

// defaultQoderImageUploadThreshold is the raw-byte size above which an inline
// base64 image is pre-uploaded (in "auto" mode) rather than sent inline. Large
// inline images bloat the chat payload and risk the Alibaba Cloud WAF / size
// limits on the chat endpoint, so big images go through the dedicated upload
// endpoint while small ones stay inline to avoid an extra round-trip.
const defaultQoderImageUploadThreshold = 65536 // 64 KiB of decoded image bytes

// qoderImageUploadMode returns how inline base64 images should be handled,
// controlled by the QODER_IMAGE_UPLOAD env var. The default matches the
// official qodercli behavior: upload every image, fall back to inline base64
// only when the upload fails.
//
//	"" / always (default) — upload every image (official behavior).
//	auto                  — upload only images past the size threshold; keep
//	                        smaller images inline (saves a round-trip).
//	0 / never / off       — never upload; always keep inline base64.
//
// Upload targets the device-token CENTER endpoint
// (center.qoder.sh/algo/api/v2/image/upload, COSY-signed over the body length)
// and always falls back to inline base64 on any failure, so image forwarding
// never breaks regardless of mode.
func qoderImageUploadMode() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("QODER_IMAGE_UPLOAD"))) {
	case "0", "off", "never", "false", "no":
		return "never"
	case "auto":
		return "auto"
	default:
		return "always"
	}
}

// qoderImageUploadThreshold is the decoded-byte size above which "auto" mode
// uploads, overridable via QODER_IMAGE_UPLOAD_THRESHOLD (bytes).
func qoderImageUploadThreshold() int {
	if v := strings.TrimSpace(os.Getenv("QODER_IMAGE_UPLOAD_THRESHOLD")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return defaultQoderImageUploadThreshold
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
	mode := qoderImageUploadMode()
	if mode == "never" {
		return
	}
	threshold := qoderImageUploadThreshold()
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
			// In auto mode small images stay inline (no extra round-trip); only
			// images past the threshold are uploaded. "always" uploads all.
			rawLen := len(data) * 3 / 4
			if mode == "auto" && rawLen <= threshold {
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
	// Content cache: identical bytes for the same user reuse a prior upload URL,
	// so a picture repeated across conversation turns is only uploaded once.
	sum := sha256.Sum256([]byte(b64))
	cacheKey := storage.UserID + ":" + hex.EncodeToString(sum[:])
	if cached, ok := qoderImageCacheGet(cacheKey); ok {
		return cached, nil
	}

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
	// Device-token accounts upload via the CENTER endpoint. qodercli's Tv() URL
	// joiner inserts an "/algo" gateway segment before the path, so the real
	// URL is center.qoder.sh/algo/api/v2/image/upload — the missing /algo is why
	// a naive center-host PUT 404s.
	url := qoderauth.QoderCenterBase + "/algo" + qoderImageUploadPath + "?request_id=" + reqID

	// Retry transient auth/server failures (mirrors qodercli's 401/403 retry).
	// Each attempt re-signs with a fresh timestamp/request-id.
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		result, status, err := e.doQoderImageUpload(ctx, authRecord, storage, url, body, boundary)
		if err == nil {
			qoderImageCachePut(cacheKey, result)
			return result, nil
		}
		lastErr = err
		// Only retry on transient auth (401/403) or server (5xx) statuses.
		if status != http.StatusUnauthorized && status != http.StatusForbidden && status < 500 {
			break
		}
	}
	return "", lastErr
}

// doQoderImageUpload performs a single COSY-signed PUT of the multipart body and
// returns the assigned URL. status is the HTTP status (0 on transport error) so
// the caller can decide whether to retry.
func (e *QoderExecutor) doQoderImageUpload(ctx context.Context, authRecord *cliproxyauth.Auth, storage *qoderauth.QoderTokenStorage, url string, body []byte, boundary string) (result string, status int, err error) {
	// The upload's COSY signature is computed over the body *length string*, not
	// the multipart bytes: qodercli signs it via
	// prepareRequest(endpoint, path, "PUT", "auth", String(body.length), _).
	// Passing the raw bytes (as a normal chat request would) yields HTTP 403
	// "Signature invalid". The real bytes are still sent as the HTTP body below.
	sigBody := []byte(strconv.Itoa(len(body)))
	headers, err := qoderauth.BuildAuthHeaders(sigBody, url, qoderauth.CosyCredentials{
		UserID:    storage.UserID,
		AuthToken: storage.Token,
		Name:      storage.Name,
		Email:     storage.Email,
		MachineID: storage.MachineID,
	})
	if err != nil {
		return "", 0, fmt.Errorf("build COSY auth: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewReader(body))
	if err != nil {
		return "", 0, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("Accept-Encoding", "identity")
	httpReq.Header.Set("AI-CLIENT-TIMESTAMP", strconv.FormatInt(time.Now().Unix(), 10))
	headers.Apply(httpReq)

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, authRecord, 60*time.Second)
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return "", 0, fmt.Errorf("upload request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", resp.StatusCode, fmt.Errorf("upload HTTP %d: %s", resp.StatusCode, truncate(string(respBody), 300))
	}

	// Match qodercli's U_i URL extraction order.
	for _, path := range []string{"url", "result.url", "result.oss_url", "data.url", "data.oss_url"} {
		if v := gjson.GetBytes(respBody, path).String(); v != "" {
			log.Infof("qoder image-upload: uploaded %d-byte multipart -> %s", len(body), v)
			return v, resp.StatusCode, nil
		}
	}
	return "", resp.StatusCode, fmt.Errorf("upload response missing url: %s", truncate(string(respBody), 300))
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
