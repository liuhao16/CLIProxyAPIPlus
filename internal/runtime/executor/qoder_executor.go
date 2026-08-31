package executor

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	qoderauth "github.com/router-for-me/CLIProxyAPI/v7/internal/auth/qoder"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

// QoderExecutor executes requests against the Qoder API with COSY authentication
type QoderExecutor struct {
	cfg *config.Config
}

// NewQoderExecutor creates a new Qoder executor
func NewQoderExecutor(cfg *config.Config) *QoderExecutor {
	return &QoderExecutor{
		cfg: cfg,
	}
}

// Identifier returns the provider identifier
func (e *QoderExecutor) Identifier() string {
	return "qoder"
}

// ExecuteStream executes a streaming request against Qoder API
func (e *QoderExecutor) ExecuteStream(ctx context.Context, authRecord *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	// Get token storage from auth record
	storage, ok := authRecord.Storage.(*qoderauth.QoderTokenStorage)
	if !ok {
		return nil, fmt.Errorf("invalid auth storage type for qoder: %T", authRecord.Storage)
	}

	// Note: Qoder device tokens are long-lived (~30 days) and the upstream
	// /algo/api/v3/user/refresh_token endpoint returns 403 for them — see
	// QoderExecutor.Refresh's no-op rationale. We deliberately do not call
	// RefreshTokenIfNeeded per request: it would just produce a 403 in the
	// log on every chat call. Token expiry is handled by the user re-running
	// --qoder-login.

	// Translate non-openai formats to chat completions before extracting messages
	payload := req.Payload
	if opts.SourceFormat != "" && opts.SourceFormat != sdktranslator.FormatOpenAI {
		payload = sdktranslator.TranslateRequest(opts.SourceFormat, sdktranslator.FormatOpenAI, req.Model, payload, false)
	}

	// Parse request to get model and messages
	var chatReq map[string]interface{}
	if err := json.Unmarshal(payload, &chatReq); err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}

	// Map model name — strip provider prefix so qoder/auto → auto
	model := req.Model
	if model == "" {
		model, _ = chatReq["model"].(string)
	}
	qoderModel, err := validateQoderModel(model, storage)
	if err != nil {
		return nil, err
	}
	reporter := helps.NewExecutorUsageReporter(ctx, e, qoderModel, authRecord)
	defer reporter.TrackFailure(ctx, &err)

	// Normalize messages: flatten Anthropic/OpenAI multipart content arrays
	// to plain strings (Qoder's chat endpoint expects content to be a string).
	// tool_calls / role:"tool" turns pass through verbatim — Qoder accepts
	// the canonical OpenAI structure and emits real tool_use events.
	messagesRaw, _ := chatReq["messages"].([]interface{})
	toolsRaw := chatReq["tools"]
	normalized, systemText := normalizeQoderMessages(messagesRaw)

	// Pre-upload large inline base64 images to Qoder and rewrite them to URL
	// references (mirrors the official qodercli flow, which uploads then keeps
	// base64 only as a fallback). Default "auto": images over the size threshold
	// upload, smaller ones stay inline. Tunable via QODER_IMAGE_UPLOAD
	// (auto/always/never) and QODER_IMAGE_UPLOAD_THRESHOLD. Any upload failure
	// keeps the inline base64, which Qoder also accepts, so image forwarding
	// never breaks.
	e.uploadQoderImagesInMessages(ctx, authRecord, storage, normalized)

	// Resolve the per-model server-side metadata (is_vl, is_reasoning,
	// max_input_tokens, ...). Failing here is a hard error — sending the
	// wrong block silently downgrades to a different model.
	modelConfig, err := buildQoderModelConfig(storage, qoderModel)
	if err != nil {
		return nil, err
	}

	isReasoning, _ := modelConfig["is_reasoning"].(bool)
	maxOutputTokens, _ := modelConfig["max_output_tokens"].(float64)

	// Last user message text — used by Qoder for the chat_context "current
	// turn" preview slot. The full conversation still goes through `messages`.
	lastUser := lastUserText(normalized)

	// Stable IDs derived from content so retries hit upstream caches.
	// session_id is stable per user+model (routing affinity).
	// chat_record_id is deterministic per payload (dedup/cache key).
	sessionID := stableHash("qoder-session", storage.UserID, qoderModel)
	recordID := stableChatRecordID(qoderModel, normalized, toolsRaw, int(maxOutputTokens))

	// Start with the model's maximum output tokens, then clamp to
	// any user-requested limit so callers can cap cost/latency/UI.
	maxTokens := 32768
	if maxOutputTokens > 0 {
		maxTokens = int(maxOutputTokens)
	}
	if userMax, ok := chatReq["max_tokens"].(float64); ok && userMax > 0 {
		if int(userMax) < maxTokens {
			maxTokens = int(userMax)
		}
	}
	if userMax, ok := chatReq["max_completion_tokens"].(float64); ok && userMax > 0 {
		if int(userMax) < maxTokens {
			maxTokens = int(userMax)
		}
	}

	reqBody := map[string]interface{}{
		"request_id":       uuid.New().String(),
		"request_set_id":   recordID,
		"chat_record_id":   recordID,
		"session_id":       sessionID,
		"stream":           true,
		"chat_task":        "FREE_INPUT",
		"is_reply":         true,
		"is_retry":         false,
		"source":           1,
		"version":          "3",
		"session_type":     "qodercli",
		"agent_id":         "agent_common",
		"task_id":          "common",
		"code_language":    "",
		"chat_prompt":      "",
		"image_urls":       nil,
		"aliyun_user_type": "",
		"system":           systemText,
		"messages":         normalized,
		"tools":            []interface{}{},
		"parameters":       map[string]interface{}{"max_tokens": maxTokens},
		"chat_context": map[string]interface{}{
			"chatPrompt": "",
			"imageUrls":  nil,
			"extra": map[string]interface{}{
				"context": []interface{}{},
				"modelConfig": map[string]interface{}{
					"key":          qoderModel,
					"is_reasoning": isReasoning,
				},
				"originalContent": lastUser,
			},
			"features": []interface{}{},
			"text":     lastUser,
		},
		"model_config": modelConfig,
		"business": map[string]interface{}{
			"product":  "cli",
			"version":  "1.0.0",
			"type":     "agent",
			"stage":    "start",
			"id":       uuid.New().String(),
			"name":     truncate(lastUser, 30),
			"begin_at": time.Now().UnixMilli(),
		},
	}
	if toolsRaw != nil {
		reqBody["tools"] = toolsRaw
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	// Encode the body to bypass Alibaba Cloud WAF pattern matching.
	// The server decodes when &Encode=1 is present in the URL.
	encodedBytes := []byte(helps.QoderEncodeBody(bodyBytes))

	modelSource, _ := modelConfig["source"].(string)
	if modelSource == "" {
		modelSource = "system"
	}

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, authRecord, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)

	creds := qoderauth.CosyCredentials{
		UserID:    storage.UserID,
		AuthToken: storage.Token,
		Name:      storage.Name,
		Email:     storage.Email,
		MachineID: storage.MachineID,
	}

	// sendInference issues one signed inference request. The COSY signature is
	// rebuilt on every call because it carries a timestamp that would go stale
	// across a queue wait.
	sendInference := func() (*http.Response, error) {
		headers, herr := qoderauth.BuildAuthHeaders(encodedBytes, qoderauth.QoderChatURLEncoded, creds)
		if herr != nil {
			return nil, fmt.Errorf("failed to build COSY auth: %w", herr)
		}
		httpReq, rerr := http.NewRequestWithContext(ctx, "POST", qoderauth.QoderChatURLEncoded, bytes.NewReader(encodedBytes))
		if rerr != nil {
			return nil, fmt.Errorf("failed to create request: %w", rerr)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		httpReq.Header.Set("Cache-Control", "no-cache")
		headers.Apply(httpReq)
		httpReq.Header.Set("X-Model-Key", qoderModel)
		httpReq.Header.Set("X-Model-Source", modelSource)
		// Disable automatic gzip — Accept-Encoding: gzip triggers signature
		// validation on the Qoder upstream and causes 403 Signature invalid.
		httpReq.Header.Set("Accept-Encoding", "identity")
		return httpClient.Do(httpReq)
	}

	// Send the request, waiting out the upstream's model queue (403 /
	// code=10605 / isQueued:true) per the resolved queue policy.
	queue := resolveQoderQueueSettings(e.cfg)
	var httpResp *http.Response
	queueDeadline := time.Now().Add(queue.maxWait)
	// queueWaitStart is set the first time this request is queued; it drives the
	// time_consumed value reported by the official queue-finish callback and
	// signals (non-zero) that a finish callback should be fired at all.
	var queueWaitStart time.Time
	var queueFinishModelKey string
	for attempt := 0; ; attempt++ {
		resp, derr := sendInference()
		if derr != nil {
			return nil, fmt.Errorf("request failed: %w", derr)
		}

		if resp.StatusCode == http.StatusOK {
			httpResp = resp
			break
		}

		// Non-200: read body once, decide whether it is a retriable queue
		// signal or a hard error.
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		qinfo, retriable := parseQoderQueue(string(body))
		if retriable && queue.enabled {
			if queueWaitStart.IsZero() {
				queueWaitStart = time.Now()
			}
			// Prefer the upstream-reported modelKey; fall back to the model we
			// asked for. Used by the queue-finish callback.
			if qinfo.modelKey != "" {
				queueFinishModelKey = qinfo.modelKey
			} else if queueFinishModelKey == "" {
				queueFinishModelKey = qoderModel
			}
			// Prefer the official queue-status polling: ask the upstream when
			// the model is ready instead of blindly re-sending the inference
			// request. Falls back to a plain backoff-and-resend when disabled
			// or when the queue signal lacks a requestSetId to poll with.
			if queue.useStatusEndpoint && recordID != "" {
				ready, werr := e.waitQoderQueue(ctx, httpClient, creds, recordID, sessionID, qoderModel, qinfo, queue, queueDeadline)
				if werr != nil {
					if errors.Is(werr, context.Canceled) || errors.Is(werr, context.DeadlineExceeded) {
						return nil, werr
					}
					// Budget exhausted or unrecoverable poll failure: surface
					// the original queue 403 to the client.
					log.Warnf("qoder: queue wait ended without ready (attempt %d): %v; giving up", attempt+1, werr)
					return nil, newQoderStatusError(resp.StatusCode, string(body))
				}
				_ = ready
				continue
			}

			wait := qinfo.backoff(queue)
			if time.Now().Add(wait).After(queueDeadline) {
				log.Warnf("qoder: queue wait budget exhausted (attempt %d, queueCount=%d, waitTime=%ds); giving up", attempt+1, qinfo.queueCount, qinfo.waitTime)
				return nil, newQoderStatusError(resp.StatusCode, string(body))
			}
			log.Infof("qoder: queued (attempt %d, queueCount=%d, waitTime=%ds); retrying in %s", attempt+1, qinfo.queueCount, qinfo.waitTime, wait)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
			continue
		}

		// Hard error — log full diagnostics and surface it.
		allow := resp.Header.Get("Allow")
		server := resp.Header.Get("Server")
		bodyPreview := truncate(string(body), 500)
		log.WithFields(log.Fields{
			"url":            qoderauth.QoderChatURL,
			"server":         server,
			"content_type":   resp.Header.Get("Content-Type"),
			"x_request_id":   resp.Header.Get("X-Request-Id"),
			"x_eagleeye_id":  resp.Header.Get("Eagleeye-Traceid"),
			"x_oss_request":  resp.Header.Get("X-Oss-Request-Id"),
			"allow":          allow,
			"body_truncated": bodyPreview,
		}).Warnf("qoder: upstream %d allow=%q server=%q body=%q", resp.StatusCode, allow, server, bodyPreview)
		return nil, newQoderStatusError(resp.StatusCode, string(body))
	}

	// Create streaming channel
	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() { _ = httpResp.Body.Close() }()
		// Fire the official queue-finish callback once the stream terminates
		// (any exit path). Best-effort usage reporting; only when this request
		// actually waited in queue and we have the identifiers to report with.
		defer func() {
			if queueWaitStart.IsZero() || !queue.reportFinish {
				return
			}
			if recordID == "" || creds.UserID == "" {
				return
			}
			mk := queueFinishModelKey
			if mk == "" {
				mk = qoderModel
			}
			e.finishQoderQueue(context.Background(), httpClient, creds, mk, recordID, time.Since(queueWaitStart))
		}()

		// Shared across all TranslateStream calls in this stream — the
		// translator carries open-block / sequence state through it; a
		// per-chunk var would re-emit message_start on every delta.
		var streamParam any

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800) // 50MB max line

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			// Skip non-data lines
			if !bytes.HasPrefix(line, []byte("data:")) {
				continue
			}

			data := bytes.TrimPrefix(line, []byte("data:"))
			data = bytes.TrimPrefix(data, []byte(" "))
			if bytes.Equal(data, []byte("[DONE]")) {
				emitDone(ctx, out, opts.SourceFormat, req.Model, opts.OriginalRequest, payload, &streamParam)
				reporter.EnsurePublished(ctx)
				return
			}

			// Parse Qoder response envelope
			var event map[string]interface{}
			if err := json.Unmarshal(data, &event); err != nil {
				continue
			}
			statusVal := 200
			if rawStatus, ok := event["statusCodeValue"]; ok {
				switch v := rawStatus.(type) {
				case float64:
					statusVal = int(v)
				case int:
					statusVal = v
				}
			}
			innerStr, _ := event["body"].(string)
			if statusVal != http.StatusOK {
				msg := innerStr
				if msg == "" {
					msg = fmt.Sprintf("upstream status %d", statusVal)
				}
				streamErr := newQoderStatusError(statusVal, msg)
				reporter.PublishFailure(ctx, streamErr)
				out <- cliproxyexecutor.StreamChunk{Err: streamErr}
				return
			}
			if innerStr == "" {
				continue
			}
			if innerStr == "[DONE]" {
				emitDone(ctx, out, opts.SourceFormat, req.Model, opts.OriginalRequest, payload, &streamParam)
				reporter.EnsurePublished(ctx)
				return
			}
			var inner map[string]interface{}
			if err := json.Unmarshal([]byte(innerStr), &inner); err != nil {
				continue
			}
			chunkBytes, err := buildOpenAIChunk(inner, model)
			if err != nil {
				continue
			}
			// Reconstruct an OpenAI-compatible SSE line ("data: {chunk}").
			// Qoder's upstream nests OpenAI chunks inside a
			// {statusCodeValue, body} envelope so unlike kimi/openai-compat/
			// codebuddy we can't forward the raw upstream line — we have to
			// rebuild the SSE frame here. The format matches what those
			// other executors feed into TranslateStream so the translators'
			// "expects data: prefix" assumption holds.
			ssePayload := append([]byte("data: "), chunkBytes...)
			if detail, ok := helps.ParseOpenAIStreamUsage(ssePayload); ok {
				reporter.Publish(ctx, detail)
			}

			// Always run through TranslateStream. When source==target
			// (OpenAI client) it strips the "data:" prefix and returns
			// raw JSON; the OpenAI handler then re-adds the SSE framing.
			// For cross-format clients (Anthropic/Gemini) it emits the
			// format-specific stream events (message_start /
			// content_block_delta / ...) directly as fully framed bytes
			// because those handlers write chunks verbatim.
			to := sdktranslator.FormatOpenAI
			from := opts.SourceFormat
			if from == "" {
				from = to
			}
			frames := sdktranslator.TranslateStream(ctx, to, from,
				req.Model, opts.OriginalRequest, payload, ssePayload, &streamParam)
			for _, frame := range frames {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: frame}:
				case <-ctx.Done():
					return
				}
			}
		}
		// Scanner loop exited naturally (EOF). Emit a terminating
		// "data: [DONE]" / Anthropic message_stop frame so the client
		// closes the stream cleanly.
		emitDone(ctx, out, opts.SourceFormat, req.Model, opts.OriginalRequest, payload, &streamParam)
		// Check for scanner errors
		if err := scanner.Err(); err != nil {
			streamErr := fmt.Errorf("scanner error: %w", err)
			reporter.PublishFailure(ctx, streamErr)
			out <- cliproxyexecutor.StreamChunk{Err: streamErr}
			return
		}
		reporter.EnsurePublished(ctx)
	}()

	return &cliproxyexecutor.StreamResult{
		Headers: httpResp.Header.Clone(),
		Chunks:  out,
	}, nil
}

// lastUserText returns the text of the last user message in the (already
// normalized) message list, or empty when there isn't one. Qoder uses this
// for the chat_context "current turn" preview slot; the full conversation
// still travels through the messages array.
func lastUserText(messages []interface{}) string {
	for i := len(messages) - 1; i >= 0; i-- {
		msgMap, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := msgMap["role"].(string); role != "user" {
			continue
		}
		if s, ok := msgMap["content"].(string); ok {
			return s
		}
		return extractContentGeneric(msgMap["content"])
	}
	return ""
}

// extractContentGeneric extracts text content from message content field
func extractContentGeneric(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, item := range v {
			if itemMap, ok := item.(map[string]interface{}); ok {
				if itemMap["type"] == "text" {
					if text, ok := itemMap["text"].(string); ok {
						parts = append(parts, text)
					}
					continue
				}
				if text, ok := itemMap["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		if content == nil {
			return ""
		}
		return fmt.Sprintf("%v", content)
	}
}

// buildQoderContent converts an OpenAI chat message's content into the shape
// Qoder's chat endpoint accepts. Text-only content collapses to a plain string
// (Qoder's default). When the content array carries images we preserve them as
// OpenAI-style image_url parts — which is exactly what Qoder's chat ContentPart
// expects (see the official qodercli chat.proto: ContentPart.image_url{url,detail},
// and its lXa transform that emits {type:"image_url",image_url:{url}}).
//
// By the time this runs the SDK translators have already normalized every
// source format to OpenAI chat, so images arrive as
// {"type":"image_url","image_url":{"url":...}} where url is either a
// data:<mime>;base64,... inline payload (Claude Code / Codex both land here)
// or a remote https URL. text/image order is preserved.
func buildQoderContent(content interface{}) interface{} {
	arr, ok := content.([]interface{})
	if !ok {
		return extractContentGeneric(content)
	}

	parts := make([]interface{}, 0, len(arr))
	textParts := make([]string, 0, len(arr))
	hasImage := false
	flushText := func() {
		if len(textParts) > 0 {
			parts = append(parts, map[string]interface{}{
				"type": "text",
				"text": strings.Join(textParts, "\n"),
			})
			textParts = textParts[:0]
		}
	}

	for _, item := range arr {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if img := qoderImageURLPart(itemMap); img != nil {
			flushText()
			parts = append(parts, img)
			hasImage = true
			continue
		}
		if text, ok := itemMap["text"].(string); ok && text != "" {
			textParts = append(textParts, text)
		}
	}

	if !hasImage {
		return strings.Join(textParts, "\n")
	}
	flushText()
	return parts
}

// qoderImageURLPart returns a clean OpenAI image_url content part for an
// incoming part, or nil when it is not a usable image. It tolerates both the
// canonical chat object form (image_url:{url,detail}) and a bare string
// image_url, and skips empty URLs (remote http(s) and data: URLs both pass
// through — Qoder accepts either as image_url.url).
func qoderImageURLPart(item map[string]interface{}) map[string]interface{} {
	if t, _ := item["type"].(string); t != "image_url" {
		return nil
	}
	url := ""
	var detail string
	switch v := item["image_url"].(type) {
	case map[string]interface{}:
		url, _ = v["url"].(string)
		detail, _ = v["detail"].(string)
	case string:
		url = v
	}
	if url == "" {
		return nil
	}
	imgURL := map[string]interface{}{"url": url}
	if detail != "" {
		imgURL["detail"] = detail
	}
	return map[string]interface{}{"type": "image_url", "image_url": imgURL}
}

// parseImageDataURL extracts the mime type and raw base64 payload from a
// data:<mime>;base64,<payload> URL. Non-data URLs (e.g. remote http links) or
// non-base64 data URLs return ok=false.
func parseImageDataURL(url string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", "", false
	}
	meta, payload, found := strings.Cut(url[len("data:"):], ",")
	if !found || payload == "" {
		return "", "", false
	}
	fields := strings.Split(meta, ";")
	mediaType = strings.TrimSpace(fields[0])
	if mediaType == "" {
		return "", "", false
	}
	isBase64 := false
	for _, f := range fields[1:] {
		if strings.EqualFold(strings.TrimSpace(f), "base64") {
			isBase64 = true
			break
		}
	}
	if !isBase64 {
		return "", "", false
	}
	return mediaType, payload, true
}

// normalizeQoderMessages clones each message and applies sanitizations
// required by Qoder's upstream:
//
//  1. Flatten content: Anthropic/OpenAI multipart content arrays
//     ([{type:"text",text:"..."}]) are collapsed to plain strings.
//
//  2. Remap system messages: Qoder rejects role="system" in the messages
//     array; system prompt content is collected and returned separately
//     so the caller can place it in the top-level "system" request field.
func normalizeQoderMessages(messages []interface{}) (normalized []interface{}, systemText string) {
	if len(messages) == 0 {
		return nil, ""
	}
	out := make([]interface{}, 0, len(messages))
	var systemParts []string
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		// Collect system messages — Qoder does not accept role="system"
		// in the messages array, so we remap them to the top-level
		// "system" request field.
		if role, _ := msgMap["role"].(string); role == "system" {
			if text := extractContentGeneric(msgMap["content"]); text != "" {
				systemParts = append(systemParts, text)
			}
			continue
		}
		cloned := make(map[string]interface{}, len(msgMap))
		for k, v := range msgMap {
			cloned[k] = v
		}
		// Preserve multimodal image parts (as OpenAI image_url) instead of
		// flattening to text — Qoder's chat endpoint accepts image_url content
		// parts. Text-only content still collapses to a plain string.
		cloned["content"] = buildQoderContent(msgMap["content"])
		out = append(out, cloned)
	}
	return out, strings.Join(systemParts, "\n\n")
}

func buildOpenAIChunk(inner map[string]interface{}, model string) ([]byte, error) {
	if inner == nil {
		return nil, fmt.Errorf("empty inner payload")
	}
	if _, ok := inner["model"]; !ok || inner["model"] == "" {
		inner["model"] = model
	}
	if choices, ok := inner["choices"].([]interface{}); ok {
		if len(choices) == 0 {
			if inner["finish_reason"] != nil || inner["stop"] != nil {
				inner["choices"] = []map[string]interface{}{{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": "stop",
				}}
			}
		}
	}
	return json.Marshal(inner)
}

// emitDone publishes the terminating SSE frame(s) for the stream. The
// upstream "[DONE]" sentinel is fed through TranslateStream so the
// client's SourceFormat dictates the actual wire bytes — "data: [DONE]\n\n"
// for OpenAI, "event: message_stop\ndata: {...}\n\n" for Anthropic, and
// the equivalent format-specific terminators for Gemini etc. This mirrors
// the pattern used by kimi_executor.
//
// param must be the same pointer the per-chunk TranslateStream calls used
// — the Anthropic translator (and others) need the carried state to know
// which content_block indices to close, the running token count, etc.
func emitDone(ctx context.Context, out chan<- cliproxyexecutor.StreamChunk,
	sourceFormat sdktranslator.Format, reqModel string, originalReq, body []byte, param *any) {
	to := sdktranslator.FormatOpenAI
	from := sourceFormat
	if from == "" {
		from = to
	}
	frames := sdktranslator.TranslateStream(ctx, to, from,
		reqModel, originalReq, body, []byte("[DONE]"), param)
	for _, frame := range frames {
		select {
		case out <- cliproxyexecutor.StreamChunk{Payload: frame}:
		case <-ctx.Done():
			return
		}
	}
}

// qoderStatusError implements StatusError for Qoder API errors
type qoderStatusError struct {
	status  int
	message string
}

func newQoderStatusError(status int, message string) *qoderStatusError {
	return &qoderStatusError{status: status, message: message}
}

func (e *qoderStatusError) Error() string {
	return fmt.Sprintf("Qoder API error %d: %s", e.status, e.message)
}

func (e *qoderStatusError) StatusCode() int {
	return e.status
}

// Qoder queue-retry tuning defaults. When the upstream places the request in
// a waiting queue it returns 403 with code=10605 and a nested JSON body
// carrying {isQueued:true, retryAfterSeconds, serviceAvailable:true, ...}.
// That is a soft/retriable signal — we wait it out rather than surfacing the
// 403 to the client. These defaults mirror the official qodercli and are all
// overridable via the `qoder.queue` config block.
const (
	// qoderQueueDefaultMaxWait matches qodercli's QODER_MODEL_QUEUE_MAX_WAIT_MS
	// default of 36e5 ms (1 hour).
	qoderQueueDefaultMaxWait = time.Hour
	// qoderQueueDefaultPoll is used when the server omits retryAfterSeconds
	// (qodercli's isl = 30s).
	qoderQueueDefaultPoll = 30 * time.Second
	// qoderQueueDefaultMinBackoff / qoderQueueDefaultMaxBackoff clamp the
	// server-supplied retryAfterSeconds (qodercli's osl: [500ms, 30s]).
	qoderQueueDefaultMinBackoff = 500 * time.Millisecond
	qoderQueueDefaultMaxBackoff = 30 * time.Second
	// qoderQueueDefaultPollTimeout bounds one queue-status request (qodercli's
	// Bor = 30s).
	qoderQueueDefaultPollTimeout = 30 * time.Second
)

// qoderQueueSettings is the resolved, ready-to-use queue configuration for a
// single request, produced by resolveQoderQueueSettings from config + defaults.
type qoderQueueSettings struct {
	enabled           bool
	maxWait           time.Duration
	pollInterval      time.Duration
	minBackoff        time.Duration
	maxBackoff        time.Duration
	pollTimeout       time.Duration
	useStatusEndpoint bool
	reportFinish      bool
}

// resolveQoderQueueSettings folds the qoder.queue config over the defaults.
// Missing / unparsable values fall back to the qodercli-aligned defaults so a
// partial config block still behaves sanely.
func resolveQoderQueueSettings(cfg *config.Config) qoderQueueSettings {
	s := qoderQueueSettings{
		enabled:           true,
		maxWait:           qoderQueueDefaultMaxWait,
		pollInterval:      qoderQueueDefaultPoll,
		minBackoff:        qoderQueueDefaultMinBackoff,
		maxBackoff:        qoderQueueDefaultMaxBackoff,
		pollTimeout:       qoderQueueDefaultPollTimeout,
		useStatusEndpoint: true,
		reportFinish:      true,
	}
	if cfg == nil {
		return s
	}
	q := cfg.Qoder.Queue
	if q.Enabled != nil {
		s.enabled = *q.Enabled
	}
	if q.UseStatusEndpoint != nil {
		s.useStatusEndpoint = *q.UseStatusEndpoint
	}
	if q.ReportFinish != nil {
		s.reportFinish = *q.ReportFinish
	}
	if d, ok := parseDurationOpt(q.MaxWait); ok {
		s.maxWait = d
	}
	if d, ok := parseDurationOpt(q.PollInterval); ok {
		s.pollInterval = d
	}
	if d, ok := parseDurationOpt(q.MinBackoff); ok {
		s.minBackoff = d
	}
	if d, ok := parseDurationOpt(q.MaxBackoff); ok {
		s.maxBackoff = d
	}
	if d, ok := parseDurationOpt(q.PollTimeout); ok {
		s.pollTimeout = d
	}
	// Guard against an inverted clamp from a bad config.
	if s.maxBackoff > 0 && s.minBackoff > s.maxBackoff {
		s.minBackoff = s.maxBackoff
	}
	return s
}

// parseDurationOpt parses a Go duration string, reporting ok=false for empty
// or invalid input (so callers keep their default).
func parseDurationOpt(v string) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 0, false
	}
	return d, true
}

// qoderQueueInfo holds the parsed "you are in a waiting queue" signal.
type qoderQueueInfo struct {
	queued           bool
	serviceAvailable bool
	retryAfter       time.Duration
	queueCount       int64
	waitTime         int64
	modelKey         string
	queueType        string
}

// parseQoderQueue digs through the (multiply-escaped) 403 error body and
// reports whether it is a retriable queue signal. The upstream nests the
// payload several layers deep, e.g.:
//
//	{"code":"403","message":"{\"code\":\"10605\",\"message\":\"{\\\"isQueued\\\":true,
//	 \\\"retryAfterSeconds\\\":30,\\\"queueCount\\\":301,\\\"serviceAvailable\\\":true,...}\"}"}
//
// gjson only sees the outer layer, so we peel `message` strings that are
// themselves JSON until we find the object carrying isQueued.
func parseQoderQueue(body string) (qoderQueueInfo, bool) {
	var info qoderQueueInfo
	// Peel up to a few layers of nested/escaped JSON-in-a-string.
	cur := strings.TrimSpace(body)
	for i := 0; i < 6 && cur != ""; i++ {
		res := gjson.Parse(cur)
		if !res.IsObject() {
			break
		}
		if q := res.Get("isQueued"); q.Exists() && q.Bool() {
			info.queued = true
			info.retryAfter = time.Duration(res.Get("retryAfterSeconds").Int()) * time.Second
			info.queueCount = res.Get("queueCount").Int()
			info.waitTime = res.Get("waitTime").Int()
			info.modelKey = res.Get("modelKey").String()
			info.queueType = res.Get("queueType").String()
			// serviceAvailable defaults to true when absent. Unlike the old
			// implementation we do NOT treat serviceAvailable:false as fatal:
			// the official qodercli keeps waiting through it, so it is still a
			// retriable queue signal.
			info.serviceAvailable = true
			if sa := res.Get("serviceAvailable"); sa.Exists() {
				info.serviceAvailable = sa.Bool()
			}
			return info, true
		}
		// Descend into the nested message string, if any.
		msg := res.Get("message")
		if !msg.Exists() {
			break
		}
		next := strings.TrimSpace(msg.String())
		if next == "" || next == cur {
			break
		}
		cur = next
	}
	return info, false
}

// backoff returns the wait duration before the next queue poll/retry,
// honoring the server's retryAfterSeconds and clamping to the configured
// [minBackoff, maxBackoff] range (qodercli behavior). When the server omits
// retryAfterSeconds, the configured pollInterval is used.
func (q qoderQueueInfo) backoff(s qoderQueueSettings) time.Duration {
	d := q.retryAfter
	if d <= 0 {
		d = s.pollInterval
	}
	if s.maxBackoff > 0 && d > s.maxBackoff {
		d = s.maxBackoff
	}
	if s.minBackoff > 0 && d < s.minBackoff {
		d = s.minBackoff
	}
	return d
}

// waitQoderQueue polls the upstream model-queue status endpoint until the
// model reports ready (isQueued:false), the total wait budget (deadline) is
// exhausted, or the context is cancelled. It mirrors the official qodercli:
// it does not fail on serviceAvailable:false — it keeps waiting through it —
// and it honors the server-supplied retryAfterSeconds between polls, clamped
// to the configured [minBackoff, maxBackoff] range.
//
// On success (model ready) it returns nil and the caller re-issues the
// inference request. It returns context errors verbatim so the caller can
// propagate cancellation; any other error means "give up and surface the
// original 403".
func (e *QoderExecutor) waitQoderQueue(
	ctx context.Context,
	httpClient *http.Client,
	creds qoderauth.CosyCredentials,
	requestSetID, sessionID, fallbackModelKey string,
	initial qoderQueueInfo,
	s qoderQueueSettings,
	deadline time.Time,
) (qoderQueueInfo, error) {
	cur := initial
	for poll := 0; ; poll++ {
		// Respect the total budget and honor cancellation while backing off.
		wait := cur.backoff(s)
		if time.Now().Add(wait).After(deadline) {
			return cur, fmt.Errorf("qoder: queue wait budget exhausted (polls=%d, queueCount=%d, waitTime=%ds)", poll, cur.queueCount, cur.waitTime)
		}
		modelKey := cur.modelKey
		if modelKey == "" {
			modelKey = fallbackModelKey
		}
		log.Infof("qoder: queued (poll %d, queueCount=%d, waitTime=%ds, serviceAvailable=%t); next status check in %s", poll, cur.queueCount, cur.waitTime, cur.serviceAvailable, wait)
		select {
		case <-ctx.Done():
			return cur, ctx.Err()
		case <-time.After(wait):
		}

		next, ready, perr := e.pollQoderQueueStatus(ctx, httpClient, creds, requestSetID, sessionID, modelKey, cur.queueType, s)
		if perr != nil {
			if errors.Is(perr, context.Canceled) || errors.Is(perr, context.DeadlineExceeded) {
				// Only propagate cancellation from the parent context; a
				// per-poll timeout is a transient failure we retry through.
				if ctx.Err() != nil {
					return cur, ctx.Err()
				}
			}
			// Transient poll failure: keep the previous queue info and retry
			// until the budget runs out, matching qodercli's tolerance for
			// intermittent status errors.
			log.Warnf("qoder: queue status poll %d failed: %v; retrying", poll, perr)
			continue
		}
		if ready {
			// Model is ready. Honor a final short retryAfter before re-issuing.
			if next.retryAfter > 0 {
				finalWait := next.retryAfter
				if s.maxBackoff > 0 && finalWait > s.maxBackoff {
					finalWait = s.maxBackoff
				}
				if !time.Now().Add(finalWait).After(deadline) {
					select {
					case <-ctx.Done():
						return next, ctx.Err()
					case <-time.After(finalWait):
					}
				}
			}
			log.Infof("qoder: model ready after %d poll(s); re-issuing inference request", poll+1)
			return next, nil
		}
		cur = next
	}
}

// pollQoderQueueStatus issues one signed GET to the queue-status endpoint and
// parses the response. It reports ready=true when the upstream says the model
// is no longer queued (isQueued:false). The returned qoderQueueInfo carries
// the fresh retryAfter/queueCount/etc for the next poll.
func (e *QoderExecutor) pollQoderQueueStatus(
	ctx context.Context,
	httpClient *http.Client,
	creds qoderauth.CosyCredentials,
	requestSetID, sessionID, modelKey, queueType string,
	s qoderQueueSettings,
) (qoderQueueInfo, bool, error) {
	q := url.Values{}
	q.Set("requestSetId", requestSetID)
	if modelKey != "" {
		q.Set("modelKey", modelKey)
	}
	if queueType != "" {
		q.Set("queueType", queueType)
	}
	statusURL := qoderauth.QoderQueueStatusURL + "?" + q.Encode()

	// The queue-status GET is COSY-signed with an empty body, like the model
	// list endpoint. BuildAuthHeaders derives the sig path from the URL.
	headers, herr := qoderauth.BuildAuthHeaders(nil, statusURL, creds)
	if herr != nil {
		return qoderQueueInfo{}, false, fmt.Errorf("failed to build COSY auth for queue status: %w", herr)
	}

	reqCtx := ctx
	var cancel context.CancelFunc
	if s.pollTimeout > 0 {
		reqCtx, cancel = context.WithTimeout(ctx, s.pollTimeout)
		defer cancel()
	}

	httpReq, rerr := http.NewRequestWithContext(reqCtx, "GET", statusURL, nil)
	if rerr != nil {
		return qoderQueueInfo{}, false, fmt.Errorf("failed to create queue status request: %w", rerr)
	}
	httpReq.Header.Set("Accept", "application/json")
	headers.Apply(httpReq)
	httpReq.Header.Set("X-Model-Key", modelKey)
	if sessionID != "" {
		httpReq.Header.Set("X-Session-Id", sessionID)
	}
	httpReq.Header.Set("Accept-Encoding", "identity")

	resp, derr := httpClient.Do(httpReq)
	if derr != nil {
		return qoderQueueInfo{}, false, derr
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)

	// A queued response still parses as a queue signal (isQueued:true). When
	// the model is ready the body reports isQueued:false (or omits it), which
	// parseQoderQueue reports as retriable=false → ready.
	info, stillQueued := parseQoderQueue(string(body))
	if stillQueued {
		return info, false, nil
	}
	// Not a queue signal. If the status endpoint returned a hard error status
	// (auth expired, etc.), treat it as a poll failure so the caller can
	// retry/give up rather than mistaking it for "ready".
	if resp.StatusCode != http.StatusOK {
		return qoderQueueInfo{}, false, fmt.Errorf("queue status HTTP %d: %s", resp.StatusCode, truncate(string(body), 200))
	}
	return info, true, nil
}

// finishQoderQueue fires the official queue-finish callback
// (POST /algo/api/v2/service/ask/finish?Encode=1) after a queued request
// completes. It mirrors the official qodercli: a best-effort usage-statistics
// ping reporting how long the model was waited on. Errors are swallowed — the
// result has already been streamed to the client and nothing here affects it.
//
// Wire format (from qodercli): the request body is
//
//	{"payload":"{\"model_key\":...,\"request_set_id\":...,\"user_id\":...,\"time_consumed\":<ms>}","encodeVersion":"1"}
//
// then run through the same QoderEncodeBody scheme as the chat endpoint (hence
// the ?Encode=1 query flag) and COSY-signed over the encoded bytes.
func (e *QoderExecutor) finishQoderQueue(
	ctx context.Context,
	httpClient *http.Client,
	creds qoderauth.CosyCredentials,
	modelKey, requestSetID string,
	waited time.Duration,
) {
	timeConsumed := waited.Milliseconds()
	if timeConsumed < 0 {
		timeConsumed = 0
	}

	inner, err := json.Marshal(map[string]interface{}{
		"model_key":      modelKey,
		"request_set_id": requestSetID,
		"user_id":        creds.UserID,
		"time_consumed":  timeConsumed,
	})
	if err != nil {
		return
	}
	outer, err := json.Marshal(map[string]interface{}{
		"payload":       string(inner),
		"encodeVersion": "1",
	})
	if err != nil {
		return
	}
	encoded := []byte(helps.QoderEncodeBody(outer))

	headers, herr := qoderauth.BuildAuthHeaders(encoded, qoderauth.QoderQueueFinishURLEncoded, creds)
	if herr != nil {
		log.Debugf("qoder: queue-finish auth build failed: %v", herr)
		return
	}

	reqCtx := ctx
	var cancel context.CancelFunc
	if e != nil { // keep a bounded timeout regardless of caller ctx
		reqCtx, cancel = context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
	}

	httpReq, rerr := http.NewRequestWithContext(reqCtx, "POST", qoderauth.QoderQueueFinishURLEncoded, bytes.NewReader(encoded))
	if rerr != nil {
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	headers.Apply(httpReq)
	httpReq.Header.Set("X-Model-Key", modelKey)
	httpReq.Header.Set("Accept-Encoding", "identity")

	resp, derr := httpClient.Do(httpReq)
	if derr != nil {
		log.Debugf("qoder: queue-finish callback failed: %v", derr)
		return
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Debugf("qoder: queue-finish callback HTTP %d", resp.StatusCode)
		return
	}
	log.Debugf("qoder: queue-finish reported (model=%s, waited=%dms)", modelKey, timeConsumed)
}

// CountTokens estimates token count for the request (placeholder implementation)
func (e *QoderExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	// Translate non-openai formats before extracting messages
	payload := req.Payload
	if opts.SourceFormat != "" && opts.SourceFormat != sdktranslator.FormatOpenAI {
		payload = sdktranslator.TranslateRequest(opts.SourceFormat, sdktranslator.FormatOpenAI, req.Model, payload, false)
	}

	// Simple estimation: 1 token ≈ 4 characters
	var chatReq map[string]interface{}
	if err := json.Unmarshal(payload, &chatReq); err != nil {
		return cliproxyexecutor.Response{}, err
	}

	messagesRaw, _ := chatReq["messages"].([]interface{})
	totalChars := 0
	for _, msg := range messagesRaw {
		if msgMap, ok := msg.(map[string]interface{}); ok {
			content := extractContentGeneric(msgMap["content"])
			totalChars += len(content)
		}
	}

	estimatedTokens := totalChars / 4
	if estimatedTokens < 1 {
		estimatedTokens = 1
	}

	response := map[string]interface{}{
		"usage": map[string]int{
			"prompt_tokens":     estimatedTokens,
			"completion_tokens": 0,
			"total_tokens":      estimatedTokens,
		},
	}

	responseBytes, _ := json.Marshal(response)
	return cliproxyexecutor.Response{
		Payload: responseBytes,
	}, nil
}

// Execute executes a non-streaming request against Qoder API
func (e *QoderExecutor) Execute(ctx context.Context, authRecord *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	// We need ExecuteStream to:
	//   1. Translate the request payload from the client's SourceFormat
	//      (Anthropic/Gemini/etc) into OpenAI before sending to Qoder.
	//   2. Emit raw OpenAI chunks so we can accumulate choices[0].delta.
	//
	// (1) requires opts.SourceFormat to stay as the original; (2) requires
	// it to be OpenAI. Resolve by translating the payload up-front, then
	// passing FormatOpenAI for both directions to ExecuteStream.
	internalReq := req
	internalOpts := opts
	if opts.SourceFormat != "" && opts.SourceFormat != sdktranslator.FormatOpenAI {
		internalReq.Payload = sdktranslator.TranslateRequest(
			opts.SourceFormat, sdktranslator.FormatOpenAI,
			req.Model, req.Payload, false)
	}
	internalOpts.SourceFormat = sdktranslator.FormatOpenAI

	streamResult, err := e.ExecuteStream(ctx, authRecord, internalReq, internalOpts)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	// Accumulate all chunks
	var content strings.Builder
	var finishReason string
	type pendingToolCall struct {
		ID        string
		Name      string
		Arguments string
	}
	pendingToolCalls := make(map[int]*pendingToolCall)

	for chunk := range streamResult.Chunks {
		if chunk.Err != nil {
			return cliproxyexecutor.Response{}, chunk.Err
		}

		// ExecuteStream was called with SourceFormat=FormatOpenAI so
		// TranslateStream strips the "data:" prefix and returns raw JSON.
		// Skip empty or [DONE] payloads.
		raw := chunk.Payload
		if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("[DONE]")) {
			continue
		}

		var oiChunk map[string]interface{}
		if err := json.Unmarshal(raw, &oiChunk); err == nil {
			if choices, ok := oiChunk["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if toolCalls, ok := delta["tool_calls"].([]interface{}); ok {
							for _, call := range toolCalls {
								callMap, ok := call.(map[string]interface{})
								if !ok {
									continue
								}
								idx := 0
								if rawIdx, ok := callMap["index"].(float64); ok {
									idx = int(rawIdx)
								}
								entry := pendingToolCalls[idx]
								if entry == nil {
									entry = &pendingToolCall{}
									pendingToolCalls[idx] = entry
								}
								if id, ok := callMap["id"].(string); ok && id != "" {
									entry.ID = id
								}
								if fn, ok := callMap["function"].(map[string]interface{}); ok {
									if name, ok := fn["name"].(string); ok && name != "" {
										entry.Name = name
									}
									if args, ok := fn["arguments"].(string); ok && args != "" {
										entry.Arguments += args
									}
								}
							}
						}
						if contentStr, ok := delta["content"].(string); ok {
							content.WriteString(contentStr)
						}
					}
					if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
						finishReason = fr
					}
				}
			}
		}
	}

	var toolCalls []map[string]interface{}
	if finishReason == "tool_calls" && len(pendingToolCalls) > 0 {
		for i := 0; i < len(pendingToolCalls); i++ {
			entry, ok := pendingToolCalls[i]
			if !ok || entry == nil {
				continue
			}
			id := entry.ID
			if id == "" {
				id = fmt.Sprintf("call_%d", time.Now().UnixNano())
			}
			args := entry.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   id,
				"type": "function",
				"function": map[string]interface{}{
					"name":      entry.Name,
					"arguments": args,
				},
			})
		}
	}

	// Build final response
	message := map[string]interface{}{
		"role":    "assistant",
		"content": content.String(),
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	response := map[string]interface{}{
		"id":      fmt.Sprintf("qoder-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   req.Model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
	}

	responseBytes, _ := json.Marshal(response)

	// Translate the Qoder OpenAI-format response back to the client's
	// expected SourceFormat. Reuse internalReq.Payload — that's already
	// the OpenAI-translated payload we computed above before calling
	// ExecuteStream, so we don't need to re-translate.
	var param any
	out := sdktranslator.TranslateNonStream(ctx, sdktranslator.FormatOpenAI, opts.SourceFormat, req.Model, opts.OriginalRequest, internalReq.Payload, responseBytes, &param)
	responseBytes = out

	return cliproxyexecutor.Response{
		Payload: responseBytes,
		Headers: streamResult.Headers,
	}, nil
}

// Refresh is a no-op for Qoder.
//
// Qoder's device-flow token (the "dt-..." string) is already long-lived
// (~30 days for the access token, ~360 days for the refresh token per
// the deviceToken/poll response). The upstream does not expose the
// classic OAuth refresh dance — every endpoint we've observed (cubk1's
// qoder2api, Veria, the official @qoder-ai/qodercli) either skips
// refresh entirely or routes through a different /jobToken exchange
// flow that requires personalToken (we don't have one).
//
// Hitting /algo/api/v3/user/refresh_token with our device token returns
// 403 "Forbidden" / errorCode=Forbidden — the endpoint is not for our
// flow. Mark the auth refreshed-now and keep going; if a real expiry
// happens the user re-runs --qoder-login.
func (e *QoderExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	if auth == nil {
		return nil, fmt.Errorf("qoder executor: auth is nil")
	}
	return auth, nil
}

// HttpRequest injects Qoder COSY authentication into the HTTP request and executes it
func (e *QoderExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	storage, ok := auth.Storage.(*qoderauth.QoderTokenStorage)
	if !ok {
		return nil, fmt.Errorf("invalid auth storage type for qoder")
	}

	// Read request body for COSY signing
	bodyBytes, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	headers, err := qoderauth.BuildAuthHeaders(
		bodyBytes,
		req.URL.String(),
		qoderauth.CosyCredentials{
			UserID:    storage.UserID,
			AuthToken: storage.Token,
			Name:      storage.Name,
			Email:     storage.Email,
			MachineID: storage.MachineID,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to build COSY auth: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	headers.Apply(req)

	req = req.WithContext(ctx)
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(req)
}

// buildQoderModelConfig returns the model_config block for a chat request,
// pulled from the cache populated by FetchQoderModels (which mirrors what
// /algo/api/v2/model/list publishes — per-model is_vl / is_reasoning /
// max_input_tokens / price_factor / strategies / ...). Returns an error
// when the cache has no entry for modelKey: that means we either never
// successfully fetched the model list for this auth, or the user asked
// for a model the server doesn't expose. Either way we should fail loudly
// rather than guess and silently get downgraded to a different model.
func buildQoderModelConfig(storage *qoderauth.QoderTokenStorage, modelKey string) (map[string]interface{}, error) {
	raw, ok := storage.GetModelConfig(modelKey)
	if !ok || len(raw) == 0 {
		keys := storage.ModelConfigKeys()
		if len(keys) == 0 {
			return nil, fmt.Errorf("qoder: model config cache is empty (model list not fetched yet); restart the service or check /algo/api/v2/model/list connectivity")
		}
		sort.Strings(keys)
		return nil, fmt.Errorf("qoder: no model_config cached for %q; known models: %s", modelKey, strings.Join(keys, ", "))
	}
	var cfg map[string]interface{}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("qoder: cached model_config for %q is invalid JSON: %w", modelKey, err)
	}
	if cfg == nil {
		return nil, fmt.Errorf("qoder: cached model_config for %q decoded to nil", modelKey)
	}
	// The cache stores the model description; ensure the key matches what
	// we're sending (handles model alias rewrites in caller).
	cfg["key"] = modelKey
	return cfg, nil
}

// FetchQoderModels retrieves the live model list from Qoder's
// /algo/api/v2/model/list endpoint and converts it into ModelInfo entries.
// Falls back to the static registry if the auth lacks credentials, the request
// fails, or the response is malformed. Mirrors the FetchKiloModels /
// FetchCursorModels pattern used by other dynamic providers.
func FetchQoderModels(ctx context.Context, auth *cliproxyauth.Auth, cfg *config.Config) []*registry.ModelInfo {
	storage, ok := auth.Storage.(*qoderauth.QoderTokenStorage)
	if !ok || storage == nil || storage.Token == "" {
		log.Debug("qoder: no token, returning static models")
		return registry.GetQoderModels()
	}

	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	headers, err := qoderauth.BuildAuthHeaders(nil, qoderauth.QoderModelListURL, qoderauth.CosyCredentials{
		UserID:    storage.UserID,
		AuthToken: storage.Token,
		Name:      storage.Name,
		Email:     storage.Email,
		MachineID: storage.MachineID,
	})
	if err != nil {
		log.Warnf("qoder: build cosy headers for model list: %v", err)
		return registry.GetQoderModels()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, qoderauth.QoderModelListURL, nil)
	if err != nil {
		log.Warnf("qoder: build model list request: %v", err)
		return registry.GetQoderModels()
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Accept-Encoding", "identity")
	headers.Apply(req)

	httpClient := helps.NewProxyAwareHTTPClient(ctx, cfg, auth, 0)
	resp, err := httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			log.Warnf("qoder: model list fetch canceled: %v", err)
		} else {
			log.Warnf("qoder: model list fetch failed: %v", err)
		}
		return registry.GetQoderModels()
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("qoder: read model list response: %v", err)
		return registry.GetQoderModels()
	}
	if resp.StatusCode != http.StatusOK {
		log.Warnf("qoder: model list returned %d: %s", resp.StatusCode, truncate(string(body), 300))
		return registry.GetQoderModels()
	}

	chat := gjson.GetBytes(body, "chat")
	if !chat.Exists() || !chat.IsArray() {
		log.Warnf("qoder: model list response missing 'chat' array")
		return registry.GetQoderModels()
	}

	now := time.Now().Unix()
	models := make([]*registry.ModelInfo, 0, 16)
	configs := make(map[string]json.RawMessage, 16)
	chat.ForEach(func(_, entry gjson.Result) bool {
		key := entry.Get("key").String()
		if key == "" {
			return true
		}
		if !entry.Get("enable").Bool() {
			return true
		}
		display := entry.Get("display_name").String()
		if display == "" {
			display = key
		}
		ctxLen := int(entry.Get("max_input_tokens").Int())
		isVL := entry.Get("is_vl").Bool()

		// Cache the raw upstream JSON for this model so ExecuteStream can
		// forward the exact model_config the server published (per-model
		// is_vl / is_reasoning / max_input_tokens / price_factor / ...).
		configs[key] = json.RawMessage(entry.Raw)

		mi := &registry.ModelInfo{
			ID:            "qoder/" + key,
			Object:        "model",
			Created:       now,
			OwnedBy:       "qoder",
			Type:          "qoder",
			DisplayName:   display,
			Description:   fmt.Sprintf("%s via Qoder", display),
			ContextLength: ctxLen,
		}
		if isVL {
			mi.SupportedInputModalities = []string{"TEXT", "IMAGE"}
		}
		// Parse thinking_config from upstream. Qoder returns per-model
		// effort levels (e.g. dmodel has only high/max, ultimate has
		// low/medium/high/max/xhigh) and a disabled key to indicate
		// whether reasoning can be turned off. Models without
		// thinking_config but with is_reasoning=true still get a
		// basic Thinking marker (no predefined levels).
		if tc := entry.Get("thinking_config"); tc.Exists() {
			ts := &registry.ThinkingSupport{}
			if tc.Get("disabled").Exists() {
				ts.ZeroAllowed = true
			}
			efforts := tc.Get("enabled.efforts")
			if efforts.Exists() && efforts.IsObject() {
				levels := make([]string, 0, 5)
				efforts.ForEach(func(key, _ gjson.Result) bool {
					levels = append(levels, key.String())
					return true
				})
				ts.Levels = levels
			}
			mi.Thinking = ts
		} else if entry.Get("is_reasoning").Bool() {
			mi.Thinking = &registry.ThinkingSupport{}
		}
		models = append(models, mi)
		return true
	})

	if len(models) == 0 {
		log.Warn("qoder: model list returned no enabled models, falling back to static")
		return registry.GetQoderModels()
	}

	storage.SetModelConfigs(configs)

	log.Infof("qoder: fetched %d models from /algo/api/v2/model/list", len(models))

	// Fetch usage alongside models so the management UI has fresh credit data.
	// Use context.Background() so the goroutine outlives the caller's context.
	go FetchQoderUsage(context.Background(), auth, cfg)

	return models
}

// stableHash returns a deterministic hex identifier from the given inputs.
func stableHash(prefix string, inputs ...string) string {
	h := sha256.New()
	h.Write([]byte(prefix))
	for _, in := range inputs {
		h.Write([]byte{0})
		h.Write([]byte(in))
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// stableChatRecordID produces a deterministic chat_record_id from the
// request payload so retries with identical content hit upstream caches.
func stableChatRecordID(model string, messages []interface{}, toolsRaw interface{}, maxTokens int) string {
	h := sha256.New()
	h.Write([]byte("qoder-record"))
	h.Write([]byte{0})
	h.Write([]byte(model))
	for _, msg := range messages {
		m, _ := msg.(map[string]interface{})
		if m == nil {
			continue
		}
		if role, _ := m["role"].(string); role != "" {
			h.Write([]byte{0})
			h.Write([]byte(role))
		}
		if content, _ := m["content"].(string); content != "" {
			h.Write([]byte{0})
			h.Write([]byte(content))
		}
	}
	if toolsRaw != nil {
		toolsJSON, _ := json.Marshal(toolsRaw)
		h.Write([]byte{0})
		h.Write(toolsJSON)
	}
	h.Write([]byte{0})
	h.Write([]byte(fmt.Sprintf("mt=%d", maxTokens)))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// FetchQoderUsage fetches the current quota usage from /api/v2/quota/usage
// and caches the result in storage.UsageInfo. It is called opportunistically
// alongside FetchQoderModels so the management UI can display credit balance
// without a separate round-trip.
func FetchQoderUsage(ctx context.Context, auth *cliproxyauth.Auth, cfg *config.Config) *qoderauth.QoderUsageInfo {
	storage, ok := auth.Storage.(*qoderauth.QoderTokenStorage)
	if !ok || storage == nil || storage.Token == "" {
		return nil
	}

	const usageURL = "https://openapi.qoder.sh/api/v2/quota/usage"
	log.Debugf("qoder: fetching usage for user %s (token len=%d)", storage.UserID, len(storage.Token))
	req, err := http.NewRequest(http.MethodGet, usageURL, nil)
	if err != nil {
		log.Debugf("qoder: build usage request: %v", err)
		return nil
	}
	req.Header.Set("Authorization", "Bearer "+storage.Token)
	req.Header.Set("Accept", "application/json")

	httpClient := helps.NewProxyAwareHTTPClient(ctx, cfg, auth, 15*time.Second)
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Debugf("qoder: usage fetch failed: %v", err)
		return nil
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		log.Debugf("qoder: usage fetch returned %d", resp.StatusCode)
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Debugf("qoder: read usage response: %v", err)
		return nil
	}

	var info qoderauth.QoderUsageInfo
	if err := json.Unmarshal(body, &info); err != nil {
		log.Debugf("qoder: parse usage response: %v", err)
		return nil
	}

	storage.SetUsageInfo(&info)
	log.Debugf("qoder: usage fetched — %.0f/%.0f %s used (%.1f%%)",
		info.UserQuota.Used, info.UserQuota.Total, info.UserQuota.Unit,
		info.TotalUsagePercentage*100)
	return &info
}

func validateQoderModel(rawModel string, storage *qoderauth.QoderTokenStorage) (string, error) {
	qoderModel := strings.TrimPrefix(rawModel, "qoder/")
	if mapped, ok := qoderauth.ModelMap[qoderModel]; ok {
		return mapped, nil
	}
	if storage != nil {
		if _, cached := storage.GetModelConfig(qoderModel); cached {
			return qoderModel, nil
		}
	}
	return "", fmt.Errorf("unsupported qoder model: %q", qoderModel)
}
