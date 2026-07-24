package handlers

import (
	"bytes"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// InjectFakeCacheUsage adds usage.cache_read_input_tokens to OpenAI-style JSON
// bodies when fake-cache is enabled. Works for both non-stream chat.completion
// objects and stream chat.completion.chunk objects that carry a usage object.
//
// Percent is clamped to [0, 100]. Value is derived from prompt_tokens, falling
// back to input_tokens when prompt_tokens is absent.
func InjectFakeCacheUsage(cfg *config.SDKConfig, body []byte) []byte {
	if cfg == nil || !cfg.FakeCache.Enabled || len(body) == 0 {
		return body
	}
	payload, ok := normalizeJSONPayload(body)
	if !ok {
		return body
	}
	usage := gjson.GetBytes(payload, "usage")
	if !usage.Exists() || !usage.IsObject() {
		return body
	}

	promptTokens := usage.Get("prompt_tokens")
	if !promptTokens.Exists() {
		promptTokens = usage.Get("input_tokens")
	}
	if !promptTokens.Exists() {
		// Still inject the field as 0 so clients always see the key.
		out, err := sjson.SetBytes(payload, "usage.cache_read_input_tokens", 0)
		if err != nil {
			return body
		}
		return out
	}

	percent := cfg.FakeCache.Percent
	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}
	prompt := promptTokens.Int()
	if prompt < 0 {
		prompt = 0
	}
	cached := prompt * int64(percent) / 100

	out, err := sjson.SetBytes(payload, "usage.cache_read_input_tokens", cached)
	if err != nil {
		return body
	}
	if cfg.FakeCache.SyncCachedTokens {
		out, _ = sjson.SetBytes(out, "usage.prompt_tokens_details.cached_tokens", cached)
		if usage.Get("input_tokens_details").Exists() {
			out, _ = sjson.SetBytes(out, "usage.input_tokens_details.cached_tokens", cached)
		}
	}
	return out
}

// ApplyStreamUsageToFirst injects fake-cache on every chunk, then copies the final
// usage object onto the first JSON chunk so clients that only read usage from the
// first SSE event still see cache_read_input_tokens / token totals.
//
// This requires the full stream to be buffered first (final usage arrives last).
func ApplyStreamUsageToFirst(cfg *config.SDKConfig, chunks [][]byte) [][]byte {
	if len(chunks) == 0 {
		return chunks
	}
	out := make([][]byte, len(chunks))
	var usageRaw string
	// Walk once: inject + capture last usage (prefer later chunks).
	for i, chunk := range chunks {
		c := InjectFakeCacheUsage(cfg, chunk)
		out[i] = c
		if payload, ok := normalizeJSONPayload(c); ok {
			if u := gjson.GetBytes(payload, "usage"); u.Exists() && u.IsObject() {
				usageRaw = u.Raw
			}
		}
	}
	if usageRaw == "" || !cfgFakeCacheEnabled(cfg) {
		// Even when disabled, out is a shallow copy of originals (Inject is no-op).
		return out
	}
	// Attach final usage to the first valid JSON chunk.
	for i, chunk := range out {
		payload, ok := normalizeJSONPayload(chunk)
		if !ok {
			continue
		}
		updated, err := sjson.SetRawBytes(payload, "usage", []byte(usageRaw))
		if err != nil {
			return out
		}
		out[i] = updated
		break
	}
	return out
}

func cfgFakeCacheEnabled(cfg *config.SDKConfig) bool {
	return cfg != nil && cfg.FakeCache.Enabled
}

func normalizeJSONPayload(body []byte) ([]byte, bool) {
	payload := bytes.TrimSpace(body)
	if len(payload) == 0 {
		return nil, false
	}
	if bytes.Equal(payload, []byte("[DONE]")) {
		return nil, false
	}
	if bytes.HasPrefix(payload, []byte("data:")) {
		payload = bytes.TrimSpace(payload[5:])
		if bytes.Equal(payload, []byte("[DONE]")) {
			return nil, false
		}
	}
	if !gjson.ValidBytes(payload) {
		return nil, false
	}
	return payload, true
}
