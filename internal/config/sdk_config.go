// Package config provides configuration management for the CLI Proxy API server.
// It handles loading and parsing YAML configuration files, and provides structured
// access to application settings including server port, authentication directory,
// debug settings, proxy configuration, and API keys.
package config

// SDKConfig represents the application's configuration, loaded from a YAML file.
type SDKConfig struct {
	// ProxyURL is the URL of an optional proxy server to use for outbound requests.
	ProxyURL string `yaml:"proxy-url" json:"proxy-url"`

	// DisableImageGeneration controls whether the built-in image_generation tool is injected/allowed.
	//
	// Supported values:
	//   - false (default): image_generation is enabled everywhere (normal behavior).
	//   - true: image_generation is disabled everywhere. The server stops injecting it, removes it from request payloads,
	//     and returns 404 for /v1/images/generations and /v1/images/edits.
	//   - "chat": disable image_generation injection for all non-images endpoints (e.g. /v1/responses, /v1/chat/completions),
	//     while keeping /v1/images/generations and /v1/images/edits enabled and preserving image_generation there.
	//   - "passthrough": do not modify the tool list on non-images endpoints — keep image_generation if the client
	//     sent it and do not inject it otherwise; on /v1/images/generations and /v1/images/edits behave like "chat".
	DisableImageGeneration DisableImageGenerationMode `yaml:"disable-image-generation" json:"disable-image-generation"`

	// GPTImage2BaseModel sets the base (mainline) model used by the legacy hosted
	// image_generation tool path when a Codex image request is not proxied directly
	// through the Image API.
	//
	// The value must start with "gpt-" (case-insensitive). If empty or invalid, the
	// default base model ("gpt-5.4-mini") is used.
	GPTImage2BaseModel string `yaml:"gpt-image-2-base-model,omitempty" json:"gpt-image-2-base-model,omitempty"`

	// VideoResultAuthCacheTTL controls how long video IDs stay pinned to the credential
	// that created them. Accepts duration strings like "30m" or "3h".
	// Empty or invalid values use the default 3h.
	VideoResultAuthCacheTTL string `yaml:"video-result-auth-cache-ttl,omitempty" json:"video-result-auth-cache-ttl,omitempty"`

	// ForceModelPrefix requires explicit model prefixes (e.g., "teamA/gemini-3-pro-preview")
	// to target prefixed credentials. When false, unprefixed model requests may use prefixed
	// credentials as well.
	ForceModelPrefix bool `yaml:"force-model-prefix" json:"force-model-prefix"`

	// RequestLog enables or disables detailed request logging functionality.
	RequestLog bool `yaml:"request-log" json:"request-log"`

	// APIKeys is a list of keys for authenticating clients to this proxy server.
	APIKeys []string `yaml:"api-keys" json:"api-keys"`

	// PassthroughHeaders controls whether upstream response headers are forwarded to downstream clients.
	// Default is false (disabled).
	PassthroughHeaders bool `yaml:"passthrough-headers" json:"passthrough-headers"`

	// Streaming configures server-side streaming behavior (keep-alives and safe bootstrap retries).
	Streaming StreamingConfig `yaml:"streaming" json:"streaming"`

	// NonStreamKeepAliveInterval controls how often blank lines are emitted for non-streaming responses.
	// <= 0 disables keep-alives. Value is in seconds.
	NonStreamKeepAliveInterval int `yaml:"nonstream-keepalive-interval,omitempty" json:"nonstream-keepalive-interval,omitempty"`

	// FakeCache is a legacy simple percent injector. Prefer CacheCompensation.
	// Kept for backward compatibility; when CacheCompensation.Enabled is true it wins.
	FakeCache FakeCacheConfig `yaml:"fake-cache" json:"fake-cache"`

	// CacheCompensation is the Anthropic-compatible prompt-cache usage strategy for
	// Grok/OpenAI-style upstreams (normalize + structure analysis + packaging
	// overhead + hit-rate compensation). Applies to OpenAI chat.completion and
	// Claude /v1/messages responses (stream + non-stream).
	CacheCompensation CacheCompensationConfig `yaml:"cache-compensation" json:"cache-compensation"`
}

// FakeCacheConfig controls synthetic prompt-cache token reporting on outbound usage.
type FakeCacheConfig struct {
	// Enabled toggles fake cache injection. Default false.
	Enabled bool `yaml:"enabled" json:"enabled"`
	// Percent is the share of prompt/input tokens reported as cache hits (0-100).
	// cache_read_input_tokens = prompt_tokens * Percent / 100.
	Percent int `yaml:"percent" json:"percent"`
	// SyncCachedTokens also overwrites usage.prompt_tokens_details.cached_tokens
	// (and input_tokens_details.cached_tokens when present) with the same value.
	// Default false: only cache_read_input_tokens is added/updated.
	SyncCachedTokens bool `yaml:"sync-cached-tokens" json:"sync-cached-tokens"`
}

// CacheCompensationConfig is the full Grok→Anthropic cache usage strategy.
type CacheCompensationConfig struct {
	Enabled bool `yaml:"enabled" json:"enabled"`
	// HitRateTarget is desired cache_read / (input+creation+read), 0..1. Default 0.80.
	HitRateTarget float64 `yaml:"hit-rate-target" json:"hit-rate-target"`
	// MinCacheableTokens minimum estimated cacheable prefix. Default 100.
	MinCacheableTokens int `yaml:"min-cacheable-tokens" json:"min-cacheable-tokens"`
	// EphemeralTTLSeconds local LRU TTL for ephemeral prefixes. Default 300.
	EphemeralTTLSeconds int `yaml:"ephemeral-ttl-seconds" json:"ephemeral-ttl-seconds"`
	// StaticTTLSeconds local LRU TTL for static prefixes. Default 3600.
	StaticTTLSeconds int `yaml:"static-ttl-seconds" json:"static-ttl-seconds"`
	// MaxEntries local LRU capacity. Default 10000.
	MaxEntries int `yaml:"max-entries" json:"max-entries"`
	// TokenMultiplier scales compensated input/output. Default 1.0.
	TokenMultiplier float64 `yaml:"token-multiplier" json:"token-multiplier"`
	// MetadataEnabled toggles packaging overhead subtraction. Default true when enabled.
	MetadataEnabled *bool `yaml:"metadata-enabled,omitempty" json:"metadata-enabled,omitempty"`
	// MinInputTokens floor after packaging compensation. Default 10.
	MinInputTokens int64 `yaml:"min-input-tokens" json:"min-input-tokens"`
	// BaseOverhead / per-* overheads for Grok packaging compensation.
	BaseOverhead          int64 `yaml:"base-overhead" json:"base-overhead"`
	PerMessageOverhead    int64 `yaml:"per-message-overhead" json:"per-message-overhead"`
	PerToolOverhead       int64 `yaml:"per-tool-overhead" json:"per-tool-overhead"`
	PerSystemOverhead     int64 `yaml:"per-system-overhead" json:"per-system-overhead"`
	ThinkingOverhead      int64 `yaml:"thinking-overhead" json:"thinking-overhead"`
	ToolResultOverhead    int64 `yaml:"tool-result-overhead" json:"tool-result-overhead"`
	InjectedSystemTokens  int64 `yaml:"injected-system-tokens" json:"injected-system-tokens"`
	// SyncOpenAICachedTokens also writes prompt_tokens_details.cached_tokens.
	SyncOpenAICachedTokens bool `yaml:"sync-openai-cached-tokens" json:"sync-openai-cached-tokens"`
}

// StreamingConfig holds server streaming behavior configuration.
type StreamingConfig struct {
	// KeepAliveSeconds controls how often the server emits SSE heartbeats (": keep-alive\n\n").
	// <= 0 disables keep-alives. Default is 0.
	KeepAliveSeconds int `yaml:"keepalive-seconds,omitempty" json:"keepalive-seconds,omitempty"`

	// BootstrapRetries controls how many times the server may retry a streaming request before any bytes are sent,
	// to allow auth rotation / transient recovery.
	// <= 0 disables bootstrap retries. Default is 0.
	BootstrapRetries int `yaml:"bootstrap-retries,omitempty" json:"bootstrap-retries,omitempty"`
}
