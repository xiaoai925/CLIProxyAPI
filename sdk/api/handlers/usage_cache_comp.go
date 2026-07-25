package handlers

import (
	"github.com/router-for-me/CLIProxyAPI/v7/internal/cachecomp"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	"github.com/tidwall/gjson"
)

// CacheCompConfigFromSDK maps YAML/config into cachecomp runtime config.
func CacheCompConfigFromSDK(cfg *config.SDKConfig) cachecomp.Config {
	if cfg == nil {
		return cachecomp.Config{}
	}
	c := cfg.CacheCompensation
	// If full strategy disabled but legacy fake-cache enabled, emulate simple percent mode.
	if !c.Enabled && cfg.FakeCache.Enabled {
		return cachecomp.Config{
			Enabled:                true,
			HitRateTarget:          float64(cfg.FakeCache.Percent) / 100.0,
			MinCacheableTokens:     1,
			MetadataEnabled:        false,
			MinInputTokens:         1,
			TokenMultiplier:        1.0,
			SyncOpenAICachedTokens: cfg.FakeCache.SyncCachedTokens,
		}
	}
	meta := true
	if c.MetadataEnabled != nil {
		meta = *c.MetadataEnabled
	}
	// creation-rate-target: nil → default 0.10 (fixed-ratio A mode);
	// explicit 0 → disable fixed creation (legacy hit-rate-only).
	creationRate := 0.10
	if c.CreationRateTarget != nil {
		creationRate = *c.CreationRateTarget
	}
	return cachecomp.Config{
		Enabled:                c.Enabled,
		HitRateTarget:          c.HitRateTarget,
		CreationRateTarget:     creationRate,
		MinCacheableTokens:     c.MinCacheableTokens,
		EphemeralTTLSeconds:    c.EphemeralTTLSeconds,
		StaticTTLSeconds:       c.StaticTTLSeconds,
		MaxEntries:             c.MaxEntries,
		TokenMultiplier:        c.TokenMultiplier,
		MetadataEnabled:        meta,
		MinInputTokens:         c.MinInputTokens,
		BaseOverhead:           c.BaseOverhead,
		PerMessageOverhead:     c.PerMessageOverhead,
		PerToolOverhead:        c.PerToolOverhead,
		PerSystemOverhead:      c.PerSystemOverhead,
		ThinkingOverhead:       c.ThinkingOverhead,
		ToolResultOverhead:     c.ToolResultOverhead,
		InjectedSystemTokens:   c.InjectedSystemTokens,
		SyncOpenAICachedTokens: c.SyncOpenAICachedTokens || cfg.FakeCache.SyncCachedTokens,
	}
}

// CacheCompEnabled reports whether any cache compensation path is active.
func CacheCompEnabled(cfg *config.SDKConfig) bool {
	if cfg == nil {
		return false
	}
	return cfg.CacheCompensation.Enabled || cfg.FakeCache.Enabled
}

// ApplyOpenAIUsageCompensation runs full strategy (or legacy fake-cache) on one OpenAI body.
func ApplyOpenAIUsageCompensation(cfg *config.SDKConfig, originalRequest, body []byte) []byte {
	if !CacheCompEnabled(cfg) {
		return body
	}
	cc := CacheCompConfigFromSDK(cfg)
	model := gjson.GetBytes(originalRequest, "model").String()
	if model == "" {
		model = gjson.GetBytes(body, "model").String()
	}
	// Prefer full strategy when CacheCompensation.Enabled.
	if cfg.CacheCompensation.Enabled {
		return cachecomp.Global.ApplyToOpenAIBody(cc, model, originalRequest, body)
	}
	// Legacy percent-only path.
	return InjectFakeCacheUsage(cfg, body)
}

// ApplyOpenAIStreamUsageCompensation buffers stream chunks when compensation is on.
func ApplyOpenAIStreamUsageCompensation(cfg *config.SDKConfig, originalRequest []byte, chunks [][]byte) [][]byte {
	if !CacheCompEnabled(cfg) {
		return chunks
	}
	cc := CacheCompConfigFromSDK(cfg)
	model := gjson.GetBytes(originalRequest, "model").String()
	if cfg.CacheCompensation.Enabled {
		return cachecomp.Global.ApplyStreamOpenAI(cc, model, originalRequest, chunks)
	}
	// Legacy: inject field + copy last usage to first
	return ApplyStreamUsageToFirst(cfg, chunks)
}

// ApplyClaudeUsageCompensation compensates Anthropic message JSON/SSE payloads.
func ApplyClaudeUsageCompensation(cfg *config.SDKConfig, originalRequest, body []byte) []byte {
	if cfg == nil || !cfg.CacheCompensation.Enabled {
		return body
	}
	cc := CacheCompConfigFromSDK(cfg)
	model := gjson.GetBytes(originalRequest, "model").String()
	return cachecomp.Global.ApplyToClaudeBody(cc, model, originalRequest, body)
}

// ApplyClaudeStreamUsageCompensation buffers Claude SSE chunks for consistent usage.
func ApplyClaudeStreamUsageCompensation(cfg *config.SDKConfig, originalRequest []byte, chunks [][]byte) [][]byte {
	if cfg == nil || !cfg.CacheCompensation.Enabled {
		return chunks
	}
	cc := CacheCompConfigFromSDK(cfg)
	model := gjson.GetBytes(originalRequest, "model").String()
	return cachecomp.Global.ApplyStreamClaude(cc, model, originalRequest, chunks)
}
