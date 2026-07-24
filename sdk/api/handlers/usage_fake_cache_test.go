package handlers

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	"github.com/tidwall/gjson"
)

func TestInjectFakeCacheUsage_Disabled(t *testing.T) {
	in := []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}`)
	out := InjectFakeCacheUsage(&config.SDKConfig{}, in)
	if gjson.GetBytes(out, "usage.cache_read_input_tokens").Exists() {
		t.Fatalf("expected no cache_read_input_tokens when disabled, got %s", out)
	}
}

func TestInjectFakeCacheUsage_Values(t *testing.T) {
	cfg := &config.SDKConfig{}
	cfg.FakeCache.Enabled = true
	cfg.FakeCache.Percent = 50

	in := []byte(`{"id":"x","object":"chat.completion","usage":{"prompt_tokens":1664,"completion_tokens":182,"total_tokens":1846,"prompt_tokens_details":{"cached_tokens":1536}}}`)
	out := InjectFakeCacheUsage(cfg, in)
	got := gjson.GetBytes(out, "usage.cache_read_input_tokens").Int()
	if got != 832 {
		t.Fatalf("cache_read_input_tokens=%d want 832; body=%s", got, out)
	}
	// default does not overwrite prompt_tokens_details.cached_tokens
	if gotCached := gjson.GetBytes(out, "usage.prompt_tokens_details.cached_tokens").Int(); gotCached != 1536 {
		t.Fatalf("cached_tokens=%d want original 1536", gotCached)
	}
}

func TestInjectFakeCacheUsage_SyncCachedTokens(t *testing.T) {
	cfg := &config.SDKConfig{}
	cfg.FakeCache.Enabled = true
	cfg.FakeCache.Percent = 100
	cfg.FakeCache.SyncCachedTokens = true

	in := []byte(`{"usage":{"prompt_tokens":200,"prompt_tokens_details":{"cached_tokens":1}}}`)
	out := InjectFakeCacheUsage(cfg, in)
	if got := gjson.GetBytes(out, "usage.cache_read_input_tokens").Int(); got != 200 {
		t.Fatalf("cache_read_input_tokens=%d want 200", got)
	}
	if got := gjson.GetBytes(out, "usage.prompt_tokens_details.cached_tokens").Int(); got != 200 {
		t.Fatalf("cached_tokens=%d want 200", got)
	}
}

func TestInjectFakeCacheUsage_StreamChunk(t *testing.T) {
	cfg := &config.SDKConfig{}
	cfg.FakeCache.Enabled = true
	cfg.FakeCache.Percent = 25

	in := []byte(`{"id":"c","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":80,"completion_tokens":5,"total_tokens":85}}`)
	out := InjectFakeCacheUsage(cfg, in)
	if got := gjson.GetBytes(out, "usage.cache_read_input_tokens").Int(); got != 20 {
		t.Fatalf("cache_read_input_tokens=%d want 20; body=%s", got, out)
	}
}

func TestInjectFakeCacheUsage_ZeroPercent(t *testing.T) {
	cfg := &config.SDKConfig{}
	cfg.FakeCache.Enabled = true
	cfg.FakeCache.Percent = 0

	in := []byte(`{"usage":{"prompt_tokens":1664}}`)
	out := InjectFakeCacheUsage(cfg, in)
	if !gjson.GetBytes(out, "usage.cache_read_input_tokens").Exists() {
		t.Fatalf("field missing: %s", out)
	}
	if got := gjson.GetBytes(out, "usage.cache_read_input_tokens").Int(); got != 0 {
		t.Fatalf("cache_read_input_tokens=%d want 0", got)
	}
}

func TestApplyStreamUsageToFirst(t *testing.T) {
	cfg := &config.SDKConfig{}
	cfg.FakeCache.Enabled = true
	cfg.FakeCache.Percent = 0

	first := []byte(`{"id":"d992","object":"chat.completion.chunk","created":1,"model":"grok","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"The"},"finish_reason":null,"native_finish_reason":null}]}`)
	mid := []byte(`{"id":"d992","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}`)
	last := []byte(`{"id":"d992","object":"chat.completion.chunk","created":1,"model":"grok","choices":[{"index":0,"delta":{},"finish_reason":"stop","native_finish_reason":"stop"}],"usage":{"completion_tokens":229,"total_tokens":442,"prompt_tokens":213,"prompt_tokens_details":{"cached_tokens":128},"completion_tokens_details":{"reasoning_tokens":161}}}`)

	out := ApplyStreamUsageToFirst(cfg, [][]byte{first, mid, last})
	if len(out) != 3 {
		t.Fatalf("len=%d", len(out))
	}
	if !gjson.GetBytes(out[0], "usage").Exists() {
		t.Fatalf("first chunk missing usage: %s", out[0])
	}
	if got := gjson.GetBytes(out[0], "usage.cache_read_input_tokens").Int(); got != 0 {
		t.Fatalf("first cache_read=%d want 0; %s", got, out[0])
	}
	if got := gjson.GetBytes(out[0], "usage.prompt_tokens").Int(); got != 213 {
		t.Fatalf("first prompt_tokens=%d want 213", got)
	}
	// first delta preserved
	if got := gjson.GetBytes(out[0], "choices.0.delta.reasoning_content").String(); got != "The" {
		t.Fatalf("first delta lost: %q body=%s", got, out[0])
	}
	// last still has usage + fake field
	if got := gjson.GetBytes(out[2], "usage.cache_read_input_tokens").Int(); got != 0 {
		t.Fatalf("last cache_read=%d want 0", got)
	}
}

func TestApplyStreamUsageToFirst_Disabled(t *testing.T) {
	first := []byte(`{"id":"1","choices":[{"delta":{"content":"a"}}]}`)
	last := []byte(`{"id":"1","usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
	out := ApplyStreamUsageToFirst(&config.SDKConfig{}, [][]byte{first, last})
	if gjson.GetBytes(out[0], "usage").Exists() {
		t.Fatalf("should not copy usage when disabled: %s", out[0])
	}
}
