package cachecomp

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestEstimateTokensCJK(t *testing.T) {
	// 3 CJK chars → ceil(3/1.5)=2
	if got := estimateTokens("你好啊"); got != 2 {
		t.Fatalf("cjk tokens=%d want 2", got)
	}
	// 8 ascii non-space → ceil(8/4)=2
	if got := estimateTokens("abcdefgh"); got != 2 {
		t.Fatalf("ascii tokens=%d want 2", got)
	}
}

func TestNormalizeUsageOpenAI(t *testing.T) {
	u := gjson.Parse(`{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":40}}`)
	got := NormalizeUsage(u)
	if got.InputTokens != 100 || got.OutputTokens != 10 || got.CacheReadInputTokens != 40 {
		t.Fatalf("got %+v", got)
	}
}

func TestCompensateHitRate(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MetadataEnabled = false
	cfg.HitRateTarget = 0.80
	cfg.CreationRateTarget = 0 // legacy: only read ratio
	stats := PromptCacheStats{CacheableTokens: 0}
	raw := AnthropicUsage{InputTokens: 1000, OutputTokens: 50}
	got := Compensate(cfg, stats, raw, nil)
	total := got.InputTokens + got.CacheCreationInputTokens + got.CacheReadInputTokens
	if total != 1000 {
		t.Fatalf("total changed: %d", total)
	}
	rate := float64(got.CacheReadInputTokens) / float64(total)
	if rate < 0.79 {
		t.Fatalf("hit rate %.3f want >=0.80; %+v", rate, got)
	}
	if got.InputTokens < 1 {
		t.Fatalf("input must remain >=1: %+v", got)
	}
}

func TestCompensateFixedRatios(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MetadataEnabled = false
	cfg.HitRateTarget = 0.80
	cfg.CreationRateTarget = 0.10
	stats := PromptCacheStats{}
	// After ApplyToOpenAIBody rebuild: full prompt as InputTokens, no upstream split.
	raw := AnthropicUsage{InputTokens: 1000, OutputTokens: 20}
	got := Compensate(cfg, stats, raw, nil)
	total := got.InputTokens + got.CacheCreationInputTokens + got.CacheReadInputTokens
	if total != 1000 {
		t.Fatalf("total=%d want 1000; %+v", total, got)
	}
	if got.CacheCreationInputTokens != 100 {
		t.Fatalf("creation=%d want 100; %+v", got.CacheCreationInputTokens, got)
	}
	if got.CacheReadInputTokens != 800 {
		t.Fatalf("read=%d want 800; %+v", got.CacheReadInputTokens, got)
	}
	if got.InputTokens != 100 {
		t.Fatalf("input=%d want 100; %+v", got.InputTokens, got)
	}
}

func TestCompensateWithCacheableHit(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MetadataEnabled = false
	cfg.HitRateTarget = 0.80
	cfg.CreationRateTarget = 0
	stats := PromptCacheStats{CacheableTokens: 600, CacheHit: true}
	raw := AnthropicUsage{InputTokens: 1000, OutputTokens: 20}
	got := Compensate(cfg, stats, raw, nil)
	if got.CacheReadInputTokens < 600 {
		// after hit-rate may be higher
		t.Fatalf("read=%d want >=600; %+v", got.CacheReadInputTokens, got)
	}
	total := got.InputTokens + got.CacheCreationInputTokens + got.CacheReadInputTokens
	if total != 1000 {
		t.Fatalf("total=%d want 1000", total)
	}
}

func TestAnalyzeAndLRU(t *testing.T) {
	e := NewEngine()
	cfg := DefaultConfig()
	cfg.MinCacheableTokens = 10
	// long system to exceed min
	req := []byte(`{"model":"grok","system":"这是一段足够长的中文系统提示用于缓存测试一二三四五六七八九十","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"x","description":"tool schema content for caching"}]}`)
	s1 := e.AnalyzeRequest(cfg, "grok", req)
	if s1.CacheableTokens < 10 {
		t.Fatalf("cacheable=%d", s1.CacheableTokens)
	}
	if s1.CacheHit {
		t.Fatalf("first should miss")
	}
	s2 := e.AnalyzeRequest(cfg, "grok", req)
	if !s2.CacheHit {
		t.Fatalf("second should hit")
	}
}

func TestApplyToOpenAIBody(t *testing.T) {
	e := NewEngine()
	cfg := DefaultConfig()
	cfg.MetadataEnabled = false
	cfg.HitRateTarget = 0.8
	cfg.CreationRateTarget = 0.1
	cfg.SyncOpenAICachedTokens = true
	orig := []byte(`{"model":"grok-4.5","messages":[{"role":"system","content":"stable system prompt for caching test content 1234567890"},{"role":"user","content":"hello"}]}`)
	body := []byte(`{"id":"x","object":"chat.completion","usage":{"prompt_tokens":1000,"completion_tokens":20,"total_tokens":1020,"completion_tokens_details":{"reasoning_tokens":5}}}`)
	out := e.ApplyToOpenAIBody(cfg, "grok-4.5", orig, body)
	if !gjson.GetBytes(out, "usage.cache_read_input_tokens").Exists() {
		t.Fatalf("missing cache_read: %s", out)
	}
	if gjson.GetBytes(out, "usage.input_tokens").Exists() {
		t.Fatalf("xAI style must not emit usage.input_tokens: %s", out)
	}
	prompt := gjson.GetBytes(out, "usage.prompt_tokens").Int()
	read := gjson.GetBytes(out, "usage.cache_read_input_tokens").Int()
	creation := gjson.GetBytes(out, "usage.cache_creation_input_tokens").Int()
	cached := gjson.GetBytes(out, "usage.prompt_tokens_details.cached_tokens").Int()
	write := gjson.GetBytes(out, "usage.prompt_tokens_details.cache_write_tokens").Int()
	creationAlias := gjson.GetBytes(out, "usage.prompt_tokens_details.cached_creation_tokens").Int()
	if prompt != 1000 {
		t.Fatalf("prompt_tokens=%d want 1000", prompt)
	}
	if read+creation > prompt {
		t.Fatalf("creation+read > prompt: c=%d r=%d p=%d", creation, read, prompt)
	}
	if cached != read {
		t.Fatalf("cached_tokens=%d want read=%d", cached, read)
	}
	if write != creation || creationAlias != creation {
		t.Fatalf("write/creation alias mismatch write=%d alias=%d creation=%d body=%s", write, creationAlias, creation, out)
	}
	if creation != 100 || read != 800 {
		t.Fatalf("fixed ratio want creation=100 read=800 got c=%d r=%d body=%s", creation, read, out)
	}
	if got := gjson.GetBytes(out, "usage.completion_tokens_details.reasoning_tokens").Int(); got != 5 {
		t.Fatalf("reasoning_tokens lost: %d body=%s", got, out)
	}
	if read == 0 && creation == 0 {
		t.Fatalf("expected cache fields non-zero after compensation: %s", out)
	}
}

func TestApplyToClaudeBody(t *testing.T) {
	e := NewEngine()
	cfg := DefaultConfig()
	cfg.MetadataEnabled = false
	orig := []byte(`{"model":"grok","system":[{"type":"text","text":"long system text for cache abcdefghijklmnop","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`)
	body := []byte(`{"type":"message","usage":{"input_tokens":500,"output_tokens":10}}`)
	out := e.ApplyToClaudeBody(cfg, "grok", orig, body)
	if !gjson.GetBytes(out, "usage.cache_read_input_tokens").Exists() && !gjson.GetBytes(out, "usage.cache_creation_input_tokens").Exists() {
		t.Fatalf("missing cache fields: %s", out)
	}
	in := gjson.GetBytes(out, "usage.input_tokens").Int()
	cr := gjson.GetBytes(out, "usage.cache_creation_input_tokens").Int()
	rd := gjson.GetBytes(out, "usage.cache_read_input_tokens").Int()
	if in+cr+rd != 500 {
		t.Fatalf("sum=%d want 500; %s", in+cr+rd, out)
	}
}

func TestApplyStreamOpenAIFirstChunk(t *testing.T) {
	e := NewEngine()
	cfg := DefaultConfig()
	cfg.MetadataEnabled = false
	orig := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}]}`)
	first := []byte(`{"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"h"}}]}`)
	last := []byte(`{"id":"1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":5,"total_tokens":205}}`)
	out := e.ApplyStreamOpenAI(cfg, "m", orig, [][]byte{first, last})
	if !gjson.GetBytes(out[0], "usage").Exists() {
		t.Fatalf("first missing usage: %s", out[0])
	}
	if gjson.GetBytes(out[0], "choices.0.delta.content").String() != "h" {
		t.Fatalf("delta lost: %s", out[0])
	}
}
