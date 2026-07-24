// Package cachecomp implements Anthropic-compatible prompt-cache usage compensation
// for Grok/OpenAI-style upstreams that do not return cache_creation/cache_read fields.
package cachecomp

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Config is the runtime compensation policy.
type Config struct {
	Enabled bool

	// HitRateTarget is desired cache_read / total input-side tokens (0..1). Default 0.80.
	HitRateTarget float64

	// MinCacheableTokens is the minimum estimated cacheable prefix size. Default 100.
	MinCacheableTokens int

	// EphemeralTTLSeconds / StaticTTLSeconds control local LRU TTL. Defaults 300 / 3600.
	EphemeralTTLSeconds int
	StaticTTLSeconds    int

	// MaxEntries caps local prompt-cache metadata entries. Default 10000.
	MaxEntries int

	// TokenMultiplier scales compensated input/output. Default 1.0.
	TokenMultiplier float64

	// Metadata compensation (Grok packaging overhead).
	MetadataEnabled        bool
	MinInputTokens         int64
	BaseOverhead           int64
	PerMessageOverhead     int64
	PerToolOverhead        int64
	PerSystemOverhead      int64
	ThinkingOverhead       int64
	ToolResultOverhead     int64
	InjectedSystemTokens   int64

	// SyncOpenAICachedTokens also writes OpenAI prompt_tokens_details.cached_tokens.
	SyncOpenAICachedTokens bool
}

// DefaultConfig returns design-doc defaults with compensation enabled.
func DefaultConfig() Config {
	return Config{
		Enabled:             true,
		HitRateTarget:       0.80,
		MinCacheableTokens:  100,
		EphemeralTTLSeconds: 300,
		StaticTTLSeconds:    3600,
		MaxEntries:          10000,
		TokenMultiplier:     1.0,
		MetadataEnabled:     true,
		MinInputTokens:      10,
		BaseOverhead:        40,
		PerMessageOverhead:  6,
		PerToolOverhead:     12,
		PerSystemOverhead:   8,
		ThinkingOverhead:    20,
		ToolResultOverhead:  15,
	}
}

// Normalize fills defaults and clamps ranges.
func (c Config) Normalize() Config {
	d := DefaultConfig()
	if c.HitRateTarget <= 0 {
		c.HitRateTarget = d.HitRateTarget
	}
	if c.HitRateTarget > 1 {
		c.HitRateTarget = 1
	}
	if c.MinCacheableTokens <= 0 {
		c.MinCacheableTokens = d.MinCacheableTokens
	}
	if c.EphemeralTTLSeconds <= 0 {
		c.EphemeralTTLSeconds = d.EphemeralTTLSeconds
	}
	if c.StaticTTLSeconds <= 0 {
		c.StaticTTLSeconds = d.StaticTTLSeconds
	}
	if c.MaxEntries <= 0 {
		c.MaxEntries = d.MaxEntries
	}
	if c.TokenMultiplier <= 0 {
		c.TokenMultiplier = 1.0
	}
	if c.MinInputTokens <= 0 {
		c.MinInputTokens = d.MinInputTokens
	}
	if c.BaseOverhead == 0 && c.MetadataEnabled {
		c.BaseOverhead = d.BaseOverhead
	}
	if c.PerMessageOverhead == 0 && c.MetadataEnabled {
		c.PerMessageOverhead = d.PerMessageOverhead
	}
	if c.PerToolOverhead == 0 && c.MetadataEnabled {
		c.PerToolOverhead = d.PerToolOverhead
	}
	if c.PerSystemOverhead == 0 && c.MetadataEnabled {
		c.PerSystemOverhead = d.PerSystemOverhead
	}
	if c.ThinkingOverhead == 0 && c.MetadataEnabled {
		c.ThinkingOverhead = d.ThinkingOverhead
	}
	if c.ToolResultOverhead == 0 && c.MetadataEnabled {
		c.ToolResultOverhead = d.ToolResultOverhead
	}
	return c
}

// AnthropicUsage is the client-facing usage triple + output.
type AnthropicUsage struct {
	InputTokens              int64
	CacheCreationInputTokens int64
	CacheReadInputTokens     int64
	OutputTokens             int64
}

// PromptCacheStats is A-layer output.
type PromptCacheStats struct {
	CacheableTokens int64
	VariableTokens  int64
	CacheHit        bool
}

// Engine holds process-local prompt-cache metadata LRU.
type Engine struct {
	mu      sync.Mutex
	entries map[string]cacheEntry
	order   []string
}

type cacheEntry struct {
	expires time.Time
}

// Global is the process-wide engine used by handlers.
var Global = NewEngine()

// NewEngine creates an empty engine.
func NewEngine() *Engine {
	return &Engine{entries: make(map[string]cacheEntry)}
}

// AnalyzeRequest performs A-layer structure analysis on the original client request.
// Works for Anthropic-style (system/messages/tools + cache_control) and OpenAI chat
// (messages/tools; whole system/developer messages treated as cacheable).
func (e *Engine) AnalyzeRequest(cfg Config, model string, originalRequest []byte) PromptCacheStats {
	cfg = cfg.Normalize()
	if !cfg.Enabled || len(originalRequest) == 0 {
		return PromptCacheStats{}
	}
	root := gjson.ParseBytes(originalRequest)

	var cacheableParts []string
	var variableParts []string
	var cacheType string // ephemeral / static

	// Anthropic system
	if sys := root.Get("system"); sys.Exists() {
		if sys.IsArray() {
			for _, block := range sys.Array() {
				text := contentText(block)
				if hasCacheControl(block) {
					cacheableParts = append(cacheableParts, text)
					if t := cacheControlType(block); t != "" {
						cacheType = t
					}
				} else {
					variableParts = append(variableParts, text)
				}
			}
		} else {
			text := sys.String()
			// bare string system → treat as cacheable stable prefix
			cacheableParts = append(cacheableParts, text)
		}
	}

	// messages
	if msgs := root.Get("messages"); msgs.IsArray() {
		for _, msg := range msgs.Array() {
			role := msg.Get("role").String()
			content := msg.Get("content")
			if content.IsArray() {
				for _, block := range content.Array() {
					text := contentText(block)
					if hasCacheControl(block) {
						cacheableParts = append(cacheableParts, text)
						if t := cacheControlType(block); t != "" {
							cacheType = t
						}
					} else if role == "system" || role == "developer" {
						// OpenAI-style system/developer → cacheable
						cacheableParts = append(cacheableParts, text)
					} else {
						variableParts = append(variableParts, text)
					}
				}
			} else {
				text := content.String()
				if role == "system" || role == "developer" {
					cacheableParts = append(cacheableParts, text)
				} else {
					variableParts = append(variableParts, text)
				}
			}
		}
	}

	// tools
	if tools := root.Get("tools"); tools.IsArray() {
		for _, tool := range tools.Array() {
			text := tool.Raw
			if hasCacheControl(tool) {
				cacheableParts = append(cacheableParts, text)
				if t := cacheControlType(tool); t != "" {
					cacheType = t
				}
			} else {
				// tool schemas are usually stable → cacheable by default
				cacheableParts = append(cacheableParts, text)
			}
		}
	}

	cacheableTokens := estimateTokens(strings.Join(cacheableParts, "\n"))
	variableTokens := estimateTokens(strings.Join(variableParts, "\n"))
	if cacheableTokens < int64(cfg.MinCacheableTokens) {
		return PromptCacheStats{VariableTokens: variableTokens + cacheableTokens}
	}
	if cacheType == "" {
		cacheType = "ephemeral"
	}
	key := promptCacheKey(model, cacheableParts, cacheType)
	ttl := time.Duration(cfg.EphemeralTTLSeconds) * time.Second
	if cacheType == "static" {
		ttl = time.Duration(cfg.StaticTTLSeconds) * time.Second
	}
	hit := e.touch(key, ttl, cfg.MaxEntries)
	return PromptCacheStats{
		CacheableTokens: cacheableTokens,
		VariableTokens:  variableTokens,
		CacheHit:        hit,
	}
}

// NormalizeUsage maps OpenAI/xAI or Anthropic raw usage into AnthropicUsage.
func NormalizeUsage(usage gjson.Result) AnthropicUsage {
	if !usage.Exists() || usage.Type == gjson.Null {
		return AnthropicUsage{}
	}
	input := firstInt(usage, "input_tokens", "prompt_tokens", "promptTokens")
	output := firstInt(usage, "output_tokens", "completion_tokens", "completionTokens")
	creation := firstInt(usage, "cache_creation_input_tokens", "cache_creation_tokens")
	read := firstInt(usage, "cache_read_input_tokens", "cache_read_tokens")
	// OpenAI cached_tokens maps to read if explicit Anthropic fields absent
	if creation == 0 && read == 0 {
		if cached := firstInt(usage, "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens"); cached > 0 {
			read = cached
		}
	}
	return AnthropicUsage{
		InputTokens:              input,
		CacheCreationInputTokens: creation,
		CacheReadInputTokens:     read,
		OutputTokens:             output,
	}
}

// Compensate runs B + split + C on normalized usage.
func Compensate(cfg Config, stats PromptCacheStats, raw AnthropicUsage, originalRequest []byte) AnthropicUsage {
	cfg = cfg.Normalize()
	if !cfg.Enabled {
		return raw
	}

	// B: metadata packaging compensation on input side only
	input := raw.InputTokens
	// If upstream already split cache into creation/read and reduced input, rebuild total input first.
	upstreamCovered := raw.CacheCreationInputTokens + raw.CacheReadInputTokens
	totalInputSide := input + upstreamCovered
	if totalInputSide <= 0 {
		totalInputSide = input
	}

	if cfg.MetadataEnabled {
		comp := metadataCompensation(cfg, originalRequest)
		totalInputSide = applyTokenCompensation(totalInputSide, comp, cfg.MinInputTokens)
	}
	if cfg.TokenMultiplier != 1.0 {
		totalInputSide = int64(math.Round(float64(totalInputSide) * cfg.TokenMultiplier))
		raw.OutputTokens = int64(math.Round(float64(raw.OutputTokens) * cfg.TokenMultiplier))
	}

	// Split creation/read
	var creation, read int64
	if raw.CacheCreationInputTokens > 0 || raw.CacheReadInputTokens > 0 {
		// Prefer upstream non-zero fields, but re-base against compensated total.
		creation = raw.CacheCreationInputTokens
		read = raw.CacheReadInputTokens
		covered := creation + read
		if covered > totalInputSide {
			// scale down proportionally
			if covered > 0 {
				creation = creation * totalInputSide / covered
				read = totalInputSide - creation
			}
		}
	} else if stats.CacheableTokens > 0 {
		cacheable := stats.CacheableTokens
		if cacheable > totalInputSide {
			cacheable = totalInputSide
		}
		if stats.CacheHit {
			read = cacheable
			creation = 0
		} else {
			creation = cacheable
			read = 0
		}
	}

	// Eliminate double counting: uncached input = total - covered
	covered := creation + read
	uncached := totalInputSide - covered
	if uncached < 1 {
		uncached = 1
		// if we forced uncached=1, shrink read first then creation
		over := covered + uncached - totalInputSide
		if over > 0 {
			if read >= over {
				read -= over
			} else {
				over -= read
				read = 0
				if creation >= over {
					creation -= over
				} else {
					creation = 0
				}
			}
		}
	}

	// C: hit-rate compensation — only move from input → read
	uncached, creation, read = applyCacheHitRateCompensation(uncached, creation, read, cfg.HitRateTarget)

	return AnthropicUsage{
		InputTokens:              uncached,
		CacheCreationInputTokens: creation,
		CacheReadInputTokens:     read,
		OutputTokens:             raw.OutputTokens,
	}
}

// ApplyToOpenAIBody injects compensated usage into OpenAI chat.completion JSON.
// originalRequest is the client request used for A/B analysis.
func (e *Engine) ApplyToOpenAIBody(cfg Config, model string, originalRequest, body []byte) []byte {
	cfg = cfg.Normalize()
	if !cfg.Enabled || len(body) == 0 {
		return body
	}
	payload, ok := stripSSE(body)
	if !ok {
		return body
	}
	usageNode := gjson.GetBytes(payload, "usage")
	if !usageNode.Exists() || !usageNode.IsObject() {
		return body
	}
	stats := e.AnalyzeRequest(cfg, model, originalRequest)
	raw := NormalizeUsage(usageNode)
	// For OpenAI, NormalizeUsage may have already subtracted cached into read and
	// left prompt_tokens as full prompt. Prefer full prompt as total input side.
	if prompt := firstInt(usageNode, "prompt_tokens", "input_tokens"); prompt > 0 {
		// If we mapped cached_tokens → read, rebuild raw.InputTokens as full prompt
		// so Compensate can re-split without double-count from OpenAI semantics.
		if raw.CacheReadInputTokens > 0 && raw.InputTokens+raw.CacheReadInputTokens == prompt {
			// extractOpenAI-style already split; keep
		} else if raw.CacheReadInputTokens > 0 && raw.InputTokens == prompt {
			// prompt is full; cached also set — treat prompt as total, clear read for re-split via A
			raw.CacheReadInputTokens = 0
			raw.CacheCreationInputTokens = 0
			raw.InputTokens = prompt
		} else if raw.InputTokens == 0 {
			raw.InputTokens = prompt
		}
	}
	comp := Compensate(cfg, stats, raw, originalRequest)
	out := payload
	// xAI / OpenAI chat.completion usage shape:
	//   prompt_tokens, completion_tokens, total_tokens,
	//   prompt_tokens_details.cached_tokens,
	//   completion_tokens_details.reasoning_tokens (preserved),
	//   plus cache_read_input_tokens / cache_creation_input_tokens for clients
	//   that understand Anthropic-style cache accounting on OpenAI payloads.
	promptTotal := comp.InputTokens + comp.CacheCreationInputTokens + comp.CacheReadInputTokens
	if promptTotal <= 0 {
		promptTotal = firstInt(usageNode, "prompt_tokens", "input_tokens")
	}
	out, _ = sjson.SetBytes(out, "usage.prompt_tokens", promptTotal)
	out, _ = sjson.SetBytes(out, "usage.completion_tokens", comp.OutputTokens)
	out, _ = sjson.SetBytes(out, "usage.total_tokens", promptTotal+comp.OutputTokens)
	out, _ = sjson.SetBytes(out, "usage.cache_read_input_tokens", comp.CacheReadInputTokens)
	out, _ = sjson.SetBytes(out, "usage.cache_creation_input_tokens", comp.CacheCreationInputTokens)
	// Always keep xAI-style prompt_tokens_details.cached_tokens in sync with cache_read.
	out, _ = sjson.SetBytes(out, "usage.prompt_tokens_details.cached_tokens", comp.CacheReadInputTokens)
	// Drop non-xAI aliases if present from prior injections.
	out, _ = sjson.DeleteBytes(out, "usage.input_tokens")
	out, _ = sjson.DeleteBytes(out, "usage.output_tokens")
	return out
}

// ApplyToClaudeBody injects compensated usage into Anthropic message JSON or SSE event JSON.
func (e *Engine) ApplyToClaudeBody(cfg Config, model string, originalRequest, body []byte) []byte {
	cfg = cfg.Normalize()
	if !cfg.Enabled || len(body) == 0 {
		return body
	}
	payload, ok := stripSSE(body)
	if !ok {
		return body
	}
	// Locate usage node: top-level, message.usage, or nested
	usagePath := ""
	if gjson.GetBytes(payload, "usage").Exists() {
		usagePath = "usage"
	} else if gjson.GetBytes(payload, "message.usage").Exists() {
		usagePath = "message.usage"
	} else {
		return body
	}
	usageNode := gjson.GetBytes(payload, usagePath)
	if !usageNode.IsObject() {
		return body
	}
	stats := e.AnalyzeRequest(cfg, model, originalRequest)
	raw := NormalizeUsage(usageNode)
	// Claude extractOpenAIUsage already subtracts cached from input; rebuild total if needed
	if raw.CacheReadInputTokens > 0 {
		// input is uncached already in some paths; rebuild total for compensation base
		raw.InputTokens = raw.InputTokens + raw.CacheReadInputTokens + raw.CacheCreationInputTokens
		// keep creation/read as upstream hints
	}
	comp := Compensate(cfg, stats, raw, originalRequest)
	out := payload
	out, _ = sjson.SetBytes(out, usagePath+".input_tokens", comp.InputTokens)
	out, _ = sjson.SetBytes(out, usagePath+".output_tokens", comp.OutputTokens)
	out, _ = sjson.SetBytes(out, usagePath+".cache_creation_input_tokens", comp.CacheCreationInputTokens)
	out, _ = sjson.SetBytes(out, usagePath+".cache_read_input_tokens", comp.CacheReadInputTokens)
	return out
}

// ApplyStreamOpenAI buffers stream chunks, compensates final usage, copies to first chunk.
func (e *Engine) ApplyStreamOpenAI(cfg Config, model string, originalRequest []byte, chunks [][]byte) [][]byte {
	if len(chunks) == 0 {
		return chunks
	}
	cfg = cfg.Normalize()
	out := make([][]byte, len(chunks))
	var lastUsage string
	for i, ch := range chunks {
		c := e.ApplyToOpenAIBody(cfg, model, originalRequest, ch)
		out[i] = c
		if payload, ok := stripSSE(c); ok {
			if u := gjson.GetBytes(payload, "usage"); u.Exists() && u.IsObject() {
				lastUsage = u.Raw
			}
		}
	}
	if lastUsage == "" || !cfg.Enabled {
		return out
	}
	for i, ch := range out {
		payload, ok := stripSSE(ch)
		if !ok {
			continue
		}
		updated, err := sjson.SetRawBytes(payload, "usage", []byte(lastUsage))
		if err != nil {
			return out
		}
		out[i] = updated
		break
	}
	return out
}

// ApplyStreamClaude copies compensated usage onto message_start and message_delta events.
func (e *Engine) ApplyStreamClaude(cfg Config, model string, originalRequest []byte, chunks [][]byte) [][]byte {
	if len(chunks) == 0 {
		return chunks
	}
	cfg = cfg.Normalize()
	out := make([][]byte, len(chunks))
	var lastUsage string
	for i, ch := range chunks {
		c := e.ApplyToClaudeBody(cfg, model, originalRequest, ch)
		out[i] = c
		if payload, ok := stripSSE(c); ok {
			if u := gjson.GetBytes(payload, "usage"); u.Exists() && u.IsObject() {
				lastUsage = u.Raw
			} else if u := gjson.GetBytes(payload, "message.usage"); u.Exists() && u.IsObject() {
				lastUsage = u.Raw
			}
		}
	}
	if lastUsage == "" || !cfg.Enabled {
		return out
	}
	// Apply final usage to message_start (message.usage) and any message_delta usage
	for i, ch := range out {
		payload, ok := stripSSE(ch)
		if !ok {
			continue
		}
		typ := gjson.GetBytes(payload, "type").String()
		var updated []byte
		var err error
		switch typ {
		case "message_start":
			updated, err = sjson.SetRawBytes(payload, "message.usage", []byte(lastUsage))
		case "message_delta":
			updated, err = sjson.SetRawBytes(payload, "usage", []byte(lastUsage))
		default:
			continue
		}
		if err != nil {
			continue
		}
		out[i] = updated
	}
	return out
}

func (e *Engine) touch(key string, ttl time.Duration, maxEntries int) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	now := time.Now()
	// purge expired occasionally
	if len(e.entries) > 0 && len(e.entries)%64 == 0 {
		e.purgeExpiredLocked(now)
	}
	if ent, ok := e.entries[key]; ok && now.Before(ent.expires) {
		ent.expires = now.Add(ttl)
		e.entries[key] = ent
		return true
	}
	// insert
	e.entries[key] = cacheEntry{expires: now.Add(ttl)}
	e.order = append(e.order, key)
	if maxEntries > 0 && len(e.order) > maxEntries {
		// drop oldest
		drop := e.order[0]
		e.order = e.order[1:]
		delete(e.entries, drop)
	}
	return false
}

func (e *Engine) purgeExpiredLocked(now time.Time) {
	for k, ent := range e.entries {
		if now.After(ent.expires) {
			delete(e.entries, k)
		}
	}
	// rebuild order without deleted keys
	next := e.order[:0]
	for _, k := range e.order {
		if _, ok := e.entries[k]; ok {
			next = append(next, k)
		}
	}
	e.order = next
}

func metadataCompensation(cfg Config, originalRequest []byte) int64 {
	root := gjson.ParseBytes(originalRequest)
	messagesCount := int64(len(root.Get("messages").Array()))
	toolsCount := int64(len(root.Get("tools").Array()))
	systemCount := int64(0)
	if sys := root.Get("system"); sys.Exists() {
		if sys.IsArray() {
			systemCount = int64(len(sys.Array()))
		} else {
			systemCount = 1
		}
	}
	hasThinking := root.Get("thinking").Exists() || root.Get("reasoning").Exists() ||
		root.Get("reasoning_effort").Exists() || strings.Contains(strings.ToLower(root.Get("model").String()), "thinking")
	hasToolResults := false
	for _, msg := range root.Get("messages").Array() {
		if msg.Get("role").String() == "tool" {
			hasToolResults = true
			break
		}
		if content := msg.Get("content"); content.IsArray() {
			for _, b := range content.Array() {
				if b.Get("type").String() == "tool_result" {
					hasToolResults = true
					break
				}
			}
		}
	}
	comp := cfg.BaseOverhead +
		messagesCount*cfg.PerMessageOverhead +
		toolsCount*cfg.PerToolOverhead +
		systemCount*cfg.PerSystemOverhead +
		cfg.InjectedSystemTokens
	if hasThinking {
		comp += cfg.ThinkingOverhead
	}
	if hasToolResults {
		comp += cfg.ToolResultOverhead
	}
	return comp
}

func applyTokenCompensation(raw, compensation, minInput int64) int64 {
	if minInput <= 0 {
		minInput = 10
	}
	out := raw - compensation
	if out < minInput {
		return minInput
	}
	return out
}

func applyCacheHitRateCompensation(input, creation, read int64, target float64) (int64, int64, int64) {
	if target < 0 {
		target = 0
	}
	if target > 1 {
		target = 1
	}
	total := input + creation + read
	if total <= 0 {
		return input, creation, read
	}
	current := float64(read) / float64(total)
	if current >= target {
		return input, creation, read
	}
	targetRead := int64(math.Ceil(target * float64(total)))
	need := targetRead - read
	maxTransferable := input - 1
	if maxTransferable < 0 {
		maxTransferable = 0
	}
	transfer := need
	if transfer > maxTransferable {
		transfer = maxTransferable
	}
	if transfer < 0 {
		transfer = 0
	}
	return input - transfer, creation, read + transfer
}

func estimateTokens(s string) int64 {
	if s == "" {
		return 0
	}
	var cjk, other int
	for _, r := range s {
		if isCJK(r) {
			cjk++
		} else if !unicode.IsSpace(r) {
			other++
		}
	}
	// CJK 1.5 char/token, other 4 char/token
	tokens := math.Ceil(float64(cjk)/1.5 + float64(other)/4.0)
	if tokens < 1 && (cjk+other) > 0 {
		return 1
	}
	return int64(tokens)
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hangul, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		(r >= 0x3000 && r <= 0x303F) // CJK punctuation
}

func contentText(block gjson.Result) string {
	if t := block.Get("text"); t.Exists() {
		return t.String()
	}
	if t := block.Get("content"); t.Exists() && t.Type == gjson.String {
		return t.String()
	}
	if block.Type == gjson.String {
		return block.String()
	}
	// include type/name for tools
	var b strings.Builder
	if n := block.Get("name"); n.Exists() {
		b.WriteString(n.String())
		b.WriteByte(' ')
	}
	if block.IsObject() || block.IsArray() {
		b.WriteString(block.Raw)
	}
	return b.String()
}

func hasCacheControl(node gjson.Result) bool {
	cc := node.Get("cache_control")
	if !cc.Exists() {
		return false
	}
	if cc.IsObject() {
		return true
	}
	return cc.Type != gjson.Null
}

func cacheControlType(node gjson.Result) string {
	t := strings.ToLower(strings.TrimSpace(node.Get("cache_control.type").String()))
	switch t {
	case "static", "ephemeral":
		return t
	default:
		if hasCacheControl(node) {
			return "ephemeral"
		}
		return ""
	}
}

func promptCacheKey(model string, parts []string, cacheType string) string {
	h := sha256.New()
	h.Write([]byte(model))
	h.Write([]byte("|grok|"))
	for _, p := range parts {
		sum := sha256.Sum256([]byte(p))
		h.Write(sum[:])
		h.Write([]byte("|"))
	}
	h.Write([]byte(cacheType))
	return hex.EncodeToString(h.Sum(nil))
}

func firstInt(node gjson.Result, paths ...string) int64 {
	for _, p := range paths {
		if v := node.Get(p); v.Exists() && v.Type != gjson.Null {
			return v.Int()
		}
	}
	return 0
}

func stripSSE(body []byte) ([]byte, bool) {
	payload := bytesTrimSpace(body)
	if len(payload) == 0 {
		return nil, false
	}
	if string(payload) == "[DONE]" {
		return nil, false
	}
	if len(payload) >= 5 && string(payload[:5]) == "data:" {
		payload = bytesTrimSpace(payload[5:])
		if string(payload) == "[DONE]" {
			return nil, false
		}
	}
	// multi-line SSE: take data: line if present
	if !gjson.ValidBytes(payload) {
		// try extract JSON object
		if i := indexByte(payload, '{'); i >= 0 {
			payload = payload[i:]
		}
		if !gjson.ValidBytes(payload) {
			return nil, false
		}
	}
	return payload, true
}

func bytesTrimSpace(b []byte) []byte {
	i, j := 0, len(b)
	for i < j {
		r, size := utf8.DecodeRune(b[i:])
		if !unicode.IsSpace(r) {
			break
		}
		i += size
	}
	for j > i {
		r, size := utf8.DecodeLastRune(b[i:j])
		if !unicode.IsSpace(r) {
			break
		}
		j -= size
	}
	return b[i:j]
}

func indexByte(b []byte, c byte) int {
	for i, v := range b {
		if v == c {
			return i
		}
	}
	return -1
}
