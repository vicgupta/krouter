package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

// Config maps config.json: where to listen, where to route.
type Config struct {
	Port      int    `json:"port"`
	InputPath string `json:"inputPath"`
	OutputURL string `json:"outputURL"`
	LogFile   string `json:"logFile"`
}

func LoadConfig(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer f.Close()

	var c Config
	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return Config{}, err
	}
	if c.Port == 0 || c.InputPath == "" || c.OutputURL == "" {
		return Config{}, fmt.Errorf("port, inputPath and outputURL are required")
	}
	if c.LogFile == "" {
		c.LogFile = "requests.log"
	}
	return c, nil
}

// AsyncLogger appends one line per request from a background goroutine.
// Log never blocks the caller: it drops with a warning when the buffer is full.
type AsyncLogger struct {
	ch chan string
}

func NewAsyncLogger(path string) (*AsyncLogger, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	l := &AsyncLogger{ch: make(chan string, 1024)}
	go func() {
		defer f.Close()
		for msg := range l.ch {
			fmt.Fprintf(f, "%s %s\n", time.Now().UTC().Format(time.RFC3339), msg)
		}
	}()
	return l, nil
}

func (l *AsyncLogger) Log(body []byte) {
	if l == nil {
		return
	}
	s := strings.ReplaceAll(string(body), "\n", " ")
	select {
	case l.ch <- s:
	default:
		log.Printf("request log full, dropping %d bytes", len(body))
	}
}

func (l *AsyncLogger) Close() {
	if l == nil {
		return
	}
	close(l.ch)
}

// Transform is a passthrough hook. Edit later to reshape payloads.
// Token accounting for this stage lives in CountTokens/ollamaCounts,
// wired in makeHandler (request in-tokens before forward,
// response out-tokens after upstream replies).
func Transform(body []byte) ([]byte, error) {
	return body, nil
}

// CountTokens approximates tokens as ceil(runes/4), the common
// ~4-chars-per-token rule. Fallback when upstream reports no counts.
func CountTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len([]rune(text)) + 3) / 4
}

// ollamaCounts extracts authoritative counts from an Ollama
// /api/generate response body.
func ollamaCounts(respBody []byte) (prompt, eval int, ok bool) {
	var v struct {
		PromptEvalCount int `json:"prompt_eval_count"`
		EvalCount       int `json:"eval_count"`
	}
	if err := json.Unmarshal(respBody, &v); err != nil {
		return 0, 0, false
	}
	if v.PromptEvalCount == 0 && v.EvalCount == 0 {
		return 0, 0, false
	}
	return v.PromptEvalCount, v.EvalCount, true
}

func responseText(respBody []byte) string {
	var v struct {
		Response string `json:"response"`
	}
	if err := json.Unmarshal(respBody, &v); err != nil || v.Response == "" {
		return string(respBody)
	}
	return v.Response
}

func makeHandler(cfg *atomic.Value, logger *AsyncLogger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c := cfg.Load().(Config)
		if r.URL.Path != c.InputPath {
			http.NotFound(w, r)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body", http.StatusBadRequest)
			return
		}

		logger.Log(body) // async, non-blocking; part of Transform stage
		out, err := Transform(body)
		if err != nil {
			http.Error(w, "transform", http.StatusUnprocessableEntity)
			return
		}

		target := c.OutputURL
		if q := r.URL.RawQuery; q != "" {
			target += "?" + q
		}
		req, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(out))
		if err != nil {
			http.Error(w, "build request", http.StatusInternalServerError)
			return
		}
		if ct := r.Header.Get("Content-Type"); ct != "" {
			req.Header.Set("Content-Type", ct)
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			http.Error(w, "post output: "+err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			http.Error(w, "read output", http.StatusBadGateway)
			return
		}

		inTok, outTok := CountTokens(string(body)), CountTokens(responseText(respBody))
		if p, e, ok := ollamaCounts(respBody); ok {
			inTok, outTok = p, e
		}
		logger.Log([]byte(fmt.Sprintf("tokens in=%d out=%d req_bytes=%d resp_bytes=%d status=%d",
			inTok, outTok, len(body), len(respBody), resp.StatusCode)))

		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(respBody)
	}
}

func watchConfig(path string, cfg *atomic.Value, interval time.Duration) {
	last, _ := modTime(path)
	for range time.Tick(interval) {
		mt, err := modTime(path)
		if err != nil || mt.Equal(last) {
			continue
		}
		last = mt
		c, err := LoadConfig(path)
		if err != nil {
			log.Printf("reload config: %v", err)
			continue
		}
		old := cfg.Load().(Config)
		cfg.Store(c)
		log.Printf("reloaded config: %+v", c)
		if c.Port != old.Port {
			log.Printf("port changed %d -> %d: restart to apply", old.Port, c.Port)
		}
	}
}

func modTime(path string) (time.Time, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return time.Time{}, err
	}
	return fi.ModTime(), nil
}

func main() {
	path := "config.json"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	c, err := LoadConfig(path)
	if err != nil {
		log.Fatal(err)
	}

	var cfg atomic.Value
	cfg.Store(c)

	logger, err := NewAsyncLogger(c.LogFile)
	if err != nil {
		log.Fatal(err)
	}
	defer logger.Close()

	go watchConfig(path, &cfg, 5*time.Second)

	addr := fmt.Sprintf(":%d", c.Port)
	log.Printf("listening on %s, %s -> %s (log %s)", addr, c.InputPath, c.OutputURL, c.LogFile)
	log.Fatal(http.ListenAndServe(addr, makeHandler(&cfg, logger)))
}
