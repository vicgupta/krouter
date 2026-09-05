package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestTransformPassthrough(t *testing.T) {
	cases := []struct {
		name string
		in   []byte
	}{
		{"empty", nil},
		{"json", []byte(`{"a":1}`)},
		{"text", []byte("hello")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Transform(tc.in)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(tc.in) {
				t.Fatalf("got %q want %q", got, tc.in)
			}
		})
	}
}

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	body := `{"port":8080,"inputPath":"/ingest","outputURL":"http://x/receive"}`
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.Port != 8080 || c.InputPath != "/ingest" || c.OutputURL != "http://x/receive" {
		t.Fatalf("bad config: %+v", c)
	}
}

func TestHandlerForwards(t *testing.T) {
	var gotBody, gotCT, gotQuery string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody, gotCT, gotQuery = string(b), r.Header.Get("Content-Type"), r.URL.RawQuery
		w.WriteHeader(201)
		_, _ = w.Write([]byte("ok"))
	}))
	defer upstream.Close()

	var cfg atomic.Value
	cfg.Store(Config{Port: 8080, InputPath: "/ingest", OutputURL: upstream.URL})
	h := makeHandler(&cfg, nil)

	req := httptest.NewRequest(http.MethodPost, "/ingest?k=v", strings.NewReader(`{"a":1}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h(rec, req)

	res := rec.Result()
	b, _ := io.ReadAll(res.Body)
	if res.StatusCode != 201 || string(b) != "ok" {
		t.Fatalf("got %d %q", res.StatusCode, b)
	}
	if gotBody != `{"a":1}` || gotCT != "application/json" || gotQuery != "k=v" {
		t.Fatalf("upstream got body=%q ct=%q q=%q", gotBody, gotCT, gotQuery)
	}
}

func TestCountTokens(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"", 0},
		{"Hi", 1},
		{"1234", 1},
		{"12345", 2},
		{`{"a":1}`, 2},
	}
	for _, tc := range cases {
		if got := CountTokens(tc.in); got != tc.want {
			t.Fatalf("CountTokens(%q) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestOllamaCounts(t *testing.T) {
	p, e, ok := ollamaCounts([]byte(`{"response":"Hello","prompt_eval_count":11,"eval_count":305}`))
	if !ok || p != 11 || e != 305 {
		t.Fatalf("got %d %d %v", p, e, ok)
	}
	if _, _, ok := ollamaCounts([]byte(`ok`)); ok {
		t.Fatal("expected not-ok for plain body")
	}
}

func TestHandlerLogsTokens(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"response":"Hello","prompt_eval_count":11,"eval_count":7}`))
	}))
	defer upstream.Close()

	dir := t.TempDir()
	l, err := NewAsyncLogger(filepath.Join(dir, "requests.log"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg atomic.Value
	cfg.Store(Config{Port: 8080, InputPath: "/ingest", OutputURL: upstream.URL})
	h := makeHandler(&cfg, l)

	req := httptest.NewRequest(http.MethodPost, "/ingest", strings.NewReader(`{"prompt":"Hi"}`))
	h(httptest.NewRecorder(), req)
	l.Close()

	p := filepath.Join(dir, "requests.log")
	for i := 0; i < 50; i++ {
		b, _ := os.ReadFile(p)
		if strings.Contains(string(b), "tokens in=11 out=7") {
			return
		}
		<-time.After(20 * time.Millisecond)
	}
	b, _ := os.ReadFile(p)
	t.Fatalf("missing token line, got %q", b)
}

func TestAsyncLoggerWritesAndNonBlocking(t *testing.T) {
	p := filepath.Join(t.TempDir(), "requests.log")
	l, err := NewAsyncLogger(p)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	l.Log([]byte(`{"hello":"world"}`))
	if time.Since(start) > time.Second {
		t.Fatal("Log blocked")
	}
	l.Close()
	// File write is async; poll briefly.
	for i := 0; i < 50; i++ {
		b, _ := os.ReadFile(p)
		if strings.Contains(string(b), `{"hello":"world"}`) {
			return
		}
		<-time.After(20 * time.Millisecond)
	}
	b, _ := os.ReadFile(p)
	t.Fatalf("log missing payload, got %q", b)
}
