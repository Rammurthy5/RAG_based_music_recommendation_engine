package handler_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/rsi03/music-rec-gateway/internal/client"
	"github.com/rsi03/music-rec-gateway/internal/handler"
)

func setupTestHandler(t *testing.T, ragServer *httptest.Server) *handler.Handler {
	t.Helper()
	ragClient := client.NewRAGClient(ragServer.URL, 5*time.Second, 5, 60*time.Second, 3)
	return handler.NewHandler(ragClient)
}

func TestHealth(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	w := httptest.NewRecorder()

	h.Health(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"status":"ok"`) {
		t.Fatalf("unexpected body: %s", body)
	}
}

func TestRecommend_EmptyQuery(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	body := `{"query":"","limit":5}`
	req := httptest.NewRequest(http.MethodPost, "/api/recommend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Recommend(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestRecommend_QueryTooLong(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	longQuery := strings.Repeat("a", 501)
	body := `{"query":"` + longQuery + `","limit":5}`
	req := httptest.NewRequest(http.MethodPost, "/api/recommend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Recommend(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestRecommend_InvalidLimit(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	body := `{"query":"happy vibes","limit":99}`
	req := httptest.NewRequest(http.MethodPost, "/api/recommend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Recommend(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestRecommend_InvalidJSON(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	body := `not json at all`
	req := httptest.NewRequest(http.MethodPost, "/api/recommend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Recommend(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestRecommend_Success(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"query":"chill","recommendations":[],"metadata":{"source":"full_rag"}}`))
	}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	body := `{"query":"chill vibes","limit":5}`
	req := httptest.NewRequest(http.MethodPost, "/api/recommend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Recommend(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
}

func TestRecommend_ServiceUnavailable(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"boom"}`))
	}))
	defer ragServer.Close()

	h := setupTestHandler(t, ragServer)
	body := `{"query":"sad songs","limit":3}`
	req := httptest.NewRequest(http.MethodPost, "/api/recommend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Recommend(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}
}
