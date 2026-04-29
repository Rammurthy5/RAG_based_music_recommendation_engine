package handler

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/rsi03/music-rec-gateway/internal/client"
)

const maxBodySize = 10 * 1024 // 10KB

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	ragClient *client.RAGClient
}

// NewHandler creates a handler with the given RAG client.
func NewHandler(ragClient *client.RAGClient) *Handler {
	return &Handler{ragClient: ragClient}
}

// Recommend proxies the recommendation request to the RAG service.
func (h *Handler) Recommend(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "request body too large or unreadable")
		return
	}

	// Validate required fields
	var req struct {
		Query   string      `json:"query"`
		Limit   *int        `json:"limit,omitempty"`
		Filters interface{} `json:"filters,omitempty"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if len(req.Query) == 0 || len(req.Query) > 500 {
		writeError(w, http.StatusBadRequest, "query must be 1-500 characters")
		return
	}
	if req.Limit != nil && (*req.Limit < 1 || *req.Limit > 20) {
		writeError(w, http.StatusBadRequest, "limit must be 1-20")
		return
	}

	result, err := h.ragClient.Recommend(r.Context(), body)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "recommendation service unavailable")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(result)
}

// Health returns a liveness check for the gateway itself.
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok","service":"api-gateway","version":"1.0.0"}`))
}

// HealthReady performs a deep check including downstream dependencies.
func (h *Handler) HealthReady(w http.ResponseWriter, r *http.Request) {
	ragHealth, err := h.ragClient.HealthCheck(r.Context())
	cbState := h.ragClient.CircuitBreakerState()

	resp := map[string]interface{}{
		"status":          "ready",
		"circuit_breaker": cbState,
		"dependencies": map[string]interface{}{
			"rag_service": "ok",
		},
	}

	if err != nil || cbState == "open" {
		resp["status"] = "degraded"
		resp["dependencies"] = map[string]interface{}{
			"rag_service": "unavailable",
		}
	} else {
		var ragStatus map[string]interface{}
		if json.Unmarshal(ragHealth, &ragStatus) == nil {
			resp["dependencies"] = map[string]interface{}{
				"rag_service": "ok",
				"weaviate":    ragStatus["weaviate"],
			}
		}
	}

	status := http.StatusOK
	if resp["status"] == "degraded" {
		status = http.StatusServiceUnavailable
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(resp)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
