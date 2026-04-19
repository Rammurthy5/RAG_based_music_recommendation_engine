package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/sony/gobreaker"
)

// RAGClient proxies requests to the Python RAG service.
type RAGClient struct {
	baseURL    string
	httpClient *http.Client
	cb         *gobreaker.CircuitBreaker
}

// NewRAGClient creates a client for the RAG service with circuit breaker.
func NewRAGClient(baseURL string) *RAGClient {
	cbSettings := gobreaker.Settings{
		Name:        "RAGService",
		MaxRequests: 3,
		Interval:    60 * time.Second,
		Timeout:     60 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > 5
		},
	}

	return &RAGClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 45 * time.Second,
		},
		cb: gobreaker.NewCircuitBreaker(cbSettings),
	}
}

// Recommend sends a recommendation request to the RAG service.
func (c *RAGClient) Recommend(ctx context.Context, body []byte) (json.RawMessage, error) {
	result, err := c.cb.Execute(func() (interface{}, error) {
		reqCtx, cancel := context.WithTimeout(ctx, 35*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL+"/recommend", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("creating request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("calling rag-service: %w", err)
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("reading response: %w", err)
		}

		if resp.StatusCode >= 500 {
			return nil, fmt.Errorf("rag-service returned %d: %s", resp.StatusCode, string(respBody))
		}

		return json.RawMessage(respBody), nil
	})

	if err != nil {
		return nil, err
	}
	return result.(json.RawMessage), nil
}

// HealthCheck calls the RAG service /health endpoint.
func (c *RAGClient) HealthCheck(ctx context.Context) (json.RawMessage, error) {
	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return nil, fmt.Errorf("creating health request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading health response: %w", err)
	}

	return json.RawMessage(body), nil
}
