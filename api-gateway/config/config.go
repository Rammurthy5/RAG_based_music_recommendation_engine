package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds all gateway configuration sourced from environment variables.
type Config struct {
	Port           string
	RAGServiceURL  string
	AllowedOrigins []string

	// Timeouts
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	IdleTimeout     time.Duration
	ProxyTimeout    time.Duration
	HealthTimeout   time.Duration
	ShutdownTimeout time.Duration

	// Rate limiting
	RateLimit int
	RateBurst int

	// Circuit breaker
	CBMaxFailures  int
	CBResetTimeout time.Duration
	CBMaxRequests  int

	// Limits
	MaxBodySize     int64
	MaxResponseSize int64
}

// Load reads configuration from environment with sensible defaults.
func Load() *Config {
	return &Config{
		Port:           envOrDefault("PORT", "8080"),
		RAGServiceURL:  envOrDefault("RAG_SERVICE_URL", "http://rag-service:8000"),
		AllowedOrigins: []string{"http://localhost:3000", "http://localhost:*"},

		ReadTimeout:     envDurationOrDefault("READ_TIMEOUT", 5*time.Second),
		WriteTimeout:    envDurationOrDefault("WRITE_TIMEOUT", 45*time.Second),
		IdleTimeout:     envDurationOrDefault("IDLE_TIMEOUT", 120*time.Second),
		ProxyTimeout:    envDurationOrDefault("PROXY_TIMEOUT", 35*time.Second),
		HealthTimeout:   envDurationOrDefault("HEALTH_TIMEOUT", 5*time.Second),
		ShutdownTimeout: envDurationOrDefault("SHUTDOWN_TIMEOUT", 10*time.Second),

		RateLimit: envIntOrDefault("RATE_LIMIT", 20),
		RateBurst: envIntOrDefault("RATE_BURST", 40),

		CBMaxFailures:  envIntOrDefault("CB_MAX_FAILURES", 5),
		CBResetTimeout: envDurationOrDefault("CB_RESET_TIMEOUT", 60*time.Second),
		CBMaxRequests:  envIntOrDefault("CB_MAX_REQUESTS", 3),

		MaxBodySize:     int64(envIntOrDefault("MAX_BODY_SIZE", 10*1024)),
		MaxResponseSize: int64(envIntOrDefault("MAX_RESPONSE_SIZE", 1024*1024)),
	}
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envIntOrDefault(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func envDurationOrDefault(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}
