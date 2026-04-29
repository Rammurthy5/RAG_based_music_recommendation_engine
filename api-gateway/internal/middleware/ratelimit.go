package middleware

import (
	"net/http"
	"sync"
	"time"
)

// RateLimiter implements a simple token-bucket rate limiter per IP.
type RateLimiter struct {
	mu       sync.Mutex
	visitors map[string]*bucket
	rate     int
	burst    int
	interval time.Duration
}

type bucket struct {
	tokens   int
	lastFill time.Time
}

// NewRateLimiter creates a rate limiter.
func NewRateLimiter(rate, burst int, interval time.Duration) *RateLimiter {
	rl := &RateLimiter{
		visitors: make(map[string]*bucket),
		rate:     rate,
		burst:    burst,
		interval: interval,
	}
	go rl.cleanup()
	return rl
}

func (rl *RateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	b, exists := rl.visitors[ip]
	now := time.Now()

	if !exists {
		rl.visitors[ip] = &bucket{tokens: rl.burst - 1, lastFill: now}
		return true
	}

	elapsed := now.Sub(b.lastFill)
	refills := int(elapsed / rl.interval)
	if refills > 0 {
		b.tokens += refills * rl.rate
		if b.tokens > rl.burst {
			b.tokens = rl.burst
		}
		b.lastFill = now
	}

	if b.tokens <= 0 {
		return false
	}

	b.tokens--
	return true
}

func (rl *RateLimiter) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		rl.mu.Lock()
		cutoff := time.Now().Add(-10 * time.Minute)
		for ip, b := range rl.visitors {
			if b.lastFill.Before(cutoff) {
				delete(rl.visitors, ip)
			}
		}
		rl.mu.Unlock()
	}
}

// Handler returns middleware that rate-limits by client IP.
func (rl *RateLimiter) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr
		if !rl.allow(ip) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":"rate limit exceeded"}`))
			return
		}
		next.ServeHTTP(w, r)
	})
}
