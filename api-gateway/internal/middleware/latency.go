package middleware

import (
	"net/http"
	"strconv"
	"time"
)

// Latency adds X-Latency-Ms response header with millisecond precision.
func Latency(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		ms := time.Since(start).Milliseconds()
		w.Header().Set("X-Latency-Ms", strconv.FormatInt(ms, 10))
	})
}
