package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"

	"github.com/rsi03/music-rec-gateway/internal/client"
	"github.com/rsi03/music-rec-gateway/internal/handler"
	"github.com/rsi03/music-rec-gateway/internal/middleware"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	ragURL := os.Getenv("RAG_SERVICE_URL")
	if ragURL == "" {
		ragURL = "http://rag-service:8000"
	}

	ragClient := client.NewRAGClient(ragURL)

	r := chi.NewRouter()

	// Middleware stack
	r.Use(middleware.RequestID)
	r.Use(chimiddleware.Logger)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.RealIP)
	r.Use(chimiddleware.Timeout(45 * time.Second))
	r.Use(middleware.NewRateLimiter(20, 40, time.Second).Handler) // 20 req/s per IP, burst 40
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"http://localhost:3000", "http://localhost:*"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Content-Type", "X-Request-ID"},
		ExposedHeaders:   []string{"X-Latency-Ms", "X-Request-ID"},
		AllowCredentials: false,
		MaxAge:           300,
	}))

	// Routes
	h := handler.NewHandler(ragClient)
	r.Get("/api/health", h.Health)
	r.Get("/api/health/ready", h.HealthReady)
	r.Post("/api/recommend", h.Recommend)

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 45 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("api-gateway listening on :%s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "server error: %v\n", err)
			os.Exit(1)
		}
	}()

	<-done
	log.Println("api-gateway shutting down…")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("graceful shutdown failed: %v", err)
	}
	log.Println("api-gateway stopped.")
}
