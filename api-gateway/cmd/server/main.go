package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"

	"github.com/rsi03/music-rec-gateway/config"
	"github.com/rsi03/music-rec-gateway/internal/client"
	"github.com/rsi03/music-rec-gateway/internal/handler"
	"github.com/rsi03/music-rec-gateway/internal/middleware"
)

func main() {
	// Structured JSON logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg := config.Load()

	ragClient := client.NewRAGClient(
		cfg.RAGServiceURL,
		cfg.ProxyTimeout,
		cfg.CBMaxFailures,
		cfg.CBResetTimeout,
		cfg.CBMaxRequests,
	)

	r := chi.NewRouter()

	// Middleware stack (order matters)
	r.Use(middleware.RequestID)
	r.Use(middleware.SecurityHeaders)
	r.Use(middleware.Logger)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.RealIP)
	r.Use(chimiddleware.Timeout(cfg.WriteTimeout))
	r.Use(middleware.NewRateLimiter(cfg.RateLimit, cfg.RateBurst, time.Second).Handler)
	r.Use(middleware.Latency)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   cfg.AllowedOrigins,
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
	r.Route("/api", func(api chi.Router) {
		api.Use(middleware.ContentTypeJSON)
		api.Post("/recommend", h.Recommend)
	})

	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      r,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		slog.Info("api-gateway starting", "port", cfg.Port, "rag_service_url", cfg.RAGServiceURL)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "server error: %v\n", err)
			os.Exit(1)
		}
	}()

	<-done
	slog.Info("api-gateway shutting down")

	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("graceful shutdown failed", "error", err)
	}
	slog.Info("api-gateway stopped")
}
