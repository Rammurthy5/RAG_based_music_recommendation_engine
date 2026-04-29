# Build stage
FROM golang:1.22-alpine AS build

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /api-gateway ./cmd/server

# Runtime stage
FROM alpine:3.20

RUN apk add --no-cache wget ca-certificates \
    && adduser -D -u 1000 appuser

COPY --from=build /api-gateway /api-gateway

USER appuser
EXPOSE 8080

ENTRYPOINT ["/api-gateway"]
