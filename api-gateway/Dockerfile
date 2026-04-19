# Build stage
FROM golang:1.22-alpine AS build

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /api-gateway ./cmd/server

# Runtime stage
FROM gcr.io/distroless/static-debian12

COPY --from=build /api-gateway /api-gateway

EXPOSE 8080

ENTRYPOINT ["/api-gateway"]
