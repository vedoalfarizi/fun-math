.PHONY: help setup up start down down-v build logs ps restart ollama-setup backend-shell frontend-shell

# Default target
help:
	@echo ""
	@echo "Agentic Math Learning System — Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "  setup           Copy .env.example to .env (skips if .env already exists)"
	@echo "  up              Build images and start all services (foreground)"
	@echo "  start           Build images and start all services (detached/background)"
	@echo "  down            Stop containers, keep data volumes"
	@echo "  down-v          Stop containers AND delete all data volumes"
	@echo "  build           Rebuild images without starting services"
	@echo "  logs            Tail logs from all services"
	@echo "  ps              Show running service status"
	@echo "  restart         Restart all services"
	@echo "  ollama-setup    Pull the required Ollama model on the host (llama3:latest)"
	@echo "  backend-shell   Open a shell inside the backend container"
	@echo "  frontend-shell  Open a shell inside the frontend container"
	@echo ""

# Copy .env.example → .env (won't overwrite if .env already exists)
setup:
	@if [ -f .env ]; then \
		echo ".env already exists — skipping. Edit it manually if needed."; \
	else \
		cp .env.example .env; \
		echo ".env created from .env.example — fill in your values before running 'make up'."; \
	fi

# Build and start all services in the foreground
up:
	docker-compose up --build

# Build and start all services in the background
start:
	docker-compose up --build -d

# Stop all containers, preserve data volumes
down:
	docker-compose down

# Stop all containers and delete all data volumes (destructive!)
down-v:
	docker-compose down -v

# Rebuild images without starting
build:
	docker-compose build

# Tail logs from all services
logs:
	docker-compose logs -f

# Show service status
ps:
	docker-compose ps

# Restart all services
restart:
	docker-compose restart

# Pull the required Ollama model on the host machine
ollama-setup:
	ollama pull llama3:latest

# Open a shell inside the running backend container
backend-shell:
	docker-compose exec backend sh

# Open a shell inside the running frontend container
frontend-shell:
	docker-compose exec frontend sh
