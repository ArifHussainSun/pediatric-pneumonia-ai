#!/bin/bash

# Production Deployment Script for Pneumonia Detection API
# Automated deployment with health checks and rollback capability

set -e

# Configuration
PROJECT_NAME="pneumonia-detection-api"
DOCKER_IMAGE="pneumonia-api"
CONTAINER_NAME="pneumonia-api-container"
API_PORT=8000
HEALTH_CHECK_URL="http://localhost:${API_PORT}/health"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."

    docker build -t ${DOCKER_IMAGE}:latest . || {
        log_error "Docker build failed"
        exit 1
    }

    log_info "Docker image built successfully"
}

# Stop existing container
stop_existing() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        log_info "Stopping existing container..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi
}

# Deploy new container
deploy_container() {
    log_info "Deploying new container..."

    docker run -d \
        --name ${CONTAINER_NAME} \
        --restart unless-stopped \
        -p ${API_PORT}:${API_PORT} \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/configs:/app/configs \
        -e API_PORT=${API_PORT} \
        ${DOCKER_IMAGE}:latest || {
        log_error "Container deployment failed"
        exit 1
    }

    log_info "Container deployed successfully"
}

# Health check
health_check() {
    log_info "Performing health check..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s ${HEALTH_CHECK_URL} > /dev/null; then
            log_info "Health check passed"
            return 0
        fi

        log_warn "Health check attempt ${attempt}/${max_attempts} failed, waiting..."
        sleep 5
        ((attempt++))
    done

    log_error "Health check failed after ${max_attempts} attempts"
    return 1
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."

    # Stop failed container
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi

    # Start previous version if available
    if docker images -q ${DOCKER_IMAGE}:previous | grep -q .; then
        log_info "Starting previous version..."
        docker run -d \
            --name ${CONTAINER_NAME} \
            --restart unless-stopped \
            -p ${API_PORT}:${API_PORT} \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/configs:/app/configs \
            ${DOCKER_IMAGE}:previous
    fi

    log_warn "Rollback completed"
}

# Tag current image as previous
tag_previous() {
    if docker images -q ${DOCKER_IMAGE}:latest | grep -q .; then
        docker tag ${DOCKER_IMAGE}:latest ${DOCKER_IMAGE}:previous
    fi
}

# Main deployment function
deploy() {
    log_info "Starting deployment of ${PROJECT_NAME}..."

    # Check prerequisites
    check_prerequisites

    # Tag current as previous
    tag_previous

    # Build new image
    build_image

    # Stop existing container
    stop_existing

    # Deploy new container
    deploy_container

    # Wait for startup
    sleep 10

    # Health check
    if ! health_check; then
        log_error "Deployment failed health check"
        rollback
        exit 1
    fi

    log_info "Deployment completed successfully!"
    log_info "API is available at: http://localhost:${API_PORT}"
}

# Show usage
usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  deploy    - Deploy the application"
    echo "  stop      - Stop the application"
    echo "  restart   - Restart the application"
    echo "  logs      - Show application logs"
    echo "  status    - Show application status"
}

# Stop application
stop_app() {
    log_info "Stopping application..."
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        log_info "Application stopped"
    else
        log_warn "No running container found"
    fi
}

# Restart application
restart_app() {
    log_info "Restarting application..."
    stop_app

    if docker images -q ${DOCKER_IMAGE}:latest | grep -q .; then
        deploy_container
        sleep 10
        health_check
        log_info "Application restarted successfully"
    else
        log_error "No image found to restart"
        exit 1
    fi
}

# Show logs
show_logs() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker logs -f ${CONTAINER_NAME}
    else
        log_warn "No running container found"
    fi
}

# Show status
show_status() {
    log_info "Application status:"

    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        echo "Status: Running"
        echo "Container ID: $(docker ps -q -f name=${CONTAINER_NAME})"
        echo "Port: ${API_PORT}"

        # Check health
        if curl -s ${HEALTH_CHECK_URL} > /dev/null; then
            echo "Health: OK"
        else
            echo "Health: FAILED"
        fi
    else
        echo "Status: Stopped"
    fi
}

# Main script
case "$1" in
    deploy)
        deploy
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    *)
        usage
        exit 1
        ;;
esac