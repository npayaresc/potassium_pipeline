# Docker Bake configuration for advanced caching
# This file enables BuildKit caching features

group "default" {
  targets = ["magnesium-api", "magnesium-pipeline", "magnesium-dev"]
}

target "magnesium-base" {
  dockerfile = "Dockerfile"
  target = "base"
  cache-from = [
    "type=registry,ref=magnesium-pipeline:cache-base",
    "type=local,src=/tmp/docker-cache"
  ]
  cache-to = [
    "type=registry,ref=magnesium-pipeline:cache-base,mode=max",
    "type=local,dest=/tmp/docker-cache,mode=max"
  ]
}

target "magnesium-api" {
  inherits = ["magnesium-base"]
  target = "production"
  tags = ["magnesium-pipeline-magnesium-api:latest"]
  cache-from = [
    "type=registry,ref=magnesium-pipeline:cache-api",
    "type=local,src=/tmp/docker-cache"
  ]
  cache-to = [
    "type=registry,ref=magnesium-pipeline:cache-api,mode=max",
    "type=local,dest=/tmp/docker-cache,mode=max"
  ]
}

target "magnesium-pipeline" {
  inherits = ["magnesium-base"]
  target = "production"
  tags = ["magnesium-pipeline-magnesium-pipeline:latest"]
  cache-from = [
    "type=registry,ref=magnesium-pipeline:cache-pipeline",
    "type=local,src=/tmp/docker-cache"
  ]
  cache-to = [
    "type=registry,ref=magnesium-pipeline:cache-pipeline,mode=max",
    "type=local,dest=/tmp/docker-cache,mode=max"
  ]
}

target "magnesium-dev" {
  inherits = ["magnesium-base"]
  target = "development"
  tags = ["magnesium-pipeline-magnesium-dev:latest"]
  cache-from = [
    "type=registry,ref=magnesium-pipeline:cache-dev",
    "type=local,src=/tmp/docker-cache"
  ]
  cache-to = [
    "type=registry,ref=magnesium-pipeline:cache-dev,mode=max",
    "type=local,dest=/tmp/docker-cache,mode=max"
  ]
}