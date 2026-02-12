# specific implementation note for TEN VAD Usage (Dockerized)

## Overview
This document outlines the plan to integrate [TEN VAD](https://github.com/TEN-framework/ten-vad) into the `agent-starter-python` project, replacing the current `silero` VAD. TEN VAD is chosen for its high performance and low latency. The project is deployed via Docker.

## Prerequisites
-   **Docker Environment**: Linux-based (Debian Bookworm Slim).
-   **Dependency Management**: `uv`.
-   **System Requirements**: `libc++1` is required by TEN VAD on Linux.

## Integration Plan

### 1. Docker & Dependency Configuration

#### Update `Dockerfile`
We must ensure `libc++1` is installed in the container.
```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    libc++1 \
    ...
```

#### Update `pyproject.toml`
*Note: The user requested to skip dependency installation for now. This step will be performed when ready to build.*
```toml
# To be added later:
# [project]
# dependencies = [
#     ...
#     "ten-vad @ git+https://github.com/TEN-framework/ten-vad.git",
# ]
```

### 3. Docker Compose & Deployment
The project is built via `Dockerfile` which installs dependencies from `pyproject.toml` using `uv`. 
**Crucial**: You must add the dependency to `pyproject.toml` and then rebuild the image.

#### 1. Add Dependency to `pyproject.toml`
Add the following line to the `dependencies` array:
```toml
"ten-vad @ git+https://github.com/TEN-framework/ten-vad.git",
```

#### 2. Rebuild with Docker Compose
Run the following command to rebuild the container with the new dependency:
```bash
docker-compose up --build
```
This ensures `uv sync` runs during the build and installs `ten-vad`.

## Steps
1.  [x] Update `Dockerfile`. (Added `libc++1`)
2.  [ ] Update `pyproject.toml` (To be done before next build).
3.  [x] Create `src/ten_vad_adapter.py`. (Direct import used; no graceful fallback).
4.  [x] Update `src/agent.py`.
