# Council of AIs: Change Review

## 1. Objective
Replace the existing `Silero VAD` implementation with `TEN VAD` (released by Agora/TEN Framework) to leverage its reported performance benefits:
*   **Lower Latency**: "Agent-Friendly" response times.
*   **Higher Accuracy**: Improved precision/recall on benchmarks.
*   **Lightweight**: Reduced computational overhead.

## 2. Changes Implemented

### A. Infrastructure (`Dockerfile`)
*   **Change**: Added `libc++1` to system dependencies.
*   **Rationale**: `TEN VAD`'s underlying C++ library requires `libc++1` on Linux environments (Debian Bookworm).
*   **Status**: ✅ Applied.

### B. Dependency Management (`pyproject.toml`)
*   **Change**: *Planned* addition of `ten-vad @ git+https://github.com/TEN-framework/ten-vad.git`.
*   **Rationale**: The package is not yet on PyPI and must be installed directly from source.
*   **Status**: ⏸️ Deferred (User instruction: "dont download dependencies now"). The user must add this line before the next build.

### C. Adapter Implementation (`src/ten_vad_adapter.py`)
*   **Change**: Created `TenLiveKitVAD` class inheriting from `livekit.agents.vad.VAD`.
*   **Key Design Decisions**:
    *   **Direct Import**: Removed graceful degradation. The module now crashes if `ten_vad` is missing (User request: "remove gracefull handle").
    *   **Buffering**: Implemented a buffering mechanism to accumulate incoming audio samples until a full frame (256 samples @ 16kHz) is available, matching TEN VAD's stride requirement.
    *   **Adapter Pattern**: adhere strictly to LiveKit's VAD interface for seamless drop-in replacement.
    *   **Hard Usage**: The implementation uses the defaults optimized for this model (16kHz sample rate, 256 sample hop size).

### D. Agent Integration (`src/agent.py`)
*   **Change**: Swapped `silero` for `TenLiveKitVAD`.
*   **Details**:
    *   Imported `TenLiveKitVAD`.
    *   Updated `prewarm` function to initialize the new VAD.
    *   Passed the new VAD instance to `AgentSession`.
*   **Status**: ✅ Applied.

## 3. Deployment Instructions (Docker Compose)
To deploy these changes, the following **MUST** be executed:

1.  **Update Configuration**:
    Add the dependency to `pyproject.toml`:
    ```toml
    "ten-vad @ git+https://github.com/TEN-framework/ten-vad.git",
    ```

2.  **Rebuild Container**:
    ```bash
    docker-compose up --build
    ```
    *Failure to rebuild will result in `ImportError: No module named 'ten_vad'`.*

## 4. Risk Assessment
*   **Dependency Stability**: Installing from a git repository (`git+https`) carries a risk if the remote repository changes or is deleted. *Mitigation: Monitor the repo or mirror it locally.*
*   **Runtime Failure**: If `libc++1` is missing, the shared library load will fail. *Mitigation: Dockerfile update included.*
*   **Latency**: If the audio pipeline introduces unusual chunk sizes, the adapter's buffering might introduce slight jitter. *Mitigation: `src/audio_pipeline.py` appears to output consistent streams, but real-world testing is advised.*

## 5. Verification Plan
*   **Build Test**: Verify `uv sync` succeeds during Docker build.
*   **Runtime Test**: Launch agent and speak. Verify VAD triggers `START_OF_SPEECH` and `END_OF_SPEECH` events in logs.
*   **Performance**: Subjectively evaluate latency compared to the previous Silero implementation.

---
**Review Status**: Ready for experimental deployment.
