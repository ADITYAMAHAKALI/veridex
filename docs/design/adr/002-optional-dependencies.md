# ADR-002: Optional Dependencies & Plugin System

## Status
Accepted

## Context
The library aims to be "lightweight", but state-of-the-art detection often requires heavy Deep Learning frameworks (PyTorch, Transformers, Diffusers). Forcing all users to install these multi-GB dependencies would hinder adoption for those who only need simple statistical signals or want to use the library in resource-constrained environments (e.g., AWS Lambda).

## Decision
We will use **Optional Dependencies** (via `extras_require` in `setup.py` / `pyproject.toml`) and a **Lazy Loading** mechanism for signals.

*   **Core:** The `veridex` core will only depend on `numpy` and `scipy`.
*   **Extras:** `veridex[text]` and `veridex[image]` will install the heavy ML libraries.
*   **Runtime Check:** Signals that require missing dependencies will raise an informative `ImportError` or `ModuleNotFoundError` *only when instantiated*.

## Consequences
### Positive
*   **Lightweight Core:** Users can install the base package quickly.
*   **Flexibility:** Users only pay the "disk space tax" for the modalities they use.

### Negative
*   **User Experience:** Users might encounter runtime errors if they forget to install the extras. We must provide clear error messages (e.g., "Please run `pip install veridex[text]` to use Binoculars").
