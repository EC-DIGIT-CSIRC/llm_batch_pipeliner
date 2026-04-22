# AGENTS

## OLLAMA

- Never start additional Ollama shard servers against a shared `OLLAMA_MODELS` directory unless `OLLAMA_NOPRUNE=true` is set for every shard process.
- When matching the main Ollama service configuration, also carry over its other relevant environment variables such as `OLLAMA_MODELS` and `OLLAMA_NUM_PARALLEL`.
