# Week 1: Core RAG Improvements

## Day 1-2: Langfuse Integration + Tracing
**Status:** In Progress
**Branch:** feature/langfuse-observability
**Priority:** High

### Requirements
- [ ] Add Langfuse client initialization
- [ ] Implement @observe decorators for query pipeline
- [ ] Track retrieval latency and chunk metadata
- [ ] Track LLM generation (tokens, cost, latency)
- [ ] Add custom scoring for user feedback
- [ ] Create observability module with tracer class
- [ ] Write unit tests for tracer (100% coverage)
- [ ] Update .env.example with Langfuse keys
- [ ] Document Langfuse setup in README

### Acceptance Criteria
- All RAG queries traced in Langfuse dashboard
- Latency, tokens, and cost tracked per query
- User feedback scores logged
- Tests pass with 100% coverage
- No hardcoded secrets

### Technical Notes
- Use langfuse==3.12.1
- Implement as separate observability module
- Make Langfuse optional (graceful degradation if disabled)
