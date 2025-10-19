# 0000 – Record Architecture Decisions
Status: Accepted
Date: 2025-10-19

## Context
We want a lightweight way to document architectural and process decisions.

## Decision
Adopt Architecture Decision Records (ADRs) stored under `docs/adr/` with sequential numbering, using the included `TEMPLATE.md`. Maintain an index in `docs/adr/README.md`. CI and pre-commit will check that new ADRs are indexed.

## Consequences
Positive:
- Clear, searchable history of decisions.
- Easier onboarding and governance.

Negative / Trade-offs:
- Slight overhead to write ADRs.

Follow-ups:
- Keep ADR index updated.
