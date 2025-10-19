# 0010 – AI Curation Policy
Status: Accepted
Date: 2025-10-19

## Context
This project follows the Software Curator methodology: AI accelerates drafting and scaffolding, while a human curator remains accountable for direction and releases. We also want a consistent AI configuration across clients.

## Decision
- All non-trivial behavior changes must include tests and, where appropriate, an ADR update.
- Pre-commit and CI enforce quick feedback and repository health.
- Provenance is documented in README (Attribution & Curation) and `AUTHORS`.
- Default AI model for curation assistance is "Claude Sonnet 4" enabled for all clients via `.curator/config.yaml`.

## Consequences
Positive:
- Transparent AI involvement and consistent results across tools.
- Repeatable local checks and CI stability.

Negative / Trade-offs:
- Slight overhead maintaining ADRs and hooks.

Follow-ups:
- Keep `.curator/config.yaml` updated if provider/model naming changes.
