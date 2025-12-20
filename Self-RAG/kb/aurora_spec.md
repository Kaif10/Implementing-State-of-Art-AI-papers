# Aurora Product Spec (v0.6)

## Ticket lifecycle
Tickets move through:
open → triaged → in_progress → (blocked) → resolved → closed

Rules:
- A ticket can move to blocked only from triaged or in_progress.
- A ticket can move to closed only from resolved.
- Tickets auto-close 14 days after resolved unless reopened.

## Priorities
- p0: production outage or severe security issue
- p1: major customer impact
- p2: normal customer issue
- p3: low urgency

## API basics
Base URL: https://api.aurora.example/v1

Authentication:
Authorization: Bearer <API_KEY>

Rate limits per workspace:
- 600 requests per minute
- burst up to 120 requests in 10 seconds

If rate limited:
- HTTP 429
- Retry-After header in seconds

## Logging restrictions
Do not log passwords, session cookies, API keys, or full request bodies that may contain Restricted data.
