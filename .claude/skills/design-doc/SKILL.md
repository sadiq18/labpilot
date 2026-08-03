---
name: design-doc
description: >-
  Write comprehensive design documents for any system, feature, or architectural work. Use this skill whenever the user is working on: system architecture, microservices design, API design, infrastructure/DevOps changes, feature design, technical proposals, architecture decisions/tradeoffs, migration strategies, or technology selections. This applies to both brand new systems and incremental changes. The skill guides you through all 12 critical sections with detailed templates: Background, Problem Statement, Functional & Non-Functional Requirements, Goals & Success Criteria, Scope, High-Level Design (with diagrams), Components & Responsibility Boundaries, Low-Level Design (with code examples), Design Choices & Tradeoffs, Observability & Monitoring, Evaluation & Testing Strategy, and Production Readiness (deployment phases, rollback, testing, disaster recovery). Always use this skill proactively when you see: "design doc", "technical design", "system design", "architecture", "design proposal", "design review", "RFC", "requirements document", "implementation plan", "deployment strategy", or when someone describes needing to document any technical decision affecting code, infrastructure, or system behavior.
compatibility: []
---

# Design Document Framework

Use this framework to create comprehensive design documents that cover all critical aspects of architectural and feature work. Follow the structure below, adapting depth as needed based on the scope (new system vs. incremental change).

## When to Skip Sections

- **High-Level Design (HLD)**: Can be skipped if the change contains only low-level details or is a minor modification to well-understood systems. Include HLD when introducing new abstractions, changing data flows, or when team members need the bigger picture.
- **Components & Responsibility**: For small changes, can be condensed into a single section if only 1-2 components are affected.
- **Observability**: For internal tooling with minimal operational burden, can be brief. Always include for user-facing or critical systems.
- **Evaluation & Testing**: For very low-risk changes, can be condensed. Always include detailed testing for systems affecting customers or critical operations.

---

## 1. Background & Context

**Purpose**: Set the scene. Why does this work matter? What's the current state?

Write 2-3 paragraphs covering:
- What problem space or business context we're in
- Current state / how things work today
- Key stakeholders or teams affected
- Links to related issues, OKRs, or strategic initiatives

**Example:**
```
We've seen a 40% increase in API request volume over the past quarter, 
pushing our current request router to 75% capacity. The router uses 
a single-threaded event loop that is becoming a bottleneck. This impacts 
latency for our ML pipeline team (primary user) and limits our ability 
to onboard new services. We're targeting 2x scale capacity by Q4.
```

---

## 2. Problem Statement

**Purpose**: Name the core problem(s) you're solving. Be specific.

State:
- What pain point or limitation exists today
- How it manifests (symptoms, metrics, user impact)
- Why current approaches fall short
- Business impact (revenue, developer velocity, customer experience, etc.)

**Example:**
```
**Problem:** Request routing latency increases linearly with volume because 
the current router processes requests sequentially.

**Symptoms:**
- P99 latency 200ms → 600ms as load increases
- ML team's batch jobs now take 30% longer
- New services can't be added without affecting existing ones

**Why it matters:** Blocks feature launches and increases operational cost.
```

---

## 3. Requirements

Separate functional and non-functional requirements. Be concrete — use metrics and acceptance criteria.

### Functional Requirements (What must the system do?)

- FR-1: Router must distribute requests across N backend services
- FR-2: Support weighted load balancing (e.g., 70% service A, 30% service B)
- FR-3: Route requests based on service tags (e.g., `service: ml-pipeline`)
- FR-4: Handle graceful degradation if a backend is unavailable

### Non-Functional Requirements (Quality, scale, performance)

- NFR-1: P99 latency must not exceed 50ms (current: 200ms for baseline load)
- NFR-2: Support 10K requests/sec (2x current peak)
- NFR-3: 99.95% availability (no unplanned downtime)
- NFR-4: Configuration changes must apply within 10 seconds
- NFR-5: Cost per request ≤ current spend (no infrastructure scaling if possible)

---

## 4. Goals & Success Criteria

**Product Goals** (what we're trying to achieve):
- Reduce request latency by 80% to unblock downstream services
- Enable N new services to be onboarded without performance degradation
- Improve developer experience for routing configuration

**Success Criteria** (measurable):
- [ ] P99 latency < 50ms at 2x current load
- [ ] Router handles 10K req/sec with <2% error rate
- [ ] Onboarding a new service takes < 1 hour (currently 4 hours)
- [ ] 99.95% uptime achieved across all services

**Anti-goals** (what we're NOT doing):
- Not building a full service mesh (out of scope, different tool for that)
- Not rewriting all service clients (routing is transparent)

---

## 5. Scope

### In-Scope

- Request routing layer redesign
- Load balancing algorithm and weighted distribution
- Configuration management for routes (YAML-based)
- Monitoring and alerting for router health
- Graceful degradation when backends are down
- Metrics/logging for debugging

### Out-of-Scope

- Service mesh (Istio, Linkerd, etc.) — separate initiative
- Circuit breaker logic (owned by individual services)
- TLS/encryption upgrades (handled upstream)
- Database query routing (different layer)

**Rationale**: Keeping scope tight allows us to deliver in one quarter and avoid dependencies on other teams.

---

## 6. High-Level Design (HLD)

### Architecture Overview

Provide a visual and textual description of the system's major components and how they interact.

```
┌─────────────┐
│  Clients    │
└──────┬──────┘
       │ (HTTP requests)
       ▼
┌──────────────────────────────┐
│  Request Router (NEW)        │
│  ┌──────────────────────┐    │
│  │ Load Balancer        │    │
│  │ (weighted round-rob) │    │
│  └──────┬───────────────┘    │
└─────────┼───────────────────┘
          │
    ┌─────┴──────┬──────────┐
    ▼            ▼          ▼
┌────────┐  ┌────────┐  ┌────────┐
│Service │  │Service │  │Service │
│   A    │  │   B    │  │   C    │
└────────┘  └────────┘  └────────┘

┌────────────────────────────────┐
│ Config Store (Redis/etcd)      │
│ - Route definitions            │
│ - Weight assignments           │
└────────────────────────────────┘
```

### Data Flow

1. Client sends HTTP request to `/api/resource`
2. Router consults Config Store for route rules
3. Router applies load balancing algorithm (weighted round-robin)
4. Request forwarded to selected backend service
5. Response returned to client

### Key Design Decisions (Rationale)

| Decision | Options Considered | Why This One | Tradeoff |
|----------|-------------------|--------------|----------|
| **Language** | Go vs Rust vs Python | Go — fast startup, good stdlib, team familiar | Slightly larger binary than Rust |
| **Config Source** | Redis vs etcd vs Consul | etcd — strong consistency, built-in watch API | Adds external dependency |
| **Load Balancing** | Round-robin vs Least Conn vs Hash | Weighted round-robin — simple, fair, predictable | May not optimize for uneven request sizes |

---

## 7. Components & Responsibility Boundaries

### Core Components

#### 1. **Request Router**
- **Responsibility**: Accept incoming requests, consult routing rules, forward to backend
- **Interfaces**:
  - Inbound: HTTP on port 8080
  - Outbound: HTTP to backend services
  - Config watch: Watches etcd for changes
- **Dependencies**: Config Store, backend services
- **Scalability**: Stateless; can run N replicas behind a load balancer

#### 2. **Config Store (etcd)**
- **Responsibility**: Store and distribute routing rules
- **Interfaces**:
  - Router watches `/config/routes` for changes
  - Admin API to update routes
- **Dependencies**: None (external service)
- **Persistence**: Replicated key-value store

#### 3. **Admin API** (future: can be simple CLI for now)
- **Responsibility**: Update routing configuration
- **Interfaces**: REST API (POST /config/routes)
- **Dependencies**: Config Store

#### 4. **Metrics & Observability**
- **Responsibility**: Emit metrics, handle logging, tracing
- **Interfaces**: Prometheus /metrics endpoint
- **Dependencies**: Prometheus, logging infrastructure

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────┐
│  Request Router (Go service)                    │
│  ┌────────────────────────────────────────────┐ │
│  │ HTTP Handler                               │ │
│  │  └─> Route Lookup                         │ │
│  │       └─> Load Balancer Selection         │ │
│  │            └─> Proxy to Backend           │ │
│  │                 └─> Record Metrics        │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │ Config Watcher                             │ │
│  │  └─> Watch etcd for changes               │ │
│  │       └─> Update in-memory routing table  │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
              │                    │
              │ watches            │ reads config
              ▼                    │
        ┌──────────┐               │
        │   etcd   │◄──────────────┘
        └──────────┘
              ▲
              │ writes config
              │
        ┌──────────────┐
        │ Admin CLI    │
        └──────────────┘
```

---

## 8. Low-Level Design (LLD)

### 8.1 Request Routing Algorithm

**Weighted Round-Robin Implementation**

```go
type Router struct {
  backends []Backend
  weights  []int      // weight for each backend
  current  int        // current index
  mu       sync.RWMutex
}

func (r *Router) SelectBackend() Backend {
  r.mu.RLock()
  defer r.mu.RUnlock()
  
  r.current = (r.current + 1) % len(r.backends)
  return r.backends[r.current]
}

// For weighted round-robin:
// If backend A has weight 7 and B has weight 3,
// return A 7 times, then B 3 times, repeat.
```

**Config Format (YAML)**

```yaml
routes:
  /api/ml-pipeline:
    backends:
      - service: ml-processor-v1
        weight: 70
      - service: ml-processor-v2
        weight: 30
    timeout: 5s
    
  /api/search:
    backends:
      - service: search-engine
        weight: 100
    timeout: 2s
```

### 8.2 Config Watcher Loop

```go
func (r *Router) WatchConfig(ctx context.Context, etcdClient *clientv3.Client) {
  watchChan := etcdClient.Watch(ctx, "/config/routes", clientv3.WithPrefix())
  
  for wresp := range watchChan {
    for _, ev := range wresp.Events {
      // Parse new config
      newConfig := parseConfig(ev.Kv.Value)
      
      // Swap routing table (atomic)
      r.mu.Lock()
      r.routingTable = newConfig
      r.mu.Unlock()
      
      log.Infof("Updated routing config: %+v", newConfig)
    }
  }
}
```

### 8.3 Request Handling Flow

```
1. Accept request
   ├─> Extract service name from URL path
   │   (e.g., /api/ml-pipeline → ml-pipeline)
   │
2. Route Lookup
   ├─> Consult routing table for ml-pipeline
   ├─> If not found → return 404
   │
3. Select Backend
   ├─> Apply weighted round-robin
   ├─> Get Backend {service, host, port}
   │
4. Proxy Request
   ├─> Forward HTTP request to backend host:port
   ├─> Set X-Forwarded-For header
   ├─> Forward response back to client
   │
5. Record Metrics
   ├─> request_count{service, backend, status}
   ├─> request_latency_ms{service, backend}
   ├─> error_count{service, backend, error_type}
```

### 8.4 Data Structures

```go
type Backend struct {
  Service string    // e.g., "ml-processor-v1"
  Host    string    // e.g., "10.0.1.5"
  Port    int       // e.g., 8080
  Weight  int       // relative weight
  Status  string    // "healthy", "degraded", "down"
}

type Route struct {
  Path     string       // e.g., "/api/ml-pipeline"
  Backends []Backend
  Timeout  time.Duration
  Retry    int          // retry count on failure
}

type RoutingTable struct {
  routes map[string]*Route
}
```

### 8.5 Concurrency & Thread Safety

- **Config updates**: Use RWMutex to allow concurrent reads with exclusive writes
- **Metrics recording**: Use atomic operations or a metrics buffer to avoid lock contention
- **Request forwarding**: Stateless; safe to run in goroutine pool

---

## 9. Design Choices & Tradeoffs

| Choice | Alternative | Why We Chose This | Tradeoff |
|--------|-------------|-------------------|----------|
| **Weighted round-robin** | Least connections, least CPU | Simple, fair, predictable latency distribution | May not be optimal for requests with varying sizes |
| **etcd for config** | File-based, Consul, ZooKeeper | Built-in watch API, strong consistency, cloud-native | Adds operational complexity (etcd cluster) |
| **In-memory routing table** | Lookup from etcd on every request | Avoids latency of external lookups, faster routing | Need active watcher to keep table in sync |
| **Separate Admin API** | Config passed as startup flags | Dynamic updates without restart, better UX | More code to maintain |
| **Prometheus metrics** | Logging only | Query-able, alerting, dashboarding | Adds dependency on Prometheus scraper |

**Risk Mitigation:**
- If etcd becomes bottleneck, cache config in-memory with periodic refresh
- If latency is still high, consider memory-mapped files for config hot-reload
- If weight distribution is unfair, switch to least-connections algorithm

---

## 10. Observability

### Metrics (Prometheus)

```
# Routing metrics
router_request_total{service="ml-pipeline", backend="ml-processor-v1", status="200"} 15430
router_request_latency_ms{service="ml-pipeline", backend="ml-processor-v1", quantile="p99"} 45

# Config sync metrics
router_config_updates_total 127
router_config_update_latency_ms 250

# Backend health
router_backend_health{service="ml-processor-v1", status="healthy"} 1
router_backend_health{service="ml-processor-v2", status="degraded"} 1
```

### Logging Strategy

**Info level:**
```
[INFO] Config update: /api/ml-pipeline → [ml-processor-v1(70%), ml-processor-v2(30%)]
[INFO] Route lookup: ml-pipeline → selected ml-processor-v1 (1240 requests)
```

**Warn level:**
```
[WARN] Backend unavailable: ml-processor-v1 (connection refused)
[WARN] Config update failed: context deadline exceeded
```

**Debug level (disabled in prod):**
```
[DEBUG] Request: GET /api/ml-pipeline → 10.0.1.5:8080
[DEBUG] Config watcher event: PUT /config/routes/ml-pipeline
```

### Tracing

Use distributed tracing (Jaeger, Datadog) to track request flow:
```
Router → Service A
  └─ Span: routing_decision (1ms)
  └─ Span: proxy_request (45ms)
  └─ Span: service_a_processing (120ms)
```

### Dashboards

**Router Health Dashboard:**
- Request rate (RPS) by service
- P50, P99 latency by backend
- Error rate by service
- Config update frequency and latency
- Backend availability (uptime %)

**Alerting Rules:**
```
alert: RouterHighLatency
expr: router_request_latency_ms{quantile="p99"} > 100
for: 5m

alert: BackendDownAllReplicas
expr: count(router_backend_health == 1) == 0
for: 1m
```

---

## 11. Evaluation & Testing Strategy

**Purpose**: Define exactly how this design will be validated and what success looks like.

### Test Scenarios

List 3-5 concrete test scenarios that will verify the design works:

| Test Scenario | Input/Setup | Expected Result | Exit Criteria |
|---------------|-----------|-----------------|----------------|
| **Happy Path** | Normal load, all systems healthy | Requests complete in <50ms p99, 100% success rate | P99 < 50ms, 0% errors |
| **Degraded Replica** | Kill one backend replica | Traffic reroutes to healthy replica, no user impact | <1s failover, <2% error spike |
| **Load Surge** | 2x normal load for 10min | System handles gracefully, latency increases <20% | P99 < 60ms under 2x load |
| **Config Update** | Modify routing rules while traffic flows | New rules take effect without restarting, no dropped requests | <10s config propagation, 0 dropped requests |
| **Chaos: Network Partition** | Simulate network lag to one backend | Requests timeout and retry to healthy backend | <5s detection, <5 retries per request |

### Performance Benchmarks

Baseline metrics to measure before and after deployment:

- **Latency**: P50, P99, P99.9 response times (ms)
- **Throughput**: Requests per second (RPS) at 95% success rate
- **Resource Usage**: CPU (%), Memory (GB), Network (Mbps) per component
- **Error Rate**: % of failed requests by error type
- **System Cost**: Infrastructure cost per RPS

### Go/No-Go Criteria for Each Phase

**Phase 1 (Canary 5% traffic):**
- [ ] P99 latency < target + 10ms
- [ ] Error rate < 0.1%
- [ ] No customer complaints
- [ ] All monitoring alerts firing correctly
- **Decision**: If all pass → proceed to Phase 2; else rollback and fix

**Phase 2 (Staged 50% traffic):**
- [ ] P99 latency within 5% of Phase 1
- [ ] Error rate < 0.05%
- [ ] Cost per RPS unchanged or improved
- [ ] Replicas staying in sync (<1s lag)
- **Decision**: If all pass → proceed to Phase 3; else scale down and investigate

**Phase 3 (100% traffic):**
- [ ] 24-hour run with <0.01% error rate
- [ ] p99 latency stable across peak hours
- [ ] Team confidence high (incident post-mortem not needed)
- [ ] Runbooks tested and validated
- **Decision**: Mark as stable; schedule deprecation of old system

### Observability Exit Criteria

Ensure these observability pieces are deployed before going live:

- [ ] All metrics emitting correctly (check Prometheus scrape)
- [ ] Dashboards rendering with expected data
- [ ] Alert rules firing on test triggers (test alert → slack/pagerduty)
- [ ] Logs being ingested and searchable
- [ ] Distributed tracing showing end-to-end request flow
- [ ] On-call runbook validated with team

---

## 12. Production Readiness

### Deployment Strategy

**Phase 1 (Week 1-2): Canary Deployment**
- Deploy router in canary environment
- Route 5% of ml-pipeline traffic to new router
- Monitor error rate, latency, resource usage
- If good, increase to 20%

**Phase 2 (Week 3): Staged Rollout**
- Deploy to all services with 100% traffic
- Keep old router running as fallback
- Monitor for 1 week

**Phase 3 (Week 4+): Deprecate Old Router**
- Remove old router once new router is stable

### Rollback Plan

If new router causes P99 latency to exceed 150ms for >5 minutes:
1. Alert on-call engineer
2. Automatically switch traffic back to old router via load balancer
3. Start incident investigation
4. Manual fix and re-deploy

### Dependencies & External Factors

| Dependency | Owner | Risk | Mitigation |
|-----------|-------|------|-----------|
| etcd cluster availability | Platform team | Config updates blocked if etcd is down | Local cache + periodic refresh |
| Network latency to backends | Infrastructure | High p99 latency | Collocate services in same datacenter |
| Backend service capacity | Service owners | Routing to overloaded service | Work with owners on capacity planning |

### Infrastructure Requirements

**Compute:**
- 3 router replicas (HA)
- 2 CPU, 1 GB memory per instance
- Runs in Kubernetes (single pod per node for load distribution)

**Storage:**
- etcd cluster: 3 nodes (quorum)
- 10 GB storage sufficient for routing config

**Network:**
- Port 8080 (incoming requests)
- Port 2379 (etcd client)
- Port 9090 (Prometheus metrics)

### Testing Strategy

**Unit Tests:**
- Weighted round-robin algorithm correctness
- Route lookup performance (< 1μs)
- Config parsing edge cases

**Integration Tests:**
- Config watcher syncs with etcd
- Routing table updates without downtime
- Request forwarding to multiple backends

**Load Tests:**
- 10K req/sec sustained
- P99 latency < 50ms
- Resource usage stays within budget

**Chaos Tests:**
- Kill backend service → route to healthy replica
- etcd unavailable → use cached config
- Network partition → graceful degradation

### Monitoring & Alerting Setup

**Pre-deployment checklist:**
- [ ] Prometheus scrape job configured
- [ ] Grafana dashboards created
- [ ] Alert rules loaded (PagerDuty integration)
- [ ] Logs being shipped to centralized logging system
- [ ] Distributed tracing enabled

**Health Checks:**
- `/health` endpoint returns 200 if router and etcd are up
- `/ready` endpoint returns 200 if routing table is initialized

### Security Considerations

**Potential Issues:**
- Unauthorized route modifications (config injection)
- DDoS via request flooding
- Man-in-the-middle on backend connections

**Mitigations:**
- etcd access control (TLS, basic auth)
- Rate limiting on Admin API (token-based, per-IP)
- TLS for communication with backends (setup separately)

### Documentation & Handoff

**What needs to be documented:**
- How to update routing configuration (runbook)
- How to add a new backend service
- How to interpret metrics and dashboards
- Incident response playbook (what to do if router is down)
- Architecture deep-dive (for onboarding new team members)

---

## Appendix: Examples & Conventions

### Config Update Runbook

```
# To add a new backend for service A:

etcdctl put /config/routes/api-service-a \
  '{"backends": [{"service":"svc-a-v1", "weight":100}]}'

# To drain traffic from a service (migration):
# Step 1: Reduce weight to 20%
etcdctl put /config/routes/api-service-a \
  '{"backends": [{"service":"svc-a-v1", "weight":20}, {"service":"svc-a-v2", "weight":80}]}'

# Monitor for 10 min, then increase if healthy
etcdctl put /config/routes/api-service-a \
  '{"backends": [{"service":"svc-a-v1", "weight":0}, {"service":"svc-a-v2", "weight":100}]}'
```

### Glossary

- **P99 latency**: 99th percentile response time (99% of requests faster than this)
- **RPS**: Requests per second
- **Canary deployment**: Route small % of traffic to new version first
- **Graceful degradation**: Service degrades but doesn't fail completely
- **In-memory cache**: Data stored in process RAM for fast lookup

---

## Next Steps

1. **Design Review**: Present to architecture team; gather feedback
2. **Implementation**: Break into sprints (config watcher, routing algorithm, testing)
3. **Testing**: Run load tests, chaos tests, security review
4. **Deployment**: Follow phased rollout plan with monitoring
5. **Post-mortem**: Review learnings 2 weeks after production deployment

