# Design — M19: an experiment is a change to its parent

**Plan:** [../14-experiments-as-deltas.md](../14-experiments-as-deltas.md) ·
**Status:** design · **Owner:** unassigned · **Supersedes:** the Jinja template pack ·
**Subsumes:** technique registry, `applied`/`candidate` split, template→labpilot coupling

---

## 1. Problem

Two mechanisms produce training code today, and both are wrong for different
reasons.

**Templates don't scale.** Seven Jinja templates cover a space that varies per
competition, per problem type, per dataset quirk. Measured on rogii: the registry
declares 12 executable techniques and **7 resolve `not_applicable` purely because
nobody wrote a gate**. The registry exists only to feed template gates — its own
docstring says so — so the whole `applied` vs `candidate` distinction is an
artifact of the mechanism rather than anything about research.

They are the third instance of a pattern rejected twice already: a curated set
answering an open-world question, after `KNOWN_TECHNIQUES` and the proposed
package allowlist.

**Whole-file regeneration is wasteful and lossy.** The parent's code goes into
the prompt — up to 120k chars, measured at **~3,376 tokens or 46% of it** — and
the contract then says *"always emit full overridden train.py"*. The system pays
to send the parent, pays again to receive a near-copy, and every regeneration is
an opportunity to silently drop something that worked. The skill's own
instruction to "keep what worked" fights its mechanism.

### Why this matters beyond cost

The partitioned template encodes `partition_suffix_holdout`, `_driver_columns()`
and three leakage gates — validation discipline fixed once, after a real leakage
bug. Under whole-file regeneration that discipline is re-derived every run and
can be lost with no metric showing it: **a leaky score looks better, not worse.**
That risk is the only reason templates still looked load-bearing.

---

## 2. The change

**An experiment is a diff against its parent, not a fresh file.**

The validation protocol then survives **whenever the delta does not touch it** —
it lives in the parent, and nothing regenerates it. That is a much better
starting position than whole-file regeneration, where the discipline is
re-derived every run.

It is *not* a guarantee, and this design should not claim one. A delta **can**
reach into `_driver_columns` or the holdout construction; the spike simply did
not, twice, on one kind of request. n=2 is a strong argument against building an
edit format. It is not a proof of safety — which is precisely why the flagging
work in §5 and exit criterion 3 are load-bearing rather than optional.

This also aligns the code artifact with the model the rest of the system already
uses. Evidence cards compare `parent_cv` to `treatment_cv`; the experiment graph
is parent → child; `technique_attribution` credits the difference. Only the code
was a fresh object each time. After this it is `parent + change`.

```
competition start   ──▶  baseline      whole file, from the dataset profile
                              │
hypothesis 1        ──▶       ├──▶ delta ──▶ child A
hypothesis 2        ──▶       ├──▶ delta ──▶ child B
hypothesis 3        ──▶  (on A) ──▶ delta ──▶ child C
```

---

## 3. Buy the edit machinery, build the research parts

An earlier draft of this design specified an anchored-edit format, uniqueness
rules, an apply algorithm and a failure ladder — roughly 60% of the document.
**A spike deleted all of it.**

`aider` already does this, tuned over years against published edit-format
benchmarks. Measured 2026-08-07 on rogii's real `train.py` (331 lines), asking
for the `SWA`-style change that produced this system's only genuine improvement:

| | nemotron-super-120b | **nemotron-ultra-550b** |
|---|---|---|
| Delta | +55 / −8 | **+24 / −8** |
| Self-doubt comments left in code | 6 | **0** |
| Validation half correct | yes | yes |
| Test half correct | no — *"Let me correct"* | **yes** |
| `_driver_columns` / `_add_partition_features` / `_known_rows` / `partition_suffix_holdout` touched | **0 lines** | **0 lines** |
| Syntax valid | yes | yes |
| Cost | $0.007 | $0.02 |

The strong model's delta is what a competent human writes: a seed loop over
validation, a seed loop over the final fit, and the submission mapping updated to
the averaged predictions. Nothing else moved.

**Both runs left the leakage discipline untouched** — §2's core requirement, met
without labpilot writing a line of edit-format code.

**The variable is model quality, not mechanism** — which is what
[M10](04-llm-tiering.md) exists to manage. But see §4: passing `--model` only
transfers the *selection*. Everything else M10 does is bypassed unless aider is
pointed at fitroute rather than at the provider.

So this design specifies an **adapter**, not an edit format.

---

## 4. Architecture

```
      plan + hypothesis + parent code
                  │
                  ▼
        copy workspace to scratch          ← never edit the workspace
                  │
                  ▼
      aider --model <codegen role>         ← borrowed edit machinery
                  │
                  ▼
        diff scratch against parent
                  │
                  ▼
             CodeProposal                  ← existing typed contract
                  │
                  ▼
    existing validation (ast.parse, path allowlist)
                  │
                  ▼
              apply_proposal               ← existing apply path
```

#### The proposal carries whole files, not a patch

`CodeFileSpec` is `path` + `content` + `action="write"`, and `apply_proposal`
writes `spec.content` after `ast.parse`. A unified diff or SEARCH/REPLACE block
does not fit it, and **must not be made to**.

So the adapter reads the *resulting files* out of the scratch copy into
`CodeProposal.files[].content`. The diff is computed too, but only as **evidence**
— it feeds validation-region flagging and the delta→card linkage in §5. It is
never the thing that gets applied.

Stated explicitly because the obvious next move is to extend `CodeProposal` with
patch operations, which would reinvent the apply path this design exists to
delete. The delta is how the *model* thinks; whole files are how the *system*
applies. Those are allowed to differ.

Running aider in a **copy** is what preserves three properties the direct-edit
alternative would lose:

* **propose-then-apply.** A bad proposal is rejected before it touches the
  workspace. Direct file edits would mean discovering damage afterwards.
* **never edit the workspace.** The standing rule the repair machinery depends
  on.
* **provenance.** The adapter records `generated_by`, model, provider and
  attempt count into `agent_invocations` like every other agent, so M14's
  instrument keeps working on the most important call in the system.

**Scope of the copy:** editable roots only — `pipeline/`, plus `config.yaml` and
the profile summary aider needs for context. **Not** `data/`, which is the bulk
of a competition tree and which generated code reads by path at run time, not at
edit time. Copying it would make every experiment pay a multi-GB copy for no
benefit.

### The proxy — and why it is a prerequisite, not a nicety

An earlier draft claimed that passing `--model` kept "routing, budget, failover
and provenance in one place". **That is true only of routing.** aider makes its
own HTTP calls, so:

| M10 capability | With `--model` alone |
|---|---|
| Role → provider selection | works — we choose it |
| Budget ledger (`record`) | **bypassed** — aider calls the provider; we never see the tokens |
| Rate limiting (`availability`) | **bypassed** — aider makes several calls per message; the ledger counts one |
| Runtime failover (`cool_down` + re-select) | **bypassed** — it lives in `RoleBoundClient.complete`, which aider never enters |
| Response cache | bypassed — aider has its own |
| `structured_output` preflight | **actively wrong** — aider needs good *editing*, not JSON mode; the precondition would exclude models that are fine for it |
| Provenance | partial — we know what we asked for, not what aider retried |

Shipping the adapter that way would quietly opt the system's most important LLM
call out of M10 — a regression dressed as a feature, and the kind that is hard to
see later.

The fix is to make aider a *client* of fitroute rather than a peer:

```
aider --openai-api-base http://localhost:PORT
          │
          ▼
   fitroute proxy  →  select_route · ledger · failover · cache · provenance
          │
          ▼
   OpenRouter / Groq / ollama
```

Then **every** aider call — including its internal retries and repo-map lookups —
goes through M10. Budget accounting becomes complete rather than an estimate
scraped from stdout; today's failover applies to codegen; provenance records what
happened rather than what was intended.

`LLMGateway` already does all of this as a Python API, so the proxy is a thin
OpenAI-compatible HTTP shell over it, and it stays inside `fitroute`, which keeps
the package extractable.

##### The `structured_output` carve-out is a deliberate catalog change

Today `MANDATORY_CAPS` unions `structured_output` into **every** role and cannot
be relaxed (PR #98) — that precondition is what made deleting the rule engines
safe in M14 phase 3. aider needs a good *editor*, not JSON mode, so the current
mandate would exclude models that are ideal for it.

Step 0 must therefore change the catalog, not rely on a proxy side-effect:

* **`codegen` opts out** of `structured_output` — its output is a file edit,
  validated by `ast.parse` and by the run itself, not by JSON parsing.
* **Every other role keeps the mandate.** `default` (the Conductor policy),
  `reasoning` and `summarize` all parse JSON, and relaxing them would reopen the
  prose-reply failure M14 phase 3 removed the net for.

Spelled out because the two wrong implementations are both plausible: keep the
global mandate and silently exclude good editors, or weaken `MANDATORY_CAPS`
globally and quietly undo phase 3.

#### Roles must survive the HTTP boundary

The proxy sees requests, not roles — so every aider call would look identical and
per-role limits, `requires`, `on_exhaustion` and `requires_strong` would collapse
into one bucket. That is M10's founding property lost at the first hop.

Encode the role in the model name:

```
aider --model labpilot/codegen --openai-api-base http://localhost:PORT
```

The proxy reads `labpilot/codegen`, resolves it through
`select_route("codegen", …)`, and substitutes the real provider and model. Call
sites still name a role, never a vendor.

#### Rate limiting through a proxy

Enforcement changes shape, and improves. Today `RoleBoundClient.complete` handles
a limit by **sleeping** until the window reopens. A proxy cannot: aider holds an
HTTP connection with its own timeout, and stalling it invites a client-side retry
that makes the pressure worse.

Server-side, in order:

1. **Route around it** — pick another eligible provider. This is the `cool_down`
   + re-select failover already in `gateway.py`, and it beats waiting: the
   request succeeds rather than stalling.
2. **Degrade**, for roles whose `on_exhaustion` permits it.
3. **`429` with `Retry-After`** only when nothing is available — the correct
   OpenAI-compatible answer, which litellm (aider's client) already honours.

Accounting also gets *more* accurate, not less: every call aider makes, including
internal retries and repo-map lookups, arrives as a request the ledger records.

**Useful beyond aider:** any external tool reached for later gets M10 for free.

#### aider is a local process, and always will be

Worth stating because it bounds what a backend can ever buy. `aider` is a CLI
that reads and writes files on local disk — verified in the spike, which ran it
via `uvx` against a local `train.py`. It has no server component. Its only
network traffic is *outbound* calls to the model, which is exactly why pointing
it at the proxy works.

```
labpilot          (local)  spawns ↓
  aider           (local)  edits files in the workspace copy
    │ HTTP
    ▼
  fitroute proxy  (local now, relocatable)
    │ HTTP
    ▼
  OpenRouter / Groq / ollama          (remote)
```

So relocating fitroute moves **LLM routing only**:

| Component | Can move to a backend? | Why |
|---|---|---|
| fitroute routing / budget / failover | **yes** — the proxy is the seam | it only needs the request |
| aider | no | must be where the files are |
| labpilot's own agents | no | orchestration is local by nature |
| training runs | no | need the dataset on disk |

That split is worth naming: a hosted fitroute gives shared budget, central policy
and a shared cache across machines. It does **not** make the system remote —
every machine still runs its own labpilot, its own aider and its own training.
Wanting all of that remote is a hosted *workspace*, a different and much larger
architecture that the proxy does not set up.

#### One chokepoint, two transports

A tempting generalisation is to route *all* labpilot LLM traffic through the
proxy, so relocating it to a shared backend later is a URL change. The goal is
right; forcing every internal call onto a socket to get it is not.

What actually delivers "change the URL and nothing breaks" is **one place where
routing decisions are made** — and `LLMGateway` already is that place. The proxy
is one adapter onto it, for clients that can only speak HTTP:

```
micro agents ──── Python ────┐
                             ├──▶ LLMGateway ──▶ select_route · ledger · failover
aider ─── HTTP ─── proxy ────┘                          │
                                                         ▼
                                              providers, or a remote fitroute
```

Relocation is then a `ProviderSpec` change — a `remote` kind pointing at a hosted
fitroute — not a rewrite, and every call site is already insulated because none
of them names a vendor.

Making every internal call HTTP would buy uniformity at the cost of: a process
lifecycle to manage (who starts the proxy, what happens on port conflict), a new
hard dependency for a path that currently cannot fail that way, and stack traces
that stop at a socket. Worth revisiting if a shared backend becomes real —
recorded here so the option is not lost.

### The seam

```python
class CodeAgent(Protocol):
    def propose(self, ctx: CodegenContext, parent: Path | None) -> CodeProposal: ...
```

Two implementations: `AiderAgent` and the existing whole-file `CodeEngineerAgent`
(baselines, and any workspace without aider). One protocol, chosen by config —
the same shape as `fitroute`, where the router was put behind a boundary so it
could be swapped without a rewrite.

---

## 5. What labpilot still has to build

The parts no coding agent knows about, because they are about *research*, not
code:

| Piece | Why only labpilot can do it |
|---|---|
| **Validation-region flagging** | Detect when a delta touches `partition_suffix_holdout`, `_driver_columns`, or the holdout construction, and record it on the evidence card |
| **Delta → evidence linkage** | The card already compares parent and treatment; it should carry *what changed*, so `technique_attribution` can be read against the actual diff |
| **Baseline vs delta routing** | No parent ⇒ whole file; parent ⇒ delta. The experiment graph already knows which |
| **Provenance capture** | Translate aider's result into an `agent_invocations` row |
| **Failure classification** | `aider_no_edit`, `aider_syntax_fail` alongside `json_shape`, so the rate is measurable |

---

## 6. Risks

**The one that would hurt.** A delta can still damage validation logic — running
in a copy makes it *reviewable*, not impossible. Mitigation is detection, not
prohibition: flag a delta whose change falls in the validation region and record
it on the card. A hypothesis *about* validation is legitimate; one that changes
validation while claiming to test a feature is a false result.

**Cost per experiment rises.** ~$0.02 against near-zero for a template render —
about $1.20 across a 60-step campaign. Acceptable, but it is real, and it is the
first time the research loop's cost scales with experiment count rather than
prompt count.

**A new runtime dependency.** `uvx --from aider-chat aider` needs no install, so
it is containable, but it is an external surface with its own CLI stability. The
`CodeAgent` protocol is what keeps this reversible: if aider becomes a problem,
the whole-file implementation is still there.

**Drift over a long chain.** Twenty deltas deep, the code is far from the
baseline and no single review saw the whole thing. The experiment graph records
the chain; a periodic whole-file re-emission is a legibility checkpoint, not a
correctness measure.

---

## 7. Testing

The failure modes are all "it looked applied and was not", so:

1. **The workspace is never touched on a rejected proposal.** Force a syntax
   error in the scratch copy; assert the workspace file is byte-identical.
2. **Validation logic survives a feature-adding delta**, byte-identical. This is
   the property that lets templates go, and the spike already demonstrates it
   twice.
3. **A delta touching validation is flagged** on the card.
4. **Provenance is recorded** — model and `generated_by` reach
   `agent_invocations`, not just the log.
5. **No aider ⇒ whole-file path**, so a workspace without it still works.
6. **A no-op aider run is a failure**, not a silent success — the same lesson as
   the stale `metrics.json` guard, which asked "is there a file?" instead of
   "did this run write one?".

---

## 8. Rollout

Behind `codegen.strategy: whole_file | delta`, defaulting to `whole_file`. Both
paths coexist while the failure rate is measured on real campaigns — the standard
M14 phase 2b set for exactly this kind of decision, and why 2b shipped
default-off with a number attached rather than a guess.

0. **fitroute OpenAI-compatible proxy.** Prerequisite, not optional — without
   it steps 1-3 opt codegen out of M10 (see §4). Independently useful and
   independently testable. Includes the `codegen` carve-out from
   `MANDATORY_CAPS`.

   **Lifecycle: an ephemeral child of the campaign,** started when
   `run_until_stop` begins and stopped in its `finally`, on an OS-assigned port
   passed to aider. Not a daemon: a long-running server needs its own
   supervision, port conflicts between concurrent campaigns become a real
   failure mode, and an orphan outlives the run that owns its budget ledger.
   Scoping it to the campaign means it cannot outlive what it is accounting for.
   A shared daemon is the right shape only once fitroute is genuinely hosted,
   which is the same trigger as the full-HTTP option below.
1. `CodeAgent` protocol + `AiderAgent` + copy/diff/propose, pointed at the proxy.
   Nothing calls it.
2. Opt-in via config; measure `aider_no_edit` and `aider_syntax_fail`.
3. Flip the default when the rate justifies it.
4. Delete templates **in the same change** that makes delta the default — never
   before. The discipline M14 phase 3 established, where a removal and the
   precondition that makes it safe must ship together.

**Open question for step 1.** The spike used a free OpenRouter model at $0.02.
Whether the `codegen` role should point somewhere stronger for delta work is a
routing decision, not a design one — and `agent_invocations` will answer it with
a rate rather than an opinion.
