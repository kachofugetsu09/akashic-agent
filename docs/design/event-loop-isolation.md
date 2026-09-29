# Event-loop execution boundaries

Related issues: #827 (stack), #828 (storage), #829 (interest), #836 (materials).

Long history reads use a private read-only SQLite transaction and connection. Nested readers of the same MessageLog in that synchronous call share its read snapshot; they do not acquire the writer's connection or decoded-message cache. Reads inside an existing write transaction still see that transaction's uncommitted rows. Owner transactions remain synchronous and atomic, and listeners wake only after commit.

MessageLog uses file-backed WAL mode so a pinned read does not delay a writer's commit. This changes the runtime journal mode, not the schema. Backups must use SQLite backup or include the SQLite sidecars; copying only the live main database file is not a snapshot. Existing databases keep their schema and data; an unsupported journal mode is rejected explicitly. Short synchronous writes can still wait on SQLite file-level contention; this change does not claim that all storage I/O is asynchronous.

Reply preparation captures the source head and full message head before awaiting history. Async warmup decodes only that fixed prefix, drains the worker on cancellation, and installs the existing incremental cache only if the original connection's data version is unchanged. An external edit during the read causes a retry. Later appends are read as a tail; external edits after warmup invalidate the cache on the next read.

Interest scoring keeps model selection and candidate embedding in the original async owner. Historical sample and prototype construction run in a drained worker without changing the formula, sample order or cutoff.

Independent context material owners prepare in a TaskGroup. Their results merge in the fixed source order with the existing output sorting and conflict checks. An ordinary failure drains all started siblings and is then reported in frozen source order. Cancellation cancels and drains siblings. Owner scopes close before either reaches the caller. Ordered transform events and the unique summary reducer remain sequential.

```text
fixed input / head
       │
       ├─ independent read or compute job
       │       └─ await actual completion, including on cancellation
       ▼
ordered merge / state-dependent transition
       ▼
durable commit → notify → next dependent operation
```

Validation is tied to real MessageLog and CompositionRoot boundaries. The new regressions fail before their fixes; the stack retains the existing per-graph publication and ordered tool-result tests.

Akasha keeps one notification event and worker per routed graph. Notifications coalesce while that graph is busy. A graph access lock orders startup, consumption and recall; the MessageMemory lock still guards each transition and publication. A starting graph reserves its embedding space before awaiting, so a concurrent graph cannot silently bind a changed space. Startup schedules these consumers without waiting for the default graph's backlog. Explicit rebuild closes graph admission, drains existing users, then closes writers and rebuilds; admission reopens even on failure. Whole-runtime shutdown cancels and drains the owned worker group before closing memories.
