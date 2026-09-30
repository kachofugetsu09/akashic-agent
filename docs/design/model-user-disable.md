# Explicit model opt-outs and catalog availability

The Models registry owns two distinct facts in its existing chat/embedding rows:

- `enabled` is the existing effective catalog availability
- `user_disabled` records an explicit user opt-out, defaulting to false

A single `enabled` bit cannot distinguish a model temporarily missing from the
provider catalog from one the user deliberately switched off. Sync must keep
refreshing discovery-owned capabilities, and a disappeared model must still
return automatically when the provider lists it again, unless the user opted out.
Changing capability ownership to `manual` would incorrectly stop those updates.

`set_model_enabled` changes both facts in one existing revision CAS. Disabling
still refuses role/default-embedding references. Explicit reopening verifies the
current credentials and model before clearing the opt-out. Failed verification
or a stale revision preserves the opt-out. Repeating the same choice is a no-op.
Choosing off for an already provider-unavailable model records a new opt-out.

Sync preserves `user_disabled`, refreshes discovery-owned capabilities, and
enables returned models only when they are not opted out. Missing discovery rows
remain disabled without creating an opt-out. Manual model ownership, model IDs,
credentials, roles and the public catalog schema are unchanged. The private
snapshot also honors the opt-out if older code has set `enabled` back to true.

## Migration and recovery

The Models-owned append-only migration `20260930_01_model_user_disabled` adds
one constrained boolean column to each existing model table. Fresh registries
are created with the current schema. Existing writable registries require this
explicit installation migration; ordinary startup does not apply the expansion.
Read-only inspection of a pre-migration registry treats the missing opt-out as
false, while a partially installed expansion is rejected.

The migration follows the existing plugin SQLite-backup pattern before an atomic
pair of ALTERs; it does not import a private Core helper. The private backup
directory contains a 0600 database and digest/integrity
manifest. Repeating the migration checks the completed column definitions and
does not rewrite data. Unknown/partial definitions fail before migration writes.
It does not infer user intent from old disabled rows or rewrite their existing
availability, credentials, IDs, bindings or model revision.

For an existing installation, stage the Models artifact and use the explicit
[operator deployment plan](operator-deployment.md): include Models in the target
set and approve `20260930_01_model_user_disabled` in the plan's `migrations` list
before activating this artifact. Review the backup/ALTER scope first. The normal
migration-bundle loader discovers it from `plugins/models/migration.catalog.toml`
and checks its file digests. This change does not depend on PR #918 or assume
that ordinary plugin updates automatically run migrations. An installation that
has not applied the approved migration must keep the prior artifact active;
starting the new Models writer directly fails with the required migration ID.

Code rollback alone does not preserve the new opt-out behavior: old sync code
does not consult the flag. Keep the additive columns and apply the fixed Models
artifact, or perform a separately authorized offline restore from the verified
backup. Do not restore a pre-migration database over later model or credential
changes without reconciling those changes first.

Validation is provided by `scripts/model_enabled_sync_scenario.py` with a real
Models composition and SQLite registry, controlled provider I/O, and temporary
data. This is not evidence of real-provider or production deployment success.
