# Archived: Kubernetes manifests

Moved from `k8s/` by VIAB#P3-2 (2026-10-08). Kosmos is a single-user CLI
(`kosmos run`) and runs no HTTP server, so these manifests, which assume a web
service on port 8000 with HTTP probes and an ingress, describe nothing that
ships. Kubernetes support beyond this archive is a non-goal of the viability
plan (evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md §9).

The image builds from the repository `Dockerfile`; its health check runs
`python -m kosmos.cli.main version`. Nothing in the repository reads these
files. `docs/deployment/deployment-guide.md` still describes them as they were.
