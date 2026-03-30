---
paths:
  - "plan/**"
---

# Plan Format Rules

Plans live under `plan/` and follow a consistent structure.

## Frontmatter

Required fields: `goal`, `version`, `date_created`, `last_updated`, `change_log` (list of `{date, version, summary}`), `owner`, `status`, `tags`.

## Status badge

Place immediately after the `# Introduction` heading:

- Done: `![Status: Done](https://img.shields.io/badge/status-Done-brightgreen)`
- In progress: `![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)`
- Planned: `![Status: Planned](https://img.shields.io/badge/status-Planned-blue)`

## Task tables

Use `✅` (not `Yes` or other text) in the Completed column.
