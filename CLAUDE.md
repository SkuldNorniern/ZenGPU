# CLAUDE.md

## Repository Rules

- Use conventional commit messages for all commits.
- Do not add extra commit comments or explanatory bodies unless explicitly requested.
- Do not add co-author lines to commits.
- Create commits when a funcition is finished or added or new baseline as added do not work on before a commit  since it can override the code making harder to track the work in order

## Planning Files

- Plan files must always be locally excluded from Git.
- Do not commit plan files.
- Do not mention plan files in commits.
- Do not reference plan files in commit messages, changelogs, or public project metadata.

## Utilities And Asset Scripts

- Shared utility, tool, and base asset-generation scripts belong in the shared `utils` directory.
- Project repositories should only contain project-specific 1:1 wrapper scripts based on the shared utility scripts.
- Keep generated asset tooling reusable and avoid duplicating base utility logic across projects.


