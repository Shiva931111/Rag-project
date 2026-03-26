# RAG Project Agent Configuration

This workspace uses conservative Copilot Agent settings to avoid tool/model compatibility failures.

## Allowed models

- GPT-4 class Copilot models
- Claude class Copilot models
- Default Copilot model selected by VS Code

## Disallowed models

- Gemini preview models (for example `Gemini 3 Flash (Preview)`)

## Tool policy

Use only tools that are available in the installed VS Code extensions. Do not reference unknown tools such as:

- `github/issue_read`
- `github.vscode-pull-request-github/issue_fetch`
- `github.vscode-pull-request-github/activePullRequest`

If a required tool is unavailable, continue with local repository analysis and report the limitation.
