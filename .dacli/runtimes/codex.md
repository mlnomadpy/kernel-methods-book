---
id: rt-codex
kind: runtime
created: 2026-07-24T22:13:01Z
created_by: a-root
name: codex
binary: /Applications/ChatGPT.app/Contents/Resources/codex
invoke_mode: stdin
invoke_args: [exec, -]
sandbox_ro_args: [--sandbox, read-only]
env_passthrough: [HOME, PATH]
model_flag: --model
---
# codex
Flags here are assumptions until `dacli runtime doctor` verifies them against the installed binary.
