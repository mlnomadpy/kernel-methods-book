---
id: rt-claude
kind: runtime
created: 2026-07-24T22:13:01Z
created_by: a-root
name: claude
binary: /Users/tahabsn/.local/bin/claude
invoke_mode: arg
invoke_flag: -p
sandbox_ro_args: [--allowedTools, "Read,Grep,Glob,LS,Bash(dacli:*)"]
env_passthrough: [HOME, PATH, USER, LOGNAME, TMPDIR]
model_flag: --model
skills_native_dir: .claude/skills
usage_format: stream-json
---
# claude
Flags here are assumptions until `dacli runtime doctor` verifies them against the installed binary.
