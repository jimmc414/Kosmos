# Kosmos X-Ray Export (Enhanced)

Portable context-efficient codebase exploration system for AI programmers.

## Enhanced Features (v2)
- **Pydantic/dataclass field extraction** - `name: str = Field(...)` now visible
- **Decorator support** - `@tool`, `@dataclass`, `@agent.register`
- **Global constants** - `SYSTEM_PROMPT = "..."`, `CONFIG = {...}`
- **Line numbers** - `def method(): ...  # L42` for precise navigation
- **Mermaid diagrams** - `--mermaid` flag for dynamic architecture visualization
- **Import verification** - Agent verifies entry points are importable

## Contents

```
export/
├── README.md                 # This file
├── WARM_START.md             # Example output (regenerate for your repo)
├── agents/
│   └── kosmos_architect.md   # Agent for generating documentation
└── kosmos-xray/              # Core skill
    ├── SKILL.md
    ├── CHEATSHEET.md
    ├── reference.md
    ├── configs/
    │   ├── ignore_patterns.json
    │   └── priority_modules.json
    ├── scripts/
    │   ├── mapper.py
    │   ├── skeleton.py
    │   └── dependency_graph.py
    ├── templates/
    │   ├── warm_start.md.template
    │   └── skeleton_format.md
    └── lib/
        ├── __init__.py
        ├── ast_utils.py
        └── token_estimator.py
```

## Installation

### 1. Copy to Target Repository

```bash
# From the export directory
cp -r kosmos-xray /path/to/target/.claude/skills/
cp -r agents /path/to/target/.claude/

# Or if .claude doesn't exist yet
mkdir -p /path/to/target/.claude/skills
mkdir -p /path/to/target/.claude/agents
cp -r kosmos-xray /path/to/target/.claude/skills/
cp agents/kosmos_architect.md /path/to/target/.claude/agents/
```

### 2. Customize Configurations

Edit `configs/ignore_patterns.json` for your repo's structure:
```json
{
  "directories": ["node_modules", "dist", "build", "__pycache__", ...],
  "extensions": [".log", ".pyc", ...],
  "files": ["*.min.js", ...]
}
```

Edit `configs/priority_modules.json` for your repo's architecture:
```json
{
  "priority_patterns": {
    "critical": {
      "patterns": ["**/core/**/*.py", "**/main.*"]
    },
    "high": {
      "patterns": ["**/models/**/*.py", "**/services/**/*.py"]
    }
  }
}
```

### 3. Generate WARM_START.md

```bash
# Test the scripts first
python .claude/skills/kosmos-xray/scripts/mapper.py --summary
python .claude/skills/kosmos-xray/scripts/skeleton.py src/ --priority critical
python .claude/skills/kosmos-xray/scripts/dependency_graph.py src/

# Then use the agent to generate documentation
# In Claude Code: @kosmos_architect generate
```

## Usage

### Direct Script Usage

```bash
# Map directory structure with token estimates
python .claude/skills/kosmos-xray/scripts/mapper.py [directory] [--summary] [--json]

# Extract Python file skeletons (95% token reduction)
python .claude/skills/kosmos-xray/scripts/skeleton.py <path> [--priority critical|high|medium|low] [--json]

# Analyze import dependencies
python .claude/skills/kosmos-xray/scripts/dependency_graph.py [directory] [--root package_name] [--focus area] [--json]
```

### Via Agent (Claude Code)

```
@kosmos_architect generate     # Create WARM_START.md
@kosmos_architect refresh      # Update existing doc
@kosmos_architect query "X"    # Answer architecture questions
```

### Via Skill Triggers

The skill responds to these phrases:
- "xray", "x-ray"
- "map structure"
- "skeleton"
- "interface"
- "architecture"
- "warm start"

## Adapting for Non-Python Projects

The scripts are Python-focused but can be adapted:

1. **mapper.py** - Works with any file types, just update `ignore_patterns.json`
2. **skeleton.py** - Python-only (AST parsing), would need rewrite for other languages
3. **dependency_graph.py** - Python-only, would need rewrite for other languages

For non-Python projects, mapper.py still provides value for token budgeting.

## Requirements

- Python 3.8+
- No external dependencies (uses stdlib only: ast, json, os, pathlib, argparse)

## License

Same as parent repository.
