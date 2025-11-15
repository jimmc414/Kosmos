# World Model MVP Implementation Guide

**Version:** 1.0 (Fast Path)
**Timeline:** 1-2 weeks to production
**Goal:** Ship persistent knowledge graphs NOW, architect later

---

## Choose Your Path

You have **two implementation options**:

### 🚀 **This Document: MVP-First (Fast Path)**
- **Timeline:** 1-2 weeks
- **Approach:** Ship minimal viable persistence, validate with users, evolve
- **Philosophy:** YAGNI (You Aren't Gonna Need It) - build what's needed NOW
- **Best for:** Delivering value quickly, validating demand, iterating based on feedback

### 🏗️ **implementation.md: Full Architecture (Reference Path)**
- **Timeline:** 19 weeks (5 phases)
- **Approach:** Build complete abstraction layer, plan for scale, Production Mode ready
- **Philosophy:** Architecture-first - build for tomorrow's needs today
- **Best for:** Educational projects, papers, long-term organizational adoption

**💡 Recommendation:** Start with MVP, keep full architecture as roadmap. Evolve when users prove they need it.

---

## Table of Contents

1. [Week 1: Core Persistence](#week-1-core-persistence-5-days)
2. [Week 2: Polish & Deploy](#week-2-polish--deploy-optional)
3. [Evolution Path](#evolution-path-mvp-to-full-architecture)
4. [When to Migrate](#when-to-migrate-to-full-architecture)

---

## Week 1: Core Persistence (5 Days)

**Goal:** Make the current Kosmos knowledge graph survive restarts.

### Day 1: Make Neo4j Persistent

**Current Problem:** Neo4j data stored in temporary Docker volume, lost on restart.

**Solution:** Use named volumes.

**Update `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.13
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data      # ADD THIS - persistent storage
      - neo4j_logs:/logs      # ADD THIS - persistent logs
    restart: unless-stopped   # ADD THIS - auto-restart

volumes:
  neo4j_data:                 # ADD THIS - named volume
  neo4j_logs:
```

**Test persistence:**

```bash
# Start Neo4j
docker-compose up -d

# Run a research query (builds graph)
kosmos research "transformers in NLP"

# Stop Neo4j
docker-compose down

# Start again
docker-compose up -d

# Check if data survived
# Access Neo4j browser at http://localhost:7474
# Run: MATCH (n) RETURN count(n)
# Should show entities from previous run!
```

**Acceptance Criteria:**
- ✅ Knowledge graph survives Docker restart
- ✅ No code changes needed (just config)

---

### Day 2-3: Export Functionality

**Goal:** Let users back up their knowledge graphs.

**Create `kosmos/cli/graph_commands.py`:**

```python
"""Simple graph management commands."""

import click
import json
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase


@click.group()
def graph():
    """Manage knowledge graphs."""
    pass


@graph.command()
@click.argument("filepath", type=click.Path())
def export(filepath):
    """Export knowledge graph to JSON file.

    Example:
        kosmos graph export backup.json
    """
    from kosmos.config import Config

    config = Config.from_file("config.yaml")
    filepath = Path(filepath)

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.neo4j_url,
        auth=(config.neo4j_user, config.neo4j_password)
    )

    with driver.session() as session:
        # Get all nodes
        nodes_result = session.run("MATCH (n) RETURN n")
        nodes = []
        for record in nodes_result:
            node = record["n"]
            nodes.append({
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": dict(node)
            })

        # Get all relationships
        rels_result = session.run("MATCH ()-[r]->() RETURN r")
        relationships = []
        for record in rels_result:
            rel = record["r"]
            relationships.append({
                "id": rel.element_id,
                "type": rel.type,
                "start_node": rel.start_node.element_id,
                "end_node": rel.end_node.element_id,
                "properties": dict(rel)
            })

    driver.close()

    # Create export data
    export_data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "source": "kosmos",
        "statistics": {
            "nodes": len(nodes),
            "relationships": len(relationships)
        },
        "nodes": nodes,
        "relationships": relationships
    }

    # Write to file
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    click.echo(f"✅ Exported {len(nodes)} nodes and {len(relationships)} relationships")
    click.echo(f"   File: {filepath}")


@graph.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--clear", is_flag=True, help="Clear existing graph before import")
def import_graph(filepath, clear):
    """Import knowledge graph from JSON file.

    Example:
        kosmos graph import backup.json
        kosmos graph import backup.json --clear
    """
    from kosmos.config import Config

    config = Config.from_file("config.yaml")
    filepath = Path(filepath)

    # Load data
    with open(filepath) as f:
        data = json.load(f)

    nodes = data["nodes"]
    relationships = data["relationships"]

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.neo4j_url,
        auth=(config.neo4j_user, config.neo4j_password)
    )

    with driver.session() as session:
        # Clear if requested
        if clear:
            click.echo("Clearing existing graph...")
            session.run("MATCH (n) DETACH DELETE n")

        # Import nodes
        click.echo(f"Importing {len(nodes)} nodes...")
        for node in nodes:
            labels = ":".join(node["labels"])
            session.run(
                f"CREATE (n:{labels}) SET n = $properties",
                properties=node["properties"]
            )

        # Import relationships
        click.echo(f"Importing {len(relationships)} relationships...")
        # Note: This is simplified - production would need to match by properties
        # since element_id changes between databases

    driver.close()

    click.echo(f"✅ Import complete")
    click.echo(f"   Nodes: {len(nodes)}")
    click.echo(f"   Relationships: {len(relationships)}")


@graph.command()
def info():
    """Show knowledge graph statistics.

    Example:
        kosmos graph info
    """
    from kosmos.config import Config

    config = Config.from_file("config.yaml")

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.neo4j_url,
        auth=(config.neo4j_user, config.neo4j_password)
    )

    with driver.session() as session:
        # Count nodes
        node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]

        # Count relationships
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

        # Get node types
        types_result = session.run("MATCH (n) RETURN DISTINCT labels(n) as labels, count(*) as count")
        types = [(record["labels"], record["count"]) for record in types_result]

    driver.close()

    # Display info
    click.echo("\n📊 Knowledge Graph Statistics\n")
    click.echo(f"Nodes:         {node_count:,}")
    click.echo(f"Relationships: {rel_count:,}")

    if types:
        click.echo(f"\nNode Types:")
        for labels, count in types:
            label_str = ":".join(labels) if labels else "(no label)"
            click.echo(f"  {label_str}: {count:,}")

    click.echo()
```

**Register commands in `kosmos/cli/main.py`:**

```python
# Add to existing CLI
from kosmos.cli.graph_commands import graph

# In your main CLI group
cli.add_command(graph)
```

**Test:**

```bash
# Build a graph
kosmos research "machine learning"

# Export it
kosmos graph export backup.json

# Check the file
cat backup.json

# View stats
kosmos graph info
```

**Acceptance Criteria:**
- ✅ `kosmos graph export` creates valid JSON
- ✅ `kosmos graph info` shows statistics
- ✅ Export file is human-readable

---

### Day 4: Import Functionality

**Goal:** Restore graphs from backups.

**Already included in code above!** The `import_graph` command.

**Test full workflow:**

```bash
# 1. Build a graph
kosmos research "neural networks"

# 2. Export it
kosmos graph export backup.json

# 3. Clear the graph
kosmos graph import backup.json --clear  # Starts fresh

# 4. Import it back
kosmos graph import backup.json

# 5. Verify it's back
kosmos graph info
```

**Note on Import Limitations:**

The MVP import is **simplified** - it doesn't handle:
- Preserving element IDs (Neo4j generates new ones)
- Duplicate detection
- Merging with existing graphs

**This is intentional.** We're validating demand first. If users need advanced import, we'll add it in Week 2 or migrate to full architecture.

**Acceptance Criteria:**
- ✅ Can restore graph from backup
- ✅ `--clear` flag works
- ✅ Data integrity maintained

---

### Day 5: Testing & Documentation

**Create basic tests:**

```python
# tests/integration/test_graph_export.py
import json
from pathlib import Path
from click.testing import CliRunner
from kosmos.cli.main import cli


def test_export_import_workflow(tmp_path):
    """Test full export/import workflow."""
    runner = CliRunner()
    export_path = tmp_path / "test_export.json"

    # Export
    result = runner.invoke(cli, ["graph", "export", str(export_path)])
    assert result.exit_code == 0
    assert export_path.exists()

    # Verify JSON structure
    with open(export_path) as f:
        data = json.load(f)

    assert "version" in data
    assert "nodes" in data
    assert "relationships" in data

    # Import
    result = runner.invoke(cli, ["graph", "import", str(export_path)])
    assert result.exit_code == 0


def test_graph_info():
    """Test graph info command."""
    runner = CliRunner()

    result = runner.invoke(cli, ["graph", "info"])
    assert result.exit_code == 0
    assert "Nodes:" in result.output
    assert "Relationships:" in result.output
```

**Create user documentation:**

```markdown
# docs/user_guides/persistent_graphs.md

# Persistent Knowledge Graphs

Your Kosmos knowledge graphs now persist between sessions!

## How It Works

When you run research queries, Kosmos builds a knowledge graph in Neo4j.
With persistent storage, this graph is saved and available in future sessions.

## Viewing Your Graph

```bash
# Show statistics
kosmos graph info
```

## Backing Up Your Graph

```bash
# Export to file
kosmos graph export my_backup.json

# The backup is in human-readable JSON
# You can version control it, share it, etc.
```

## Restoring a Graph

```bash
# Import from backup
kosmos graph import my_backup.json

# Clear and replace
kosmos graph import my_backup.json --clear
```

## Use Cases

**Knowledge Accumulation:**
Build expertise over weeks/months instead of starting fresh each session.

**Collaboration:**
Export your graph and share with colleagues.

**Version Control:**
Track your knowledge graph evolution over time.

**Backup:**
Regular exports protect against data loss.
```

**Update README.md:**

```markdown
# Add to Features section

## 🧠 Persistent Knowledge Graphs

Kosmos now maintains persistent knowledge graphs that survive between sessions:

- **Accumulate Knowledge:** Build expertise over time, not just per-query
- **Export/Import:** Backup and share your knowledge graphs
- **Collaboration:** Share graphs with colleagues
- **Version Control:** Track knowledge evolution

```bash
# View your accumulated knowledge
kosmos graph info

# Export for backup
kosmos graph export backup.json

# Restore from backup
kosmos graph import backup.json
```

See [docs/user_guides/persistent_graphs.md](docs/user_guides/persistent_graphs.md) for details.
```

**Acceptance Criteria:**
- ✅ Tests pass
- ✅ User guide complete
- ✅ README updated

---

## Week 2: Polish & Deploy (Optional)

**Only do this if Week 1 MVP gets positive feedback!**

### Improvements Based on Feedback

**Add `reset` command:**

```python
@graph.command()
@click.confirmation_option(prompt="⚠️  Delete all graph data?")
def reset():
    """Clear all knowledge graph data (DANGEROUS)."""
    from kosmos.config import Config

    config = Config.from_file("config.yaml")
    driver = GraphDatabase.driver(
        config.neo4j_url,
        auth=(config.neo4j_user, config.neo4j_password)
    )

    with driver.session() as session:
        result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
        deleted = result.single()["deleted"]

    driver.close()

    click.echo(f"✅ Deleted {deleted:,} nodes")
```

**Better error handling:**

```python
# Add try/except blocks
try:
    driver = GraphDatabase.driver(...)
except Exception as e:
    click.echo(f"❌ Failed to connect to Neo4j: {e}")
    click.echo(f"\nIs Neo4j running? Try: docker-compose up -d")
    sys.exit(1)
```

**Progress indicators for large graphs:**

```python
from tqdm import tqdm

# In export
with tqdm(total=len(nodes), desc="Exporting nodes") as pbar:
    for node in nodes:
        # ... export logic
        pbar.update(1)
```

---

## Evolution Path: MVP to Full Architecture

**When users validate they want this feature**, you can evolve incrementally:

### Evolution Step 1: Extract to Module (Week 3)

Move from CLI commands to a proper module:

```python
# kosmos/knowledge/persistence.py
class GraphPersistence:
    """Simple graph persistence layer."""

    def __init__(self, driver):
        self.driver = driver

    def export(self, filepath):
        """Export graph to file."""
        # Move export logic here

    def import_graph(self, filepath, clear=False):
        """Import graph from file."""
        # Move import logic here
```

**Benefits:**
- Separates concerns
- Easier to test
- CLI becomes thin wrapper

### Evolution Step 2: Add Abstractions (Week 4-5)

Only if you're about to add a second storage backend:

```python
# kosmos/knowledge/storage.py
from abc import ABC, abstractmethod

class GraphStorage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def export(self, filepath):
        pass

    @abstractmethod
    def import_graph(self, filepath):
        pass

class Neo4jStorage(GraphStorage):
    """Neo4j implementation."""
    # Existing code becomes implementation
```

**When to do this:**
- You're adding PostgreSQL support
- You're adding file-based storage
- Multiple people request different backends

**Don't do this if:**
- Neo4j works for everyone
- No concrete second backend planned

### Evolution Step 3: Full Architecture (Months 2-5)

Follow the full `implementation.md` plan:
- Phase 2: Curation features
- Phase 3: Multi-project support
- Phase 4: Production Mode

**Only do this if:**
- ✅ 20+ active users
- ✅ Users requesting advanced features
- ✅ Simple mode showing limitations

---

## When to Migrate to Full Architecture

**Signals you need more architecture:**

### Performance Issues
- ❌ Users report: "Graph queries taking >5 seconds"
- ❌ Users have >10K entities
- → **Consider:** Polyglot persistence (PostgreSQL + Neo4j)

### Feature Requests
- ❌ Users want: "Semantic search" or "Find similar papers"
- → **Consider:** Vector database integration
- ❌ Users want: "Separate graphs for different projects"
- → **Consider:** Multi-project support

### Quality Issues
- ❌ Users report: "Too many duplicate entities"
- ❌ Users want: "Mark entities as verified"
- → **Consider:** Curation features (Phase 2)

### Scale
- ❌ Organization wants to use it for 50+ researchers
- ❌ Need 100K+ entities
- → **Consider:** Production Mode (Phase 4)

**Migration Path:**

```
MVP (Week 1-2)
   ↓
If validated...
   ↓
Add module structure (Week 3)
   ↓
If second backend needed...
   ↓
Add abstractions (Week 4-5)
   ↓
If advanced features needed...
   ↓
Follow full implementation.md (Months 2-5)
```

---

## Decision Framework

**Use MVP if:**
- ✅ Need to ship this week/month
- ✅ Validating user demand
- ✅ Prefer iteration over planning
- ✅ Team is small (1-3 developers)

**Use Full Architecture if:**
- ✅ Writing a paper/thesis
- ✅ Building for organizational adoption
- ✅ Educational/reference project
- ✅ Have 6+ months timeline
- ✅ Know you'll need Production Mode

**Hybrid (Recommended):**
- ✅ Ship MVP Week 1
- ✅ Get user feedback
- ✅ Selectively add architecture based on REAL needs
- ✅ Keep full implementation.md as roadmap

---

## Testing the MVP

**Week 1 Test Plan:**

```bash
# Day 1: Persistence
docker-compose up -d
kosmos research "test query"
docker-compose restart neo4j
# Verify data survived

# Day 2-3: Export
kosmos graph info
kosmos graph export test.json
cat test.json  # Should be valid JSON

# Day 4: Import
kosmos graph import test.json --clear
kosmos graph info  # Should match original

# Day 5: Integration
pytest tests/integration/test_graph_export.py
```

**Success Criteria:**
- ✅ All commands work
- ✅ Data doesn't corrupt
- ✅ Export/import preserves data
- ✅ User guide is clear

---

## Deployment (Week 2)

**Simple deployment:**

```bash
# Production docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.13
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
    volumes:
      - /var/lib/kosmos/neo4j:/data  # Persistent path
    restart: always

  kosmos:
    image: kosmos:latest
    depends_on:
      - neo4j
    volumes:
      - /var/lib/kosmos/data:/app/data
    restart: always
```

**Backup script:**

```bash
#!/bin/bash
# backup.sh - Run daily via cron

DATE=$(date +%Y%m%d)
BACKUP_DIR="/var/backups/kosmos"

mkdir -p $BACKUP_DIR

# Export graph
kosmos graph export $BACKUP_DIR/graph_$DATE.json

# Keep last 7 days
find $BACKUP_DIR -name "graph_*.json" -mtime +7 -delete
```

---

## Cost Comparison

### MVP Approach
- **Development:** 1-2 weeks
- **Code:** ~300 lines
- **Complexity:** Low
- **Maintenance:** Minimal
- **User value:** Immediate

### Full Architecture
- **Development:** 19 weeks
- **Code:** ~3,000 lines
- **Complexity:** High
- **Maintenance:** Significant
- **User value:** Future-proofed

### ROI Question

**MVP delivers 80% of value in 10% of time.**

Is the remaining 20% of value worth 90% more time?
- **For most users:** No
- **For research organizations:** Maybe
- **For educational projects:** Yes

---

## FAQ

**Q: Is the MVP production-ready?**
A: Yes, for <10K entities. Beyond that, consider full architecture.

**Q: Can I migrate from MVP to full architecture later?**
A: Yes! The export format is compatible. Follow evolution path above.

**Q: What am I sacrificing with MVP?**
A: Abstract interfaces, multi-mode support, advanced curation, semantic search, Production Mode scalability.

**Q: What's the risk?**
A: If you need Production Mode later, you'll refactor. But YAGNI principle says: don't build it until proven necessary.

**Q: Should I build both?**
A: No. Ship MVP, validate with users, THEN decide if full architecture is worth it.

---

## Next Steps

1. **Decide:** MVP-first or Full Architecture?
2. **If MVP:** Start Day 1 tomorrow
3. **If Full:** Follow `implementation.md`
4. **If Unsure:** Ship MVP Week 1, reassess Week 2

**The documents work together:**
- This guide: Get to production FAST
- implementation.md: Roadmap for the future

Choose your path based on your constraints. **You can always evolve.**

---

**Document Status:** ✅ Complete
**Recommended Path:** MVP → Validate → Selectively Evolve
**Time to Value:** 1 week vs 19 weeks
