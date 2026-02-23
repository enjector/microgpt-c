---
description: Maintain a Kanban board (KANBAN.md) to track tasks across Backlog, In Progress, and Done — preserving context between sessions.
---

# Kanban Workflow

Use `KANBAN.md` at the repo root as a persistent task-tracking board. This maintains context across conversations and sessions.

## When to Use

- At the **start of a session**: read `KANBAN.md` to understand current priorities and in-progress work.
- When **starting new work**: create a task card in the board, move it to In Progress.
- When **finishing work**: move the task to Done, update its outcome in Spear format.
- When the **user adds a request**: add it to Backlog with the next available ID.

## Board Structure

The board has three columns as markdown tables:

### Backlog
Tasks not yet started. Format: `| ID | Title | Priority |`

### In Progress
Tasks actively being worked on. Format: `| ID | Title | Started |`

### Done
Completed tasks. Format: `| ID | Title | Completed |`

## Task Details

Below the board, each task gets a detail section using the **/spear** format:

```
### K-<ID>: <Title>

- **Point:** One-sentence bottom line — what this task achieves.
- **Picture:** A relatable analogy that makes the goal obvious.
- **Proof:** The concrete evidence or metric that proves it's done.
- **Push:** The next action or follow-up this enables.

**Outcome:** <filled in when moved to Done — what actually happened>
```

## How to Manage

1. **Read the board** at the start of each session to pick up context.
2. **Add tasks** to Backlog when the user mentions new work.
3. **Move tasks** to In Progress when you start working on them.
4. **Move tasks** to Done when complete, and fill in the Outcome.
5. **Assign IDs** sequentially: K-1, K-2, K-3, etc.
6. **Priority** is one of: 🔴 High, 🟡 Medium, 🟢 Low.
7. **Run `/run-definition-of-done`** before marking significant tasks as Done.

## ID Format

Task IDs use the prefix `K-` followed by a sequential number: `K-1`, `K-2`, `K-3`, etc.

## Example

```markdown
## Backlog
| ID | Title | Priority |
|----|-------|----------|
| K-5 | Implement 5x5 Hex variant | 🟢 Low |

## In Progress
| ID | Title | Started |
|----|-------|---------|
| K-4 | Expand Red Donkey corpus to 1000+ | Feb 24 |

## Done
| ID | Title | Completed |
|----|-------|-----------|
| K-3 | Hex topology uplift | Feb 23 |

---

### K-3: Hex topology uplift
- **Point:** Hex win rate jumped from 4% to 27% with zero engine changes.
- **Picture:** Like giving a blind chess player a description of the board shape instead of just listing pieces.
- **Proof:** 27% win rate vs random (6.75× improvement), parse errors dropped 50%→17%.
- **Push:** Try virtual connection templates or 5×5 board for further gains.

**Outcome:** Shipped in commit ea87a68. BFS connectivity features + topological Judge + MCTS corpus. All docs updated, discussion #3 comment posted.
```
