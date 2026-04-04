# Development Loop

Finite loop. Runs until all milestones in ROADMAP.md are complete (all 56 test cases pass).

## Setup (every iteration)
```
git config user.email "tony.lin@intel.com"
git config user.name "hlin99"
```

## Each Iteration

1. Pull latest code
2. Read `ROADMAP.md` — find the next incomplete milestone
3. Read `docs/DESIGN.md` — the authoritative spec, follow it exactly
4. Read `docs/DESIGN_PRINCIPLES.md` — follow the rules
5. Check open issues/PRs — handle unmerged PRs first (fix CI failures, merge if ready)
6. If all milestones done → report completion and stop
7. Create GitHub Issue: what to implement, which test cases to cover
8. Create branch, implement code + tests
9. Pass lint: `ruff check src tests && isort --check src tests`
10. Create PR (body contains `Closes #N`)
11. Wait for CI green. Fix failures. Never merge red CI.
12. Self-review against DESIGN.md test cases
13. Squash merge
14. Update ROADMAP.md, push to main
15. Go to step 1

## Rules
- Committer must be `hlin99 <tony.lin@intel.com>`
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format
- DESIGN.md is the source of truth — implement exactly what it says
