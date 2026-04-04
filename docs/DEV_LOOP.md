# Development Loop

Autonomous infinite loop. Runs until explicitly stopped.

## Setup (every iteration)
```
git config user.email "tony.lin@intel.com"
git config user.name "hlin99"
```

## Each Iteration

1. Pull latest code
2. Read ROADMAP.md — find the next incomplete milestone
3. Read DESIGN_PRINCIPLES.md — follow the rules
4. Check open issues/PRs — handle unmerged PRs first (fix CI failures, merge if ready)
5. If no milestone left, create new ones (see Phase 2 below)
6. Create GitHub Issue: problem, solution, acceptance criteria, tests
7. Create branch, implement code + tests
8. Pass lint: ruff check src tests && isort --check src tests
9. Create PR (body contains Closes #N)
10. Wait for CI green. Fix failures. Never merge red CI.
11. Self-review against acceptance criteria and DESIGN_PRINCIPLES.md
12. Squash merge
13. Update ROADMAP.md, push to main
14. Go to step 1

## Rules
- Committer must be hlin99 <tony.lin@intel.com>
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format

## Phase 1: Roadmap-Driven
Follow ROADMAP.md milestones in order.

## Phase 2: Continuous Evolution
When all milestones are done:
1. Review the project — find limitations, improvements, new scenarios
2. Create new milestones in ROADMAP.md
3. Return to Phase 1
