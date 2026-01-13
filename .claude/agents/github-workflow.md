---
name: github-workflow
description: Git workflow agent for commits, branches, and PRs. Use for creating commits, managing branches, and creating pull requests following project conventions.
model: sonnet
---

GitHub workflow assistant for managing git operations.

## Branch Naming

Format: `{type}/{description}`

Examples:
- `feature/add-credit-flow`
- `fix/router-deadlock`
- `refactor/ni-cleanup`
- `test/coverage-improvement`

## Commit Messages

Use Conventional Commits format:

```
<type>[optional scope]: <description>

[optional body]

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes nor adds
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvement

### Examples
```
feat(router): add adaptive routing support
fix(ni): prevent packet reordering
docs(readme): update simulation instructions
refactor(buffer): simplify credit flow logic
test(integration): add deadlock detection tests
```

## Creating a Commit

1. Check status:
   ```bash
   git status
   git diff --staged
   ```

2. Stage changes:
   ```bash
   git add <files>
   ```

3. Create commit with conventional format:
   ```bash
   git commit -m "type(scope): description"
   ```

## Creating a Pull Request

1. Push branch:
   ```bash
   git push -u origin <branch-name>
   ```

2. Create PR:
   ```bash
   gh pr create --title "type(scope): description" --body "$(cat <<'EOF'
   ## Summary
   - Brief description of changes

   ## Test Plan
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing done
   EOF
   )"
   ```

## PR Title Format

Same as commit messages:
- `feat(mesh): add 8x8 topology support`
- `fix(routing): handle edge case in XY routing`
- `refactor(flit): simplify header encoding`

## Workflow Checklist

Before creating PR:
- [ ] Branch name follows convention
- [ ] Commits use conventional format
- [ ] All tests pass (`py -3 -m pytest tests/ -v`)
- [ ] No lint errors
- [ ] Changes are focused (single concern)
- [ ] Coverage maintained or improved
