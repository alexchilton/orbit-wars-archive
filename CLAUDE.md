# Rules for this project

## Testing discipline — NON-NEGOTIABLE

- Every theory about what is wrong MUST have a pytest unit test before any code change.
- One theory = one test. Write it. Run it. Read the output. Then fix.
- No code changes without a failing test that proves the problem first.
- No "I think the issue is X" without `assert X` in a test that fails.
- After fixing, the test must pass. That is the definition of done.
