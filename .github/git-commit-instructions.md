# Commit messages

Format: `<type>(<scope>): <summary>`

- type: feat | fix | perf | refactor | test | docs | build | chore
- scope: the module touched (e.g. solver, shader, sampler, validation, cmake).
  Omit if the change spans several modules.
- summary: imperative mood, lowercase, no trailing period, max 50 characters.
- Name the concrete thing changed. Write "fix(shader): clamp lambda denominator
  to avoid NaN", not "fix bug in shader".
- Use `fix` for correctness and `perf` for speed. Never mix them.
- Add a body only if the *why* is not obvious from the diff: 1–3 lines,
  wrapped at 72 characters. No file-by-file lists.
- If a numerical result or reference value changes, state it in the body
  (e.g. "max density error 3.1% -> 0.4%").
- Avoid vague words: update, improve, various, minor, changes.

Examples:
feat(sampler): add affine coupling layer
test(validation): compare posterior mean against analytic Gaussian
perf(solver): move neighbour search to shared memory
