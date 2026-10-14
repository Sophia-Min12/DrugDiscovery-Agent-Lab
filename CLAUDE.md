# DrugAgent-Lab — working agreements

## Commits
- **Never add `Co-Authored-By: Claude ...` trailers.** Commits are authored by the repo
  owner alone, so GitHub shows a single avatar.
- No "Generated with Claude Code" footers in commit messages or PR descriptions.

## Language
- All repo content — README, docstrings, comments, commit messages — is written in English.

## Conventions
- Every number in the README comes from a file in `experiments/results/`, produced by a
  script in `experiments/`. Change the code, re-run the script, then change the prose.
- Randomness is always seeded. A test that cannot be re-run to the same number is not
  a test.
- Prefer testing against a **closed-form or independently-known answer** (atomic masses,
  Ertl's fragment table, a hand-computed BH correction, planted truth) over testing a
  function against its own output.
- A value that depends on a method is never stored or returned without the method's name.
- When a measurement contradicts the prose, correct the prose and keep the failing case
  as a regression test.
