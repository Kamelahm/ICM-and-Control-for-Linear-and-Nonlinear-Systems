# results/

**Intentionally empty.**

The original package shipped tables here that were `run_all.py --quick` output
(2-3 trials, truncated sweeps) produced by the pre-fix code. They read as
results but were smoke tests, and every number in them is superseded by the
corrections in `../CHANGELOG.md`.

Generate before quoting anything:

    python run_all.py --quick     # ~minutes, checks the pipeline runs
    python run_all.py             # paper settings

Nothing in this directory should reach the manuscript without having been
produced at paper settings.
