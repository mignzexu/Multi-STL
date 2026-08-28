# Metrics Plotting Tool

## TL;DR
> **Summary**: Add a root-level Python matplotlib CLI, `plot_metrics.py`, that reads one run's `metrics.json` from `--work_dirs/-wd` and `--ex_name/-ex`, then writes PNG training/validation charts under that run's `TPro/` directory.
> **Deliverables**:
> - `plot_metrics.py` with argparse entrypoint mirroring the repo's existing `visualization.py` style
> - `tests/test_plot_metrics.py` with `unittest` + `tempfile` coverage
> - Generated output contract: `TPro/Loss.png`, `TPro/LR.png`, `TPro/<valid_group>/MAE_RMSE.png`, and `TPro/<valid_group>/<LIST_METRIC>.png`
> **Effort**: Short
> **Parallel**: NO
> **Critical Path**: Task 1 → Tasks 2-4 → Task 5 → Final Verification

## Context
### Original Request
The user needs a Python `.py` tool that scans model run project metrics, creates a `TPro` folder, and plots training/validation loss, learning rate, and validation metric charts from `metrics.json` using epochs as the x-axis.

### Interview Summary
- Use Python `matplotlib`.
- Use an argparse CLI similar to `visualization.py`.
- CLI must take two inputs: workdir address and project/run name.
- Use PNG output only.
- Missing/malformed `metrics.json`: skip with warning, no crash/traceback.
- Add `unittest` coverage.

### Metis Review (gaps addressed)
- Exact CLI locked: `python plot_metrics.py -wd work_dirs -ex drift_w3s` with aliases `--work_dirs` and `--ex_name`.
- Existing `TPro/*.png` files are overwritten silently.
- Missing/malformed selected run exits successfully after warning; no traceback.
- Partial/invalid metric points are skipped with warnings; a chart is skipped only when no valid series remains.
- Force `matplotlib.use("Agg")` before importing pyplot.
- Warnings go to `stderr` using `print(..., file=sys.stderr)` for simple stable tests.

## Work Objectives
### Core Objective
Create a reusable training-metrics plotting CLI for one selected run directory.

### Deliverables
- `plot_metrics.py`
- `tests/test_plot_metrics.py`

### Definition of Done (verifiable conditions with commands)
- `python -m unittest tests.test_plot_metrics` passes.
- `python plot_metrics.py -wd <tmp_work_dirs> -ex drift_w3s` creates expected PNG files for a synthetic metrics fixture.
- `python plot_metrics.py -wd work_dirs -ex drift_w3s` runs without traceback in the real repo environment when matplotlib is installed.

### Must Have
- Numeric epoch sorting for string keys (`"1"`, `"2"`, `"10"`).
- `Loss.png` contains train loss and valid loss if present.
- `LR.png` contains `train.lr` if present.
- For each dictionary group under `valid` except `loss`, create `TPro/<group>/`.
- `MAE_RMSE.png` combines `mae` and `rmse`; never write `MSE.png`.
- Each list-valued metric becomes one PNG titled by metric uppercase, with one curve per list index.
- Automatically distinguish curve colors by relying on matplotlib's default color cycle or colormap.
- Close every figure after saving.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Do not modify `utils/metrics_v2.py`, metrics writing, training, testing, report, or visualization pipelines.
- Do not add dependencies beyond Python standard library and matplotlib.
- Do not add PDF/SVG/PPT/CSV/JSON outputs.
- Do not recursively process all runs or support multiple run names in this plan.
- Do not require human visual inspection as acceptance criteria.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: TDD with existing repo style: `unittest`.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.omo/evidence/task-{N}-{slug}.{ext}`.

## Execution Strategy
### Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 foundation CLI/helper contracts.
Wave 2: Task 2 metrics loading and extraction.
Wave 3: Task 3 top-level Loss/LR plotting.
Wave 4: Task 4 validation metric group plotting.
Wave 5: Task 5 integration hardening and full smoke verification.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Task 2.
- Task 2 blocks Task 3.
- Task 3 blocks Task 4.
- Task 4 blocks Task 5.
- Task 5 blocks Final Verification.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 1 task → `quick`
- Wave 2 → 1 task → `quick`
- Wave 3 → 1 task → `quick`
- Wave 4 → 1 task → `quick`
- Wave 5 → 1 task → `quick`
- Final Verification → 4 review tasks → oracle / unspecified-high / unspecified-high / deep

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Add CLI skeleton, constants, warning behavior, and fixture test harness

  **What to do**: Create `plot_metrics.py` at repo root. Add `matplotlib.use("Agg")` before importing `matplotlib.pyplot`. Implement argparse with `-wd/--work_dirs` and `-ex/--ex_name`, resolving metrics path as `<work_dirs>/<ex_name>/metrics.json` and output path as `<work_dirs>/<ex_name>/TPro`. Add `warn(message)` that writes stable warning text to `stderr`. Create `tests/test_plot_metrics.py` with tempfile fixture helpers that write synthetic `metrics.json` matching `work_dirs/drift_w3s/metrics.json` shape.
  **Must NOT do**: Do not implement every plot in this task; only scaffold CLI/helpers enough for tests to import safely.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: bounded new-file scaffold plus tests.
  - Skills: [`test-driven-development`] - Use tests first for CLI/helper contract.
  - Omitted: [`frontend-design`] - No UI/frontend work.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: Task 2 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `visualization.py` - argparse style with work directory and experiment name flags.
  - Pattern: `work_dirs/drift_w3s/metrics.json` - real metrics schema shape.
  - Test: `models/tests/test_migration_contracts.py` - existing `unittest` style.
  - Guardrail: `utils/metrics_v2.py` - active writer; read only, do not modify.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_plot_metrics` runs and imports `plot_metrics.py` without opening GUI windows.
  - [ ] Test asserts parser accepts `-wd /tmp/work_dirs -ex drift_w3s` and resolves `<work_dirs>/drift_w3s/metrics.json`.
  - [ ] Test asserts warning helper writes text containing `WARNING` to stderr.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: CLI parse happy path
    Tool: Bash
    Steps: python -m unittest tests.test_plot_metrics
    Expected: parser/helper tests pass with exit code 0
    Evidence: .omo/evidence/task-1-cli-unittest.txt

  Scenario: Warning helper failure path
    Tool: Bash
    Steps: run unittest case that captures stderr from warn("bad metrics")
    Expected: captured stderr contains WARNING and bad metrics
    Evidence: .omo/evidence/task-1-warning-unittest.txt
  ```

  **Commit**: NO | Message: `feat(metrics): add plotting cli scaffold` | Files: `plot_metrics.py`, `tests/test_plot_metrics.py`

- [ ] 2. Implement metrics loading, numeric epoch sorting, and robust series extraction

  **What to do**: In `plot_metrics.py`, add helper functions to load JSON, numerically sort epoch keys, extract scalar series by path (`train.loss`, `valid.loss`, `train.lr`), extract validation groups as dictionary values under `valid` except `loss`, and skip invalid epochs/values with warnings. Tests must cover out-of-order epoch keys `"10"`, `"2"`, `"1"`, missing file, invalid JSON, empty metrics, missing train/valid keys, and nonnumeric values.
  **Must NOT do**: Do not fail the entire command for one bad point; do not sort lexicographically.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: pure helper functions and focused tests.
  - Skills: [`test-driven-development`] - Lock parsing behavior before plotting.
  - Omitted: [`debugging`] - No runtime bug yet.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Task 3 | Blocked By: Task 1

  **References**:
  - Pattern: `reports/build_editable_ppt.py` - existing metrics consumer and epoch sorting behavior.
  - Pattern: `reports/build_multi_stl_experiment_ppt.py` - work_dirs scan and loss extraction reference.
  - API/Type: `work_dirs/drift_w3s/metrics.json` - epoch-key and nested metric schema.

  **Acceptance Criteria**:
  - [ ] Unit test proves epochs are ordered `[1, 2, 10]` for keys `"10"`, `"2"`, `"1"`.
  - [ ] Unit test proves missing/invalid JSON returns a skip result and emits warning without traceback.
  - [ ] Unit test proves scalar extraction returns separate x/y arrays using only valid numeric points.

  **QA Scenarios**:
  ```
  Scenario: Numeric epoch sort
    Tool: Bash
    Steps: python -m unittest tests.test_plot_metrics
    Expected: test fixture with keys "10", "2", "1" extracts x-axis [1, 2, 10]
    Evidence: .omo/evidence/task-2-epoch-sort.txt

  Scenario: Malformed metrics skip
    Tool: Bash
    Steps: unittest writes invalid JSON metrics.json and calls loader
    Expected: no exception; stderr warning includes invalid JSON; result indicates skipped
    Evidence: .omo/evidence/task-2-malformed-json.txt
  ```

  **Commit**: NO | Message: `feat(metrics): parse metrics series robustly` | Files: `plot_metrics.py`, `tests/test_plot_metrics.py`

- [ ] 3. Implement Loss and LR PNG generation with tests

  **What to do**: Add plotting helper that creates a line chart with title, x-label `Epoch`, y-label, legend when more than one series exists, grid, `fig.savefig(path, dpi=300, bbox_inches="tight")`, and `plt.close(fig)`. Implement top-level `Loss.png` with train/valid loss series and `LR.png` with learning rate. Tests must run against tempfile output and assert PNG files exist and are non-empty.
  **Must NOT do**: Do not require visual approval; do not save non-PNG formats.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: isolated plotting helper and output assertions.
  - Skills: [`test-driven-development`] - Test output file creation and no non-PNG outputs.
  - Omitted: [`visual-qa`] - Acceptance is file generation, not visual design evaluation.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Task 4 | Blocked By: Task 2

  **References**:
  - Pattern: `dataset/SDweather/vis/vis_SD.py` - matplotlib save/close style.
  - Pattern: `work_dirs/drift_w3s/metrics.json` - `train.loss`, `valid.loss`, `train.lr` paths.

  **Acceptance Criteria**:
  - [ ] Unit test creates `TPro/Loss.png` and file size is greater than 0.
  - [ ] Unit test creates `TPro/LR.png` and file size is greater than 0.
  - [ ] Unit test verifies generated files under top-level `TPro/` all end with `.png`.

  **QA Scenarios**:
  ```
  Scenario: Loss and LR happy path
    Tool: Bash
    Steps: python -m unittest tests.test_plot_metrics
    Expected: tempfile run contains non-empty TPro/Loss.png and TPro/LR.png
    Evidence: .omo/evidence/task-3-loss-lr-unittest.txt

  Scenario: Missing LR failure path
    Tool: Bash
    Steps: unittest fixture omits train.lr and invokes top-level plot generation
    Expected: Loss.png still generated; LR.png skipped with WARNING; no traceback
    Evidence: .omo/evidence/task-3-missing-lr.txt
  ```

  **Commit**: NO | Message: `feat(metrics): plot loss and learning rate` | Files: `plot_metrics.py`, `tests/test_plot_metrics.py`

- [ ] 4. Implement validation metric group plotting with tests

  **What to do**: For each validation group dictionary under `valid` except `loss`, create `TPro/<group>/`. Plot `mae` and `rmse` together as `MAE_RMSE.png` when either series exists; skip `mse` always. For every list-valued metric such as `csi`, `pod`, `far`, `hss`, create `<METRIC_UPPER>.png` with one curve per list index labeled like `CSI[0]`, `CSI[1]`. If list lengths vary, plot available numeric values per index and warn about missing/invalid values.
  **Must NOT do**: Do not plot scalar metrics other than `mae`/`rmse`; do not create `MSE.png`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: bounded metric-group plotting rules.
  - Skills: [`test-driven-development`] - Tests define exact output layout.
  - Omitted: [`xlsx`] - No spreadsheet output.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: Task 5 | Blocked By: Task 3

  **References**:
  - API/Type: `work_dirs/drift_w3s/metrics.json` - `valid.w10.mae/rmse/mse/csi/pod/far/hss` structure.
  - Guardrail: User requirement - `mse` must not be plotted.

  **Acceptance Criteria**:
  - [ ] Unit test creates `TPro/w10/MAE_RMSE.png` for `mae`/`rmse`.
  - [ ] Unit test creates `TPro/w10/CSI.png`, `POD.png`, `FAR.png`, and `HSS.png` when those list metrics are present.
  - [ ] Unit test proves `TPro/w10/MSE.png` does not exist even when `mse` exists in metrics.
  - [ ] Unit test proves an empty validation group creates no PNG and emits warning.

  **QA Scenarios**:
  ```
  Scenario: Validation group happy path
    Tool: Bash
    Steps: python -m unittest tests.test_plot_metrics
    Expected: w10 folder contains MAE_RMSE.png and list metric PNGs; all non-empty
    Evidence: .omo/evidence/task-4-valid-group-unittest.txt

  Scenario: MSE exclusion edge case
    Tool: Bash
    Steps: unittest fixture includes only valid.w10.mse and invokes group plotting
    Expected: no MSE.png is written; warning notes no plottable metrics
    Evidence: .omo/evidence/task-4-mse-skipped.txt
  ```

  **Commit**: NO | Message: `feat(metrics): plot validation metric groups` | Files: `plot_metrics.py`, `tests/test_plot_metrics.py`

- [ ] 5. Integrate CLI main flow and full smoke tests

  **What to do**: Wire `main()` so `python plot_metrics.py -wd <work_dirs> -ex <ex_name>` loads the selected run, creates/overwrites `TPro`, generates all applicable plots, and returns `0` after warnings for missing/malformed data. Add CLI-level unittest or subprocess test using tempfile workdir and synthetic project name. Verify no outputs outside `TPro` and no non-PNG files. Run against real `work_dirs/drift_w3s` if matplotlib is available.
  **Must NOT do**: Do not add multi-run support, recursive scanning, README edits, or optional plotting flags.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: integration and smoke verification only.
  - Skills: [`test-driven-development`] - Add subprocess-level regression tests.
  - Omitted: [`git-master`] - Commit not requested.

  **Parallelization**: Can Parallel: NO | Wave 5 | Blocks: Final Verification | Blocked By: Task 4

  **References**:
  - Pattern: `visualization.py` - CLI invocation convention: `-wd` and `-ex` style.
  - Pattern: `README.md` - documented train/test/visualization command style.
  - Real fixture: `work_dirs/drift_w3s/metrics.json` - optional real-run smoke input.

  **Acceptance Criteria**:
  - [ ] `python -m unittest tests.test_plot_metrics` exits 0.
  - [ ] `python plot_metrics.py -wd <tempdir>/work_dirs -ex drift_w3s` exits 0 and writes the expected PNG tree.
  - [ ] Missing metrics selected run exits 0, writes warning to stderr, and does not create traceback output.
  - [ ] `find <tempdir>/work_dirs/drift_w3s/TPro -type f ! -name "*.png"` returns no files.

  **QA Scenarios**:
  ```
  Scenario: End-to-end tempfile CLI
    Tool: Bash
    Steps: python -m unittest tests.test_plot_metrics
    Expected: subprocess test creates TPro/Loss.png, TPro/LR.png, TPro/w10/MAE_RMSE.png, TPro/w10/CSI.png; exit code 0
    Evidence: .omo/evidence/task-5-cli-smoke.txt

  Scenario: Missing selected project failure path
    Tool: Bash
    Steps: python plot_metrics.py -wd /tmp/nonexistent_work_dirs -ex missing_run 2> .omo/evidence/task-5-missing-run-stderr.txt; test $? -eq 0
    Expected: stderr contains WARNING; no Python traceback appears
    Evidence: .omo/evidence/task-5-missing-run-stderr.txt
  ```

  **Commit**: NO | Message: `feat(metrics): integrate plotting cli` | Files: `plot_metrics.py`, `tests/test_plot_metrics.py`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Do not commit unless the user explicitly requests it.
- If later requested, use one commit after all verification passes: `feat(metrics): add training metrics plotting tool`.
- Intended files: `plot_metrics.py`, `tests/test_plot_metrics.py`.

## Success Criteria
- The selected run command `python plot_metrics.py -wd work_dirs -ex drift_w3s` can generate `TPro` plots from a real metrics file without traceback.
- Automated unittest coverage proves parsing, output file creation, `mse` exclusion, malformed JSON warning behavior, partial invalid-point handling, and numeric epoch sorting.
- No source files outside the new tool and new test file are modified.
