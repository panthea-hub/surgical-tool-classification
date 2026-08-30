# Engineering Decision Log

This document records significant engineering and machine learning decisions made during the surgical tool classification project. It captures the context, alternatives, reasoning, and consequences behind each finalized decision so that future contributors can understand why the project evolved as it did.

This is a living record. It should be updated whenever a meaningful design, workflow, data, modeling, evaluation, infrastructure, or implementation decision is proposed or finalized. Decisions are numbered sequentially and should not be deleted; a later entry should mark an outdated decision as superseded when necessary.

## Decision Entry Template

Copy the following template when adding a new decision.

### Decision `<number>`

#### Date

`<date>`

#### Status

`Proposed | Approved | Rejected | Superseded`

#### Category

`Data | Model | Training | Evaluation | Infrastructure | Documentation | Repository | Experiment Tracking | <other>`

#### Decision

`<short decision title>`

#### Background

`<problem or context that required a decision>`

#### Alternatives Considered

- `<alternative 1>`
- `<alternative 2>`

#### Decision Made

`<selected approach>`

#### Reasoning

`<reason the selected approach was chosen>`

#### Consequences

**Advantages**

- `<expected advantage>`

**Disadvantages or tradeoffs**

- `<expected disadvantage or tradeoff>`

#### Related Files

- `<file path>`

#### Related Change Log Entry

`<version or change-log entry>`

---

## Decision 001

### Date

2026-08-30

### Status

Approved

### Category

Infrastructure, Repository, Documentation

### Decision

Adopt a GitHub-centered development workflow with local development and Colab GPU execution

### Background

The project requires a workflow that supports local code review and editing, version-controlled collaboration, and access to GPU resources for machine learning training. The dataset is too large and unsuitable for normal Git tracking, and the Mac and Google Colab environments require different storage locations.

A consistent process was needed before model improvements began so that each change could be reviewed, reproduced, transferred between environments, and associated with a clear engineering record.

### Alternatives Considered

- Develop and run everything locally on the Mac, including model training.
- Develop entirely in Google Colab and use notebooks as the primary source of code.
- Transfer files manually between the Mac and Colab without a central repository.
- Store the dataset directly in GitHub with the project source code.
- Use GitHub as the central source of code while keeping environment-specific dataset copies outside Git.
- Combine multiple unrelated improvements into large commits without a structured change record.

### Decision Made

- Use GitHub as the central source repository.
- Use VS Code for local development, inspection, and review.
- Use Google Colab when GPU-backed training or evaluation is needed.
- Maintain the dataset separately on the Mac and in Google Drive.
- Exclude datasets and generated checkpoints from normal Git tracking.
- Implement improvements incrementally using one logical Git commit whenever practical.
- Maintain structured project, change-log, architecture, pipeline, repository, roadmap, and decision documentation.

### Reasoning

GitHub provides a shared, auditable source of truth for the codebase and its history. VS Code supports efficient local inspection and controlled edits, while Google Colab provides accessible GPU resources without requiring the local machine to perform expensive training.

Keeping separate dataset copies avoids committing large data files and allows each environment to access the dataset through its native storage. Incremental commits and structured documentation make changes easier to review, test, explain, and reverse if necessary.

### Consequences

**Advantages**

- Code changes are synchronized and traceable through GitHub.
- Local development and GPU execution can use the environments best suited to each task.
- Large datasets remain outside the source repository.
- Logical commits support professional review and controlled experimentation.
- The change log and decision log preserve engineering context beyond commit messages.

**Disadvantages or tradeoffs**

- Dataset copies on the Mac and Google Drive must be maintained consistently.
- Environment-specific paths must be handled through centralized configuration.
- Changes made in Colab must be committed and synchronized carefully to avoid branch conflicts.
- Reproducibility depends on recording configuration and environment details that are not yet fully automated.

### Related Files

- `README.md`
- `.gitignore`
- `docs/project_plan.md`
- `docs/change_log.md`
- `docs/project_architecture.md`
- `docs/repository_structure.md`
- `docs/pipeline_overview.md`
- `docs/project_roadmap.md`

### Related Change Log Entry

`v0.1.0 — Initial project setup`
