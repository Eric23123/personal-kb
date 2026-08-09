# Personal KB — Course Onboarding Guide

**Version:** 1.0.0 · **Last updated:** 2026-07-21

How to onboard a new Personal course into the knowledge base. Follow this guide
in order — every step is gated by the checks listed.

---

## 1. Naming Conventions

### 1.1 Course codes

Use the official Personal course code with a space: `PERSONAL-ALPHA`, `PERSONAL-RESEARCH`,
`PERSONAL-EPSILON`. Keep the collection identifier stable; do not abbreviate or
lowercase unless slugified for URIs.

### 1.2 Slugs (URI-safe identifiers)

Derived from the course code by lowercasing and replacing spaces with hyphens:

| Course code  | Slug          |
|-------------|---------------|
| PERSONAL-ALPHA  | `personal-alpha`  |
| PERSONAL-RESEARCH   | `personal-research`   |
| PERSONAL-EPSILON  | `personal-epsilon`  |

### 1.3 Source files

Place course material in a dedicated directory under the project root:

```
<project_root>/
  courses/
    <course_slug>/
      lectures/
        lecture_01_slides.pdf
        lecture_01_transcript.txt
        lecture_02_slides.pdf
        ...
      textbooks/
        <author>_<title_slug>.pdf
      homework/
        hw1.pdf
        hw2.pdf
      exams/
        midterm.pdf
        final.pdf
      derived/
        lecture_01_ocr.txt        # OCR output from scanned slides
        lecture_01_diagrams.json  # Extracted diagram descriptions
        notes/
          lecture_01_notes.md     # Generated Obsidian notes
```

### 1.4 Manifest entries

Every source gets an entry in `config/source_manifest.json` (or a per-course
manifest). Required fields:

```json
{
  "source_path": "courses/personal-alpha/lectures/lecture_01_slides.pdf",
  "source_hash": "<full 64-char SHA-256>",
  "source_type": "slides",
  "course": "PERSONAL-ALPHA",
  "lecture": 1,
  "semester": "FA26",
  "date": "2026-08-26",
  "corpus": "course"
}
```

**Field rules:**

| Field | Required | Notes |
|-------|----------|-------|
| `source_path` | ✅ | Relative to project root |
| `source_hash` | ✅ | Full SHA-256 of file bytes |
| `source_type` | ✅ | One of: `textbook`, `slides`, `transcript`, `homework`, `exam`, `ocr_text`, `diagram_description`, `synthesis` |
| `course` | ✅ | Official code with space |
| `lecture` | ⚠️ | Integer 1-1000. Omit for homework, exams, and multi-lecture textbooks |
| `semester` | Recommended | `FA26`, `SP27`, etc. |
| `date` | Recommended | ISO date YYYY-MM-DD |
| `corpus` | ✅ | `course` for real material, `test` for TEST fixtures |

**`source_type` catalog:**

| Type | Use for | Has lecture? |
|------|---------|-------------|
| `textbook` | Textbook chapters, reference PDFs | Usually no (multi-lecture) |
| `slides` | Lecture slide decks | ✅ |
| `transcript` | Whisper/MOSS transcript output | ✅ |
| `homework` | Problem sets | No (use `assignment` field) |
| `exam` | Exams, quizzes | No |
| `ocr_text` | OCR output from scanned documents | Varies |
| `diagram_description` | Qwen3-VL extracted descriptions | ✅ |
| `synthesis` | DeepSeek V4 Pro generated notes | ✅ |

### 1.5 Derived/OCR files

When a source is scanned (no digital text), add the derived text:

```json
{
  "source_path": "courses/personal-alpha/lectures/lecture_02_ocr.txt",
  "source_hash": "<sha256 of the .txt file>",
  "source_type": "ocr_text",
  "course": "PERSONAL-ALPHA",
  "lecture": 2,
  "semester": "FA26",
  "corpus": "course",
  "derived_text_path": "courses/personal-alpha/lectures/lecture_02_scanned.pdf",
  "derived_hash": "<sha256 of the original scanned PDF>"
}
```

---

## 2. Course Namespaces

### 2.1 OpenViking URI structure

Every resource is indexed under `viking://resources/personal-kb/`. The URI
format is:

```
viking://resources/personal-kb/<course_slug>/<source_type_slug>/<stem>-<12hexdigest>
```

Examples:
```
viking://resources/personal-kb/personal-alpha/slides/lecture-01-slides-a1b2c3d4e5f6
viking://resources/personal-kb/personal-alpha/transcript/lecture-01-transcript-f6e5d4c3b2a1
viking://resources/personal-kb/personal-research/homework/homework-1-0a1b2c3d4e5f
```

The 12-character hex digest is derived from SHA-256 of the course + source_type
+ source_hash identity string. It ensures content-addressed identity: changing
the file changes the URI.

### 2.2 Namespace registry

See `config/course_namespaces.yaml` for the authoritative mapping of course
codes to slugs and namespaces. Update it when onboarding a new course.

### 2.3 Namespace isolation

- Course material: `viking://resources/personal-kb/<course_slug>/` — indexed in OpenViking
- Personal study state: Hindsight with tags `course:<code>`, `scope:course|assessment`
- Cross-course references: OpenViking search across the personal-kb namespace

---

## 3. Obsidian Output Structure

### 3.1 Vault location

Generated notes land in `~/obsidian-vault/Personal/`. The directory structure:

```
~/obsidian-vault/Personal/
  PERSONAL-ALPHA/
    Lectures/
      Lecture 01 - Introduction.md
      Lecture 02 - Requirements.md
      ...
    Notes/
      Transfer Functions.md
      Stability Criteria.md
      ...
    Homework/
      HW1 Review.md
      HW2 Review.md
    Exams/
      Midterm Review.md
  PERSONAL-RESEARCH/
    Lectures/
      ...
    Notes/
      ...
```

### 3.2 Note template

Every generated note follows this structure:

```markdown
---
course: PERSONAL-ALPHA
lecture: 1
source_type: slides
source_uri: viking://resources/personal-kb/personal-alpha/slides/lecture-01-a1b2c3
extraction_engine: deepseek-v4-pro
generated: 2026-08-26T15:30:00Z
tags:
  - personal
  - personal-alpha
  - lecture-01
---

# Lecture 1: Introduction to Personal Knowledge Collection

## Key Concepts

- **Concept one**: Explanation with source citation.
- **Concept two**: Explanation with source citation.

## Definitions

| Term | Definition |
|------|-----------|
| Term A | Definition from lecture. |

## Diagrams

*[Diagram description or reference to source]*

## Action Items

- Review topic X before next lecture.
- Practice deriving Y.

---

*Source: `viking://resources/personal-kb/personal-alpha/slides/lecture-01-a1b2c3`*
```

### 3.3 Installed vault artifacts

The active vault is configured by the `--vault-root` argument; a common local
default is `~/obsidian-vault/`.

- Reusable templates: `Templates/Personal Lecture Note.md`, `Templates/Personal Concept Note.md`, `Templates/Personal Assessment Review.md`, and `Templates/Personal Course Index.md`.
- Templates core-plugin configuration: `.obsidian/core-plugins.json` enables `templates`; `.obsidian/templates.json` sets the default folder to `Templates`.
- Registered-course index: `Personal/Registered Courses.md`.
- Each registered course has physical `Lectures/`, `Notes/`, `Homework/`, `Exams/`, and `Diagrams/` folders plus a `Course Index.md`.
- The current registered plan is PERSONAL-ALPHA, PERSONAL-BETA, PERSONAL-PROJECT, PERSONAL-RESEARCH, PERSONAL-GAMMA, PERSONAL-DELTA, PERSONAL-EPSILON, PERSONAL-ZETA, and PERSONAL-ETA.

### 3.4 Frontmatter fields

| Field | Required | Notes |
|-------|----------|-------|
| `course` | ✅ | Official course code |
| `lecture` | ⚠️ | Omit for non-lecture notes |
| `source_type` | ✅ | The source material type |
| `source_uri` | ✅ | OpenViking URI of source |
| `extraction_engine` | ✅ | `deepseek-v4-pro` for synthesis, `whisper-large-v3-turbo` for transcripts |
| `generated` | ✅ | ISO 8601 timestamp |
| `tags` | ✅ | At minimum: `personal`, `<course-slug>`, `<lecture-slug>` |

### 3.5 Naming rules

- Lecture notes: `Lecture NN - <Title>.md` (two-digit zero-padded lecture number)
- Concept notes: `<Concept Name>.md` (title case, no lecture prefix)
- Homework reviews: `HW<N> Review.md`
- Exam reviews: `<Exam Name> Review.md`

---

## 4. Onboarding Checklist

For each new course, complete in order:

- [ ] **Register the course** — Add entry to `config/course_namespaces.yaml`
- [ ] **Create directory structure** — `courses/<course_slug>/` with subdirs
- [ ] **Write manifest** — Create `config/manifests/<course_slug>.json` or add to `config/source_manifest.json`
- [ ] **Compute hashes** — Run `python scripts/ingestion/source_manifest.py validate --manifest <path>` to verify all hashes
- [ ] **Run preflight** — `python scripts/ingestion/preflight.py` (Gate A)
- [ ] **Run e2e gate offline** — `python test_runs/e2e_gate.py` (Gate A)
- [ ] **Run e2e gate live** — `python test_runs/e2e_gate.py --live` (Gate B)
- [ ] **Ingest** — `python scripts/ingestion/source_manifest.py` (or call `ingest_manifest()`)
- [ ] **Verify** — Search for a known fact, check source read-back
- [ ] **Generate notes** — Run note generator for each lecture
- [ ] **Sync to Obsidian** — Run `python scripts/ops/obsidian_sync.py --once` or keep `--watch` running; verify notes appear under the registered course path

---

## 5. Quick Reference

| What | Where |
|------|-------|
| Source files | `courses/<slug>/` |
| Manifest | `config/source_manifest.json` or `config/manifests/<slug>.json` |
| Namespace registry | `config/course_namespaces.yaml` |
| Manifest template | `config/course_manifest_template.json` |
| Obsidian output | `~/obsidian-vault/Personal/<Course Code>/` |
| Pre-ingestion gate | `python scripts/ingestion/preflight.py` |
| E2E gate | `python test_runs/e2e_gate.py [--live]` |
| Ingestion log | `logs/ingestion_<date>.jsonl` |
| Snapshot before/after | `python scripts/ops/resource_inventory.py snapshot` |
| Cleanup test resources | `python scripts/ops/test_cleanup.py --test-only` |
