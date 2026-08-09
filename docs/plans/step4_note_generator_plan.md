# Step 4: Lecture Note Generator — Detailed Plan

**Status:** IMPLEMENTED for text synthesis; all 7 pre-semester priorities complete. Note generator produces structured markdown with DeepSeek V4 Pro. Production rollout awaits real course files (late August 2026). 5 doable items identified for pre-school window: laptop pipeline, expanded benchmark, batch ingestion, connectivity test, embedding A/B test. See STATUS.md for full tracking.
**Created:** 2026-07-13
**Depends on:** Step 1 (Whisper/MOSS), Step 2 (GLM-OCR/pymupdf), Step 3 (Hindsight)

---

## Goal

Takes a lecture transcript + slide PDF and generates structured, study-ready lecture notes with key concepts, formulas, examples, diagrams, and action items. Output goes to both Obsidian vault and Hindsight.

---

## 4.1 Input Sources

| Input | Source | Format |
|---|---|---|
| Lecture audio transcript | Whisper/MOSS via `transcribe.py` | Timestamped text |
| Lecture slide PDF | Professor's slides | Digital text or scanned images |
| Diagram descriptions | GLM-OCR/Qwen3-VL via `diagrams.py` | Structured text |
| Chunked facts | Hindsight via `chunker.py` + `ingest.py` | Searchable facts |

The generator combines all four into a single cohesive document.

---

## 4.2 Output Format (Obsidian Markdown)

```markdown
---
date: 2026-09-05
course: PERSONAL-ALPHA
lecture: 1
topic: Introduction to Control Systems
tags: [lecture-notes, control-theory, feedback]
sources: [Test.m4a, lecture_slides.pdf]
---

# Lecture 1: Introduction to Control Systems
**Course:** PERSONAL-ALPHA — Personal Knowledge Collection
**Date:** 2026-09-05
**Duration:** 9 min 10 s

## Key Concepts
- **Control system**: mechanism that alters future behavior toward a desired state
- **Open-loop**: input independent of output (e.g., dishwasher, sprinkler)
- **Closed-loop (feedback)**: input adjusted based on measured output
- **Transfer function**: DG/(1+DGH) — relates output to input through feedback

## Definitions
| Term | Definition |
|------|-----------|
| Plant | System to be controlled |
| Controller | Processes error signal into corrective input |
| Sensor | Measures actual system output |
| Reference signal | Desired state / commanded value |
| Error term | Difference between reference and measured output |

## Formulas
- Error: E = V - H·Y
- Output: Y = D·G·E
- Transfer function: Y/V = DG/(1+DGH)

## Examples (from lecture)
### Dishwasher (open-loop → closed-loop)
- Open-loop: fixed wash time regardless of cleanliness
- Closed-loop: sensor measures plate cleanliness, controller adjusts wash time
- Reference: desired cleanliness level

### Car cruise control
- Sensor: speedometer
- Reference: set speed (e.g., 100 mph)
- On hill: speed drops → positive error → controller increases throttle
- On downhill: speed increases → negative error → controller reduces throttle

## Diagrams
### Block Diagram — Feedback Control System
![Block Diagram](diagrams/lecture1_block.png)
> V → Σ → D → G → Y, with feedback Y → H → Σ. Transfer function: Y/V = DG/(1+DGH)

## Professor Notes
- "Can we make any plant G behave like anything we choose just by adding feedback?"
- "Can we turn a Pinto into a Ferrari just by applying more gas?"
- These questions hint at fundamental limitations of feedback control

## Action Items
- [ ] Review Chapter 3: Feedback Control fundamentals
- [ ] Practice block diagram reduction
- [ ] Solve HW1 Problems 1-3

## Related Topics
- [[Lecture 2 - Frequency Response]]
- [[Transfer Functions]]
- [[Block Diagram Reduction]]
```

---

## 4.3 Generation Pipeline

```
Transcript (Whisper/MOSS)
        │
        ▼
Slide PDF (GLM-OCR/pymupdf)
        │
        ├── Text facts ──────────────┐
        ├── Diagram descriptions ────┤
        └── Chunked lecture facts ───┤
                                     ▼
                            LLM synthesis prompt
                            "Combine these into
                             structured lecture notes"
                                     │
                                     ▼
                            Structured Markdown
                            (Obsidian format)
                                     │
                            ┌────────┴────────┐
                            ▼                 ▼
                      Obsidian vault    Hindsight retain
                      (browsable)       (searchable)
```

---

## 4.4 LLM Synthesis Prompt

```
You are generating structured lecture notes for {course_name} ({course_code}).

## Source Material

### Transcript (from {engine}):
{transcript_text}

### Slide Content (OCR'd):
{slide_text}

### Diagram Descriptions:
{diagram_descriptions}

### Key Facts (extracted earlier):
{chunked_facts}

## Instructions

Generate comprehensive lecture notes in this exact format:

1. **Key Concepts** — bullet list of main ideas (3-7 items)
2. **Definitions** — table of terms and definitions
3. **Formulas** — all equations mentioned, in LaTeX notation
4. **Examples** — professor's examples with explanations
5. **Diagrams** — descriptions of visual content from slides
6. **Professor Notes** — interesting quotes, hints, or questions from the professor
7. **Action Items** — study tasks based on the lecture content
8. **Related Topics** — wikilinks to related lectures or concepts

Rules:
- Be concise but complete — capture everything important
- Use Obsidian markdown syntax ([[wikilinks]], tables, code blocks)
- Preserve professor examples exactly as stated
- Include LaTeX for all formulas: $formula$ for inline, $$formula$$ for display
- Don't invent content — only use what's in the source material
- Tag action items with checkboxes: - [ ]
```

---

## 4.5 Implementation Steps

| Step | Task | Details |
|---|---|---|
| 4.5.1 | Build `note_generator.py` | Script that reads transcript + facts + diagrams → calls LLM → outputs markdown |
| 4.5.2 | Build note template | Obsidian-compatible markdown template with frontmatter |
| 4.5.3 | Test with Test.m4a | Generate notes from the control systems lecture |
| 4.5.4 | Test with Freq Response PDF | Generate notes from the OCR'd frequency response slides |
| 4.5.5 | Integrate with Obsidian | Save output to `~/obsidian-vault/Personal/{course}/` |
| 4.5.6 | Retain into Hindsight | Store generated notes as a summary fact |
| 4.5.7 | Add wikilink generation | Auto-link related lectures and topics |
| 4.5.8 | Test batch generation | Process multiple lectures at once |

---

## 4.6 Obsidian Integration

**Vault structure:**
```
~/obsidian-vault/
├── Personal/
│   ├── PERSONAL-ALPHA/
│   │   ├── Lecture 01 - Intro to Control Systems.md
│   │   ├── Lecture 02 - Frequency Response.md
│   │   ├── ...
│   │   └── diagrams/
│   │       ├── lecture1_block.png
│   │       └── lecture2_bode.png
│   ├── PERSONAL-BETA/
│   │   └── ...
│   ├── PERSONAL-RESEARCH/
│   │   └── ...
│   └── _templates/
│       └── lecture_template.md
```

**Auto-sync:**
- Each generated note goes to `~/obsidian-vault/Personal/{course}/`
- Diagram images extracted from PDFs go to `diagrams/` subfolder
- Wikilinks auto-generated to connect related lectures
- Frontmatter with tags for Obsidian search and Dataview queries

---

## 4.7 Testing Checklist

- [ ] Generate notes from Test.m4a transcript (single speaker, no slides)
- [ ] Generate notes from Freq Response Part I (scanned PDF, diagrams)
- [ ] Generate notes from Freq Response Part II (scanned PDF, Nyquist)
- [ ] Verify Key Concepts section has 3-7 items
- [ ] Verify Formulas section captures all equations
- [ ] Verify Examples section preserves professor examples
- [ ] Verify Diagrams section describes OCR'd visual content
- [ ] Verify Obsidian frontmatter is valid
- [ ] Verify wikilinks point to existing or future notes
- [ ] Verify notes save to correct Obsidian vault path
- [ ] Verify summary fact retained into Hindsight

---

## 4.8 Usage

```powershell
# Generate notes from transcript + PDF
python note_generator.py transcript.txt slides.pdf \
  --course "PERSONAL-ALPHA" --lecture 1 \
  --topic "Intro to Control Systems" \
  --output ~/obsidian-vault/Personal/PERSONAL-ALPHA/Lecture01.md

# Generate from existing facts (already in Hindsight)
python note_generator.py --from-hindsight \
  --course "FREQ_RESP" --lecture 1 \
  --output ~/obsidian-vault/Personal/FREQ_RESP/Lecture01.md
```
