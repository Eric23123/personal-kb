# Step 3: Hindsight Integration — Detailed Plan

**Status:** IMPLEMENTED with fail-closed bank isolation and structured provenance. All 7 pre-semester priorities complete. Hindsight production bank (`hermes-history`) guarded by `allow_production=True` requirement. Batch ingestion (#D3) identified as doable before school. See STATUS.md for full tracking.
**Created:** 2026-07-13
**Depends on:** Step 1 (Whisper/MOSS), Step 2 (pymupdf/GLM-OCR/Qwen3-VL)

---

## Goal

Ingest lecture transcripts, OCR'd documents, and diagram descriptions into Hindsight memory with structured metadata, enabling natural language search across all course material.

---

## 3.1 Metadata Schema

Every fact retained into Hindsight needs structured metadata for filtering and retrieval.

**Schema:**
```
{
  "course": "PERSONAL-ALPHA",           // Course code
  "course_name": "Systems Engineering",
  "lecture": 3,                      // Lecture number (0 for textbook/hw)
  "source_type": "lecture|textbook|homework|exam|notes|diagram",
  "topic": "closed-loop control",   // Topic/chapter name
  "date": "2026-09-05",             // When it was taught/assigned
  "source_file": "Test.m4a",        // Original filename
  "engine": "whisper|moss|glm-ocr|pymupdf|qwen3-vl" // Extraction method
}
```

**Source types:**
| Type | Description | Example |
|---|---|---|
| `lecture` | Transcribed audio | "closed-loop control uses feedback..." |
| `textbook` | OCR'd textbook chapter | "Chapter 3: Feedback Control" |
| `homework` | OCR'd homework solutions | "Problem 2.1: Transfer function..." |
| `exam` | OCR'd exam solutions | "Midterm Q5: Root locus analysis" |
| `notes` | Your own study notes | "Key takeaway: feedback reduces error" |
| `diagram` | Vision model description of images | "Block diagram: V→Σ→D→G→Y with feedback through H" |

---

## 3.2 Fact Chunking Strategy

Hindsight stores discrete facts, not raw documents. Long transcripts need to be chunked into self-contained, searchable units.

**Chunking rules:**
1. **Semantic boundaries** — split at topic changes, not arbitrary word counts
2. **Self-contained** — each fact should make sense without surrounding context
3. **One concept per fact** — "closed-loop control uses feedback to reduce error" vs dumping a whole paragraph
4. **Include key terms** — include technical vocabulary for searchability
5. **Preserve examples** — professor examples (dishwasher, car cruise control) are valuable

**Chunk size target:** 50-200 words per fact

**Chunking pipeline:**
```
Raw transcript → Segment by topic → Extract key concepts → Format as facts
```

---

## 3.3 Diagram/Picture Handling

Engineering courses rely heavily on visual content. Text extraction alone misses diagrams, charts, and figures.

### Pipeline

```
PDF page
  ├── Has text? → pymupdf (text extraction)
  ├── Has images/diagrams? → Extract images → Vision model → Structured description
  └── Is scanned? → GLM-OCR (text + layout recognition)
```

### Image Extraction from PDFs

```python
import pymupdf

doc = pymupdf.open("lecture_slides.pdf")
for page_num, page in enumerate(doc):
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = pymupdf.Pixmap(doc, xref)
        img_path = f"temp/page{page_num}_img{img_index}.png"
        pix.save(img_path)
        # Process with vision model
```

### Diagram-Specific Prompts for Qwen3-VL

| Diagram Type | Prompt |
|---|---|
| **Block diagram** | "Describe this block diagram: identify all blocks, their labels, connections/arrows, input/output flow, and any equations shown." |
| **Flowchart** | "Describe this flowchart: identify each step, decision point, arrows showing flow, start/end points." |
| **Circuit diagram** | "Describe this circuit: identify components (resistors, capacitors, etc.), connections, values, and the overall circuit function." |
| **Graph/plot** | "Describe this graph: identify axes labels, units, data series, trends, key data points, and any equations fitted to the data." |
| **Equation/formula** | "Read all equations and formulas in this image. Output in standard mathematical notation." |
| **MATLAB/Simulink** | "Describe this simulation output: identify the system being simulated, parameters shown, key observations from the plots." |
| **General** | "Describe this technical diagram in detail: what type is it, what are the components, what relationships are shown, what labels/values appear?" |

### Storing Diagram Facts in Hindsight

Each diagram gets stored as a fact with `source_type: diagram`:

```
Content: "Block diagram of feedback control system: Reference signal V enters summing junction Σ, which produces error E = V - H*Y. Error feeds into Controller D, output goes to Plant G, producing output Y. Sensor H feeds Y back to summing junction. Transfer function: Y/V = DG/(1+DGH)."
Metadata: {
  "course": "PERSONAL-ALPHA",
  "lecture": 1,
  "source_type": "diagram",
  "topic": "closed-loop control",
  "diagram_type": "block_diagram"
}
```

### Linking Text and Diagram Facts

When a transcript mentions a diagram and the slides contain it, both facts are stored and linked via shared metadata (same course, lecture, topic). This enables:

- Search "block diagram" → returns both verbal explanation AND visual description
- Search "transfer function" → returns equation from text AND diagram showing the derivation

---

## 3.4 Ingestion Pipeline (Complete)

```
Audio file ──→ Whisper/MOSS ──→ Raw transcript
                                    │
PDF file ──→ pymupdf (digital text)
         ──→ GLM-OCR (scanned pages)
         ──→ Qwen3-VL (diagrams/images)
                                    │
                                    ▼
                            LLM chunking prompt
                            "Extract key concepts,
                             formulas, examples,
                             diagram descriptions
                             as discrete study facts"
                                    │
                                    ▼
                            Hindsight retain()
                            (with metadata)
                                    │
                                    ▼
                            PostgreSQL + embeddings
                            (searchable via recall)
```

### LLM Chunking Prompt Template

```
You are processing course material for {course_name} ({course_code}).
Lecture #{lecture}: {topic}
Source type: {source_type}

Extract key study facts from this content. Each fact should be:
- Self-contained (understandable without context)
- Focused on one concept, formula, example, or diagram
- Include technical terms for searchability
- Preserve professor examples and explanations

For diagrams/images, describe:
- The type of diagram
- All components and their labels
- Connections/relationships
- Any equations or values shown

Format each fact as:
FACT: <the fact>
TOPIC: <topic tag>
TYPE: <text|formula|example|diagram>

Content:
{segment_text_or_diagram_description}
```

---

## 3.5 Hindsight API Integration

**Retain (store facts):**
```python
hindsight_retain(
    content="Closed-loop control uses feedback to drive error to zero. Transfer function: DG/(1+DGH).",
    context="PERSONAL-ALPHA Lecture 1: closed-loop control",
    tags=["systems-engineering", "control-theory", "feedback", "transfer-function"]
)
```

**Recall (search facts):**
```python
hindsight_recall(query="transfer function feedback control system")
# Returns relevant facts with metadata
```

**Reflect (synthesize insights):**
```python
hindsight_reflect(query="What are the key differences between open-loop and closed-loop control?")
# Synthesizes a comprehensive answer from stored facts
```

---

## 3.6 Implementation Steps

| Step | Task | Details |
|---|---|---|
| 3.6.1 | Define course catalog | Config file mapping course codes to names, semesters, expected lectures |
| 3.6.2 | Build chunking script | Python script: transcript → LLM → structured facts |
| 3.6.3 | Build diagram extraction script | PDF → extract images → Qwen3-VL → diagram descriptions |
| 3.6.4 | Build ingestion script | Reads chunks → calls `hindsight_retain()` with metadata |
| 3.6.5 | Test with Test.m4a | Ingest control systems lecture, verify facts stored correctly |
| 3.6.6 | Test diagram extraction | Find a lecture PDF with diagrams, extract and describe images |
| 3.6.7 | Test recall queries | Search "transfer function", "open-loop", "block diagram" |
| 3.6.8 | Test reflect | Ask "summarize lecture 1" — verify synthesis from facts |
| 3.6.9 | Build batch ingestion | Process multiple files at once |
| 3.6.10 | Document patterns | Save working retain/recall/reflect patterns as a skill |

---

## 3.7 Testing Checklist

- [ ] Retain a single fact → verify it appears in recall
- [ ] Retain 10 facts from Test.m4a → verify all searchable
- [ ] Recall "transfer function" → returns relevant fact
- [ ] Recall "dishwasher example" → returns open-loop example
- [ ] Extract diagrams from a lecture slide PDF
- [ ] Retain diagram descriptions → verify searchable
- [ ] Recall "block diagram" → returns diagram description
- [ ] Reflect "summarize control systems intro" → synthesizes from facts
- [ ] Filter by metadata: only facts from "PERSONAL-ALPHA"
- [ ] Filter by metadata: only "diagram" source_type
- [ ] Ingest second lecture → verify cross-lecture search works
- [ ] Verify facts don't duplicate on re-ingestion

---

## 3.8 Course Catalog Config

```yaml
courses:
  PERSONAL-ALPHA:
    name: "Personal Knowledge Collection"
    semester: "Fall 2026"
    lectures: 28
    textbook: "Systems Engineering Handbook"
  PERSONAL-BETA:
    name: "Systems Engineering Project Management"
    semester: "Fall 2026"
    lectures: 28
  PERSONAL-RESEARCH:
    name: "Introduction to Engineering Probability"
    semester: "Fall 2026"
    lectures: 28
```
