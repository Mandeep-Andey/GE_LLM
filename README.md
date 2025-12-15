# GE_LLM — Character Interaction Network Extraction for *Middlemarch*

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An LLM-powered pipeline for extracting, validating, and visualizing character interaction networks from George Eliot's *Middlemarch*.**

---

## 📖 Project Overview

**GE_LLM** is a sophisticated Natural Language Processing (NLP) pipeline that uses Large Language Models (LLMs) to automatically extract character interactions from literary texts. The system:

1. **Processes** the full text of *Middlemarch* (8 books, ~80+ chapters)
2. **Extracts** character interactions using a local LLM (via Ollama)
3. **Validates** extracted entities against a curated canonical character list with alias resolution
4. **Builds** a weighted graph representing the character network
5. **Analyzes** the graph using network science metrics (centrality, community detection)
6. **Visualizes** the results as interactive HTML network diagrams

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GE_LLM Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  Raw Text    │───▶│   Chunking   │───▶│   LLM Call   │                  │
│   │  (book_X.txt)│    │  (Adaptive)  │    │  (Ollama)    │                  │
│   └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                  │                          │
│                                                  ▼                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  Canonical   │◀───│   Alias      │◀───│   Pydantic   │                  │
│   │  Validation  │    │   Resolution │    │   Parsing    │                  │
│   └──────┬───────┘    └──────────────┘    └──────────────┘                  │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  NetworkX    │───▶│   Analysis   │───▶│   PyVis      │                  │
│   │  Graph (.gml)│    │   Reports    │    │   HTML Viz   │                  │
│   └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

```
GE_LLM/
├── src/                          # Core Python package
│   ├── __init__.py               # Package initializer
│   ├── character_mapper.py       # Alias-to-canonical name resolution
│   ├── data_preprocessor.py      # Text loading utilities
│   ├── graph_manager.py          # Graph construction, analysis & visualization
│   ├── llm_client.py             # Ollama API client with resilient parsing
│   ├── prompt_manager.py         # LLM prompt templates and formatting
│   ├── schemas.py                # Pydantic models for type validation
│   ├── settings.py               # Configuration loader (YAML → Python)
│   └── utils.py                  # (Reserved for future utilities)
│
├── data/                         # Input data
│   └── Middlemarch/              # Raw book text files
│       ├── book_1.txt
│       ├── book_2.txt
│       └── ...                   # Books 3-8
│
├── llm_results/                  # LLM extraction output (JSON per chapter)
│   ├── book_1/
│   │   ├── chapter_000.json
│   │   ├── chapter_001.json
│   │   └── ...
│   └── book_2/ ...
│
├── graph_artifacts/              # Serialized NetworkX graphs (.gml)
│   ├── book_1_graph.gml
│   ├── book_2_graph.gml
│   └── ...
│
├── analysis_reports/             # Final outputs (text reports + HTML visualizations)
│   ├── book_1/
│   │   ├── analysis_report.txt
│   │   └── book_1_network.html
│   └── book_2/ ...
│
├── run_llm_extraction.py         # Main LLM extraction script
├── build_graph.py                # Graph construction from LLM results
├── analyze_graph.py              # Graph analysis and visualization
├── analyze_all.sh                # Batch processing script for all books
│
├── test_llm.py                   # LLM client test suite
├── test_llm_context.py           # Context-awareness test suite
│
├── config.yaml                   # Central configuration file
├── char_alias.json               # Canonical character list with aliases (80+ characters)
├── pyproject.toml                # Python project metadata & dependencies
└── uv.lock                       # Dependency lock file
```

---

## 📦 Module Documentation

### `src/settings.py` — Configuration Management

**Purpose:** Centralizes all configuration by loading `config.yaml` and exposing settings as typed Python attributes.

**Key Class:**
```python
class Settings:
    def __init__(self, config_path: str = "config.yaml")
```

**Exposed Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `PROJECT_ROOT` | `Path` | Root directory of the project |
| `BOOKS_DIR` | `Path` | Directory containing raw book text files |
| `CHARACTER_FILE` | `Path` | Path to `char_alias.json` |
| `RESULTS_DIR` | `Path` | Output directory for LLM extraction results |
| `GRAPH_ARTIFACTS_DIR` | `Path` | Output directory for `.gml` graph files |
| `LLM_MODEL` | `str` | Ollama model name (e.g., `"qwen3:8b"`) |
| `LLM_HOST` | `str` | Ollama server URL (e.g., `"http://localhost:11434"`) |
| `FAST_TOKENIZER` | `str` | HuggingFace tokenizer for token counting |
| `CHUNK_TOKEN_LIMIT` | `int` | Maximum tokens per chunk sent to LLM (default: 256) |
| `CHUNK_OVERLAP_SENTENCES` | `int` | Sentence overlap between chunks for context continuity |
| `TOP_N_ANALYSIS` | `int` | Number of top results to show in reports |

---

### `src/character_mapper.py` — Entity Resolution

**Purpose:** Resolves character mentions (including nicknames, titles, and informal references) to their canonical names for consistent graph construction.

**Key Class:**
```python
class CharacterMapper:
    def __init__(self, file_path: str)
    def get_canonical_name(self, mention: str) -> Optional[str]
```

**How It Works:**
1. Loads `char_alias.json` containing 80+ characters with their aliases
2. Builds a case-insensitive lookup dictionary mapping every alias to its canonical name
3. Provides `get_canonical_name()` to resolve any mention

**Example Mappings:**
| Mention | Canonical Name |
|---------|----------------|
| `"Dodo"` | `"Dorothea Brooke"` |
| `"Mrs. Casaubon"` | `"Dorothea Brooke"` |
| `"the Vicar"` | `"Reverend Camden Farebrother"` |
| `"uncle"` | `"Mr. Arthur Brooke"` |
| `"Rosy"` | `"Rosamond Vincy"` |

The alias file contains rich mappings for formal/informal names, titles, nicknames, and even evolving names (e.g., "Miss Brooke" → "Mrs. Casaubon" → "Mrs. Ladislaw" for Dorothea).

---

### `src/data_preprocessor.py` — Text Loading

**Purpose:** Provides utility functions for loading raw text files from the data directory.

**Key Function:**
```python
def load_books(directory_path: str) -> Dict[str, str]
```

**Behavior:**
- Scans the specified directory for all `.txt` files
- Returns a dictionary mapping filename → full text content
- Files are sorted alphabetically for consistent processing order

---

### `src/schemas.py` — Data Validation Models

**Purpose:** Defines Pydantic models that enforce strict typing and validation on LLM outputs.

**Models:**

```python
# Controlled vocabulary for interaction types
InteractionType = Literal["Direct Dialogue", "Physical Action", "Observation", "Memory/Reference"]

class Interaction(BaseModel):
    """A single character-to-character interaction extracted from text."""
    character_1: str
    character_2: str
    interaction_type: InteractionType  # Enforced vocabulary
    evidence_snippet: str              # Exact quote from source text (3-5 words)

class LLMInteractionOutput(BaseModel):
    """The complete response structure expected from the LLM."""
    interactions: List[Interaction]
```

**Validation Benefits:**
- Rejects malformed LLM outputs automatically
- Enforces the 4-type interaction taxonomy
- Ensures evidence snippets are captured for auditability

---

### `src/prompt_manager.py` — Prompt Engineering

**Purpose:** Constructs carefully engineered prompts that guide the LLM to produce structured, valid JSON outputs.

**Key Class:**
```python
class PromptManager:
    def __init__(self, canonical_character_list: List[str])
    def create_interaction_prompt(self, paragraph_text: str, active_characters: List[str] = None) -> str
```

**Prompt Structure:**
The generated prompt includes:
1. **Task definition** — Clear instruction to extract character interactions
2. **Strict rules** — JSON schema, canonical name requirements, interaction type vocabulary
3. **Few-shot example** — A complete input/output example for the LLM to follow
4. **Context buffer** — Recently mentioned characters to help resolve pronouns
5. **Valid characters list** — Complete canonical character list injected into prompt
6. **Target paragraph** — The actual text to analyze

**Context Awareness:**
The `active_characters` parameter allows passing recently mentioned characters from previous chunks, enabling the LLM to correctly resolve pronouns like "she" or "he" when the antecedent appeared in an earlier chunk.

---

### `src/llm_client.py` — LLM Communication

**Purpose:** Handles all communication with the local Ollama LLM server, including request formatting, response parsing, and error recovery.

**Key Class:**
```python
class LLMClient:
    def __init__(self, host: str)
    def get_llm_response(self, model_name: str, prompt: str) -> Optional[LLMInteractionOutput]
```

**Resilient Parsing Strategy:**

The client implements a **"Trust, but Verify"** approach:

1. **Request** — Sends prompt to Ollama's `/api/generate` endpoint with `format: "json"`
2. **Parse JSON** — Decodes the raw response string
3. **Extract interactions** — Gets the `interactions` list from the response
4. **Individual validation** — Validates each interaction object against the Pydantic schema
5. **Key healing** — Fixes common LLM typos (trailing commas, extra spaces in keys)
6. **Graceful degradation** — Skips malformed interactions while preserving valid ones

**Error Handling:**
- Connection failures → Returns `None`, logged to console
- Invalid JSON → Returns `None`, logs the raw output for debugging
- Missing `interactions` key → Returns `None` with warning
- Malformed individual interactions → Skipped with warning, valid ones preserved

---

### `src/graph_manager.py` — Graph Construction & Analysis

**Purpose:** The core module for building, analyzing, and visualizing character interaction networks using NetworkX and PyVis.

**Key Class:**
```python
class GraphManager:
    def __init__(self, edges: List[Tuple[str, str]])
    
    @classmethod
    def from_gml(cls, gml_path: Path) -> 'GraphManager'
    
    def generate_full_analysis_report(self, top_n: int = 10) -> str
    def generate_chapter_wise_report(self, chapter_data: Dict[int, List[...]], top_n: int = 5) -> str
    def save_interactive_visualization(self, output_path: Path)
```

**Graph Construction:**
- Takes a list of `(character_1, character_2)` tuples
- Counts edge frequencies using `collections.Counter`
- Creates a weighted, undirected NetworkX graph

**Analysis Capabilities:**

| Metric | Description |
|--------|-------------|
| **Degree Centrality** | Characters with the most direct connections |
| **Betweenness Centrality** | Characters who bridge different social groups |
| **Eigenvector Centrality** | Characters connected to other well-connected characters |
| **Community Detection** | Groups characters into social clusters (Louvain algorithm) |
| **Chapter-wise Analysis** | Tracks which characters dominate each chapter |

**Visualization Features:**
- Interactive HTML using PyVis
- Force-directed layout (ForceAtlas2)
- Node size scaled by degree centrality
- Node color coded by community membership
- Tooltips showing character name, community, and interaction count
- Legend for community colors
- Interactive physics controls

---

## 🛠️ Top-Level Scripts

### `run_llm_extraction.py` — Main Extraction Pipeline

**Purpose:** The primary entry point that orchestrates the entire LLM extraction process.

**Usage:**
```bash
uv run run_llm_extraction.py [--force-rerun]
```

**Pipeline Stages:**

1. **Setup**
   - Loads configuration from `config.yaml`
   - Downloads NLTK punkt tokenizer (with SSL workaround)
   - Initializes HuggingFace tokenizer for token counting

2. **Data Loading**
   - Loads all book text files from `data/Middlemarch/`
   - Splits each book into chapters using regex: `Chapter \d+`

3. **Adaptive Chunking**
   ```python
   def create_adaptive_chunks(sentences, tokenizer, token_limit, overlap_sentences)
   ```
   - Splits chapter text into sentences (NLTK)
   - Groups sentences into chunks that fit within `CHUNK_TOKEN_LIMIT` (256 tokens)
   - Maintains sentence overlap for context continuity

4. **LLM Processing**
   - Iterates through each chunk
   - Maintains an "Active Character Buffer" (last 5 mentioned characters)
   - Sends context-aware prompts to the LLM
   - Collects and validates responses

5. **Deduplication**
   - Removes duplicate interactions (same characters + same evidence snippet)
   - Uses character-order-independent comparison

6. **Output**
   - Saves JSON files per chapter to `llm_results/book_X/chapter_XXX.json`

**Incremental Processing:**
- Skips chapters that already have result files
- Use `--force-rerun` to start fresh

---

### `build_graph.py` — Graph Artifact Builder

**Purpose:** Constructs a serialized NetworkX graph from the LLM extraction results.

**Usage:**
```bash
uv run build_graph.py book_1
```

**Process:**
1. Loads all chapter JSON files for the specified book
2. For each interaction:
   - Resolves both character names to canonical forms
   - Skips self-loops and invalid characters
   - Accumulates edge weights and metadata
3. Attaches rich metadata to edges:
   ```python
   {
       "type": "Direct Dialogue",
       "sentiment": "Neutral",  
       "location": "Unknown",
       "evidence": "Come here, Dorothea"
   }
   ```
4. Serializes to GML format (`.gml`) for portability

---

### `analyze_graph.py` — Analysis & Visualization

**Purpose:** Generates comprehensive analysis reports and interactive visualizations.

**Usage:**
```bash
uv run analyze_graph.py book_1
```

**Outputs:**
1. **Text Report** (`analysis_report.txt`)
   - Network statistics (nodes, edges)
   - Top relationships by interaction count
   - Centrality rankings (degree, betweenness, eigenvector)
   - Chapter-by-chapter character importance

2. **Interactive Visualization** (`book_X_network.html`)
   - Force-directed graph layout
   - Draggable, zoomable interface
   - Community-colored nodes
   - Size-scaled by importance

---

### `analyze_all.sh` — Batch Processing

**Purpose:** Convenience script to process all books in sequence.

**Usage:**
```bash
bash analyze_all.sh
```

**Behavior:**
- Finds all subdirectories in `llm_results/`
- For each book:
  1. Runs `build_graph.py` to create the graph artifact
  2. Runs `analyze_graph.py` to generate reports and visualizations

---

## ⚙️ Configuration Reference

### `config.yaml`

```yaml
data:
  books_directory: "./data/Middlemarch"      # Raw text input
  character_file: "./char_alias.json"        # Canonical character list
  llm_results_dir: "./llm_results"           # LLM output directory
  graph_artifacts_dir: "./graph_artifacts"   # Graph serialization directory

models:
  llm_model: "qwen3:8b"                      # Ollama model name
  llm_host: "http://localhost:11434"         # Ollama server address
  fast_tokenizer_for_counting: "bert-base-cased"  # HuggingFace tokenizer

processing:
  chunk_token_limit: 256                     # Max tokens per LLM call
  chunk_overlap_sentences: 1                 # Context overlap

analysis:
  top_n_results: 10                          # Results to show in reports
```

---

### `char_alias.json` Structure

```json
[
    {
        "canonical_name": "Dorothea Brooke",
        "aliases": [
            "Dorothea", "dorothea",
            "Miss Brooke", "miss brooke",
            "Dodo", "dodo",
            "Mrs. Casaubon", "mrs. casaubon",
            "Mrs. Ladislaw", "mrs. ladislaw"
        ]
    },
    ...
]
```

The file contains **80+ characters** with comprehensive alias coverage including:
- Formal titles (Mr., Mrs., Dr., Sir, Reverend)
- Nicknames (Dodo, Rosy, Kitty)
- Role-based references (the Vicar, the banker, the Mayor)
- Marriage name changes

---

## 🧪 Testing

### `test_llm.py` — Model Comparison Tests

**Purpose:** Benchmarks different LLM models on a sample chapter.

**Tests:**
- Response time measurement
- JSON validity
- Interaction count comparison

**Usage:**
```bash
uv run test_llm.py
```

---

### `test_llm_context.py` — Context Awareness Tests

**Purpose:** Validates that the context buffer improves pronoun resolution.

**Test Cases:**
1. **Without context** — Process a paragraph with pronouns ("she said")
2. **With context** — Same paragraph, but with active character buffer

**Expected Result:** Context-aware processing correctly resolves pronouns to character names.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **Ollama** running locally with a model installed (e.g., `qwen3:8b`)
- **uv** package manager (or use `pip` with a virtual environment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GE_LLM.git
cd GE_LLM

# Install dependencies
uv sync

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Running the Pipeline

```bash
# Step 1: Extract interactions (runs LLM on all chapters)
uv run run_llm_extraction.py

# Step 2: Build graphs and generate reports for all books
bash analyze_all.sh

# Step 3: View the results
open analysis_reports/book_1/book_1_network.html
```

---

## 📊 Output Examples

### Sample Analysis Report

```
--- Character Network Analysis Report ---

Total Characters (Nodes): 47
Total Unique Relationships (Edges): 156

--- Top 10 Relationships by Interaction Count ---
  42    | Dorothea Brooke -- Reverend Edward Casaubon
  38    | Dorothea Brooke -- Celia Brooke
  31    | Tertius Lydgate -- Rosamond Vincy
  ...

--- Top 10 Characters by Degree Centrality ---
  Dorothea Brooke               | Score: 0.4783
  ...

--- Top 5 Most Important Characters by Chapter ---
Chapter 1:
  1. Dorothea Brooke            (Score: 0.6667)
  2. Celia Brooke               (Score: 0.5000)
  ...
```

### Interactive Visualization

The HTML visualization features:
- **Dark theme** with white text
- **Color-coded communities** (Louvain clustering)
- **Size-scaled nodes** (larger = more central)
- **Interactive physics** (drag nodes, zoom, pan)
- **Tooltips** with character details

---

## 🔧 Extending the Pipeline

### Adding New Books

1. Place raw text files in `data/YourBook/`
2. Update `config.yaml` → `books_directory`
3. Create a new `char_alias.json` for your book's characters
4. Run the pipeline

### Using Different LLMs

1. Install the model in Ollama: `ollama pull llama3:8b`
2. Update `config.yaml` → `models.llm_model`
3. Adjust `chunk_token_limit` based on model context window

### Custom Interaction Types

Modify `src/schemas.py`:
```python
InteractionType = Literal["Direct Dialogue", "Physical Action", "Observation", "Memory/Reference", "Your New Type"]
```

Update the prompt in `src/prompt_manager.py` to include the new type.

---

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `networkx` | Graph data structures and algorithms |
| `pyvis` | Interactive HTML network visualizations |
| `python-louvain` | Community detection (Louvain algorithm) |
| `matplotlib` | Color mapping for communities |
| `pydantic` | Data validation and serialization |
| `pyyaml` | Configuration file parsing |
| `nltk` | Sentence tokenization |
| `transformers` | Fast tokenization for chunk sizing |
| `requests` | HTTP client for Ollama API |
| `tqdm` | Progress bars |

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **George Eliot Archive** — For inspiring this digital humanities project
- **Ollama** — For enabling local LLM inference
- **NetworkX** — For powerful graph analysis capabilities

---

*Documentation generated by analyzing the complete GE_LLM codebase.*
