# C++ Documentation Generator

Documentation generator for C++ codebases using academic papers as reference.

## Installation

```bash
cd cpp_doc_generator

uv venv

uv sync

source .venv/bin/activate
```

## Running

```bash
python main.py
```

## Configuration

Edit `config.yaml` to set:
- Input C++ repository path
- Papers directory path
- LLM provider (Ollama/Gemini)
- Output settings

## Output

Generated documentation will be saved in the `output/` directory:
- `functions/` - Individual function documentation