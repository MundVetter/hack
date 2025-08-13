# VulnFlow

VulnFlow is a Python CLI that builds a code dataflow graph from sources (user input) to sinks (sensitive operations like DB/LLM/subprocess) and flags potential vulnerabilities (e.g., SQL injection).

## Install

```bash
python -m pip install -e .
# or
python -m pip install -r requirements.txt
```

## OpenAI and Modal (optional)

- Set OpenAI API key if you want AI explanations:
```bash
export OPENAI_API_KEY="<your-openai-key>"
```
- Set Modal credentials if you want to run in the cloud:
```bash
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```

## Usage

Analyze a local path:
```bash
vulnflow analyze --path /path/to/repo --graph-out /tmp/graph.graphml --json
```

Analyze a GitHub repo (shallow clone to temp):
```bash
vulnflow analyze --repo https://github.com/pallets/flask --openai-explain --graph-out flask.graphml
```

Run via Modal cloud:
```bash
vulnflow analyze --repo https://github.com/psf/requests --modal --json
```

## Output

- Findings printed to console (optionally JSON).
- Graph written to GraphML; open with tools like yEd or Gephi.

## Notes

- Initial analyzer focuses on Python. JS/TS hooks exist but are minimal.
- Taint analysis is heuristic and intra-procedural; treat results as guidance, not proof.