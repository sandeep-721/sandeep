# Universal RAG

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended
- Qdrant
- Node.js/npm for MCP Inspector

## Development

### Start the API

```powershell
.\.venv\Scripts\python.exe .\server\launcher.py
```




The API runs at:

http://127.0.0.1:8000

## Tests

Run the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```





## Production configuration

Set these environment variables before starting the API:

- `UNIVERSAL_RAG_HOST` — defaults to `127.0.0.1`
- `UNIVERSAL_RAG_PORT` — defaults to `8000`
- `UNIVERSAL_RAG_API_KEY` — protects `/search` when set
- `UNIVERSAL_RAG_API_URL` — MCP API URL
- `HF_TOKEN` — optional Hugging Face authentication token

For production, keep `UNIVERSAL_RAG_API_KEY` secret and do not commit it to Git.
