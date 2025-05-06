source venv/bin/activate
python scripts/specs_docs_generation.py
sphinx-apidoc -o docs/source/ immuneML/
