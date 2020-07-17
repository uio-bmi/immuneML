source venv4/bin/activate
python scripts/specs_docs_generation.py
sphinx-apidoc -o docs/source/ source/
deactivate