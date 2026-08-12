# Pretranslation Pipeline

### About

This is a proof-of-concept machine translation pipeline using:
- remote LLM `Claude Sonnet 5.x`
- augmented retrieval of glossary terms using the `multilingual-e5-small` embedding model
- manual glossary embedding
- context-aware translation for CSV (15 lines before + 15 after, see global vars)

### Requirements

Python

```bash
pip install openai sentence-transformers numpy pandas python-dateutil anthropic
```

The script will download additional required models upon the first launch.

Run `python try.py -v 4` to monoitor the initial download progress.

### Usage

```
python try.py [OPTIONS]
```

```
OPTIONS:
  -s, --string TEXT     Translate a single string
  -f, --file CSV        Translate a CSV file (reads column 1, writes to column 2 (or 'en'))
  -ow, --overwrite      Overwrite existing translations in the CSV
  -v, --verbosity 0-4   Set output verbosity (default: 3)
                            0 = silent (only errors)
                            1 = translation output + progress
                            2 = + embedding status
                            3 = + glossary matches
                            4 = + LLM reasoning
```

EXAMPLES:
  # Translate a single string with default verbosity
  python try.py -s "Текст для перевода"

  # Translate a CSV file quietly
  python try.py -f data.csv -v 0

  # Translate with full debug output and overwrite existing translations
  python try.py -f "test2.csv" -v 4 -ow

  # Default behavior (translates hardcoded text)
  python try.py

### Batch CSV translation

```
python try_many.py [path_to_csv_folder_1, path_to_csv_folder_2, ...]
```

Translate all CSV in a specific folder or folders

```
python try_many.py
```

Translate all CSV in ./ttr/
