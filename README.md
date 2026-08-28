# Simple Ollama Agent

Chat mit einem lokalen Ollama-Modell, das Tools ausführen kann. Ersatz für
`ollama launch claude`, wenn man nur Tool-Calling braucht - ohne MCP, Skills
oder Claude Codes ~50.000 Token Systemprompt.

Nur Standardbibliothek, keine Abhängigkeiten.

## Start

```bash
python3 agent.py
```

Im Chat: `/reset` leert die Historie, `/exit` beendet. Strg-C bricht eine
laufende Antwort ab, ohne das Programm zu beenden.

## Optionen

| Flag | Default | Zweck |
|---|---|---|
| `--model` | `qwen3.8-27b-uncensored` | Modell-Tag |
| `--no-think` | Reasoning **an** | Reasoning abschalten - spürbar schneller |
| `--no-stream` | Streaming **an** | Antwort erst am Stück ausgeben |
| `--yes` | aus | Keine Rückfragen - Vorsicht, siehe unten |
| `--ctx` | `32768` | `num_ctx`. Auf `131072` setzen, wenn parallel Open WebUI läuft, sonst lädt Ollama das Modell bei jedem Wechsel neu |
| `--max-rounds` | `10` | Obergrenze aufeinanderfolgender Tool-Runden |

## Tools

| Tool | Rückfrage | Beschreibung |
|---|---|---|
| `read_file` | nein | Datei lesen |
| `list_dir` | nein | Verzeichnis auflisten, optional Glob-Muster (rekursiv) |
| `fetch_url` | nein | URL abrufen, HTML-Tags entfernt |
| `write_file` | **ja** | Datei schreiben, legt fehlende Ordner an |
| `run_command` | **ja** | Shell-Befehl, Timeout 120 s |

Ausgaben werden bei 20.000 Zeichen abgeschnitten, damit ein `cat` auf eine
große Datei nicht den Kontext sprengt.

## Sicherheit

`write_file` und `run_command` fragen vor der Ausführung nach. Das ist die
einzige Bremse - das verwendete Modell ist abliterated und hat keine eigene.
`--yes` schaltet die Rückfrage ab; nur benutzen, wenn klar ist, was der
Agent tun soll.

## Tests

```bash
python3 -m unittest test_agent
```
