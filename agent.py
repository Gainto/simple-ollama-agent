#!/usr/bin/env python3
"""Minimaler Ollama-Agent mit Tool-Ausfuehrung.

Chattet mit einem lokalen Ollama-Modell und fuehrt dessen Tool-Aufrufe aus.
Nur Standardbibliothek - keine externen Abhaengigkeiten.
"""
import argparse
import glob as globmod
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.8-27b-uncensored"
MAX_OUTPUT = 20000

# Tools, die vor der Ausfuehrung eine Bestaetigung brauchen
NEEDS_CONFIRM = {"write_file", "run_command"}

GRAU = "\033[90m"
GELB = "\033[33m"
CYAN = "\033[36m"
AUS = "\033[0m"


# --- Tools ---------------------------------------------------------------

def read_file(path):
    try:
        with open(os.path.expanduser(path)) as f:
            return f.read()[:MAX_OUTPUT]
    except OSError as e:
        return f"Fehler beim Lesen von {path}: {e}"


def write_file(path, content):
    path = os.path.expanduser(path)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"{len(content)} Zeichen nach {path} geschrieben."
    except OSError as e:
        return f"Fehler beim Schreiben von {path}: {e}"


def list_dir(path, pattern=None):
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return f"Fehler: {path} ist kein Verzeichnis."
    if pattern:
        treffer = sorted(
            os.path.relpath(p, path)
            for p in globmod.glob(os.path.join(path, "**", pattern), recursive=True)
        )
    else:
        treffer = sorted(os.listdir(path))
    if not treffer:
        return f"Keine Treffer in {path}."
    return "\n".join(treffer)[:MAX_OUTPUT]


def run_command(command):
    try:
        p = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return f"Fehler: '{command}' nach 120 s abgebrochen."
    teile = [f"exitcode: {p.returncode}"]
    if p.stdout.strip():
        teile.append(f"stdout:\n{p.stdout.strip()}")
    if p.stderr.strip():
        teile.append(f"stderr:\n{p.stderr.strip()}")
    return "\n".join(teile)[:MAX_OUTPUT]


def fetch_url(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ollama-agent"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rohtext = r.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError) as e:
        return f"Fehler beim Abruf von {url}: {e}"
    ohne_skripte = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", rohtext)
    text = re.sub(r"(?s)<[^>]+>", " ", ohne_skripte)
    return re.sub(r"\s+", " ", text).strip()[:MAX_OUTPUT]


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_command": run_command,
    "fetch_url": fetch_url,
}


def tool_schemas():
    def fn(name, beschreibung, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": beschreibung,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }

    s = {"type": "string"}
    return [
        fn("read_file", "Liest eine Textdatei und gibt ihren Inhalt zurueck.",
           {"path": {**s, "description": "Pfad zur Datei"}}, ["path"]),
        fn("write_file", "Schreibt Text in eine Datei und legt fehlende Ordner an.",
           {"path": {**s, "description": "Pfad zur Datei"},
            "content": {**s, "description": "Zu schreibender Inhalt"}}, ["path", "content"]),
        fn("list_dir", "Listet ein Verzeichnis auf, optional rekursiv per Glob-Muster.",
           {"path": {**s, "description": "Verzeichnis"},
            "pattern": {**s, "description": "Glob wie *.py, optional"}}, ["path"]),
        fn("run_command", "Fuehrt einen Shell-Befehl aus und gibt Ausgabe und Exitcode zurueck.",
           {"command": {**s, "description": "Der Befehl"}}, ["command"]),
        fn("fetch_url", "Ruft eine URL ab und gibt den Text ohne HTML-Tags zurueck.",
           {"url": {**s, "description": "Die URL"}}, ["url"]),
    ]


# --- Ausfuehrung ---------------------------------------------------------

def execute_tool(name, args, confirm):
    fn = TOOLS.get(name)
    if fn is None:
        return f"Fehler: unbekanntes Tool '{name}'."
    if name in NEEDS_CONFIRM:
        beschreibung = args.get("command") or args.get("path", "")
        if not confirm(f"{name}: {beschreibung}"):
            return f"Vom Benutzer abgelehnt: {name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Fehler: falsche Argumente fuer {name}: {e}"


# --- Transport -----------------------------------------------------------

def _request(payload):
    return urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def post_ollama(payload):
    with urllib.request.urlopen(_request(payload), timeout=1800) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def stream_ollama(payload):
    """Liefert die NDJSON-Zeilen der Streaming-Antwort."""
    with urllib.request.urlopen(_request(payload), timeout=1800) as r:
        for rohzeile in r:
            yield rohzeile.decode("utf-8", "replace")


def consume_stream(lines, on_thinking=None, on_content=None):
    """Setzt die Streaming-Zeilen zu einer Nachricht zusammen.

    Ruft `on_thinking` / `on_content` fuer jedes Teilstueck auf, sobald es
    eintrifft, und gibt am Ende die vollstaendige Assistant-Nachricht zurueck.
    """
    gedanken, text, aufrufe = [], [], []
    for zeile in lines:
        if isinstance(zeile, bytes):
            zeile = zeile.decode("utf-8", "replace")
        zeile = zeile.strip()
        if not zeile:
            continue
        chunk = json.loads(zeile)
        if "error" in chunk:
            raise RuntimeError(chunk["error"])
        nachricht = chunk.get("message") or {}
        stueck_gedanke = nachricht.get("thinking")
        if stueck_gedanke:
            gedanken.append(stueck_gedanke)
            if on_thinking:
                on_thinking(stueck_gedanke)
        stueck_text = nachricht.get("content")
        if stueck_text:
            text.append(stueck_text)
            if on_content:
                on_content(stueck_text)
        aufrufe.extend(nachricht.get("tool_calls") or [])
    ergebnis = {"role": "assistant", "content": "".join(text)}
    if gedanken:
        ergebnis["thinking"] = "".join(gedanken)
    if aufrufe:
        ergebnis["tool_calls"] = aufrufe
    return ergebnis


# --- Agent-Loop ----------------------------------------------------------

def chat(messages, model, post=post_ollama, stream_post=stream_ollama, stream=False,
         confirm=None, think=False, num_ctx=32768, max_rounds=10,
         on_tool=None, on_thinking=None, on_content=None, on_request=None):
    """Fuehrt den Agent-Loop, bis eine Antwort ohne Tool-Aufruf kommt.

    Haengt alle Zwischenschritte an `messages` an (wird mutiert).
    """
    if confirm is None:
        confirm = lambda m: True  # noqa: E731
    for _ in range(max_rounds):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tool_schemas(),
            "stream": stream,
            "think": think,
            "options": {"num_ctx": num_ctx},
        }
        if on_request:
            on_request()
        if stream:
            nachricht = consume_stream(stream_post(payload), on_thinking, on_content)
        else:
            antwort = post(payload)
            if "error" in antwort:
                return f"Fehler von Ollama: {antwort['error']}"
            nachricht = antwort["message"]
        messages.append(nachricht)
        aufrufe = nachricht.get("tool_calls") or []
        if not aufrufe:
            return nachricht.get("content", "")
        for aufruf in aufrufe:
            f = aufruf["function"]
            name, args = f["name"], f.get("arguments") or {}
            if on_tool:
                on_tool(name, args)
            ergebnis = execute_tool(name, args, confirm)
            messages.append({"role": "tool", "name": name, "content": ergebnis})
    return f"Abgebrochen: mehr als {max_rounds} Tool-Runden ohne Antwort."


# --- Terminal-Darstellung ------------------------------------------------

class Spinner:
    """Zeigt waehrend der Wartezeit einen Indikator mit Sekundenzaehler.

    Noetig, weil vor dem ersten Token die Prompt-Verarbeitung laeuft - dabei
    kommt vom Modell nichts, was man streamen koennte.
    """

    RAHMEN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text="denkt nach"):
        self.text = text
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread or not sys.stdout.isatty():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._lauf, daemon=True)
        self._thread.start()

    def _lauf(self):
        beginn = time.time()
        for zeichen in itertools.cycle(self.RAHMEN):
            if self._stop.is_set():
                return
            sys.stdout.write(
                f"\r{GRAU}{zeichen} {self.text}… {int(time.time() - beginn)}s{AUS}"
            )
            sys.stdout.flush()
            self._stop.wait(0.1)

    def stop(self):
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


class Ausgabe:
    """Schreibt Reasoning und Antwort waehrend des Streamings ins Terminal."""

    def __init__(self, spinner):
        self.spinner = spinner
        self.modus = None

    def _wechsel(self, modus, kopf):
        self.spinner.stop()
        if self.modus == modus:
            return
        if self.modus is not None:
            sys.stdout.write(f"{AUS}\n")
        sys.stdout.write(kopf)
        self.modus = modus

    def gedanke(self, stueck):
        self._wechsel("denken", f"{GRAU}│ ")
        sys.stdout.write(stueck.replace("\n", f"\n{GRAU}│ "))
        sys.stdout.flush()

    def antwort(self, stueck):
        self._wechsel("text", "")
        sys.stdout.write(stueck)
        sys.stdout.flush()

    def tool(self, name, args):
        self._wechsel("tool", "")
        argtext = json.dumps(args, ensure_ascii=False)[:120]
        sys.stdout.write(f"{GRAU}  -> {name}({argtext}){AUS}\n")
        sys.stdout.flush()
        self.modus = None

    def abschluss(self):
        if self.modus is not None:
            sys.stdout.write(f"{AUS}\n")
        self.modus = None


# --- CLI -----------------------------------------------------------------

def frage_nach(text):
    antwort = input(f"  {GELB}? {text}{AUS}  ausfuehren? [j/N] ").strip().lower()
    return antwort in ("j", "ja", "y", "yes")


def main(argv=None):
    p = argparse.ArgumentParser(description="Ollama-Chat mit Tool-Ausfuehrung.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-think", dest="think", action="store_false",
                   help="Reasoning abschalten (schneller)")
    p.add_argument("--no-stream", dest="stream", action="store_false",
                   help="Antwort erst am Stueck ausgeben")
    p.add_argument("--yes", action="store_true", help="Nicht nachfragen")
    p.add_argument("--ctx", type=int, default=32768, help="num_ctx")
    p.add_argument("--max-rounds", type=int, default=10)
    p.set_defaults(think=True, stream=True)
    a = p.parse_args(argv)

    confirm = (lambda m: True) if a.yes else frage_nach
    messages = []
    print(f"Modell: {a.model}  |  Tools: {', '.join(TOOLS)}")
    print("/reset leert die Historie, /exit beendet.\n")
    while True:
        try:
            eingabe = input(f"{CYAN}du>{AUS} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not eingabe:
            continue
        if eingabe == "/exit":
            return 0
        if eingabe == "/reset":
            messages.clear()
            print("Historie geleert.\n")
            continue
        messages.append({"role": "user", "content": eingabe})
        spinner = Spinner()
        ausgabe = Ausgabe(spinner)
        try:
            text = chat(
                messages, model=a.model, confirm=confirm, think=a.think,
                stream=a.stream, num_ctx=a.ctx, max_rounds=a.max_rounds,
                on_request=spinner.start,
                on_tool=ausgabe.tool,
                on_thinking=ausgabe.gedanke if a.stream else None,
                on_content=ausgabe.antwort if a.stream else None,
            )
        except KeyboardInterrupt:
            spinner.stop()
            print("\n  abgebrochen\n")
            continue
        except (OSError, RuntimeError) as e:
            spinner.stop()
            print(f"\n  Fehler: {e}\n")
            continue
        finally:
            spinner.stop()
        ausgabe.abschluss()
        if not a.stream:
            print(f"\n{text}")
        print()


if __name__ == "__main__":
    sys.exit(main())
