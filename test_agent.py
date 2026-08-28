"""Tests fuer den Ollama-Agent."""
import os
import tempfile
import unittest
from unittest import mock

import agent


class ReadFileTest(unittest.TestCase):
    def test_gibt_dateiinhalt_zurueck(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w") as f:
                f.write("Hallo Welt")
            self.assertEqual(agent.read_file(p), "Hallo Welt")

    def test_meldet_fehlende_datei_als_text(self):
        with tempfile.TemporaryDirectory() as d:
            result = agent.read_file(os.path.join(d, "fehlt.txt"))
            self.assertIn("Fehler", result)


class WriteFileTest(unittest.TestCase):
    def test_schreibt_inhalt_in_datei(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "neu.txt")
            agent.write_file(p, "Inhalt")
            with open(p) as f:
                self.assertEqual(f.read(), "Inhalt")

    def test_legt_fehlende_verzeichnisse_an(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tief", "neu.txt")
            agent.write_file(p, "x")
            self.assertTrue(os.path.exists(p))


class ListDirTest(unittest.TestCase):
    def test_listet_eintraege(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.txt"), "w").close()
            open(os.path.join(d, "b.py"), "w").close()
            result = agent.list_dir(d)
            self.assertIn("a.txt", result)
            self.assertIn("b.py", result)

    def test_filtert_nach_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.txt"), "w").close()
            open(os.path.join(d, "b.py"), "w").close()
            result = agent.list_dir(d, "*.py")
            self.assertIn("b.py", result)
            self.assertNotIn("a.txt", result)

    def test_meldet_fehlenden_pfad_als_text(self):
        result = agent.list_dir("/gibt/es/nicht")
        self.assertIn("Fehler", result)


class RunCommandTest(unittest.TestCase):
    def test_gibt_stdout_zurueck(self):
        self.assertIn("hallo", agent.run_command("echo hallo"))

    def test_meldet_exitcode_bei_fehler(self):
        result = agent.run_command("exit 3")
        self.assertIn("3", result)


class FetchUrlTest(unittest.TestCase):
    def test_gibt_seitentext_zurueck(self):
        antwort = mock.MagicMock()
        antwort.read.return_value = b"<p>Inhalt</p>"
        antwort.__enter__ = lambda s: s
        antwort.__exit__ = lambda s, *a: None
        with mock.patch.object(agent.urllib.request, "urlopen", return_value=antwort):
            self.assertIn("Inhalt", agent.fetch_url("http://example.com"))

    def test_meldet_netzwerkfehler_als_text(self):
        with mock.patch.object(agent.urllib.request, "urlopen", side_effect=OSError("weg")):
            self.assertIn("Fehler", agent.fetch_url("http://example.com"))


class ToolSchemasTest(unittest.TestCase):
    def test_enthaelt_alle_fuenf_tools(self):
        namen = {s["function"]["name"] for s in agent.tool_schemas()}
        self.assertEqual(
            namen,
            {"read_file", "write_file", "list_dir", "run_command", "fetch_url"},
        )


class ExecuteToolTest(unittest.TestCase):
    def test_ruft_tool_mit_argumenten_auf(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w") as f:
                f.write("da")
            result = agent.execute_tool("read_file", {"path": p}, confirm=lambda m: True)
            self.assertEqual(result, "da")

    def test_meldet_unbekanntes_tool(self):
        result = agent.execute_tool("gibtsnicht", {}, confirm=lambda m: True)
        self.assertIn("Fehler", result)

    def test_fragt_vor_shell_befehl_nach(self):
        gefragt = []
        agent.execute_tool(
            "run_command", {"command": "echo x"}, confirm=lambda m: gefragt.append(m) or True
        )
        self.assertEqual(len(gefragt), 1)

    def test_fuehrt_nicht_aus_wenn_abgelehnt(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "nein.txt")
            result = agent.execute_tool(
                "write_file", {"path": p, "content": "x"}, confirm=lambda m: False
            )
            self.assertFalse(os.path.exists(p))
            self.assertIn("abgelehnt", result.lower())

    def test_fragt_bei_lesen_nicht_nach(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w") as f:
                f.write("da")
            gefragt = []
            agent.execute_tool(
                "read_file", {"path": p}, confirm=lambda m: gefragt.append(m) or True
            )
            self.assertEqual(gefragt, [])


class ChatLoopTest(unittest.TestCase):
    def test_gibt_text_zurueck_wenn_kein_tool_call(self):
        post = mock.Mock(return_value={"message": {"role": "assistant", "content": "Antwort"}})
        messages = [{"role": "user", "content": "hi"}]
        text = agent.chat(messages, model="m", post=post, confirm=lambda m: True)
        self.assertEqual(text, "Antwort")
        self.assertEqual(post.call_count, 1)

    def test_fuehrt_tool_aus_und_fragt_erneut(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w") as f:
                f.write("Dateiinhalt")
            post = mock.Mock(
                side_effect=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "read_file", "arguments": {"path": p}}}
                            ],
                        }
                    },
                    {"message": {"role": "assistant", "content": "Da steht Dateiinhalt"}},
                ]
            )
            messages = [{"role": "user", "content": "lies"}]
            text = agent.chat(messages, model="m", post=post, confirm=lambda m: True)
            self.assertEqual(text, "Da steht Dateiinhalt")
            self.assertEqual(post.call_count, 2)

    def test_haengt_tool_ergebnis_an_historie(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w") as f:
                f.write("XYZ")
            post = mock.Mock(
                side_effect=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "read_file", "arguments": {"path": p}}}
                            ],
                        }
                    },
                    {"message": {"role": "assistant", "content": "fertig"}},
                ]
            )
            messages = [{"role": "user", "content": "lies"}]
            agent.chat(messages, model="m", post=post, confirm=lambda m: True)
            rollen = [m["role"] for m in messages]
            self.assertIn("tool", rollen)
            tool_msg = next(m for m in messages if m["role"] == "tool")
            self.assertEqual(tool_msg["content"], "XYZ")

    def test_bricht_nach_max_runden_ab(self):
        endlos = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "run_command", "arguments": {"command": "true"}}}],
            }
        }
        post = mock.Mock(return_value=endlos)
        messages = [{"role": "user", "content": "los"}]
        text = agent.chat(
            messages, model="m", post=post, confirm=lambda m: True, max_rounds=3
        )
        self.assertEqual(post.call_count, 3)
        self.assertIn("abgebrochen", text.lower())

    def test_sendet_tools_und_think_flag(self):
        post = mock.Mock(return_value={"message": {"role": "assistant", "content": "ok"}})
        agent.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            post=post,
            confirm=lambda m: True,
            think=False,
            num_ctx=32768,
        )
        payload = post.call_args[0][0]
        self.assertEqual(payload["model"], "m")
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_ctx"], 32768)
        self.assertEqual(len(payload["tools"]), 5)


if __name__ == "__main__":
    unittest.main()


def zeilen(*chunks):
    """Baut NDJSON-Zeilen, wie Ollama sie beim Streaming liefert."""
    import json as j
    return [j.dumps(c) for c in chunks]


class ConsumeStreamTest(unittest.TestCase):
    def test_setzt_content_aus_chunks_zusammen(self):
        nachricht = agent.consume_stream(
            zeilen(
                {"message": {"content": "Hallo "}, "done": False},
                {"message": {"content": "Welt"}, "done": True},
            )
        )
        self.assertEqual(nachricht["content"], "Hallo Welt")

    def test_meldet_content_deltas_an_callback(self):
        stuecke = []
        agent.consume_stream(
            zeilen(
                {"message": {"content": "ab"}, "done": False},
                {"message": {"content": "cd"}, "done": True},
            ),
            on_content=stuecke.append,
        )
        self.assertEqual(stuecke, ["ab", "cd"])

    def test_trennt_thinking_von_content(self):
        gedanken, text = [], []
        nachricht = agent.consume_stream(
            zeilen(
                {"message": {"thinking": "ueberlege"}, "done": False},
                {"message": {"content": "Antwort"}, "done": True},
            ),
            on_thinking=gedanken.append,
            on_content=text.append,
        )
        self.assertEqual(gedanken, ["ueberlege"])
        self.assertEqual(text, ["Antwort"])
        self.assertEqual(nachricht["thinking"], "ueberlege")
        self.assertEqual(nachricht["content"], "Antwort")

    def test_sammelt_tool_calls_ueber_chunks(self):
        nachricht = agent.consume_stream(
            zeilen(
                {"message": {"content": ""}, "done": False},
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "/x"}}}
                        ],
                    },
                    "done": True,
                },
            )
        )
        self.assertEqual(len(nachricht["tool_calls"]), 1)
        self.assertEqual(nachricht["tool_calls"][0]["function"]["name"], "read_file")

    def test_ignoriert_leere_zeilen(self):
        nachricht = agent.consume_stream(
            ["", '{"message": {"content": "x"}, "done": true}', "  "]
        )
        self.assertEqual(nachricht["content"], "x")

    def test_gibt_rolle_assistant_zurueck(self):
        nachricht = agent.consume_stream(zeilen({"message": {"content": "x"}, "done": True}))
        self.assertEqual(nachricht["role"], "assistant")

    def test_meldet_fehler_im_stream(self):
        with self.assertRaises(RuntimeError):
            agent.consume_stream(['{"error": "kaputt"}'])


class ChatStreamingTest(unittest.TestCase):
    def test_nutzt_stream_post_wenn_stream_aktiv(self):
        stream_post = mock.Mock(
            return_value=zeilen({"message": {"content": "Antwort"}, "done": True})
        )
        text = agent.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            stream=True,
            stream_post=stream_post,
            confirm=lambda m: True,
        )
        self.assertEqual(text, "Antwort")
        self.assertEqual(stream_post.call_count, 1)
        self.assertIs(stream_post.call_args[0][0]["stream"], True)

    def test_fuehrt_tool_aus_und_streamt_zweite_runde(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w") as f:
                f.write("INHALT")
            stream_post = mock.Mock(
                side_effect=[
                    zeilen(
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {"function": {"name": "read_file", "arguments": {"path": p}}}
                                ],
                            },
                            "done": True,
                        }
                    ),
                    zeilen({"message": {"content": "fertig"}, "done": True}),
                ]
            )
            messages = [{"role": "user", "content": "lies"}]
            text = agent.chat(
                messages, model="m", stream=True, stream_post=stream_post,
                confirm=lambda m: True,
            )
            self.assertEqual(text, "fertig")
            self.assertEqual(stream_post.call_count, 2)
            tool_msg = next(m for m in messages if m["role"] == "tool")
            self.assertEqual(tool_msg["content"], "INHALT")

    def test_reicht_callbacks_durch(self):
        gedanken, text = [], []
        stream_post = mock.Mock(
            return_value=zeilen(
                {"message": {"thinking": "denk"}, "done": False},
                {"message": {"content": "sag"}, "done": True},
            )
        )
        agent.chat(
            [{"role": "user", "content": "hi"}],
            model="m", stream=True, stream_post=stream_post,
            confirm=lambda m: True,
            on_thinking=gedanken.append, on_content=text.append,
        )
        self.assertEqual(gedanken, ["denk"])
        self.assertEqual(text, ["sag"])
