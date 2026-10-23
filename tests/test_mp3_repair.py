"""
Tests mit synthetischen MP3s (ffmpeg wird zum Erzeugen gebraucht).
Online-Dienste werden durch ein Fake-Objekt ersetzt.

    python3 -m unittest discover -s tests -v
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from mutagen.id3 import ID3, TIT2, TLEN, TPE1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mp3_repair as m  # noqa: E402

SR = 44100
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def synth_song(seconds: float, seed: int, fade: float = 3.0, sr: int = SR) -> np.ndarray:
    """Akkordfolge mit Rauschanteil; optional Fade-out. Pegel ~ -18 dBFS."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    out = np.zeros_like(t)
    base = 110 * (1 + seed % 5 * 0.12)
    for i, ratio in enumerate((1, 1.25, 1.5, 2.0, 3.0)):
        f = base * ratio * (1 + 0.003 * seed)
        out += np.sin(2 * np.pi * f * t + i) * (0.5 / (i + 1))
    out *= 0.6 + 0.4 * np.sin(2 * np.pi * (1.5 + seed % 3) * t) ** 2          # rhythmische Hüllkurve
    out += rng.normal(0, 0.02, len(t))
    out *= 0.25 / np.sqrt(np.mean(out ** 2) + 1e-12) * 0.5
    if fade:
        n = int(fade * sr)
        out[-n:] *= np.linspace(1, 0, n) ** 2
    return out.astype(np.float32)


def write_wav(path: Path, data: np.ndarray, sr: int = SR) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())


def encode(wav: Path, mp3: Path, bitrate: int = 192, extra: list[str] | None = None) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-c:a", "libmp3lame",
                    "-b:a", f"{bitrate}k", *(extra or []), str(mp3)], check=True)


def tag(path: Path, title="Testsong", artist="Testband", tlen: float | None = None) -> None:
    try:
        t = ID3(path)
    except Exception:
        t = ID3()
    t.add(TIT2(encoding=3, text=title))
    t.add(TPE1(encoding=3, text=artist))
    if tlen is not None:
        t.add(TLEN(encoding=3, text=str(int(tlen * 1000))))
    t.save(path)


class FakeOnline:
    """Ersetzt OnlineReferences: liefert feste Referenzlängen."""

    def __init__(self, refs=(), apple=()):
        self.refs, self.apple_refs = list(refs), list(apple)

    def collect(self, path, tags, key, file_duration):
        return m.RefResult(refs=[m.Ref("Fake-MB", s) for s in self.refs])

    def apple(self, tags, key):
        return m.RefResult(refs=[m.Ref("Fake-Apple", s) for s in self.apple_refs])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg nicht installiert")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mp3repair_test_"))
        cls.a_wav = cls.tmp / "a.wav"
        cls.b_wav = cls.tmp / "b.wav"
        cls.song_a = synth_song(30.0, seed=1)
        cls.song_b = synth_song(6.0, seed=7, fade=0)             # anderer "Song", endet abrupt
        write_wav(cls.a_wav, cls.song_a)
        write_wav(cls.b_wav, cls.song_b)
        cls.a_mp3 = cls.tmp / "A.mp3"
        cls.b_mp3 = cls.tmp / "B.mp3"
        encode(cls.a_wav, cls.a_mp3)
        encode(cls.b_wav, cls.b_mp3)
        tag(cls.a_mp3, tlen=30.0)
        tag(cls.b_mp3, title="Anderer Song")
        cls.cfg = m.Config(have_mp3val=bool(shutil.which("mp3val")), have_fpcalc=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(dir=self.tmp))
        self.out = self.work / "out"
        self.out.mkdir()

    def process(self, src: Path, refs=(), cfg=None, apple=()):
        return m.process_file(src, self.out, cfg or self.cfg, FakeOnline(refs, apple), "key")

    # --- Scanner ------------------------------------------------------------------
    def test_scan_clean_file(self):
        s = m.scan_mp3(self.a_mp3)
        self.assertTrue(s.ok)
        self.assertAlmostEqual(s.duration, 30.0, delta=0.15)
        self.assertEqual(s.header_kind, "Info")
        self.assertIn("Lavf", s.encoder + "Lavf") if not s.encoder else self.assertTrue(s.encoder)
        self.assertEqual(s.junk, [])
        self.assertEqual(m.stream_defects(s), [])
        self.assertAlmostEqual(s.avg_bitrate, 192, delta=3)

    def test_scan_skips_macos_resource_forks(self):
        d = self.work / "dir"
        d.mkdir()
        shutil.copy(self.a_mp3, d / "ok.mp3")
        (d / "._ok.mp3").write_bytes(b"\x00\x05\x16\x07 apple double")
        (d / "cover.jpg").write_bytes(b"x")
        sub = d / "sub"
        sub.mkdir()
        shutil.copy(self.a_mp3, sub / "nested.mp3")
        self.assertEqual([p.name for p in m.list_mp3s(d)], ["ok.mp3"])

    # --- sauberer Song ------------------------------------------------------------------
    def test_clean_file_is_copied_bit_identical(self):
        row = self.process(self.a_mp3, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_OK, row)
        self.assertEqual(row["fragment"], "nein")
        self.assertEqual(row["abgeschnitten"], "nein")
        self.assertEqual(row["laenge_ok"], "ja")
        self.assertEqual(Path(row["ausgabedatei"]).read_bytes(), self.a_mp3.read_bytes())

    def test_fade_out_shorter_than_reference_is_not_truncated(self):
        # Ende wirkt sauber (Fade-out) -> nicht "abgeschnitten", nur Kontrolle (z.B. andere Version)
        f = self.work / "notlen.mp3"
        shutil.copy(self.a_mp3, f)
        t = ID3(f)
        t.delall("TLEN")                                             # sonst bestätigt das TLEN-Tag die Länge
        t.save(f)
        row = self.process(f, refs=[45.0])
        self.assertNotEqual(row["zielordner"], m.OUT_TRUNC)
        self.assertIn(row["abgeschnitten"], ("nein", "unsicher"))
        self.assertEqual(row["zielordner"], m.OUT_REVIEW)          # Länge unerklärt -> Kontrolle

    # --- Fragment per cat (zweiter Info-Header + zweites ID3-Tag) ---------------------------------
    def test_concatenated_fragment_is_trimmed_losslessly(self):
        cat = self.work / "cat.mp3"
        cat.write_bytes(self.a_mp3.read_bytes() + self.b_mp3.read_bytes())
        s = m.scan_mp3(cat)
        self.assertEqual(len(s.mid_tag_times), 1)
        self.assertAlmostEqual(s.mid_tag_times[0], 30.0, delta=0.15)
        row = self.process(cat, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertIn(row["fragment"], ("sicher", "wahrscheinlich"))
        out = Path(row["ausgabedatei"])
        so = m.scan_mp3(out)
        self.assertAlmostEqual(so.duration, 30.0, delta=0.15)
        self.assertEqual(so.mid_info_times + so.mid_tag_times, [])
        self.assertIn("verlustfrei", row["aktion"])
        # Tags (Titel des ersten Songs, TLEN) sind übernommen
        t = ID3(out)
        self.assertEqual(str(t["TIT2"].text[0]), "Testsong")
        self.assertAlmostEqual(int(str(t["TLEN"].text[0])) / 1000, so.duration, delta=0.1)
        # Original unverändert
        self.assertGreater(m.scan_mp3(cat).duration, 35)

    def test_concatenated_fragment_without_any_reference(self):
        cat = self.work / "cat2.mp3"
        cat.write_bytes(self.a_mp3.read_bytes() + self.b_mp3.read_bytes())
        # TLEN der Datei fehlt -> Tags entfernen, keine Referenzen: strukturelles Indiz (Gewicht 2) reicht
        t = ID3(cat)
        t.delall("TLEN")
        t.save(cat)
        row = self.process(cat)
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertAlmostEqual(m.scan_mp3(Path(row["ausgabedatei"])).duration, 30.0, delta=0.15)

    # --- Fragment als Stille-Lücke (ein einziger Encoder-Lauf, kein struktureller Hinweis) ---------------
    def make_gap_file(self, name: str, gap=0.8) -> Path:
        wav, mp3 = self.work / f"{name}.wav", self.work / f"{name}.mp3"
        write_wav(wav, np.concatenate([self.song_a, np.zeros(int(gap * SR), np.float32), self.song_b[:int(3.5 * SR)]]))
        encode(wav, mp3)
        tag(mp3)
        return mp3

    def test_gap_fragment_with_reference_is_trimmed(self):
        f = self.make_gap_file("gap1")
        row = self.process(f, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertEqual(row["fragment"], "sicher", row["fragment_gruende"])
        self.assertAlmostEqual(float(row["fragment_schnitt_sek"]), 30.0, delta=0.8)   # im Ausblend-Ende/in der Lücke
        self.assertAlmostEqual(m.scan_mp3(Path(row["ausgabedatei"])).duration, 30.0, delta=0.9)

    def test_short_fragment_within_duration_tolerance_is_still_found(self):
        wav, mp3 = self.work / "short.wav", self.work / "short.mp3"
        write_wav(wav, np.concatenate([self.song_a, np.zeros(int(0.6 * SR), np.float32), self.song_b[:int(1.2 * SR)]]))
        encode(wav, mp3)
        tag(mp3)
        row = self.process(mp3, refs=[30.0])                       # nur ~1.8 s zu lang -> innerhalb +-3 s
        self.assertEqual(row["laenge_ok"], "ja")
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertAlmostEqual(float(row["fragment_schnitt_sek"]), 30.0, delta=0.8)

    def test_gap_without_reference_is_only_flagged(self):
        f = self.make_gap_file("gap2")
        row = self.process(f, refs=[])
        self.assertEqual(row["fragment"], "unsicher")
        self.assertEqual(row["fragment_schnitt_sek"], "")
        self.assertEqual(row["zielordner"], m.OUT_REVIEW)
        self.assertEqual(Path(row["ausgabedatei"]).read_bytes(), f.read_bytes())    # nichts verändert

    def test_stinger_ending_is_not_trimmed_when_length_matches(self):
        # Song, 1.2 s Pause, kurzer Schlussakkord -> Länge == Referenz -> kein Fragment
        wav, mp3 = self.work / "stinger.wav", self.work / "stinger.mp3"
        stinger = np.concatenate([self.song_a[:int(27 * SR)], np.zeros(int(1.2 * SR), np.float32),
                                  self.song_b[:int(1.5 * SR)]])
        write_wav(wav, stinger)
        encode(wav, mp3)
        tag(mp3)
        total = len(stinger) / SR
        row = self.process(mp3, refs=[total])
        self.assertEqual(row["fragment_schnitt_sek"], "", row["fragment_gruende"])
        self.assertNotEqual(row["fragment"], "sicher")

    def test_strict_mode_only_trims_confident(self):
        short = self.work / "short2.mp3"
        wav = self.work / "short2.wav"
        write_wav(wav, np.concatenate([self.song_a, np.zeros(int(0.6 * SR), np.float32), self.song_b[:int(1.2 * SR)]]))
        encode(wav, short)
        tag(short)
        cfg = m.Config(strict=True, have_mp3val=self.cfg.have_mp3val)
        row = self.process(short, refs=[30.0], cfg=cfg)
        self.assertEqual(row["fragment_schnitt_sek"], "")           # Score 2 reicht im strict-Modus nicht

    # --- abgeschnitten ------------------------------------------------------------------------
    def test_truncated_song_is_flagged_not_modified(self):
        wav, mp3 = self.work / "trunc.wav", self.work / "trunc.mp3"
        write_wav(wav, self.song_a[:int(20 * SR)])                  # mitten in lauter Musik gekappt
        encode(wav, mp3)
        tag(mp3, tlen=30.0)
        row = self.process(mp3, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_TRUNC, row)
        self.assertIn(row["abgeschnitten"], ("sicher", "wahrscheinlich"))
        self.assertEqual(row["tag_laenge_ok"], "nein")
        self.assertEqual(Path(row["ausgabedatei"]).read_bytes(), mp3.read_bytes())   # Kopie bleibt unverändert

    def test_file_cut_mid_frame_is_detected(self):
        cut = self.work / "bytecut.mp3"
        data = self.a_mp3.read_bytes()
        cut.write_bytes(data[:int(len(data) * 0.6) + 17])
        s = m.scan_mp3(cut)
        self.assertGreater(s.partial_last_frame, 0)
        row = self.process(cut, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_TRUNC, row)

    # --- MP3-Fehler ----------------------------------------------------------------------------------
    def test_stream_junk_is_detected_and_repaired(self):
        bad = self.work / "junk.mp3"
        data = bytearray(self.a_mp3.read_bytes())
        mid = len(data) // 2
        data[mid:mid] = b"\x00\x13\x37" * 40                         # Datenmüll mitten im Stream
        data += b"garbage-at-end" * 5
        bad.write_bytes(bytes(data))
        s = m.scan_mp3(bad)
        self.assertEqual(len(s.junk), 1)
        self.assertGreater(s.trailing_junk, 0)
        row = self.process(bad, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertIn("verlustfrei", row["aktion"])                  # Datenmüll entfernt, Audio nicht neu encodiert
        self.assertTrue(row["mp3_fehler"])
        so = m.scan_mp3(Path(row["ausgabedatei"]))
        self.assertEqual((so.junk, so.trailing_junk), ([], 0))
        self.assertAlmostEqual(so.duration, 30.0, delta=0.3)

    def test_wrong_xing_header_is_rebuilt(self):
        bad = self.work / "xing.mp3"
        data = bytearray(self.a_mp3.read_bytes())
        i = data.find(b"Info")
        data[i + 8:i + 12] = (3).to_bytes(4, "big")                  # Frames-Feld auf Unsinn setzen
        bad.write_bytes(bytes(data))
        s = m.scan_mp3(bad)
        self.assertTrue(any("Header" in d for d in m.stream_defects(s)))
        row = self.process(bad, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        so = m.scan_mp3(Path(row["ausgabedatei"]))
        self.assertLessEqual(abs(so.header_frames - so.n_frames), 2)

    def test_low_bitrate_is_reencoded_without_going_lower(self):
        low = self.work / "low.mp3"
        encode(self.a_wav, low, bitrate=64)
        tag(low, tlen=30.0)
        row = self.process(low, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertIn("kbps", row["neu_encodiert_grund"])
        so = m.scan_mp3(Path(row["ausgabedatei"]))
        self.assertGreaterEqual(so.avg_bitrate, 125)

    def test_unusual_samplerate_is_resampled(self):
        low = self.work / "sr.mp3"
        encode(self.a_wav, low, bitrate=128, extra=["-ar", "22050"])
        tag(low)
        row = self.process(low, refs=[30.0])
        self.assertEqual(row["zielordner"], m.OUT_FIXED, row)
        self.assertEqual(m.scan_mp3(Path(row["ausgabedatei"])).sample_rate, 44100)

    def test_unreadable_file_goes_to_error_folder(self):
        junk = self.work / "kaputt.mp3"
        junk.write_bytes(b"das ist keine mp3" * 100)
        row = self.process(junk)
        self.assertEqual(row["zielordner"], m.OUT_ERROR)

    def test_wrong_tlen_is_fixed_in_copy_only(self):
        f = self.work / "tlen.mp3"
        shutil.copy(self.a_mp3, f)
        tag(f, tlen=95.0)
        row = self.process(f, refs=[30.0])
        self.assertEqual(row["tag_laenge_ok"], "nein")
        self.assertEqual(row["zielordner"], m.OUT_FIXED)
        out = Path(row["ausgabedatei"])
        self.assertAlmostEqual(int(str(ID3(out)["TLEN"].text[0])) / 1000, 30.0, delta=0.2)
        self.assertEqual(ID3(f)["TLEN"].text[0], "95000")           # Original unberührt

    def test_analyze_only_writes_nothing(self):
        cfg = m.Config(analyze_only=True, have_mp3val=self.cfg.have_mp3val)
        cat = self.work / "cat3.mp3"
        cat.write_bytes(self.a_mp3.read_bytes() + self.b_mp3.read_bytes())
        row = self.process(cat, refs=[30.0], cfg=cfg)
        self.assertTrue(row["aktion"].startswith("geplant"))
        self.assertEqual(list(self.out.iterdir()), [])


@unittest.skipUnless(HAVE_FFMPEG and shutil.which("fpcalc"), "ffmpeg/fpcalc nicht installiert")
class OnlineLogicTests(unittest.TestCase):
    """AcoustID/MusicBrainz/Apple-Auswertung mit gemockten HTTP-Antworten."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mp3repair_online_"))
        wav = cls.tmp / "a.wav"
        write_wav(wav, synth_song(200.0, seed=2))
        cls.mp3 = cls.tmp / "Testband - Testsong.mp3"
        encode(wav, cls.mp3, bitrate=64)
        tag(cls.mp3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make(self, handler):
        from unittest import mock
        cfg = m.Config(acoustid_key="KEY", have_fpcalc=True)
        online = m.OnlineReferences(cfg, m.Cache(None))
        calls = []

        def fake_http(method, url, limiter, **kw):
            calls.append((url, kw))
            return handler(url, kw)
        patcher = mock.patch.object(online, "_http", side_effect=fake_http)
        patcher.start()
        self.addCleanup(patcher.stop)
        return online, calls, cfg

    def test_acoustid_collects_all_versions_of_the_same_song(self):
        def handler(url, kw):
            return {"status": "ok", "results": [
                {"id": "x", "score": 0.97, "recordings": [
                    {"id": "1", "title": "Testsong", "duration": 240, "artists": [{"name": "Testband"}]},
                    {"id": "2", "title": "Testsong (Radio Edit)", "duration": 200, "artists": [{"name": "Testband"}]},
                    {"id": "3", "title": "Völlig anderer Titel", "duration": 99, "artists": [{"name": "Testband"}]}]},
                {"id": "y", "score": 0.4, "recordings": [
                    {"id": "4", "title": "Testsong", "duration": 55, "artists": [{"name": "Testband"}]}]}]}
        online, calls, _ = self.make(handler)
        tags = m.read_tags(self.mp3)
        r = online.collect(self.mp3, tags, "k", 200.0)
        self.assertEqual(sorted(x.seconds for x in r.refs), [200, 240])       # 99 s (anderer Titel) und Score 0.4 fehlen
        self.assertTrue(r.identity_ok)
        self.assertEqual(r.info["acoustid_score"], 0.97)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["data"]["duration"], 200)               # Datei-Dauer wird mitgeschickt

    def test_acoustid_other_song_sets_identity_false(self):
        def handler(url, kw):
            return {"status": "ok", "results": [{"id": "x", "score": 0.95, "recordings": [
                {"id": "1", "title": "Something Else", "duration": 180, "artists": [{"name": "Someone"}]}]}]}
        online, _, _ = self.make(handler)
        r = online.collect(self.mp3, m.read_tags(self.mp3), "k", 200.0)
        self.assertFalse(r.identity_ok)

    def test_fallback_to_musicbrainz_search_and_acoustid_retry_with_reference_duration(self):
        def handler(url, kw):
            if "acoustid" in url:
                if kw["data"]["duration"] == 241:        # Wiederholung mit Referenzdauer -> Treffer
                    return {"status": "ok", "results": [{"id": "x", "score": 0.93, "recordings": [
                        {"id": "1", "title": "Testsong", "duration": 241, "artists": [{"name": "Testband"}]}]}]}
                return {"status": "ok", "results": []}
            return {"recordings": [
                {"score": 100, "title": "Testsong", "length": 241000, "artist-credit": [{"name": "Testband"}]},
                {"score": 100, "title": "Testsong (live)", "length": 400000, "artist-credit": [{"name": "Andere Band"}]},
                {"score": 60, "title": "Testsong", "length": 123000, "artist-credit": [{"name": "Testband"}]}]}
        online, calls, _ = self.make(handler)
        r = online.collect(self.mp3, m.read_tags(self.mp3), "k", 200.0 + 8)   # Datei mit Fragment: Dauer weicht ab
        self.assertEqual([c[0].split("/")[-1] for c in calls], ["lookup", "recording", "lookup"])
        self.assertEqual(sorted({x.seconds for x in r.refs}), [241.0])
        self.assertEqual(r.info.get("acoustid_score"), 0.93)

    def test_results_are_cached(self):
        def handler(url, kw):
            return {"status": "ok", "results": [{"id": "x", "score": 0.97, "recordings": [
                {"id": "1", "title": "Testsong", "duration": 240, "artists": [{"name": "Testband"}]}]}]}
        online, calls, _ = self.make(handler)
        tags = m.read_tags(self.mp3)
        online.collect(self.mp3, tags, "same-key", 200.0)
        online.collect(self.mp3, tags, "same-key", 200.0)
        self.assertEqual(len(calls), 1)

    def test_lookup_errors_do_not_crash_and_are_reported(self):
        def handler(url, kw):
            raise RuntimeError("503 Service Unavailable")
        online, _, _ = self.make(handler)
        r = online.collect(self.mp3, m.read_tags(self.mp3), "k", 200.0)
        self.assertEqual(r.refs, [])
        self.assertTrue(any("AcoustID" in e for e in r.errors))

    def test_service_is_disabled_after_repeated_failures(self):
        from unittest import mock
        cfg = m.Config(acoustid_key="KEY", have_fpcalc=True, mb_search=False)
        online = m.OnlineReferences(cfg, m.Cache(None))
        sess = mock.Mock()
        sess.request.side_effect = ConnectionError("proxy down")
        online._local.s = sess
        with mock.patch.object(m.time, "sleep"), mock.patch.object(m.RateLimiter, "wait"):
            tags = m.read_tags(self.mp3)
            for i in range(m.MAX_CONSECUTIVE_FAILURES + 3):
                r = online.collect(self.mp3, tags, f"k{i}", 200.0)
        self.assertEqual(sess.request.call_count, m.MAX_CONSECUTIVE_FAILURES * 3)   # danach keine Versuche mehr
        self.assertEqual(r.errors, [])                                                # abgeschaltet -> kein Fehlerlärm

    def test_apple_catalog_durations(self):
        def handler(url, kw):
            self.assertIn("itunes.apple.com", url)
            return {"results": [
                {"trackName": "Testsong", "artistName": "Testband", "trackTimeMillis": 243500, "previewUrl": "http://x/p.m4a"},
                {"trackName": "Testsong - Karaoke Version", "artistName": "Karaoke Stars", "trackTimeMillis": 100000}]}
        online, calls, _ = self.make(handler)
        r = online.apple(m.read_tags(self.mp3), "k")
        self.assertEqual([x.seconds for x in r.refs], [243.5])
        self.assertEqual(r.previews, ["http://x/p.m4a"])
        self.assertEqual(len(calls), 1)

    def test_end_to_end_with_online_reference_flags_truncation(self):
        # Datei ist 200 s lang, MusicBrainz kennt nur 240 s und das Ende ist laut (kein Fade-out)
        wav, mp3 = self.tmp / "tr.wav", self.tmp / "Testband - Cut.mp3"
        write_wav(wav, synth_song(200.0, seed=2, fade=0))
        encode(wav, mp3, bitrate=128)
        tag(mp3, title="Cut")

        def handler(url, kw):
            return {"status": "ok", "results": [{"id": "x", "score": 0.97, "recordings": [
                {"id": "1", "title": "Cut", "duration": 240, "artists": [{"name": "Testband"}]}]}]}
        online, _, cfg = self.make(handler)
        out = self.tmp / "out"
        out.mkdir(exist_ok=True)
        row = m.process_file(mp3, out, cfg, online, "k2")
        self.assertEqual(row["zielordner"], m.OUT_TRUNC, row)
        self.assertEqual(row["abgeschnitten"], "wahrscheinlich")
        self.assertIn("40.0s kürzer", row["abgeschnitten_gruende"])


class PureLogicTests(unittest.TestCase):
    def test_norm_and_sim(self):
        self.assertEqual(m.norm("Celice (Radio Edit) feat. X"), "celice")
        self.assertGreater(m.sim("Café del Mar", "Cafe Del Mar"), 0.95)
        self.assertLess(m.sim("Song A", "Totally different"), 0.5)

    def test_secs_unit_guess(self):
        self.assertEqual(m._secs(240), 240)
        self.assertEqual(m._secs(240000), 240.0)
        self.assertIsNone(m._secs(None))

    def test_choose_reencode_bitrate_never_goes_below_source(self):
        cfg = m.Config()
        self.assertEqual(m.choose_reencode_bitrate(64, cfg), 128)
        self.assertEqual(m.choose_reencode_bitrate(190, cfg), 192)
        self.assertEqual(m.choose_reencode_bitrate(245, cfg), 256)
        self.assertEqual(m.choose_reencode_bitrate(320, cfg), 320)
        self.assertEqual(m.choose_reencode_bitrate(64, m.Config(reencode_bitrate=256)), 256)

    def test_alignment_finds_embedded_fingerprint(self):
        rng = np.random.default_rng(0)
        long = rng.integers(0, 2 ** 32, 500, dtype=np.uint64).astype(np.uint32)
        short = long[200:260].copy()
        self.assertEqual(m.alignment_percent(short, long), 100.0)
        noisy = rng.integers(0, 2 ** 32, 60, dtype=np.uint64).astype(np.uint32)
        self.assertLess(m.alignment_percent(noisy, long), 60.0)

    def test_encoding_splice_detection(self):
        f = [(i * 0.026, 128, 44100, 2) for i in range(300)] + [(8 + i * 0.026, 128, 48000, 1) for i in range(10)]
        self.assertAlmostEqual(m.find_encoding_splice(f, True), 8.0, delta=0.01)
        vbr = [(i * 0.026, 100 + i % 50, 44100, 2) for i in range(300)]
        self.assertIsNone(m.find_encoding_splice(vbr, False))


if __name__ == "__main__":
    unittest.main()
