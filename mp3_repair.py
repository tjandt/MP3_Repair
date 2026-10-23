#!/usr/bin/env python3
"""
mp3_repair.py
=============

Analysiert alle MP3-Dateien EINES Ordners (nicht rekursiv), erkennt abgeschnittene
Songs, angehängte Fremd-Fragmente und MP3-Stream-Fehler und schreibt reparierte
KOPIEN in einen neuen Ordner. Die Originale werden nie verändert.

Ziel: pro Song eine möglichst saubere MP3, die Apple iTunes Match / Apple Music
(Titel-Abgleich per Audio-Fingerprint) gut gegen den Katalog abgleichen kann.

Was geprüft wird
----------------
1. ID3-Tags (Titel, Interpret, Album, TLEN) werden gelesen, aber NICHT verändert
   (Ausnahme: TLEN wird in geänderten Dateien an die neue Länge angepasst).
2. Echte Länge der Datei (aus den MP3-Frames, nicht aus einem evtl. defekten
   Header) gegen: TLEN-Tag, AcoustID/MusicBrainz (Fingerprint), MusicBrainz-
   Textsuche (aus Tags) und optional die Apple-Music-Katalog-Dauer. Toleranz
   +/- 3 s (``--tolerance``).
3. Angehängte Fragmente: mehrere unabhängige Indizien werden bewertet
   (zweiter Xing/Info-Header oder ID3-Tag mitten im Stream, mehr Frames als der
   Header angibt, Encoding-Wechsel am Ende, "Stille -> Neustart"-Muster in der
   Lautstärkekurve, Schnittpunkt passt zur Referenzlänge). Nur bei genügend
   Indizien wird automatisch getrimmt; unsichere Fälle landen zur Kontrolle im
   Ordner ``04_manuell_pruefen``.
4. Abgeschnittene Songs (Ende mitten im Frame, Header kündigt mehr an als
   vorhanden, kürzer als TLEN/Referenz, Ende ohne Ausblendung). Ein abge-
   schnittener Song lässt sich nicht "reparieren" - er wird markiert.
5. MP3-Fehler: Sync-Verluste/Datenmüll, Müll am Ende, kaputte/fehlende
   Xing-Header, Dekodierfehler (ffmpeg), mp3val-Warnungen. Reparatur zuerst
   verlustfrei (Remux, Audio bleibt bit-identisch), nur wenn das nicht reicht
   oder die Datei für Apple ungeeignet ist (< 96 kbps, ungewöhnliche Samplerate,
   kein MP3-Codec) wird neu encodiert (LAME, nie mit niedrigerer Bitrate als die
   Quelle). Es werden bewusst KEINE Klangänderungen (Loudness-Normalisierung,
   Denoise, Filter) angewendet - sie würden den Fingerprint-Abgleich verschlechtern.

Ausgabe-Unterordner (jede Datei landet in genau EINEM davon, Details im CSV):
    01_ok_unveraendert/   keine Auffälligkeiten (bit-identische Kopie)
    02_repariert/         Fragment getrimmt / MP3-Fehler behoben / neu encodiert
    03_abgeschnitten/     Song ist unvollständig -> Ersatz beschaffen
    04_manuell_pruefen/   unsichere Befunde, nichts automatisch verändert
    05_nicht_lesbar/      Datei nicht lesbar oder Reparatur fehlgeschlagen

Voraussetzungen (macOS / Homebrew):
    brew install ffmpeg mp3val chromaprint
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

AcoustID-Fingerprinting braucht einen kostenlosen API-Key (https://acoustid.org/api-key):
    export ACOUSTID_API_KEY=...      oder   --acoustid-key ...

Aufruf:
    python3 mp3_repair.py ~/Music/1996 ~/Music/1996_repariert --analyze-only   # nur Report
    python3 mp3_repair.py ~/Music/1996 ~/Music/1996_repariert                  # Kopien erzeugen
"""

from __future__ import annotations

import argparse
import bisect
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

try:
    import numpy as np
    from mutagen.id3 import ID3, TLEN
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Fehlende Python-Abhängigkeit ({exc.name}). Bitte 'pip install -r requirements.txt'.")

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # Online-Abfragen werden dann übersprungen


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
DEFAULT_TOLERANCE_SEC = 3.0        # erlaubte Abweichung Länge vs. Metadaten/Referenz

ANALYSIS_SR = 22050                # Downsampling für die Lautstärkekurve
WINDOW_SEC = 0.1                   # Fensterbreite der Lautstärkekurve
MIN_GAP_SEC = 0.2                  # kürzeste Stille-Lücke, die als Songgrenze gilt
MIN_FRAGMENT_SEC = 0.2             # kürzestes Fragment nach der Lücke
GAP_SEARCH_BEFORE_REF_SEC = 8.0    # Lückensuche ab (kürzeste Referenz - X)
GAP_SEARCH_NO_REF_SEC = 60.0       # ohne Referenz: nur die letzten X Sekunden
ABRUPT_CHECK_SEC = 1.0             # Länge des Endstücks für den Ausblendungs-Check
ABRUPT_LEVEL_DB = 10.0             # Endstück näher als X dB am Songpegel = "abrupt"

GROUP_SEC = 1.5                    # Indizien innerhalb X s gelten als derselbe Schnitt
REF_AGREE_SEC = 1.5                # Schnitt passt zur Referenz, wenn |Schnitt-Ref| <= X
MIN_CUT_SEC = 10.0                 # nie vor X Sekunden schneiden
HEADER_EXTRA_FRAMES = 3            # so viele Frames mehr als im Header = Indiz

MIN_ITUNES_BITRATE_KBPS = 96       # Apples Untergrenze fürs Matching/Hochladen
STANDARD_SAMPLE_RATES = {44100, 48000}
STANDARD_BITRATES_KBPS = [96, 112, 128, 160, 192, 224, 256, 320]
REENCODE_MIN_BITRATE_KBPS = 128
REENCODE_SAMPLE_RATE = 44100

OUT_OK = "01_ok_unveraendert"
OUT_FIXED = "02_repariert"
OUT_TRUNC = "03_abgeschnitten"
OUT_REVIEW = "04_manuell_pruefen"
OUT_ERROR = "05_nicht_lesbar"

USER_AGENT = "mp3-repair/1.0 ({contact})"
LABELS = {0: "nein", 1: "unsicher", 2: "wahrscheinlich"}


def label(score: int) -> str:
    return "sicher" if score >= 3 else LABELS[max(0, score)]


@dataclass
class Config:
    tol: float = DEFAULT_TOLERANCE_SEC
    acoustid_key: str | None = None
    mb_search: bool = True
    apple: str = "suspicious"          # off | suspicious | all
    country: str = "de"
    verify_preview: bool = False
    strict: bool = False
    reencode_bitrate: int | None = None
    analyze_only: bool = False
    copy_unchanged: bool = True
    contact: str = "private-script"
    have_mp3val: bool = False
    have_fpcalc: bool = False


# ---------------------------------------------------------------------------
# Kleine Helfer
# ---------------------------------------------------------------------------
def run(cmd: list[str], timeout: int = 900, binary: bool = False) -> subprocess.CompletedProcess:
    if binary:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)


def norm(s: str | None) -> str:
    """Vergleichsform für Titel/Interpret: ohne Akzente, Klammerzusätze, feat.-Teil, Satzzeichen."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\b(feat|ft|featuring)\b.*", " ", s)
    s = s.replace("&", " and ")
    return re.sub(r"[\W_]+", " ", s).strip()


def sim(a: str | None, b: str | None) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def fast_copy(src: Path, dst: Path) -> None:
    """Kopie; auf macOS/APFS als Clone (kostet keinen zusätzlichen Platz)."""
    if sys.platform == "darwin":
        r = subprocess.run(["cp", "-cp", str(src), str(dst)], capture_output=True)
        if r.returncode == 0:
            return
    shutil.copy2(src, dst)


class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next > now:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


class Cache:
    """Merkt sich Online-Abfragen, damit ein erneuter Lauf nicht alles neu abfragt."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict = {}
        self._lock = threading.Lock()
        self._dirty = 0
        if path and path.is_file():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, key: str, sub: str):
        return self.data.get(key, {}).get(sub)

    def put(self, key: str, sub: str, value) -> None:
        with self._lock:
            self.data.setdefault(key, {})[sub] = value
            self._dirty += 1
            if self._dirty >= 25:
                self._save_locked()

    def _save_locked(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = 0

    def save(self) -> None:
        with self._lock:
            self._save_locked()


# ---------------------------------------------------------------------------
# ID3-Tags
# ---------------------------------------------------------------------------
@dataclass
class Tags:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    tlen: float | None = None          # Sekunden
    has_id3: bool = False


def read_tags(path: Path) -> Tags:
    t = Tags()
    try:
        id3 = ID3(path)
        t.has_id3 = True

        def first(key):
            f = id3.get(key)
            return str(f.text[0]).strip() if f is not None and f.text else None

        t.title, t.artist, t.album = first("TIT2"), first("TPE1"), first("TALB")
        raw = first("TLEN")
        if raw:
            t.tlen = float(raw) / 1000.0
    except Exception:
        pass
    if not (t.title and t.artist) and " - " in path.stem:  # Fallback: "Interpret - Titel.mp3"
        a, _, b = path.stem.partition(" - ")
        t.artist = t.artist or a.strip()
        t.title = t.title or b.strip()
    return t


def write_tags(src: Path, dst: Path, new_duration: float | None, fresh: bool) -> bool:
    """Überträgt alle ID3v2-Frames (inkl. Cover) von src nach dst; TLEN wird, falls vorhanden,
    auf new_duration gesetzt. Gibt True zurück, wenn TLEN geändert wurde."""
    try:
        tags = ID3(src)
    except Exception:
        return False
    changed = False
    if new_duration is not None and "TLEN" in tags:
        old = tags["TLEN"].text[0] if tags["TLEN"].text else None
        new = str(int(round(new_duration * 1000)))
        if str(old) != new:
            tags.setall("TLEN", [TLEN(encoding=3, text=new)])
            changed = True
    version = 3 if tags.version[1] < 4 else 4
    if fresh:
        tags.save(dst, v1=0, v2_version=version)
    else:
        tags.save(dst, v2_version=version)
    return changed


# ---------------------------------------------------------------------------
# MP3-Frame-Scanner (eigener Parser: unabhängig von evtl. defekten Headern)
# ---------------------------------------------------------------------------
_BR_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_BR_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
_SR = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


class Hdr(NamedTuple):
    offset: int
    length: int
    bitrate: int
    sr: int
    mode: int
    ver: int       # 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
    spf: int       # Samples pro Frame
    crc: bool


def parse_header(d: bytes, i: int) -> Hdr | None:
    if i < 0 or i + 4 > len(d) or d[i] != 0xFF or (d[i + 1] & 0xE0) != 0xE0:
        return None
    b1, b2, b3 = d[i + 1], d[i + 2], d[i + 3]
    ver, layer = (b1 >> 3) & 3, (b1 >> 1) & 3
    if ver == 1 or layer != 1:                      # nur MPEG-1/2/2.5 Layer III
        return None
    br_idx, sr_idx = b2 >> 4, (b2 >> 2) & 3
    if br_idx in (0, 15) or sr_idx == 3:
        return None
    bitrate = (_BR_V1 if ver == 3 else _BR_V2)[br_idx]
    sr = _SR[ver][sr_idx]
    spf = 1152 if ver == 3 else 576
    length = (spf // 8) * bitrate * 1000 // sr + ((b2 >> 1) & 1)
    return Hdr(i, length, bitrate, sr, (b3 >> 6) & 3, ver, spf, (b1 & 1) == 0)


def _plausible(d: bytes, pos: int, end: int) -> Hdr | None:
    """Header, dessen Folge-Frame ebenfalls passt (schützt vor Scheintreffern im Datenmüll)."""
    h = parse_header(d, pos)
    if h is None:
        return None
    nxt = pos + h.length
    if nxt + 4 > end:
        return h
    h2 = parse_header(d, nxt)
    return h if h2 and h2.ver == h.ver and h2.sr == h.sr else None


def id3v2_size(d: bytes, pos: int) -> int:
    if d[pos:pos + 3] != b"ID3" or len(d) < pos + 10:
        return 0
    s = d[pos + 6:pos + 10]
    if any(b & 0x80 for b in s):
        return 0
    return 10 + ((s[0] << 21) | (s[1] << 14) | (s[2] << 7) | s[3]) + (10 if d[pos + 5] & 0x10 else 0)


def _xing_info(d: bytes, h: Hdr):
    """(Art, Frames, Bytes, Encoder) wenn der Frame ein Xing/Info/VBRI-Header ist, sonst None."""
    ch = 1 if h.mode == 3 else 2
    side = (32 if ch != 1 else 17) if h.ver == 3 else (17 if ch != 1 else 9)
    off = h.offset + 4 + (2 if h.crc else 0) + side
    tag = d[off:off + 4]
    if tag in (b"Xing", b"Info") and off + 8 <= h.offset + h.length:
        flags = int.from_bytes(d[off + 4:off + 8], "big")
        p, frames, nbytes = off + 8, None, None
        if flags & 1:
            frames, p = int.from_bytes(d[p:p + 4], "big"), p + 4
        if flags & 2:
            nbytes, p = int.from_bytes(d[p:p + 4], "big"), p + 4
        if flags & 4:
            p += 100
        if flags & 8:
            p += 4
        enc = re.sub(r"[^\x20-\x7e]", "", d[p:p + 9].decode("latin-1")).strip()
        return tag.decode(), frames, nbytes, enc
    voff = h.offset + 4 + 32
    if d[voff:voff + 4] == b"VBRI":
        return "VBRI", int.from_bytes(d[voff + 14:voff + 18], "big"), int.from_bytes(d[voff + 10:voff + 14], "big"), ""
    return None


@dataclass
class FrameScan:
    ok: bool = False
    n_frames: int = 0
    duration: float = 0.0
    avg_bitrate: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    mpeg: str = ""
    is_cbr: bool = True
    header_kind: str = ""
    header_frames: int | None = None
    header_duration: float | None = None
    encoder: str = ""
    junk: list = field(default_factory=list)         # (Offset, Länge) mitten im Stream
    trailing_junk: int = 0                           # Bytes (ungleich 0) hinter dem letzten Frame
    partial_last_frame: int = 0                      # Bytes eines abgeschnittenen letzten Frames
    mid_info_times: list = field(default_factory=list)
    mid_tag_times: list = field(default_factory=list)
    splice_time: float | None = None
    frame_times: list = field(default_factory=list)      # Startzeit je Audio-Frame
    frame_offsets: list = field(default_factory=list)    # Byte-Offset je Audio-Frame
    frame_ends: list = field(default_factory=list)       # Byte-Ende je Audio-Frame


def scan_mp3(path: Path) -> FrameScan:
    s = FrameScan()
    try:
        d = path.read_bytes()
    except OSError:
        return s
    pos = 0
    while True:
        sz = id3v2_size(d, pos)
        if not sz:
            break
        pos += sz
    end = len(d)
    if end - pos >= 128 and d[end - 128:end - 125] == b"TAG":          # ID3v1
        end -= 128
    if end - pos >= 32 and d[end - 32:end - 24] == b"APETAGEX":        # APEv2
        size = int.from_bytes(d[end - 20:end - 16], "little")
        has_header = int.from_bytes(d[end - 12:end - 8], "little") & (1 << 31)
        end -= min(end - pos, size + (32 if has_header else 0))

    frames: list[tuple] = []
    elapsed, audio_bytes, spf_seen, ver_seen = 0.0, 0, 1152, 3
    in_sync, first = False, True
    trailing = (0, 0)
    while pos + 4 <= end:
        h = parse_header(d, pos) if in_sync else _plausible(d, pos, end)
        if h is None:
            tag_len = id3v2_size(d, pos)
            if tag_len:
                s.mid_tag_times.append(elapsed)
                pos, in_sync = pos + tag_len, False
                continue
            if d[pos:pos + 3] == b"TAG" and pos + 128 <= end:
                s.mid_tag_times.append(elapsed)
                pos, in_sync = pos + 128, False
                continue
            j, found = pos + 1, -1
            while True:
                j = d.find(b"\xff", j, end - 3)
                if j == -1:
                    break
                if _plausible(d, j, end):
                    found = j
                    break
                j += 1
            if found == -1:
                trailing = (pos, end - pos)
                pos = end
                break
            s.junk.append((pos, found - pos))
            pos, in_sync = found, True
            continue
        in_sync = True
        if pos + h.length > end:
            s.partial_last_frame = end - pos
            break
        info = _xing_info(d, h)
        if first:
            first = False
            if info:
                s.header_kind, s.header_frames, _, s.encoder = info
                spf_seen, ver_seen = h.spf, h.ver
                pos += h.length
                continue
        elif info and info[0] in ("Xing", "Info"):
            s.mid_info_times.append(elapsed)
            pos += h.length
            continue
        frames.append((elapsed, h.bitrate, h.sr, 1 if h.mode == 3 else 2, pos, pos + h.length))
        elapsed += h.spf / h.sr
        audio_bytes += h.length
        spf_seen, ver_seen = h.spf, h.ver
        pos += h.length

    if trailing[1] and d[trailing[0]:trailing[0] + trailing[1]].strip(b"\x00"):
        s.trailing_junk = trailing[1]

    s.n_frames = len(frames)
    if not frames:
        return s
    s.ok = True
    s.duration = elapsed
    s.avg_bitrate = audio_bytes * 8 / elapsed / 1000 if elapsed else None
    s.sample_rate = Counter(f[2] for f in frames).most_common(1)[0][0]
    s.channels = Counter(f[3] for f in frames).most_common(1)[0][0]
    s.mpeg = {3: "MPEG-1", 2: "MPEG-2", 0: "MPEG-2.5"}[ver_seen]
    s.is_cbr = Counter(f[1] for f in frames).most_common(1)[0][1] >= 0.98 * len(frames)
    if s.header_frames is not None:
        s.header_duration = s.header_frames * spf_seen / s.sample_rate
    s.splice_time = find_encoding_splice(frames, s.is_cbr)
    s.frame_times = [f[0] for f in frames]
    s.frame_offsets = [f[4] for f in frames]
    s.frame_ends = [f[5] for f in frames]
    return s


def find_encoding_splice(frames: list, is_cbr: bool) -> float | None:
    """Wechsel von Samplerate/Kanalmodus (bei CBR auch Bitrate) bis zum Dateiende -> gespleißtes Fragment."""
    if len(frames) < 50:
        return None
    key = (lambda f: (f[2], f[3], f[1])) if is_cbr else (lambda f: (f[2], f[3]))
    dom, cnt = Counter(key(f) for f in frames).most_common(1)[0]
    if cnt < 0.8 * len(frames):
        return None
    i = len(frames)
    while i > 0 and key(frames[i - 1]) != dom:
        i -= 1
    return frames[i][0] if 0 < i <= len(frames) - 3 else None


# ---------------------------------------------------------------------------
# Dekodieren + Lautstärkekurve
# ---------------------------------------------------------------------------
def decode_mono(path: Path) -> tuple[np.ndarray, list[str]]:
    """Dekodiert die Datei einmal komplett. Rückgabe: (Samples, Dekodierfehler)."""
    try:
        r = run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
                 "-ar", str(ANALYSIS_SR), "-f", "f32le", "-"], timeout=1800, binary=True)
    except Exception as exc:
        return np.zeros(0, dtype=np.float32), [f"ffmpeg: {exc}"]
    errs = [l.strip() for l in r.stderr.decode(errors="replace").splitlines() if l.strip()]
    usable = len(r.stdout) // 4 * 4
    return np.frombuffer(r.stdout[:usable], dtype=np.float32), errs


@dataclass
class Profile:
    db: np.ndarray
    win: float
    level: float          # typischer Pegel des Songs (Median der nicht stillen Fenster)
    quiet_thr: float      # darunter gilt ein Fenster als (fast) Stille
    decoded_duration: float


def build_profile(samples: np.ndarray, sr: int = ANALYSIS_SR, win: float = WINDOW_SEC) -> Profile | None:
    w = int(round(sr * win))
    n = len(samples)
    if n < w * 5:
        return None
    full = n // w
    x = samples[:full * w].reshape(full, w).astype(np.float64)
    rms = list(np.sqrt((x * x).mean(axis=1) + 1e-12))
    rem = samples[full * w:]
    if len(rem) >= int(0.02 * sr):          # letztes Teilfenster NICHT verwerfen (kurze Fragmente!)
        r = rem.astype(np.float64)
        rms.append(float(np.sqrt((r * r).mean() + 1e-12)))
    db = 20.0 * np.log10(np.asarray(rms) + 1e-9)
    active = db[db > -55]
    if len(active) < 10:
        return None
    level = float(np.median(active))
    return Profile(db, win, level, min(level - 25.0, -38.0), n / sr)


def last_active_index(p: Profile, t_end: float) -> int:
    i = min(len(p.db), int(round(t_end / p.win)))
    while i > 0 and p.db[i - 1] < p.quiet_thr:
        i -= 1
    return i


def music_end(p: Profile, t_end: float) -> float:
    """Zeitpunkt, an dem vor t_end der letzte hörbare Ton endet."""
    return last_active_index(p, t_end) * p.win


def ends_abruptly(p: Profile, t_end: float) -> bool:
    """True, wenn die Musik vor t_end noch laut ist (kein Ausklingen/Fade-out)."""
    i1 = last_active_index(p, t_end)
    seg = p.db[max(0, i1 - int(ABRUPT_CHECK_SEC / p.win)):i1]
    return len(seg) > 0 and float(seg.mean()) >= p.level - ABRUPT_LEVEL_DB


@dataclass
class Gap:
    start: float
    end: float
    cut: float


def find_gaps(p: Profile, search_start: float) -> list[Gap]:
    """Muster 'Stille-Lücke -> danach wieder (überwiegend) laute Musik bis Dateiende'."""
    db, w, n = p.db, p.win, len(p.db)
    quiet = db < p.quiet_thr
    out, i = [], max(1, int(search_start / w))
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        run_len = (j - i) * w
        if run_len >= MIN_GAP_SEC and j < n:
            rest = db[j:]
            if len(rest) * w >= MIN_FRAGMENT_SEC and float((rest >= p.level - 22).mean()) >= 0.6:
                out.append(Gap(i * w, j * w, i * w + min(0.4, run_len / 2)))
        i = j
    return out


# ---------------------------------------------------------------------------
# Bewertung: Fragment / Abschnitt / Länge
# ---------------------------------------------------------------------------
@dataclass
class Ref:
    source: str
    seconds: float
    own: bool = False      # True = steht in der Datei selbst (TLEN)


@dataclass
class Verdict:
    nearest: Ref | None = None
    diff: float | None = None                 # Ist-Länge minus nächste Referenz
    length_ok: bool | None = None             # Ist-Länge passt zu mindestens einer Referenz
    frag_score: int = 0
    frag_reasons: list = field(default_factory=list)
    cut: float | None = None                  # Schnittpunkt (nur wenn automatisch getrimmt wird)
    trunc_score: int = 0
    trunc_reasons: list = field(default_factory=list)
    final_length: float = 0.0
    explained_by_silence: bool = False
    unresolved_length: bool = False


def evaluate(actual: float, scan: FrameScan, profile: Profile | None, refs: list[Ref], cfg: Config) -> Verdict:
    v = Verdict(final_length=actual)
    tol = cfg.tol
    if refs:
        v.nearest = min(refs, key=lambda r: abs(actual - r.seconds))
        v.diff = actual - v.nearest.seconds
        v.length_ok = abs(v.diff) <= tol

    # ---- Fragment am Ende ---------------------------------------------------
    # Kandidat: (Zeit, Begründung, Gewicht, Priorität)
    cands: list[tuple] = []
    for t in scan.mid_info_times:
        cands.append((t, "zweiter Xing/Info-Header im Stream (aneinandergehängte Datei)", 2, 4))
    for t in scan.mid_tag_times:
        cands.append((t, "ID3-Tag mitten im Stream (aneinandergehängte Datei)", 2, 4))
    if scan.header_frames is not None and scan.n_frames - scan.header_frames >= HEADER_EXTRA_FRAMES:
        t = scan.header_duration
        cands.append((t, "mehr Frames vorhanden als der Header angibt", 1, 3))
    if scan.splice_time is not None:
        cands.append((scan.splice_time, "Encoding-Wechsel (Bitrate/Samplerate/Kanäle) am Ende", 1, 2))
    if profile is not None:
        below = [r.seconds for r in refs if r.seconds < actual - 0.3]
        start = max(0.0, min(below) - GAP_SEARCH_BEFORE_REF_SEC) if below else max(0.0, actual - GAP_SEARCH_NO_REF_SEC)
        for g in find_gaps(profile, start):
            cands.append((g.cut, "Stille-Lücke gefolgt von neuem lautem Audio bis Dateiende", 1, 1))

    cands.sort(key=lambda c: c[0])
    groups: list[list[tuple]] = []
    for c in cands:
        if groups and c[0] - groups[-1][0][0] <= GROUP_SEC:
            groups[-1].append(c)
        else:
            groups.append([c])

    best = None
    for g in groups:
        main = max(g, key=lambda c: c[3])
        t = main[0]
        if t < MIN_CUT_SEC or actual - t < 0.15:
            continue
        weights = {}
        for c in g:
            weights[c[1]] = max(weights.get(c[1], 0), c[2])
        score, reasons = sum(weights.values()), list(weights)
        near = min((r for r in refs if r.seconds < actual - 0.3), key=lambda r: abs(r.seconds - t), default=None)
        if (near and abs(near.seconds - t) <= REF_AGREE_SEC
                and actual - near.seconds >= max(0.5, 0.5 * (actual - t))):
            score += 1
            reasons.append(f"Schnittpunkt {t:.1f}s passt zur Referenz {near.source} {near.seconds:.1f}s")
            if actual - near.seconds > tol:
                score += 1
                reasons.append(f"Datei {actual - near.seconds:.1f}s länger als Referenz")
        if best is None or score > best[0]:
            best = (score, t, reasons)
    if best:
        v.frag_score, v.frag_reasons = best[0], best[2]
        if best[0] >= (3 if cfg.strict else 2):
            v.cut = best[1]
            v.final_length = best[1]
    end_t = v.final_length

    # ---- Abgeschnitten --------------------------------------------------------
    confirms = any(abs(end_t - r.seconds) <= tol for r in refs)
    tr_score, tr = 0, []
    if scan.partial_last_frame:
        tr_score += 1
        tr.append("Datei endet mitten in einem MP3-Frame")
    if scan.header_frames is not None and scan.header_frames - scan.n_frames >= HEADER_EXTRA_FRAMES:
        tr_score += 2
        tr.append(f"Header kündigt {scan.header_duration:.1f}s an, vorhanden sind {scan.duration:.1f}s")
    if not confirms:
        tlen = next((r for r in refs if r.own), None)
        others = [r for r in refs if not r.own]
        if tlen and end_t < tlen.seconds - tol:
            tr_score += 2
            tr.append(f"{tlen.seconds - end_t:.1f}s kürzer als TLEN-Tag")
        elif others and all(end_t < r.seconds - tol for r in others):
            near_o = max(others, key=lambda r: r.seconds)
            tr_score += 1
            tr.append(f"{near_o.seconds - end_t:.1f}s kürzer als alle Referenzen ({near_o.source})")
        if profile is not None and ends_abruptly(profile, end_t):
            tr_score += 1
            tr.append("Ende ohne Ausblenden, mitten in lauter Musik")
    v.trunc_score, v.trunc_reasons = tr_score, tr

    # ---- Länge bleibt unerklärt? ------------------------------------------------
    if refs and not confirms and tr_score < 2:
        if profile is not None:
            stop = music_end(profile, end_t)
            if any(abs(stop - r.seconds) <= tol for r in refs):
                v.explained_by_silence = True
        v.unresolved_length = not v.explained_by_silence
    return v


# ---------------------------------------------------------------------------
# Referenzlängen aus dem Netz (AcoustID/MusicBrainz, Apple)
# ---------------------------------------------------------------------------
@dataclass
class RefResult:
    refs: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    identity_ok: bool | None = None
    errors: list = field(default_factory=list)
    previews: list = field(default_factory=list)


def _secs(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v / 1000.0 if v > 20000 else v


def _add_ref(out: RefResult, source: str, seconds: float | None) -> None:
    if seconds and not any(r.source == source and abs(r.seconds - seconds) < 1.0 for r in out.refs):
        out.refs.append(Ref(source, float(seconds)))


class ServiceDisabled(Exception):
    """Dienst wurde nach wiederholten Fehlern für den Rest des Laufs abgeschaltet."""


MAX_CONSECUTIVE_FAILURES = 5


class OnlineReferences:
    def __init__(self, cfg: Config, cache: Cache):
        self.cfg, self.cache = cfg, cache
        self._local = threading.local()
        self.lim_acoustid, self.lim_mb, self.lim_apple = RateLimiter(0.4), RateLimiter(1.1), RateLimiter(3.2)
        self._fails: dict = {}
        self._lock = threading.Lock()

    def _session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
            self._local.s.headers["User-Agent"] = USER_AGENT.format(contact=self.cfg.contact)
        return self._local.s

    def _http(self, method: str, url: str, limiter: RateLimiter, **kw) -> dict:
        host = url.split("/")[2]
        if self._fails.get(limiter, 0) >= MAX_CONSECUTIVE_FAILURES:
            raise ServiceDisabled(host)
        last = None
        for attempt in range(3):
            limiter.wait()
            try:
                r = self._session().request(method, url, timeout=30, **kw)
                if r.status_code in (429, 503):
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                self._fails[limiter] = 0
                return data
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        with self._lock:
            self._fails[limiter] = self._fails.get(limiter, 0) + 1
            if self._fails[limiter] == MAX_CONSECUTIVE_FAILURES:
                print(f"\nWARNUNG: {host} antwortet nicht ({MAX_CONSECUTIVE_FAILURES} Fehlschläge in Folge) - "
                      f"wird für den Rest des Laufs übersprungen. Ergebnis ist trotzdem im Report, "
                      f"nur ohne diese Referenzquelle.", flush=True)
        raise RuntimeError(f"{type(last).__name__}: {str(last)[:100]}")

    # -- AcoustID ------------------------------------------------------------
    def _acoustid(self, path: Path, duration_override: int | None = None) -> dict:
        r = run(["fpcalc", "-json", "-length", "120", str(path)], timeout=300)
        fp = json.loads(r.stdout)
        duration = duration_override or int(round(float(fp["duration"])))
        resp = self._http("POST", "https://api.acoustid.org/v2/lookup", self.lim_acoustid, data={
            "client": self.cfg.acoustid_key, "duration": duration,
            "fingerprint": fp["fingerprint"], "meta": "recordings"})
        results = []
        for res in resp.get("results", []):
            recs = [{"title": rec.get("title"), "duration": _secs(rec.get("duration")),
                     "artist": ", ".join(a.get("name", "") for a in rec.get("artists", []) or [])}
                    for rec in res.get("recordings", []) or []]
            results.append({"score": res.get("score", 0), "recordings": recs})
        return {"results": results}

    def _apply_acoustid(self, ac: dict, tags: Tags, out: RefResult) -> None:
        best = None
        for res in ac.get("results", []):
            for rec in res["recordings"]:
                ts, as_ = sim(tags.title, rec.get("title")), sim(tags.artist, rec.get("artist"))
                rank = (ts + 0.5 * as_, res["score"])
                if best is None or rank > best[0]:
                    best = (rank, res, rec, ts, as_)
        if best is None:
            return
        _, res, rec, ts, as_ = best
        out.info.update(acoustid_score=round(res["score"], 3), acoustid_titel=rec.get("title"),
                        acoustid_interpret=rec.get("artist"))
        if tags.title and tags.artist:
            out.identity_ok = ts >= 0.6 and as_ >= 0.5
        for r in ac["results"]:
            if r["score"] < 0.8:
                continue
            for rc in r["recordings"]:
                if rc["duration"] and (not tags.title or sim(tags.title, rc.get("title")) >= 0.8
                                       or (not tags.artist and rc is rec)):
                    _add_ref(out, "AcoustID/MusicBrainz", rc["duration"])

    # -- MusicBrainz-Textsuche über die Tags ---------------------------------------
    def _mb_search(self, tags: Tags) -> list:
        esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
        q = f'recording:"{esc(tags.title)}" AND artist:"{esc(tags.artist)}"'
        resp = self._http("GET", "https://musicbrainz.org/ws/2/recording", self.lim_mb,
                          params={"query": q, "fmt": "json", "limit": 10})
        hits = []
        for rec in resp.get("recordings", []):
            credit = ", ".join(c.get("name", "") for c in rec.get("artist-credit", []) if isinstance(c, dict))
            if (rec.get("score", 0) >= 90 and rec.get("length") and sim(tags.title, rec.get("title")) >= 0.85
                    and sim(tags.artist, credit) >= 0.6):
                hits.append(_secs(rec["length"]))
        return hits

    def collect(self, path: Path, tags: Tags, key: str, file_duration: float) -> RefResult:
        out, cfg = RefResult(), self.cfg
        if requests is None:
            return out
        ac = None
        if cfg.acoustid_key and cfg.have_fpcalc:
            ac = self.cache.get(key, "acoustid")
            if ac is None:
                try:
                    ac = self._acoustid(path)
                    self.cache.put(key, "acoustid", ac)
                except ServiceDisabled:
                    pass
                except Exception as exc:
                    out.errors.append(f"AcoustID: {exc}")
            if ac:
                self._apply_acoustid(ac, tags, out)
        if not out.refs and cfg.mb_search and tags.title and tags.artist:
            hits = self.cache.get(key, "mb")
            if hits is None:
                try:
                    hits = self._mb_search(tags)
                    self.cache.put(key, "mb", hits)
                except ServiceDisabled:
                    hits = []
                except Exception as exc:
                    out.errors.append(f"MusicBrainz: {exc}")
                    hits = []
            for h in hits:
                _add_ref(out, "MusicBrainz-Suche", h)
            # Ein angehängtes Fragment verfälscht die Dauer, mit der AcoustID filtert -> mit Referenzdauer wiederholen
            if (hits and ac is not None and not ac.get("results") and cfg.acoustid_key and cfg.have_fpcalc
                    and file_duration > 130):
                retry = self.cache.get(key, "acoustid_retry")
                if retry is None:
                    try:
                        retry = self._acoustid(path, duration_override=int(round(hits[0])))
                        self.cache.put(key, "acoustid_retry", retry)
                    except ServiceDisabled:
                        pass
                    except Exception as exc:
                        out.errors.append(f"AcoustID (Wiederholung): {exc}")
                if retry:
                    self._apply_acoustid(retry, tags, out)
        return out

    # -- Apple-Katalog (iTunes Search API) ---------------------------------------------
    def apple(self, tags: Tags, key: str) -> RefResult:
        out = RefResult()
        if requests is None or not (tags.title and tags.artist):
            return out
        hits = self.cache.get(key, "apple")
        if hits is None:
            hits = []
            title_plain = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*$", "", tags.title).strip()
            queries = [f"{tags.artist} {tags.title}"]
            if title_plain and title_plain != tags.title:
                queries.append(f"{tags.artist} {title_plain}")
            countries = [self.cfg.country] + [c for c in ("de", "us") if c != self.cfg.country]
            try:
                for country in countries:
                    for q in queries:
                        resp = self._http("GET", "https://itunes.apple.com/search", self.lim_apple, params={
                            "term": q, "media": "music", "entity": "song", "limit": 10, "country": country})
                        for r in resp.get("results", []):
                            if (r.get("trackTimeMillis") and sim(tags.title, r.get("trackName")) >= 0.85
                                    and sim(tags.artist, r.get("artistName")) >= 0.6):
                                hits.append({"duration": r["trackTimeMillis"] / 1000.0,
                                             "preview": r.get("previewUrl"), "country": country})
                        if hits:
                            break
                    if hits:
                        break
                self.cache.put(key, "apple", hits)
            except ServiceDisabled:
                pass
            except Exception as exc:
                out.errors.append(f"Apple: {exc}")
        for h in hits:
            _add_ref(out, "Apple Music", h["duration"])
            if h.get("preview"):
                out.previews.append(h["preview"])
        return out

    def verify_preview(self, path: Path, previews: list) -> float | None:
        """Vergleicht den Fingerprint von Apples ~30-s-Vorschau mit der lokalen Datei (0-100 %)."""
        local = fingerprint_raw(path, 600)
        if local is None:
            return None
        best = None
        with tempfile.TemporaryDirectory() as tmp:
            for i, url in enumerate(previews[:3]):
                try:
                    self.lim_apple.wait()
                    data = self._session().get(url, timeout=30)
                    data.raise_for_status()
                    p = Path(tmp) / f"preview{i}.m4a"
                    p.write_bytes(data.content)
                    short = fingerprint_raw(p, 60)
                except Exception:
                    continue
                pct = alignment_percent(short, local) if short is not None else None
                if pct is not None and (best is None or pct > best):
                    best = pct
        return best


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def fingerprint_raw(path: Path, length: int) -> np.ndarray | None:
    try:
        r = run(["fpcalc", "-raw", "-length", str(length), str(path)], timeout=300)
    except Exception:
        return None
    m = re.search(r"FINGERPRINT=([\d,\-]+)", r.stdout)
    if not m:
        return None
    vals = [int(x) & 0xFFFFFFFF for x in m.group(1).split(",") if x]
    return np.array(vals, dtype=np.uint32) if vals else None


def alignment_percent(short: np.ndarray, long: np.ndarray) -> float | None:
    """Schiebt den kurzen Fingerprint über den langen; beste Übereinstimmung in % (50 % = Zufall)."""
    n, m = len(short), len(long)
    if n == 0 or m < n:
        return None
    idx = np.arange(n)[None, :] + np.arange(m - n + 1)[:, None]
    x = np.ascontiguousarray(long[idx] ^ short[None, :])
    bits = _POPCOUNT8[x.view(np.uint8)].sum(axis=1, dtype=np.int64)
    return 100.0 * (1.0 - float(bits.min()) / n / 32.0)


# ---------------------------------------------------------------------------
# Fehlerprüfung + Reparatur
# ---------------------------------------------------------------------------
def mp3val_issues(path: Path) -> list[str]:
    try:
        r = run(["mp3val", str(path)], timeout=300)
    except Exception as exc:
        return [f"mp3val: {exc}"]
    out = []
    for line in r.stdout.splitlines():
        m = re.match(r'^(WARNING|ERROR): ".*?": (.*)$', line.strip())
        if m and "No supported tags" not in m.group(2):          # fehlende Tags sind kein MP3-Fehler
            out.append(re.sub(r"\s*\(offset 0x[0-9a-fA-F]+\)", "", m.group(2)).strip())
    return out


def probe_codec(path: Path) -> str | None:
    try:
        r = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                 "-of", "csv=p=0", str(path)], timeout=60)
        return r.stdout.strip() or None
    except Exception:
        return None


def stream_defects(scan: FrameScan) -> list[str]:
    out = []
    if scan.junk:
        out.append(f"{len(scan.junk)}x Sync-Verlust ({sum(l for _, l in scan.junk)} Byte Datenmüll im Stream)")
    if scan.trailing_junk:
        out.append(f"{scan.trailing_junk} Byte Datenmüll am Dateiende")
    if scan.partial_last_frame:
        out.append("unvollständiger letzter Frame")
    if scan.header_frames is not None and abs(scan.header_frames - scan.n_frames) > 2:
        out.append(f"Xing/Info-Header passt nicht zur Frame-Zahl ({scan.header_frames} vs. {scan.n_frames})")
    if scan.ok and not scan.is_cbr and not scan.header_kind:
        out.append("VBR-Datei ohne Xing-Header (Dauer/Bitrate werden falsch angezeigt)")
    if scan.mid_info_times:
        out.append(f"{len(scan.mid_info_times)}x zusätzlicher Xing/Info-Header mitten im Stream")
    return out


def choose_reencode_bitrate(avg: float | None, cfg: Config) -> int:
    if cfg.reencode_bitrate:
        return cfg.reencode_bitrate
    if not avg:
        return 192
    target = next((b for b in STANDARD_BITRATES_KBPS if b >= avg - 1), 320)   # nie unter der Quelle
    return min(320, max(REENCODE_MIN_BITRATE_KBPS, target))


def ffmpeg_build(src: Path, dst: Path, mode: str, cut: float | None, bitrate: int, resample: bool) -> bool:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:a:0"]
    if cut is not None:
        cmd += ["-t", f"{cut:.3f}"]
    if mode == "copy":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"]
        if resample:
            cmd += ["-ar", str(REENCODE_SAMPLE_RATE)]
    cmd += ["-map_metadata", "-1", "-id3v2_version", "0", "-write_xing", "1", str(dst)]
    try:
        r = run(cmd, timeout=1800)
        return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0
    except Exception:
        return False


def clean_stream_bytes(path: Path, scan: FrameScan, cut: float | None = None) -> bytes | None:
    """Verlustfreier Neuaufbau des Streams auf Byte-Ebene: nur valide Audio-Frames (ohne Datenmüll,
    abgeschnittenen Rest-Frame, ID3/Info-Frames mitten im Stream), optional nur die Frames vor `cut`.
    Deterministischer als `ffmpeg -c copy -t` genau auf einer Frame-Grenze."""
    if not scan.frame_offsets:
        return None
    n = len(scan.frame_times) if cut is None else bisect.bisect_left(scan.frame_times, cut - 1e-6)
    if n <= 0:
        return None
    d = path.read_bytes()
    parts, start, end = [], scan.frame_offsets[0], scan.frame_ends[0]
    for k in range(1, n):
        if scan.frame_offsets[k] == end:
            end = scan.frame_ends[k]
        else:
            parts.append(d[start:end])
            start, end = scan.frame_offsets[k], scan.frame_ends[k]
    parts.append(d[start:end])
    return b"".join(parts)


def verify_output(path: Path, expect: float, mode: str) -> tuple[bool, str]:
    s = scan_mp3(path)
    if not s.ok:
        return False, "Ergebnis enthält keine MP3-Frames"
    if s.junk or s.trailing_junk or s.partial_last_frame or s.mid_info_times:
        return False, "Ergebnis enthält noch Stream-Fehler"
    if s.header_frames is not None and abs(s.header_frames - s.n_frames) > 2:
        return False, "Ergebnis hat inkonsistenten Header"
    if abs(s.duration - expect) > max(1.0, 0.01 * expect):
        return False, f"Länge {s.duration:.1f}s statt erwarteter {expect:.1f}s"
    _, errs = decode_mono(path)
    if errs:
        return False, f"Dekodierfehler: {errs[0]}"
    return True, ""


# ---------------------------------------------------------------------------
# Eine Datei verarbeiten
# ---------------------------------------------------------------------------
def fmt(x, nd=1):
    return "" if x is None else round(float(x), nd)


def process_file(path: Path, out_root: Path, cfg: Config, online: OnlineReferences | None,
                 cache_key: str) -> dict:
    row: dict = {"datei": path.name}
    hints: list[str] = []

    tags = read_tags(path)
    row.update(titel=tags.title, interpret=tags.artist, album=tags.album, tag_laenge_sek=fmt(tags.tlen))

    scan = scan_mp3(path)
    codec = probe_codec(path)
    samples, decode_errs = decode_mono(path)
    profile = build_profile(samples)
    row.update(codec=codec, bitrate_kbps=fmt(scan.avg_bitrate, 0), samplerate=scan.sample_rate,
               kanaele=scan.channels, vbr="ja" if scan.ok and not scan.is_cbr else "nein",
               encoder=scan.encoder, mpeg=scan.mpeg)

    if scan.ok:
        actual = scan.duration
    elif len(samples):
        actual = len(samples) / ANALYSIS_SR
    else:
        row["fehler"] = "; ".join(decode_errs[:3]) or "keine Audiodaten gefunden"
        row["aktion"] = "Datei nicht lesbar - unverändert kopiert"
        return finish(row, path, out_root, cfg, Verdict(), hints, None, 0.0, False, scan, None, failed=True)
    row["ist_laenge_sek"] = fmt(actual)
    if tags.tlen is not None:
        row["abweichung_tag_sek"] = fmt(actual - tags.tlen)
        row["tag_laenge_ok"] = "ja" if abs(actual - tags.tlen) <= cfg.tol else "nein"

    issues = stream_defects(scan)
    if decode_errs:
        issues.append(f"{len(decode_errs)} Dekodierfehler (ffmpeg): {decode_errs[0][:120]}")
    mv = mp3val_issues(path) if cfg.have_mp3val else []
    if mv:
        issues.append("mp3val: " + " / ".join(mv[:3]) + (" ..." if len(mv) > 3 else ""))
    row["mp3_fehler"] = " | ".join(issues)

    # --- Referenzlängen ---------------------------------------------------------
    refs: list[Ref] = [Ref("TLEN-Tag", tags.tlen, own=True)] if tags.tlen else []
    info: dict = {}
    identity_ok = None
    errors: list[str] = []
    previews: list[str] = []
    if online:
        rr = online.collect(path, tags, cache_key, actual)
        refs += rr.refs
        info, identity_ok, errors = rr.info, rr.identity_ok, rr.errors
    v = evaluate(actual, scan, profile, refs, cfg)

    suspicious = (not refs or v.length_ok is False or v.frag_score or v.trunc_score or bool(issues))
    if online and cfg.apple != "off" and (cfg.apple == "all" or suspicious):
        ar = online.apple(tags, cache_key)
        errors += ar.errors
        previews = ar.previews
        if ar.refs:
            refs += ar.refs
            v = evaluate(actual, scan, profile, refs, cfg)
    if online and cfg.verify_preview and previews and cfg.have_fpcalc:
        pct = online.verify_preview(path, previews)
        row["apple_vorschau_match_pct"] = fmt(pct)
        if pct is not None and pct < 65:
            hints.append(f"Apple-Vorschau passt nur zu {pct:.0f}% -> vermutlich andere Version/Mix")

    row.update(referenzen=" | ".join(f"{r.source} {r.seconds:.1f}s" for r in refs),
               naechste_referenz_sek=fmt(v.nearest.seconds) if v.nearest else "",
               abweichung_referenz_sek=fmt(v.diff),
               laenge_ok="" if v.length_ok is None else ("ja" if v.length_ok else "nein"),
               acoustid_score=info.get("acoustid_score", ""), acoustid_titel=info.get("acoustid_titel", ""),
               acoustid_interpret=info.get("acoustid_interpret", ""),
               identitaet_passt_zu_tags="" if identity_ok is None else ("ja" if identity_ok else "nein"),
               abgeschnitten=label(v.trunc_score), abgeschnitten_gruende=" | ".join(v.trunc_reasons),
               fragment=label(v.frag_score), fragment_gruende=" | ".join(v.frag_reasons),
               fragment_schnitt_sek=fmt(v.cut, 2))
    if errors:
        hints.append("Abfrage-Fehler: " + "; ".join(errors))
    if not refs:
        hints.append("keine Referenzlänge gefunden (kein Tag-TLEN, kein Online-Treffer)")
    if v.explained_by_silence:
        hints.append("Längenabweichung erklärt sich durch Stille am Ende")
    if v.unresolved_length:
        hints.append("Länge passt zu keiner Referenz und ist nicht anderweitig erklärt "
                     "(andere Version/Edit? oder Fragment ohne klaren Schnittpunkt)")
    if identity_ok is False and (info.get("acoustid_score") or 0) >= 0.9:
        hints.append("AcoustID erkennt einen anderen Song als die Tags angeben")
    if v.frag_score == 1:
        hints.append("einzelnes Fragment-Indiz, nicht automatisch getrimmt")

    # --- Reparaturplan ------------------------------------------------------------
    reasons: list[str] = []
    if codec not in (None, "mp3") or not scan.ok:
        reasons.append(f"kein MP3-Audio (Codec {codec})")
    if scan.ok and scan.avg_bitrate is not None and scan.avg_bitrate < MIN_ITUNES_BITRATE_KBPS:
        reasons.append(f"Bitrate {scan.avg_bitrate:.0f} kbps < {MIN_ITUNES_BITRATE_KBPS} kbps (Apple-Minimum)")
    if scan.ok and scan.sample_rate not in STANDARD_SAMPLE_RATES:
        reasons.append(f"ungewöhnliche Samplerate {scan.sample_rate} Hz")
    need_rebuild = bool(issues) or v.cut is not None
    final_len = v.final_length
    row["neu_encodiert_grund"] = "; ".join(reasons)

    tlen_wrong = tags.tlen is not None and abs(final_len - tags.tlen) > cfg.tol and v.trunc_score < 2
    if not (need_rebuild or reasons):
        # Audio bleibt unangetastet; nur TLEN ggf. korrigieren
        row["laenge_neu_sek"] = fmt(final_len)
        row["aktion"] = "keine Änderung nötig"
        return finish(row, path, out_root, cfg, v, hints, None, final_len, tlen_wrong, scan, identity_ok)

    if cfg.analyze_only:
        todo = []
        if v.cut is not None:
            todo.append(f"Trimmen auf {v.cut:.1f}s")
        if reasons:
            todo.append("Neu-Encodierung")
        elif issues:
            todo.append("MP3-Fehler reparieren")
        row["aktion"] = "geplant: " + ", ".join(todo)
        return finish(row, path, out_root, cfg, v, hints, None, final_len, tlen_wrong, scan, identity_ok, planned=True)

    bitrate = choose_reencode_bitrate(scan.avg_bitrate, cfg)
    resample = scan.sample_rate not in STANDARD_SAMPLE_RATES
    modes = ["encode"] if reasons else ["copy", "encode"]
    actions: list[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix=".mp3repair_", dir=out_root))
    try:
        built, expect = None, v.cut if v.cut is not None else actual
        for mode in modes:
            cand = tmp_dir / f"audio_{mode}.mp3"
            src, cut_arg = path, v.cut
            if mode == "copy":
                raw = clean_stream_bytes(path, scan, v.cut)
                if raw is None:
                    actions.append("copy: kein sauberer Stream aufbaubar")
                    continue
                src = tmp_dir / "clean_input.mp3"
                src.write_bytes(raw)
                cut_arg = None
            if not ffmpeg_build(src, cand, mode, cut_arg, bitrate, resample):
                actions.append(f"{mode}: ffmpeg fehlgeschlagen")
                continue
            ok, why = verify_output(cand, expect, mode)
            if ok:
                built = cand
                if mode == "copy":
                    actions.append("verlustfrei neu verpackt (Audio unverändert)")
                else:
                    actions.append(f"neu encodiert ({bitrate} kbps CBR" + (f", {REENCODE_SAMPLE_RATE} Hz)" if resample else ")"))
                break
            actions.append(f"{mode}: {why}")
        if built is None:
            row["fehler"] = "; ".join(actions)
            return finish(row, path, out_root, cfg, v, hints, None, final_len, False, scan, identity_ok, failed=True)
        if v.cut is not None:
            actions.insert(0, f"Fragment abgeschnitten bei {v.cut:.1f}s ({fmt(actual - v.cut)}s entfernt)")
        new_len = scan_mp3(built).duration
        row["laenge_neu_sek"] = fmt(new_len)
        row["aktion"] = "; ".join(actions)
        return finish(row, path, out_root, cfg, v, hints, built, new_len, False, scan, identity_ok)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def finish(row, path, out_root, cfg, v, hints, built, new_len, tlen_wrong, scan, identity_ok,
           planned=False, failed=False) -> dict:
    """Wählt den Zielordner, legt die Kopie ab (Tags übernehmen, TLEN anpassen)."""
    wrong_song = identity_ok is False and (row.get("acoustid_score") or 0) >= 0.9
    review = bool(v.frag_score == 1 or v.trunc_score == 1 or v.unresolved_length or wrong_song)
    if failed:
        target = OUT_ERROR
    elif v.trunc_score >= 2:
        target = OUT_TRUNC
    elif review:
        target = OUT_REVIEW
    elif built is not None or "geplant" in str(row.get("aktion", "")) or tlen_wrong:
        target = OUT_FIXED
    else:
        target = OUT_OK
    row["zielordner"] = target
    row["hinweis"] = " | ".join(hints)
    if cfg.analyze_only or planned:
        if tlen_wrong and row.get("aktion") == "keine Änderung nötig":
            row["aktion"] = "geplant: TLEN-Tag korrigieren"
        return row
    if target == OUT_OK and not cfg.copy_unchanged:
        row["ausgabedatei"] = ""
        return row
    dest_dir = out_root / target
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    tmp_dest = dest.with_name(dest.name + ".part")
    try:
        if built is not None:
            shutil.copy2(built, tmp_dest)
            write_tags(path, tmp_dest, new_len, fresh=True)
        else:
            fast_copy(path, tmp_dest)
            if tlen_wrong and write_tags(path, tmp_dest, new_len, fresh=False):
                row["aktion"] = "TLEN-Tag korrigiert (Audio unverändert)"
        tmp_dest.replace(dest)
        row["ausgabedatei"] = str(dest)
    except Exception as exc:
        tmp_dest.unlink(missing_ok=True)
        row["fehler"] = f"Ausgabe fehlgeschlagen: {exc}"
        row["zielordner"] = OUT_ERROR
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "datei", "zielordner", "titel", "interpret", "album",
    "tag_laenge_sek", "ist_laenge_sek", "abweichung_tag_sek", "tag_laenge_ok",
    "referenzen", "naechste_referenz_sek", "abweichung_referenz_sek", "laenge_ok",
    "acoustid_score", "acoustid_titel", "acoustid_interpret", "identitaet_passt_zu_tags",
    "apple_vorschau_match_pct",
    "abgeschnitten", "abgeschnitten_gruende", "fragment", "fragment_gruende", "fragment_schnitt_sek",
    "mp3_fehler", "codec", "mpeg", "bitrate_kbps", "vbr", "samplerate", "kanaele", "encoder",
    "neu_encodiert_grund", "aktion", "laenge_neu_sek", "hinweis", "fehler", "ausgabedatei",
]


def list_mp3s(folder: Path) -> list[Path]:
    """Nur *.mp3 direkt im Ordner. macOS-Metadatenfiles ('._name.mp3') und versteckte Dateien werden übersprungen."""
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".mp3" and not p.name.startswith("."))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MP3-Ordner analysieren, Fragmente/Fehler erkennen und reparierte Kopien erzeugen.")
    ap.add_argument("input_folder", type=Path)
    ap.add_argument("output_folder", type=Path)
    ap.add_argument("--acoustid-key", default=os.environ.get("ACOUSTID_API_KEY"),
                    help="AcoustID API-Key (oder Umgebungsvariable ACOUSTID_API_KEY)")
    ap.add_argument("--analyze-only", action="store_true", help="nur Report schreiben, keine Dateien erzeugen")
    ap.add_argument("--report", type=Path, help="CSV-Report (Default: <output>/mp3_report.csv)")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_SEC, help="Längentoleranz in s (Default 3)")
    ap.add_argument("--no-musicbrainz-search", action="store_true", help="keine MusicBrainz-Textsuche über die Tags")
    ap.add_argument("--apple", choices=["off", "suspicious", "all"], default="suspicious",
                    help="Apple-Katalogdauer abfragen: nie / nur bei Auffälligkeiten (Default) / immer")
    ap.add_argument("--country", default="de", help="bevorzugter Apple-Storefront (Default de)")
    ap.add_argument("--verify-preview", action="store_true",
                    help="Apples 30-s-Vorschau per Fingerprint mit der Datei vergleichen (braucht --apple != off)")
    ap.add_argument("--strict", action="store_true", help="nur bei 'sicher' automatisch trimmen")
    ap.add_argument("--reencode-bitrate", type=int, help="feste CBR-Bitrate für Neu-Encodierung (Default: wie Quelle, min. 128)")
    ap.add_argument("--no-copy-unchanged", action="store_true", help="fehlerfreie Dateien nicht in den Ausgabeordner kopieren")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="parallele Dateien (Default bis 4)")
    ap.add_argument("--limit", type=int, help="nur die ersten N Dateien (zum Testen)")
    ap.add_argument("--refresh-cache", action="store_true", help="Online-Abfrage-Cache ignorieren")
    ap.add_argument("--contact", default=os.environ.get("MP3_REPAIR_CONTACT", "private-script"),
                    help="Kontakt (E-Mail/URL) für den User-Agent der MusicBrainz-Abfragen")
    a = ap.parse_args(argv)

    if not a.input_folder.is_dir():
        sys.exit(f"Eingabeordner nicht gefunden: {a.input_folder}")
    if a.input_folder.resolve() == a.output_folder.resolve():
        sys.exit("Ausgabeordner muss sich vom Eingabeordner unterscheiden (Originale bleiben unberührt).")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"'{tool}' fehlt. Installieren mit: brew install ffmpeg")

    cfg = Config(tol=a.tolerance, acoustid_key=a.acoustid_key, mb_search=not a.no_musicbrainz_search,
                 apple=a.apple, country=a.country, verify_preview=a.verify_preview, strict=a.strict,
                 reencode_bitrate=a.reencode_bitrate, analyze_only=a.analyze_only,
                 copy_unchanged=not a.no_copy_unchanged, contact=a.contact,
                 have_mp3val=bool(shutil.which("mp3val")), have_fpcalc=bool(shutil.which("fpcalc")))
    if not cfg.have_mp3val:
        print("Hinweis: 'mp3val' nicht gefunden (brew install mp3val) - Stream-Prüfung läuft nur mit eigenem Parser + ffmpeg.")
    if cfg.acoustid_key and not cfg.have_fpcalc:
        print("Hinweis: 'fpcalc' nicht gefunden (brew install chromaprint) - AcoustID-Fingerprinting entfällt.")
    if not cfg.acoustid_key:
        print("Hinweis: kein AcoustID-Key -> nur Tag-/MusicBrainz-Textsuche/Apple-Referenzen und Datei-Analyse.")
    if requests is None:
        print("Hinweis: Paket 'requests' fehlt -> keine Online-Abfragen.")

    files = list_mp3s(a.input_folder)
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit("Keine MP3-Dateien im Ordner gefunden.")
    a.output_folder.mkdir(parents=True, exist_ok=True)
    report_path = a.report or a.output_folder / "mp3_report.csv"
    cache = Cache(None if a.refresh_cache else a.output_folder / ".mp3_repair_cache.json")
    online = OnlineReferences(cfg, cache) if requests is not None else None

    print(f"{len(files)} MP3-Dateien in {a.input_folder} " + ("(nur Analyse)" if cfg.analyze_only else f"-> {a.output_folder}"))
    lock, counter = threading.Lock(), [0]

    def work(path: Path) -> dict:
        st = path.stat()
        key = f"{path.name}|{st.st_size}|{st.st_mtime_ns}"
        try:
            row = process_file(path, a.output_folder, cfg, online, key)
        except Exception as exc:  # eine kaputte Datei darf den Lauf nie abbrechen
            row = {"datei": path.name, "zielordner": OUT_ERROR, "fehler": f"{type(exc).__name__}: {exc}"}
        with lock:
            counter[0] += 1
            flags = [f"{n}: {row[n]}" for n in ("abgeschnitten", "fragment") if row.get(n) not in (None, "", "nein")]
            if row.get("mp3_fehler"):
                flags.append("MP3-Fehler")
            print(f"[{counter[0]}/{len(files)}] {path.name} -> {row.get('zielordner')}" + (f"  ({', '.join(flags)})" if flags else ""),
                  flush=True)
        return row

    rows: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            rows = list(pool.map(work, files))
    except KeyboardInterrupt:
        print("\nAbgebrochen - bisherige Online-Abfragen sind im Cache gespeichert, einfach erneut starten.")
        cache.save()
        return 130
    finally:
        cache.save()

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_dir = Counter(r.get("zielordner") for r in rows)
    print("\n--- Zusammenfassung ---")
    for d, text in ((OUT_OK, "ok, unverändert"), (OUT_FIXED, "repariert"), (OUT_TRUNC, "abgeschnitten"),
                    (OUT_REVIEW, "manuell prüfen"), (OUT_ERROR, "nicht lesbar/fehlgeschlagen")):
        print(f"{d:22s} {by_dir.get(d, 0):5d}  {text}")
    print(f"Fragmente erkannt: {sum(1 for r in rows if r.get('fragment') not in (None, '', 'nein'))}, "
          f"davon getrimmt: {sum(1 for r in rows if r.get('fragment_schnitt_sek') not in (None, ''))}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
