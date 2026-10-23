# mp3_repair – MP3-Ordner analysieren und für iTunes Match aufbereiten

Ein Skript (`mp3_repair.py`) ersetzt die Einzelskripte `mp3_analyzer.py`, `mp3_master_processor.py`,
`mp3_deep_analyzer.sh` und `verify_match.py`. Es verarbeitet **nur die `.mp3`-Dateien direkt im
angegebenen Ordner** (keine Unterordner, `._*`-Dateien von macOS und andere Dateitypen werden ignoriert)
und schreibt **Kopien** in einen neuen Ordner. Die Originale werden nie verändert.

## Installation (macOS, Homebrew)

```bash
brew install ffmpeg mp3val chromaprint
cd mp3-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ACOUSTID_API_KEY=dein_key          # https://acoustid.org/api-key
export MP3_REPAIR_CONTACT=du@example.com  # MusicBrainz verlangt einen Kontakt im User-Agent
```

## Benutzung

```bash
# 1. Nur analysieren – schreibt nur den Report, keine Audiodateien
python3 mp3_repair.py ~/Music/1996 ~/Music/1996_repariert --analyze-only

# 2. Report ansehen (mp3_report.csv in Numbers/Excel), dann Kopien erzeugen
python3 mp3_repair.py ~/Music/1996 ~/Music/1996_repariert

# Kleiner Test mit den ersten 10 Dateien
python3 mp3_repair.py ~/Music/1996 /tmp/test --limit 10
```

Online-Abfragen werden in `<ausgabe>/.mp3_repair_cache.json` zwischengespeichert; ein abgebrochener Lauf
(Strg+C) kann einfach neu gestartet werden. Weitere Optionen: `python3 mp3_repair.py --help`
(u. a. `--tolerance`, `--strict`, `--apple off|suspicious|all`, `--verify-preview`, `--reencode-bitrate`,
`--no-copy-unchanged`, `--workers`).

## Ergebnis-Ordner

Jede Datei landet in genau **einem** Ordner; alle Befunde stehen zusätzlich im CSV.

| Ordner | Bedeutung |
|---|---|
| `01_ok_unveraendert/` | keine Auffälligkeiten, bit-identische Kopie (auf APFS ohne Platzverbrauch) |
| `02_repariert/` | Fragment abgeschnitten, MP3-Fehler behoben und/oder für Apple neu encodiert |
| `03_abgeschnitten/` | Song ist unvollständig – lässt sich nicht reparieren, Ersatz besorgen. Kopie unverändert |
| `04_manuell_pruefen/` | unsichere Befunde, es wurde nichts automatisch verändert |
| `05_nicht_lesbar/` | Datei nicht lesbar oder Reparatur fehlgeschlagen (Original unverändert kopiert) |

## Was geprüft wird

**Länge:** Die echte Länge wird aus den MP3-Frames bestimmt (nicht aus einem evtl. defekten Header) und
mit allen verfügbaren Referenzen verglichen, Toleranz ±3 s: `TLEN`-Tag, AcoustID/MusicBrainz
(Fingerprint), MusicBrainz-Textsuche über Interpret/Titel, optional Apples Katalogdauer. Eine Datei gilt als
passend, wenn **eine** Referenz im Toleranzbereich liegt (Radio-Edit/Album-Version haben verschiedene
Längen). Im CSV: `tag_laenge_ok`, `abweichung_tag_sek`, `referenzen`, `abweichung_referenz_sek`, `laenge_ok`.

**Fragment am Ende / abgeschnitten** werden über Punkte bewertet, nicht über ein einzelnes Merkmal:

| Indiz | Punkte |
|---|---|
| zweiter Xing/Info-Header oder ID3-Tag mitten im Stream (aneinandergehängte Dateien) | 2 |
| mehr Frames, als der Header angibt / Encoding-Wechsel (Samplerate, Kanäle, bei CBR Bitrate) am Ende | 1 |
| Stille-Lücke, danach wieder laute Musik bis Dateiende | 1 |
| Schnittpunkt liegt bei einer Referenzlänge und die Datei ist länger als diese | 1 (+1 wenn > 3 s länger) |

Ab **2 Punkten** wird automatisch getrimmt (`--strict`: ab 3). Ein einzelnes Indiz ergibt „unsicher“ →
`04_manuell_pruefen`. Dadurch wird z. B. ein Song mit Pause und kurzem Schlussakkord nicht fälschlich
gekürzt, wenn seine Länge zur Referenz passt.

Abgeschnitten (`abgeschnitten`-Spalte, ab 2 Punkten → `03_abgeschnitten`): Ende mitten im Frame, Header kündigt
mehr Frames an als vorhanden (2), kürzer als das `TLEN`-Tag (2) bzw. als alle MusicBrainz-/Apple-Referenzen (1),
Ende ohne Ausblenden in lauter Musik (1). Ein Song, der kürzer als die Referenz ist, aber sauber ausblendet,
gilt nicht als abgeschnitten, sondern wird zur Kontrolle vorgelegt (evtl. andere Version).

**MP3-Fehler:** Sync-Verluste/Datenmüll im Stream, Müll am Dateiende, abgeschnittener letzter Frame, falsche
oder fehlende Xing/Info-Header, Dekodierfehler (ffmpeg), mp3val-Warnungen.

## Reparatur – so wenig wie möglich verändern

1. **Verlustfrei (Standard):** Der Stream wird aus den gültigen Frames neu zusammengesetzt (Müll und Fragment
   fallen weg) und ffmpeg baut den Xing-Header neu auf. Das Audio bleibt bit-identisch. Alle ID3-Tags inkl. Cover
   werden übernommen; ein vorhandenes `TLEN` wird auf die neue Länge gesetzt.
2. **Neu encodieren (LAME) nur wenn nötig:** Bitrate < 96 kbps (Apples Untergrenze), Samplerate nicht
   44,1/48 kHz, kein MP3-Codec, oder die verlustfreie Reparatur besteht die Prüfung nicht. Die Bitrate wird nie
   unter die der Quelle gesenkt (mind. 128 kbps).
3. Jedes Ergebnis wird geprüft (Länge, Header, Dekodierung); schlägt das fehl, bleibt es bei der Original-Kopie in
   `05_nicht_lesbar/`.

Bewusst **nicht** gemacht (anders als in `mp3_master_processor.py`): Loudness-Normalisierung, Highpass, Denoise,
Umbenennen, Tag-Überschreiben und das Verschieben der Originale. Klangänderungen verschlechtern den
Fingerprint-Abgleich, und Tags sind laut Aufgabenstellung gut gepflegt. Abweichende Titel/Interpreten von
AcoustID werden nur im CSV gemeldet (`identitaet_passt_zu_tags`).

## Grenzen

* Ein Track, den Apple nicht im Katalog hat (Bootleg, Fan-Edit, andere Mix-Version), wird von iTunes Match
  hochgeladen statt gematcht – das ist kein Dateifehler. `--verify-preview` vergleicht dazu Apples 30-s-Vorschau
  per Fingerprint mit deiner Datei (`apple_vorschau_match_pct`: ≥ 80 % = sehr wahrscheinlich gleiche Aufnahme).
* Ein angehängtes Fragment ohne jedes strukturelle Merkmal (gleicher Encoder, keine Lücke) lässt sich nur mit
  einer Referenzlänge sicher erkennen. Ohne Referenz wird es als „unsicher“ gemeldet, nicht getrimmt.
* Die Apple-Such-API erlaubt nur ca. 20 Anfragen/Minute; deshalb standardmäßig nur bei auffälligen Dateien
  (`--apple suspicious`). MusicBrainz: max. 1 Anfrage/s, AcoustID ca. 2–3/s – bei mehreren tausend Dateien
  dauert der erste Lauf entsprechend (Cache beschleunigt weitere Läufe).
* Schwellwerte (Stille-Pegel, Mindestlücke, Punkte) stehen als Konstanten am Anfang von `mp3_repair.py`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Die Tests erzeugen synthetische MP3s mit ffmpeg (Fragment per `cat`, per Stille-Lücke, abgeschnitten,
Datenmüll, kaputter Header, niedrige Bitrate …); die Online-Dienste werden gemockt.
