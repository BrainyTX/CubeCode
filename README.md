# CubeCode Resonance Simulator

Dieses Repository enthält eine referenzielle Python-Implementierung des gekoppelten **Geometrie/Resonanz**-Motors (Motor A) und der **achtphasigen Schwingungsdynamik** (Motor B), wie in der Spezifikation beschrieben. Das Modell baut auf einem kubischen Gitter auf, das statische Spiegel- und Summenregeln mit einem zeitdiskreten Kuramoto-ähnlichen Entrainment kombiniert.

## Merkmale

- Front-basierte Aktualisierung der statischen Resonanzwerte inklusive Spiegelzwang.
- Achtphasige Oszillation mit Pacemaker, Nachbarschaftskopplung und Amplitudenaufbau.
- Kopplung zwischen Motor A und Motor B über amplituden-gewichtete Sum-to-Target-Regeln.
- Messung des globalen Kuramoto-Order-Parameters und einfacher FFT-Spektren.
- Automatische Auswahl einer CUDA/CuPy-Ausführung, fallback auf reine NumPy-CPU.

## Installation

```bash
pip install -e .
```

### CUDA-Unterstützung

Wenn eine CUDA-fähige GPU vorhanden ist und `cupy` installiert wurde, nutzt die Simulation sie automatisch. Ohne CuPy oder bei gesetzter Umgebungvariablen `CUBE_SIM_DISABLE_CUDA=1` läuft alles auf der CPU.

Für Visualisierung und Notebook-Experimente:

```bash
pip install -e .[notebook]
```

## Schnelleinstieg

```python
from cube_sim import Simulation, SimulationConfig

config = SimulationConfig.from_dict(
    {
        "grid": {"N": 65, "neighborhood": "faces"},
        "oscillation": {"kappa": 0.35, "delta": 0.02, "alpha": 0.15, "beta": 0.05},
        "run": {"max_iters_static": 200, "max_ticks_dynamic": 1000, "log_every": 25},
    }
)

sim = Simulation(config)
logs = sim.run()
print(logs[-1])
```

`logs` enthält periodische Schnappschüsse (z. B. Order-Parameter, Amplitudenmittelwerte, Änderungsquoten `delta_v`/`delta_p`) und liefert auf Wunsch komplette Schalen-Histogramme (`shells`). Der vollständige Zustand kann jederzeit über `sim.snapshot()` entnommen werden.

## Operativer Leitfaden

### 1. Live-Telemetrie (jeder Takt)

Pflichtmetriken, die direkt während des Laufs aufgezeichnet werden sollten:

- **Kuramoto-Order `R(t)`**: \(\left|\frac{1}{N^3}\sum_p e^{i\pi P[p]/4}\right|\). Ein steigender und anschließend periodischer Verlauf zeigt Lock-In.
- **Amplitudenstatistik**: Mittelwert \(\bar A(t)\) und Maximum \(A_{\max}(t)\); beide dürfen weder aussterben noch numerisch explodieren.
- **Schalenhistogramme**: Verteilung von `V` und `P` für jede Radius-Schale \(r\). Fixpunkte wie 5·5-Horizonte oder 6↔4-Achsenspiegel werden so sichtbar.
- **Änderungsquoten**: Anteil der Zellen mit geänderten `V`- bzw. `P`-Werten pro Takt als Konvergenzindikator.

Optionale Erweiterungen:

- Zeitreihen einzelner Referenzpunkte (z. B. Achsenschnitt) für `(V, P, A)`.
- Autokorrelation von `R(t)` zur Periodenabschätzung.
- FFT von `R(t)` oder \(\bar A(t)\) zur Bestimmung der dominanten Frequenz \(f_*\).

### 2. Snapshots & Visualisierung (alle _k_ Takte)

- Heatmaps der Ebenen `z=0`, `y=0`, `x=0` für `V`, `P`, `A`.
- Optional Isosurfaces (z. B. `A = const.` oder `V ∈ {5,6}`) als PLY/VTK.
- Summen und Spezialmomente je Schale (z. B. 202-, 232-, 515-Knoten), um erwartete Muster zu bestätigen.

### 3. Parameter-Dashboard

Halte zu jedem Lauf die wichtigsten Parameter fest, damit Sweeps reproduzierbar bleiben:

- **Grid**: `N` (ungerade, z. B. 65/129) und Nachbarschaft (faces/edges/corners).
- **Radialmetrik**: `r_inf` (kubisch) oder `r_2` (rund).
- **Bootsequenz** `S(r)` und Zielsumme `T(r)=9`.
- **Mirror**: `identical` (ruhige Muster) vs. `complement` (6↔4, 5↔5). Wechseln, falls keine Stabilität entsteht.
- **Oszillation**: Phasenanzahl (8), Kopplung \(\kappa\), Dämpfung `delta`, Aufbau `alpha`, Puls-Boost `beta`, sowie `T_drive`.
- **Sweep-Plan**: `T_drive` zwischen 0.5 s und 2.0 s in 0.05-Schritten durchlaufen.

**Lock-In-Kriterium:** `R(t)` steigt über ≈0.6 und zeigt eine stabile Periodizität; FFT liefert einen dominanten Peak (idealerweise nahe 1 Hz).

### 4. Frühe Warnzeichen & Gegenmaßnahmen

- **Diffuse Muster**: \(\kappa\) erhöhen, `delta` erhöhen, Mirror auf `identical`, Nachbarschaft auf 6 reduzieren.
- **Amplitude explodiert**: `delta` erhöhen, `alpha`/`beta` senken, optional `T(r)`-Kopplung aussetzen oder `S(r)` glätten.
- **Amplitude stirbt ab**: `alpha`/`beta` erhöhen, `S(r)` mit strenger Monotonie versehen.
- **Fehlende Spiegel/Horizonte**: Mirror auf `complement` (10−v) stellen; Phasenspiegel optional antisymmetrisch (`P -> P+4 mod 8`).

### 5. Pfadabfragen (nach Stabilisierung)

- Knoten: alle Zellen mit `V > 0`.
- Kanten: 6er-Nachbarschaft, Kosten z. B. \(\alpha |ΔP| + \beta /(1 + A(q)) + \gamma\mathbf 1[V(q)=0]\).
- Algorithmus: A* mit Manhattan-Heuristik. Ergebnis enthält Pfad, Gesamtkosten und die `(V,P,A)`-Spur („Relationskette“).

### 6. Dateiausgaben

- `metrics.csv`: Zeitstempel, `R`, \(\bar A\), \(A_{\max}\), Änderungsquoten, ggf. \(f_*\).
- `shells.json`: Histogramme, Summen und Spezialmomente je Schale.
- `slices/`: PNGs der Heatmaps für `V`, `P`, `A` auf den drei Ebenen.
- `isos/`: Optionale PLY/VTK-Isosurfaces.
- `paths/`: JSON-Dokumente mit Pfadinformationen.

### 7. Minimaler Runbook-Ablauf

1. Start mit `N=65`, `r_inf`, Nachbarschaft=6, Mirror=`complement`, `T_drive=1.0`.
2. 500–1500 Takte simulieren, Logs/Slices alle 25 Takte sichern.
3. `T_drive` von 0.5 bis 2.0 in 0.05-Schritten sweepen; je Lauf 500 Takte, `R` und FFT-Peak protokollieren.
4. Parameterkombination mit höchstem `R` und saubersten Schnitten auswählen.
5. Pfad-Tool auf 2–3 Punktpaare anwenden.
6. Erkenntnisse festhalten (z. B. welche Parameter 5·5-Horizonte oder 6↔4-Achsen erzeugen).

### 8. Beobachtungshinweise

Erfolgreiches Einrasten zeigt sich durch geglättete Pyramidenflanken, ausgeprägte 6↔4-Komplementachsen, Bienenwaben/Torus-Schichten, wiederkehrende Knotentypen (232, 464, 768/776) sowie periodische `R(t)`-Verläufe.

## Tests

```bash
pytest
```

## Weiterführende Ideen

- Ablegen von Schnitten oder Isosurfaces über `sim.snapshot()` in Visualisierungstools.
- Sweep über `T_drive` und `kappa`, um Resonanzzungen zu identifizieren.
- Erweiterung der Mirror-Konfiguration um diagonale oder antisymmetrische Phasenabbildungen.
- Ergänzung eines Pathfinder-Moduls, das Energiekosten auf Basis der Amplituden verwendet.
