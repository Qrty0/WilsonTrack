"""Interface en ligne de commande de WilsonTrack.

Exemples :
    python -m wilsontrack.cli analyser fond.mp4 --px-par-mm 5.12
    python -m wilsontrack.cli analyser kcl.mp4 --echantillon KCl --masse 50 --px-par-mm 5.12
    python -m wilsontrack.cli comparer "<dossier KCl>" "<dossier fond>"
    python -m wilsontrack.cli synthetique essai.mp4 --duree 60 --source 40
    python -m wilsontrack.cli valider --duree 30
    python -m wilsontrack.cli cameras
    python -m wilsontrack.cli nuit --camera 0 --jusqua 07:00 --video reduite --px-par-mm 5.12
    python -m wilsontrack.cli lot video1.mp4 video2.mp4 --condition KCl
    python -m wilsontrack.cli surveiller "C:\\Vidéos reçues"

Code de sortie : 0 = succès, 1 = erreur, 2 = arguments invalides, 130 = interrompu (Ctrl+C).
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import textwrap
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from . import __version__

WRAP = 100
ENV_MAX_FRAMES = "WILSONTRACK_MAX_FRAMES"   # limite cachée du nombre d'images (tests rapides)


# =========================================================================== #
# Sorties console
# =========================================================================== #
def _setup_stdio() -> None:
    """UTF-8 sur stdout/stderr (accents, symboles α β μ), avec remplacement si impossible.

    ``line_buffering`` : la progression s'affiche aussitôt, même redirigée vers un fichier.
    Tolère ``sys.stdout is None`` (exécutable Windows sans console).
    """
    for s in (sys.stdout, sys.stderr):
        if s is None or not hasattr(s, "reconfigure"):
            continue
        try:
            s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (ValueError, OSError):
            pass


def _out(msg: str = "") -> None:
    print(msg, flush=True)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(text, width=WRAP, initial_indent=indent, subsequent_indent=indent)


def _fmt_s(s: float) -> str:
    if not math.isfinite(s):
        return "?"
    if s < 90:
        return f"{s:.0f} s"
    return f"{int(s // 60)} min {int(round(s % 60)):02d} s"


class _Progress:
    """Affiche l'avancement tous les ~10 % (fonction de rappel ``progress(fraction)``)."""

    def __init__(self, label: str, calls_per_frame_block: int = 10):
        self.label = label
        self.t0 = time.perf_counter()
        self.next = 0.1
        self.calls = 0
        self.block = calls_per_frame_block

    def __call__(self, frac) -> bool:
        self.calls += 1
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            frac = 0.0
        el = time.perf_counter() - self.t0
        if frac <= 0.0:                               # durée totale inconnue
            if self.calls % 30 == 0:
                _out(f"    {self.label} : {self.calls * self.block} images traitées ({_fmt_s(el)})")
            return False
        if frac + 1e-9 >= self.next:
            pct = min(100, int(frac * 100 + 1e-6) // 10 * 10)
            rest = el * (1.0 - frac) / frac if frac < 1.0 else 0.0
            _out(f"    {self.label} : {pct:3d} %   ({_fmt_s(el)} écoulées, reste ≈ {_fmt_s(rest)})")
            while self.next <= frac + 1e-9:
                self.next += 0.1
        return False                                   # ne jamais interrompre


# =========================================================================== #
# argparse en français
# =========================================================================== #
_FR_ERRORS = [
    (r"^the following arguments are required: (.+)$", r"arguments obligatoires manquants : \1"),
    (r"^unrecognized arguments: (.+)$", r"arguments non reconnus : \1"),
    (r"^argument (.+?): invalid choice: (.+?) \(choose from (.+)\)$", r"argument \1 : choix invalide \2 (choix possibles : \3)"),
    (r"^argument (.+?): invalid \S+ value: (.+)$", r"argument \1 : valeur invalide \2"),
    (r"^argument (.+?): expected one argument$", r"argument \1 : une valeur est attendue"),
    (r"^argument (.+?): expected at least one argument$", r"argument \1 : au moins une valeur est attendue"),
    (r"^argument (.+?): not allowed with argument (.+)$", r"argument \1 : incompatible avec \2"),
    (r"^argument (.+?): ignored explicit argument (.+)$", r"argument \1 : valeur inattendue \2"),
    (r"^ambiguous option: (.+?) could match (.+)$", r"option ambiguë : \1 peut désigner \2"),
    (r"^argument (.+?): (.+)$", r"argument \1 : \2"),
]


def _fr_error(msg: str) -> str:
    for pat, rep in _FR_ERRORS:
        if re.match(pat, msg, flags=re.S):
            return re.sub(pat, rep, msg, count=1, flags=re.S)
    return msg


def _q(s: str) -> str:
    """Guillemets français avec espaces insécables (jamais séparés du mot à l'affichage de l'aide)."""
    return f"« {s} »"


def _desc(text: str) -> str:
    """Description d'une commande, coupée à 79 colonnes (le formateur conserve les retours à la ligne)."""
    return textwrap.fill(text, width=79)


class _Formatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        return super().add_usage(usage, actions, groups, "utilisation : " if prefix is None else prefix)


class _Parser(argparse.ArgumentParser):
    """ArgumentParser avec aide et messages d'erreur en français."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", _Formatter)
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "arguments"
        self._optionals.title = "options"
        self.add_argument("-h", "--help", "--aide", action="help", default=argparse.SUPPRESS,
                          help="afficher cette aide et quitter")

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog} : erreur : {_fr_error(str(message))}\n")


# --------------------------------------------------------------------------- #
# Types d'arguments (messages en français ; virgule décimale acceptée)
# --------------------------------------------------------------------------- #
def _parse_number(txt: str) -> float:
    return float(str(txt).strip().replace(",", ".").replace(" ", "").replace(" ", ""))


def _float_type(lo: float | None = None, hi: float | None = None, strict_lo: bool = False):
    def conv(txt: str) -> float:
        try:
            v = _parse_number(txt)
        except ValueError:
            raise argparse.ArgumentTypeError(f"nombre invalide « {txt} »") from None
        if not math.isfinite(v):
            raise argparse.ArgumentTypeError(f"nombre invalide « {txt} »")
        if lo is not None and (v < lo or (strict_lo and v == lo)):
            raise argparse.ArgumentTypeError(f"doit être {'>' if strict_lo else '≥'} {lo:g} (reçu {v:g})")
        if hi is not None and v > hi:
            raise argparse.ArgumentTypeError(f"doit être ≤ {hi:g} (reçu {v:g})")
        return v
    conv.__name__ = "nombre"
    return conv


def _int_type(lo: int | None = None):
    def conv(txt: str) -> int:
        try:
            v = int(str(txt).strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"entier invalide « {txt} »") from None
        if lo is not None and v < lo:
            raise argparse.ArgumentTypeError(f"doit être ≥ {lo} (reçu {v})")
        return v
    conv.__name__ = "entier"
    return conv


def _fold(s: str) -> str:
    """Minuscules sans accents (comparaison tolérante)."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).casefold().strip()


def _sample_type(txt: str) -> str:
    """Type d'échantillon : clé exacte de ``physics.SAMPLES`` ou début/partie non ambigu (« KCl », « banane »)."""
    from .physics import SAMPLES
    keys = list(SAMPLES)
    f = _fold(txt)
    for k in keys:
        if _fold(k) == f:
            return k
    cands = [k for k in keys if f and _fold(k).startswith(f)] or [k for k in keys if f and f in _fold(k)]
    if len(cands) == 1:
        return cands[0]
    choices = " ; ".join(f"« {k} »" for k in keys)
    if len(cands) > 1:
        raise argparse.ArgumentTypeError(f"« {txt} » est ambigu. Choix possibles : {choices}")
    raise argparse.ArgumentTypeError(f"type d'échantillon inconnu « {txt} ». Choix possibles : {choices}")


def _roi_type(txt: str) -> list:
    """« x0,y0,x1,y1 » (fractions 0–1 de la largeur / hauteur) → polygone normalisé (rectangle)."""
    msg = ("format attendu : x0,y0,x1,y1 en fractions de 0 à 1 de la largeur et de la hauteur de l'image "
           "(ex. 0.1,0.1,0.9,0.9)")
    parts = [p for p in re.split(r"[;,\s]+", str(txt).strip()) if p]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(msg)
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        raise argparse.ArgumentTypeError("les coordonnées de la zone doivent être comprises entre 0 et 1")
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        raise argparse.ArgumentTypeError("zone d'intérêt trop petite (au moins 2 % de la largeur et de la hauteur)")
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# =========================================================================== #
# Utilitaires d'analyse
# =========================================================================== #
LEVEL_TAGS = {"succes": "RÉSULTAT", "attention": "ATTENTION", "info": "INFO"}


def _print_conclusions(items) -> None:
    if not items:
        _out("  (aucune conclusion)")
        return
    for c in items:
        _out(f"  [{LEVEL_TAGS.get(c.level, 'INFO')}] {c.title}")
        _out(_wrap(c.text))


def _print_counts(s: dict) -> None:
    from . import physics
    from .analysis.stats import CLASSES
    tot = s["total"]
    area = max(float(s.get("area_cm2") or 0.0), 1e-9)
    _out(f"  {'Type de trace':28s} {'N':>5s}   {'Taux (/min)':>15s}   {'IC 68 % (/min)':>15s}   {'/cm²/min':>9s}")
    for c in CLASSES:
        r = s["classes"][c]["rate"]
        _out(f"  {physics.CLASS_INFO[c]['nom']:28s} {r.n:5d}   {r.per_min:7.2f} ± {r.err:5.2f}   "
             f"{r.lo:6.2f} – {r.hi:6.2f}   {s['classes'][c]['per_cm2_min']:9.4f}")
    _out(f"  {'Total':28s} {tot.n:5d}   {tot.per_min:7.2f} ± {tot.err:5.2f}   "
         f"{tot.lo:6.2f} – {tot.hi:6.2f}   {tot.per_min / area:9.4f}")


def _short_sample(sample_type: str) -> str:
    return sample_type.split(" (")[0].strip() or sample_type


def _unique_name(out_dir: Path, name: str) -> str:
    """Évite d'écraser un enregistrement existant : « nom », « nom (2) », « nom (3) »…"""
    from .analysis.run import safe_name
    cand, k = name, 2
    while (out_dir / safe_name(cand)).exists():
        cand = f"{name} ({k})"
        k += 1
    return cand


def _load_run(path_txt: str):
    from .analysis.run import RunResult
    p = Path(path_txt)
    if p.is_file() and p.name.lower() == "run.json":
        p = p.parent
    if not (p / "run.json").is_file():
        raise ValueError(f"« {p} » n'est pas un dossier d'enregistrement WilsonTrack (run.json introuvable).")
    if not (p / "traces.csv").is_file():
        raise ValueError(f"« {p} » : fichier traces.csv introuvable.")
    return RunResult.load(p)


def _env_max_frames() -> int:
    v = os.environ.get(ENV_MAX_FRAMES, "").strip()
    try:
        return max(0, int(v)) if v else 0
    except ValueError:
        return 0


# =========================================================================== #
# Commandes
# =========================================================================== #
def cmd_analyser(args) -> int:
    from .analysis import report
    from .analysis.conclusions import conclude_run
    from .analysis.run import analyze_video
    from .config import AppConfig, RunInfo, default_output_dir

    videos = [Path(v) for v in args.videos]
    missing = [str(v) for v in videos if not v.is_file()]
    if missing:
        for m in missing:
            _err(f"Erreur : vidéo introuvable « {m} »")
        return 1
    if args.fin is not None and args.fin <= (args.debut or 0.0):
        _err("Erreur : --fin doit être supérieur à --debut.")
        return 2

    cfg = AppConfig.load(args.config) if args.config else AppConfig()
    ri = cfg.run
    if args.echantillon is not None:
        ri.sample_type = args.echantillon
    if args.masse is not None:
        ri.sample_mass_g = args.masse
    if args.condition is not None and args.condition.strip():
        ri.condition = args.condition.strip()
    elif args.echantillon not in (None, "Aucun") and ri.condition == RunInfo().condition:
        ri.condition = _short_sample(args.echantillon)       # ex. « KCl », « Banane »
    if args.essai is not None:
        ri.trial = args.essai
    if args.px_par_mm is not None:
        cfg.physics.px_per_mm = args.px_par_mm
    if args.roi is not None:
        cfg.detection.roi = args.roi
    out_dir = Path(args.sortie) if args.sortie else (Path(cfg.output_dir) if cfg.output_dir else default_output_dir())
    max_frames = args.max_images if args.max_images is not None else _env_max_frames()

    _out(f"WilsonTrack {__version__} — analyse de {len(videos)} vidéo(s)")
    _out(f"  Condition : {ri.condition}   Échantillon : {ri.sample_type}"
         + (f" ({ri.sample_mass_g:g} g)" if ri.sample_mass_g > 0 else ""))
    if cfg.physics.px_per_mm > 0:
        _out(f"  Calibration : {cfg.physics.px_per_mm:g} px/mm")
    else:
        _out(f"  Calibration : aucune — champ de vision supposé de {cfg.physics.default_fov_mm:g} mm de large "
             "(indiquez --px-par-mm pour des longueurs fiables)")
    if ri.sample_type != "Aucun" and ri.sample_mass_g <= 0:
        _out("  Remarque : masse de l'échantillon non indiquée (--masse) : l'activité attendue du K-40 ne sera "
             "pas calculée.")
    roi = cfg.detection.roi or []
    if len(roi) >= 3:
        xs, ys = [float(q[0]) for q in roi], [float(q[1]) for q in roi]
        _out(f"  Zone d'intérêt : polygone de {len(roi)} sommets, x {min(xs):g}–{max(xs):g}, "
             f"y {min(ys):g}–{max(ys):g} (fractions de l'image)")
    if max_frames:
        _out(f"  Limite : {max_frames} images par vidéo")
    _out(f"  Dossier de sortie : {out_dir}")

    base_trial = ri.trial
    failures = 0
    for k, video in enumerate(videos):
        vcfg = cfg.copy()
        vcfg.run.trial = base_trial + k
        _out("")
        _out(f"=== Vidéo {k + 1}/{len(videos)} : {video.name} (essai {vcfg.run.trial}) ===")
        t0 = time.perf_counter()
        try:
            run, _det = analyze_video(str(video), vcfg, progress=_Progress("analyse"), max_frames=max_frames,
                                      start_s=args.debut or 0.0, end_s=args.fin or 0.0)
        except (OSError, ValueError, RuntimeError) as e:
            _err(f"  Erreur pendant l'analyse de « {video} » : {e}")
            failures += 1
            continue
        el = time.perf_counter() - t0
        _out(f"  Analyse terminée en {_fmt_s(el)} : {run.n_frames} images, {run.duration_s:.1f} s de vidéo analysées"
             + (f" ({run.duration_s / el:.2f}× le temps réel)" if el > 0 else ""))
        wanted = run.name
        run.name = _unique_name(out_dir, wanted)
        if run.name != wanted:
            _out(f"  Le dossier « {wanted} » existe déjà : nouveau dossier « {run.name} ».")
        try:
            folder = run.save(out_dir)
        except OSError as e:
            _err(f"  Erreur : impossible d'enregistrer les résultats dans « {out_dir} » : {e}")
            failures += 1
            continue
        s, concl = conclude_run(run)
        _out("")
        _out(f"  Résultats — {s['total'].n} traces en {run.duration_s / 60:.2f} min sur ≈ {s['area_cm2']:.0f} cm²"
             + ("" if run.calibrated else " (échelle non calibrée)"))
        _print_counts(s)
        _out("")
        _out("  Conclusions :")
        _print_conclusions(concl)
        _out("")
        _out(f"  Données : {folder}")
        _out("            (run.json, traces.csv, traces.xlsx, vignettes/)")
        if not args.pas_de_rapport:
            try:
                rp = report.build_run_report(run)
                _out(f"  Rapport : {rp}")
            except Exception as e:                   # l'analyse est déjà sauvegardée : ne pas la perdre
                _err(f"  Attention : rapport HTML non généré ({type(e).__name__} : {e})")
    return 1 if failures else 0


def cmd_comparer(args) -> int:
    from .analysis import report
    from .analysis.conclusions import conclude_comparison, expected_sample
    from .analysis.stats import CLASSES

    sample = _load_run(args.echantillon)
    background = _load_run(args.fond)
    extra = [_load_run(p) for p in (args.autre or [])]
    if Path(sample.folder).resolve() == Path(background.folder).resolve():
        _out("Attention : l'échantillon et le fond sont le même enregistrement.")
    comps, concl = conclude_comparison(sample, background)

    _out(f"WilsonTrack {__version__} — comparaison échantillon / fond")
    for lab, r in (("Échantillon", sample), ("Fond", background)):
        _out(f"  {lab:12s}: {r.name} — {r.condition}, {r.duration_s / 60:.2f} min, {r.total} traces"
             + ("" if r.calibrated else " (non calibré)"))
    for i, r in enumerate(extra):
        _out(f"  {'Autre ' + str(i + 1):12s}: {r.name} — {r.condition}, {r.duration_s / 60:.2f} min, {r.total} traces")
    exp = expected_sample(sample)
    if exp is not None:
        _out(f"  Activité attendue (K-40) : {exp.as_text()}")
    _out("")
    labels = dict(report.GROUP_LABELS)
    _out(f"  {'Type de trace':26s} {'Échantillon (/min)':>19s} {'Fond (/min)':>16s} {'Différence':>17s} "
         f"{'Rapport':>8s} {'p':>9s} {'σ':>6s}  Significatif")
    for key in list(CLASSES) + ["fines", "total"]:
        cp = comps[key]
        ratio = "—" if (cp.a.n == 0 and cp.b.n == 0) else ("∞" if math.isinf(cp.ratio) else f"×{cp.ratio:.2f}")
        diff_err = f"{cp.diff_err:.2f}" if math.isfinite(cp.diff_err) else "—"
        _out(f"  {labels.get(key, key):26s} {cp.a.per_min:10.2f} ± {cp.a.err:5.2f} {cp.b.per_min:8.2f} ± {cp.b.err:5.2f} "
             f"{cp.diff:+9.2f} ± {diff_err:>5s} {ratio:>8s} {report._fmt_p(cp.p_value):>9s} {cp.sigma:5.1f}σ  "
             f"{'OUI' if cp.significant else 'non'}")
    _out("  (test binomial exact unilatéral ; significatif = excès de l'échantillon avec p < 0,01)")
    note = report.area_note(sample, background)
    if note:
        _out(_wrap(note, "  "))
    al2 = comps.get("alpha_bilateral")
    if al2 is not None:
        _out(f"  Contrôle des alphas (test bilatéral) : p = {report._fmt_p(al2.p_value)}")
    _out("")
    _out("  Conclusions :")
    _print_conclusions(concl)
    _out("")

    if args.sortie:
        out = Path(args.sortie)
        if out.is_dir():
            out = out / "comparaison.html"
    else:
        out = Path(sample.folder) / "comparaison.html"
    rp = report.build_comparison_report(sample, background, out, extra_runs=extra)
    _out(f"  Rapport de comparaison : {rp}")
    return 0


def cmd_synthetique(args) -> int:
    from .synth import SynthConfig, generate_video

    out = Path(args.sortie)
    if out.suffix.lower() != ".mp4":
        _out(f"Remarque : la vidéo est encodée en MP4 (mp4v) même si l'extension est « {out.suffix or '(aucune)'} ».")
    scfg = SynthConfig(duration_s=args.duree, source_beta_per_min=args.source, seed=args.graine)
    _out(f"WilsonTrack {__version__} — génération d'une vidéo synthétique")
    _out(f"  {scfg.width}×{scfg.height} px, {scfg.fps:g} images/s, {scfg.duration_s:g} s, graine {scfg.seed}, "
         f"source β : {scfg.source_beta_per_min:g} /min, champ {scfg.fov_mm:g} mm → {scfg.px_per_mm:.3f} px/mm")
    t0 = time.perf_counter()
    gt = generate_video(out, scfg, progress=_Progress("génération", calls_per_frame_block=30))
    el = time.perf_counter() - t0
    n = {}
    for g in gt["tracks"]:
        key = g["cls"] + (" (source)" if g.get("source") else "")
        n[key] = n.get(key, 0) + 1
    _out(f"  Terminé en {_fmt_s(el)} : {len(gt['tracks'])} traces simulées — "
         + ", ".join(f"{k} : {v}" for k, v in sorted(n.items())))
    _out(f"  Vidéo : {out}")
    _out(f"  Vérité terrain : {out}.gt.json")
    _out(f"  Pour l'analyser : python -m wilsontrack.cli analyser \"{out}\" --px-par-mm {scfg.px_per_mm:g}")
    return 0


def cmd_valider(args) -> int:
    from . import evaluate
    from .analysis.run import analyze_video
    from .config import AppConfig
    from .synth import SynthConfig, generate_video

    scfg = SynthConfig(duration_s=args.duree, source_beta_per_min=args.source, seed=args.graine)
    _out(f"WilsonTrack {__version__} — validation sur une vidéo synthétique (vérité terrain connue)")
    _out(f"  {scfg.duration_s:g} s, graine {scfg.seed}, source β : {scfg.source_beta_per_min:g} /min, "
         f"calibration exacte {scfg.px_per_mm:.3f} px/mm")
    with tempfile.TemporaryDirectory(prefix="wilsontrack_valider_", ignore_cleanup_errors=True) as td:
        path = Path(td) / f"synthetique_graine{scfg.seed}.mp4"
        t0 = time.perf_counter()
        gt = generate_video(path, scfg, progress=_Progress("génération", calls_per_frame_block=30))
        _out(f"  Vidéo générée en {_fmt_s(time.perf_counter() - t0)} ({len(gt['tracks'])} traces simulées)")
        cfg = AppConfig()
        cfg.physics.px_per_mm = float(gt["px_per_mm"])
        cfg.run.condition = "Validation synthétique"
        t0 = time.perf_counter()
        run, _det = analyze_video(str(path), cfg, progress=_Progress("analyse"),
                                  max_frames=args.max_images or _env_max_frames())
        el = time.perf_counter() - t0
        _out(f"  Analyse en {_fmt_s(el)} ({run.duration_s:.1f} s de vidéo analysées)")
        res = evaluate.evaluate(run.events, gt)
    _out("")
    _out(evaluate.format_report(res))
    counts = run.counts()
    true_n = {}
    for g in gt["tracks"]:
        if g["length_mm"] >= 5.0:
            true_n[g["cls"]] = true_n.get(g["cls"], 0) + 1
    _out("")
    _out("  Comptages par type (vrai ≥ 5 mm → détecté) : "
         + ", ".join(f"{c} {true_n.get(c, 0)} → {counts.get(c, 0)}"
                     for c in ("alpha", "beta", "muon", "proton", "incertain")))
    return 0


def _app_config(args):
    """AppConfig de base (préréglage --config) avec condition / échantillon / masse / calibration."""
    from .config import AppConfig, RunInfo
    cfg = AppConfig.load(args.config) if getattr(args, "config", None) else AppConfig()
    ri = cfg.run
    if getattr(args, "echantillon", None) is not None:
        ri.sample_type = args.echantillon
    if getattr(args, "masse", None) is not None:
        ri.sample_mass_g = args.masse
    cond = getattr(args, "condition", None)
    if cond is not None and cond.strip():
        ri.condition = cond.strip()
    elif getattr(args, "echantillon", None) not in (None, "Aucun") and ri.condition == RunInfo().condition:
        ri.condition = _short_sample(args.echantillon)
    if getattr(args, "px_par_mm", None) is not None:
        cfg.physics.px_per_mm = args.px_par_mm
    return cfg


def _resolution_type(txt: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{2,5})\s*[x×X*]\s*(\d{2,5})\s*", str(txt))
    if not m:
        raise argparse.ArgumentTypeError(f"résolution invalide « {txt} » (format LARGEURxHAUTEUR, ex. 1280x720)")
    return int(m.group(1)), int(m.group(2))


def _hhmm_type(txt: str) -> str:
    from .longrun import parse_end_time
    try:
        hm = parse_end_time(txt)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    if hm is None:
        raise argparse.ArgumentTypeError("heure de fin vide (format HH:MM, ex. 07:00)")
    return f"{hm[0]:02d}:{hm[1]:02d}"


def cmd_nuit(args) -> int:
    from .longrun import (NightSessionConfig, estimate_disk_breakdown, fmt_bytes, fmt_duration, fmt_num,
                          free_disk_bytes, run_session)

    app_cfg = _app_config(args)
    d = {"camera_index": args.camera, "segment_min": args.segment_min, "record_video": args.video,
         "snapshot_every_min": args.photo_min, "autosave_every_min": args.sauvegarde_min,
         "dead_after_min": args.alerte_min, "keep_awake": not args.sans_anti_veille,
         "condition": app_cfg.run.condition, "sample": app_cfg.run.sample_type,
         "mass_g": app_cfg.run.sample_mass_g, "px_per_mm": app_cfg.physics.px_per_mm,
         "output_dir": args.sortie or app_cfg.output_dir or ""}
    if args.jusqua:
        d.update(duration_h=0.0, end_time=args.jusqua)
    elif args.duree_h is not None:
        d["duration_h"] = args.duree_h
    cfg = NightSessionConfig.from_dict(d)
    errs = cfg.validate()
    if errs:
        for e in errs:
            _err(f"Erreur : {e}")
        return 2

    if args.simulation:
        from .gui.worker import SynthSource
        from .synth import SynthConfig
        planned = cfg.planned_seconds() or 3600.0
        sc = SynthConfig(width=640, height=360, fps=15.0, duration_s=planned + 60.0, seed=args.graine,
                         rates_per_min={"alpha": 8, "beta": 14, "muon": 16, "proton": 0.3})
        factory = lambda: SynthSource(sc, realtime=False)  # noqa: E731
        name = "simulation accélérée"
        frame_wh, fps = (sc.width, sc.height), sc.fps
    else:
        from .capture import CameraSource
        w, h = args.resolution or (1280, 720)
        factory = lambda: CameraSource(args.camera, w, h, args.ips, args.pilote)  # noqa: E731
        name = f"camera:{args.camera}"
        frame_wh, fps = (w, h), args.ips

    est = estimate_disk_breakdown(cfg, frame_wh, fps)
    out_root = cfg.output_dir or "(dossier de résultats par défaut)"
    _out(f"WilsonTrack {__version__} — session de longue durée ({name})")
    _out(f"  {cfg.describe()}")
    _out(f"  Condition : {app_cfg.run.condition}   Échantillon : {app_cfg.run.sample_type}"
         + (f" ({fmt_num(app_cfg.run.sample_mass_g, 1)} g)" if app_cfg.run.sample_mass_g > 0 else ""))
    _out(f"  Espace disque estimé (prudent) : {fmt_bytes(est['total'])} — libre : "
         f"{fmt_bytes(free_disk_bytes(cfg.output_dir or '.'))}")
    _out(f"  Dossier de sortie : {out_root}")
    _out("  Ctrl+C pour arrêter : les résultats obtenus sont sauvegardés et le rapport est produit.")
    _out("")

    last = {"t": -1e9}

    def on_status(st: dict):
        now = time.monotonic()
        if now - last["t"] < args.affichage_s:
            return
        last["t"] = now
        rem = st["remaining_s"]
        _out(f"  [{datetime.now():%H:%M}] écoulé {fmt_duration(st['elapsed_s'])}"
             + (f", reste {fmt_duration(rem)}" if rem >= 0 else "")
             + f" — {st['n_tracks']} traces ({fmt_num(st['rate_per_min'], 1)} /min) — "
               f"{st.get('checkpoints', 0)} sauvegarde(s) — disque libre {fmt_bytes(st['disk_free_bytes'])}")

    run = run_session(cfg, app_cfg, factory, source_name=name, on_status=on_status,
                      on_warning=lambda m: _err(f"  ATTENTION : {m}"), make_report=not args.pas_de_rapport)
    _out("")
    s = run.session or {}
    _out(f"Session terminée ({s.get('stop_reason', '?')}) : {run.total} traces valides en "
         f"{fmt_duration(run.duration_s)} d'analyse.")
    cnt = run.counts()
    if cnt:
        _out("  " + ", ".join(f"{k} : {v}" for k, v in sorted(cnt.items())))
    _out(f"  Dossier : {run.folder}")
    _out("  (run.json, traces.csv, traces.xlsx, vignettes/, photos/, video/, journal.txt"
         + ("" if args.pas_de_rapport else ", rapport.html") + ")")
    return 0


def _expand_files(items: list[str]) -> list[str]:
    """Développe les jokers (* ?) : l'invite de commandes Windows ne le fait pas."""
    import glob
    out = []
    for it in items:
        if any(ch in it for ch in "*?["):
            hits = sorted(glob.glob(it))
            out.extend(hits if hits else [it])
        else:
            out.append(it)
    return out


def _run_batch(batch: dict, app_cfg, watch_once: bool = False, poll_s: float = 5.0, stable_s: float = 10.0) -> int:
    from .batch import STATUS_DONE, STATUS_ERROR, SUMMARY_CSV, BatchRunner
    from .longrun import fmt_duration

    shown: dict[str, int] = {}

    def on_jobs(jobs: list):
        for j in jobs:
            if j["status"] != "en cours":
                continue
            step = int(j["progress"] * 10)
            if step > shown.get(j["path"], -1) and step > 0:
                shown[j["path"]] = step
                eta = j["eta_s"]
                _out(f"    {j['name']} : {step * 10:3d} %" + (f" (reste ≈ {fmt_duration(eta)})" if eta >= 0 else ""))

    def on_done(j: dict):
        if j["status"] == STATUS_DONE:
            cnt = ", ".join(f"{k} {v}" for k, v in sorted(j["counts"].items())) or "aucune trace"
            _out(f"  ✓ {j['name']} : {j['total']} traces en {fmt_duration(j['duration_s'])} ({cnt})")
            _out(f"      → {j['result_folder']}")
            for w in j.get("warnings", []):
                _out(f"      remarque : {w}")
        else:
            _err(f"  ✗ {j['name']} : {j['status']}" + (f" — {j['error']}" if j["error"] else ""))

    runner = BatchRunner(batch, app_cfg, on_jobs_changed=on_jobs, on_job_finished=on_done,
                         on_error=lambda m: _err(f"  Attention : {m}"), on_info=lambda m: _out(f"  {m}"),
                         poll_s=poll_s, stable_s=stable_s, watch_once=watch_once)
    runner.start()
    try:
        while runner.is_alive():
            runner.join(0.5)
    except KeyboardInterrupt:
        _err("\nArrêt demandé : la vidéo en cours est abandonnée, les résultats déjà obtenus sont conservés.")
        runner.stop()
        runner.join()
    jobs = runner.queue.jobs()
    n_ok = sum(1 for j in jobs if j.status == STATUS_DONE)
    n_err = sum(1 for j in jobs if j.status == STATUS_ERROR)
    _out("")
    _out(f"Lot terminé : {n_ok} vidéo(s) analysée(s), {n_err} en erreur, "
         f"{len(jobs) - n_ok - n_err} non traitée(s).")
    if n_ok or n_err:
        _out(f"  Résumé : {runner.out_dir / SUMMARY_CSV}")
    return 1 if n_err else 0


def cmd_lot(args) -> int:
    app_cfg = _app_config(args)
    files = _expand_files(args.fichiers)
    batch = {"files": files, "condition": app_cfg.run.condition, "sample": app_cfg.run.sample_type,
             "mass_g": app_cfg.run.sample_mass_g, "px_per_mm": app_cfg.physics.px_per_mm,
             "output_dir": args.sortie or app_cfg.output_dir or "", "make_reports": not args.pas_de_rapport}
    _out(f"WilsonTrack {__version__} — analyse d'un lot de {len(files)} vidéo(s)")
    return _run_batch(batch, app_cfg)


def cmd_surveiller(args) -> int:
    folder = Path(args.dossier)
    if not folder.is_dir():
        _err(f"Erreur : dossier introuvable « {folder} »")
        return 1
    app_cfg = _app_config(args)
    batch = {"files": [], "watch_folder": str(folder), "condition": app_cfg.run.condition,
             "sample": app_cfg.run.sample_type, "mass_g": app_cfg.run.sample_mass_g,
             "px_per_mm": app_cfg.physics.px_per_mm, "output_dir": args.sortie or app_cfg.output_dir or "",
             "make_reports": not args.pas_de_rapport}
    _out(f"WilsonTrack {__version__} — surveillance de « {folder} »")
    _out("  Chaque nouvelle vidéo complète est analysée automatiquement ; les vidéos déjà traitées sont "
         f"mémorisées dans {_q('.wilsontrack_traitees.json')}.")
    if not args.une_fois:
        _out("  Ctrl+C pour arrêter.")
    return _run_batch(batch, app_cfg, watch_once=args.une_fois, poll_s=args.intervalle, stable_s=args.stabilite)


def cmd_cameras(args) -> int:
    import cv2

    from .capture import list_cameras
    _out(f"Recherche des caméras (index 0 à {args.max - 1})…")
    log = getattr(getattr(cv2, "utils", None), "logging", None)   # masque les avertissements d'OpenCV
    old = None
    try:
        if log is not None:
            old = log.getLogLevel()
            log.setLogLevel(log.LOG_LEVEL_ERROR)
    except Exception:
        log = None
    try:
        cams = list_cameras(max_index=args.max)
    finally:
        if log is not None and old is not None:
            try:
                log.setLogLevel(old)
            except Exception:
                pass
    if not cams:
        _out("Aucune caméra détectée. Vérifiez le branchement USB et qu'aucun autre logiciel n'utilise la caméra.")
        return 0
    for c in cams:
        _out(f"  [{c['index']}] {c['label']}")
    _out(f"{len(cams)} caméra(s) détectée(s).")
    return 0


# =========================================================================== #
# Analyseur d'arguments
# =========================================================================== #
EPILOG = """exemples :
  python -m wilsontrack.cli analyser fond.mp4 --px-par-mm 5.12
  python -m wilsontrack.cli analyser kcl.mp4 --echantillon KCl --masse 50 --px-par-mm 5.12
  python -m wilsontrack.cli comparer "<dossier KCl>" "<dossier fond>"
  python -m wilsontrack.cli synthetique essai.mp4 --duree 60 --source 40
  python -m wilsontrack.cli valider --duree 30
  python -m wilsontrack.cli nuit --camera 0 --jusqua 07:00 --px-par-mm 5.12
  python -m wilsontrack.cli lot "D:\\vidéos\\*.mp4" --condition KCl --echantillon KCl --masse 50
  python -m wilsontrack.cli surveiller "C:\\Vidéos reçues" --sortie "C:\\Résultats"

Aide d'une commande : python -m wilsontrack.cli COMMANDE --aide"""


def build_parser() -> argparse.ArgumentParser:
    from .physics import SAMPLES

    p = _Parser(prog="wilsontrack",
                description=_desc("WilsonTrack — détection, classification et analyse des traces de particules "
                                  "filmées dans une chambre de Wilson (Cégep Marie-Victorin)."),
                epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"WilsonTrack {__version__}",
                   help="afficher la version et quitter")
    sub = p.add_subparsers(dest="commande", metavar="COMMANDE", title="commandes", parser_class=_Parser)

    # ---- analyser ----------------------------------------------------------
    samples = " ; ".join(_q(k) for k in SAMPLES).replace("%", "%%")   # argparse formate l'aide avec %
    a = sub.add_parser(
        "analyser", help="analyser une ou plusieurs vidéos et produire les résultats et le rapport",
        description=_desc("Analyse chaque vidéo (détection, suivi, classification), enregistre les résultats "
                          "(run.json, traces.csv, traces.xlsx, vignettes) et le rapport HTML, puis affiche les "
                          "comptages et les conclusions. Plusieurs vidéos : même condition, numéros d'essai "
                          "successifs à partir de --essai."))
    a.add_argument("videos", nargs="+", metavar="VIDEO", help="fichier(s) vidéo (mp4, avi, mov, webm…)")
    a.add_argument("--condition", metavar="TXT",
                   help=f"condition expérimentale (ex. {_q('Fond (rayons cosmiques)')}, {_q('KCl')}, "
                        f"{_q('Banane')}) ; par défaut : celle du préréglage, ou le nom court de l'échantillon")
    a.add_argument("--echantillon", metavar="TYPE", type=_sample_type,
                   help=f"type d'échantillon (le début du nom suffit, ex. {_q('KCl')}, {_q('banane')}) : {samples}")
    a.add_argument("--masse", metavar="G", type=_float_type(0.0), help="masse de l'échantillon en grammes")
    a.add_argument("--essai", metavar="N", type=_int_type(1), help="numéro d'essai (défaut : 1)")
    a.add_argument("--px-par-mm", dest="px_par_mm", metavar="F", type=_float_type(0.0),
                   help="calibration : pixels (de la vidéo d'origine) par millimètre ; 0 = non calibré")
    a.add_argument("--roi", metavar="x0,y0,x1,y1", type=_roi_type,
                   help="zone d'intérêt rectangulaire en fractions (0 à 1) de la largeur et de la hauteur")
    a.add_argument("--config", metavar="PRESET.json", help="préréglage de paramètres (fichier JSON de l'application)")
    a.add_argument("--sortie", metavar="DOSSIER",
                   help=f"dossier où créer le dossier de l'enregistrement (défaut : {_q('WilsonTrack Résultats')} "
                        "dans Documents)")
    a.add_argument("--pas-de-rapport", dest="pas_de_rapport", action="store_true",
                   help="ne pas générer le rapport HTML")
    a.add_argument("--debut", metavar="S", type=_float_type(0.0), help="commencer l'analyse à S secondes")
    a.add_argument("--fin", metavar="S", type=_float_type(0.0, strict_lo=True), help="arrêter l'analyse à S secondes")
    a.add_argument("--max-images", dest="max_images", type=_int_type(1), default=None, help=argparse.SUPPRESS)
    a.set_defaults(func=cmd_analyser)

    # ---- comparer ----------------------------------------------------------
    c = sub.add_parser(
        "comparer", help=f"comparer un enregistrement {_q('échantillon')} au bruit de fond",
        description=_desc("Compare les taux de traces d'un enregistrement avec échantillon (KCl, banane…) à ceux "
                          "d'un enregistrement de bruit de fond (test binomial exact), affiche les conclusions et "
                          "écrit un rapport HTML de comparaison."))
    c.add_argument("echantillon", metavar="DOSSIER_ECHANTILLON", help="dossier de l'enregistrement avec échantillon")
    c.add_argument("fond", metavar="DOSSIER_FOND", help="dossier de l'enregistrement du bruit de fond")
    c.add_argument("--sortie", metavar="FICHIER.html",
                   help="rapport à écrire (défaut : comparaison.html dans le dossier de l'échantillon)")
    c.add_argument("--autre", metavar="DOSSIER", action="append",
                   help="autre enregistrement à ajouter au graphique de comparaison (option répétable)")
    c.set_defaults(func=cmd_comparer)

    # ---- synthetique -------------------------------------------------------
    s = sub.add_parser(
        "synthetique", help="générer une vidéo synthétique de chambre de Wilson (avec vérité terrain)",
        description=_desc("Génère une vidéo réaliste simulée (alphas, électrons, muons, protons, brouillard, "
                          "bruit) et sa vérité terrain (<vidéo>.gt.json). Utile pour tester l'application sans "
                          "chambre."))
    s.add_argument("sortie", metavar="SORTIE.mp4", help="fichier vidéo à créer")
    s.add_argument("--duree", metavar="S", type=_float_type(1.0), default=60.0, help="durée en secondes (défaut : 60)")
    s.add_argument("--source", metavar="BETA_PAR_MIN", type=_float_type(0.0), default=0.0,
                   help="traces β supplémentaires par minute émises par une source simulée (ex. KCl) ; défaut : 0")
    s.add_argument("--graine", metavar="N", type=_int_type(0), default=1,
                   help="graine du générateur aléatoire (même graine = même vidéo) ; défaut : 1")
    s.set_defaults(func=cmd_synthetique)

    # ---- valider -----------------------------------------------------------
    v = sub.add_parser(
        "valider", help="mesurer la fiabilité de la détection sur une vidéo synthétique",
        description=_desc("Génère une vidéo synthétique temporaire, l'analyse avec la calibration exacte et "
                          "compare les traces détectées à la vérité terrain (rappel, précision, matrice de "
                          "confusion)."))
    v.add_argument("--duree", metavar="S", type=_float_type(1.0), default=60.0, help="durée en secondes (défaut : 60)")
    v.add_argument("--graine", metavar="N", type=_int_type(0), default=1, help="graine aléatoire (défaut : 1)")
    v.add_argument("--source", metavar="BETA_PAR_MIN", type=_float_type(0.0), default=0.0,
                   help="traces β par minute d'une source simulée (défaut : 0)")
    v.add_argument("--max-images", dest="max_images", type=_int_type(1), default=None, help=argparse.SUPPRESS)
    v.set_defaults(func=cmd_valider)

    # ---- cameras -----------------------------------------------------------
    k = sub.add_parser("cameras", help="lister les caméras détectées",
                       description=_desc("Liste les caméras (webcams USB, caméras virtuelles…) utilisables par "
                                         "l'application."))
    k.add_argument("--max", metavar="N", type=_int_type(1), default=6,
                   help="nombre d'index de caméra à tester (défaut : 6)")
    k.set_defaults(func=cmd_cameras)

    def common_run_options(sp, with_output_help: str):
        sp.add_argument("--condition", metavar="TXT",
                        help=f"condition expérimentale (ex. {_q('Fond (rayons cosmiques)')}, {_q('KCl')})")
        sp.add_argument("--echantillon", metavar="TYPE", type=_sample_type,
                        help=f"type d'échantillon (le début du nom suffit, ex. {_q('KCl')}, {_q('banane')})")
        sp.add_argument("--masse", metavar="G", type=_float_type(0.0), help="masse de l'échantillon en grammes")
        sp.add_argument("--px-par-mm", dest="px_par_mm", metavar="F", type=_float_type(0.0),
                        help="calibration : pixels (de l'image d'origine) par millimètre ; 0 = non calibré")
        sp.add_argument("--config", metavar="PRESET.json", help="préréglage de paramètres (fichier JSON de l'application)")
        sp.add_argument("--sortie", metavar="DOSSIER", help=with_output_help)
        sp.add_argument("--pas-de-rapport", dest="pas_de_rapport", action="store_true",
                        help="ne pas générer le rapport HTML")

    # ---- nuit --------------------------------------------------------------
    n = sub.add_parser(
        "nuit", help="filmer et analyser toute une nuit (session de longue durée, sans interface)",
        description=_desc("Filme la chambre pendant des heures et analyse en direct : sauvegardes intermédiaires "
                          "régulières, photos périodiques, vidéo découpée en segments, journal, alerte si la "
                          "chambre ne produit plus de traces, arrêt automatique à l'heure prévue, puis résultats "
                          "définitifs et rapport. L'ordinateur est empêché de se mettre en veille (pas l'écran)."))
    n.add_argument("--camera", metavar="N", type=_int_type(0), default=0, help="index de la caméra (défaut : 0)")
    g = n.add_mutually_exclusive_group()
    g.add_argument("--duree-h", dest="duree_h", metavar="H", type=_float_type(0.0, strict_lo=True),
                   help="durée de la session en heures (ex. 8 ou 7,5) ; défaut : 8 h")
    g.add_argument("--jusqua", metavar="HH:MM", type=_hhmm_type,
                   help="heure de fin (ex. 07:00 ; une heure déjà passée aujourd'hui désigne demain)")
    n.add_argument("--video", choices=["aucune", "reduite", "complete"], default="reduite",
                   help="vidéo enregistrée : aucune, réduite (moitié de la taille, 10 img/s ; défaut) ou complète")
    n.add_argument("--segment-min", dest="segment_min", metavar="M", type=_float_type(0.0), default=30.0,
                   help="durée d'un fichier vidéo en minutes (défaut : 30 ; 0 = un seul fichier)")
    n.add_argument("--photo-min", dest="photo_min", metavar="M", type=_float_type(0.0), default=10.0,
                   help="une photo (brute + annotée) toutes les M minutes (défaut : 10 ; 0 = jamais)")
    n.add_argument("--sauvegarde-min", dest="sauvegarde_min", metavar="M", type=_float_type(0.0, strict_lo=True),
                   default=5.0, help="point de sauvegarde toutes les M minutes (défaut : 5)")
    n.add_argument("--alerte-min", dest="alerte_min", metavar="M", type=_float_type(0.0), default=20.0,
                   help="alerte si aucune trace pendant M minutes d'analyse (défaut : 20 ; 0 = jamais)")
    n.add_argument("--resolution", metavar="LxH", type=_resolution_type,
                   help="résolution demandée à la caméra (défaut : 1280x720)")
    n.add_argument("--ips", metavar="N", type=_float_type(1.0), default=30.0,
                   help="images par seconde demandées à la caméra (défaut : 30)")
    n.add_argument("--pilote", choices=["DirectShow", "Media Foundation", "auto"], default="DirectShow",
                   help="pilote de caméra Windows (défaut : DirectShow)")
    n.add_argument("--sans-anti-veille", dest="sans_anti_veille", action="store_true",
                   help="ne pas empêcher la mise en veille de l'ordinateur")
    n.add_argument("--simulation", action="store_true",
                   help="chambre SIMULÉE accélérée au lieu de la caméra (pour essayer la commande)")
    n.add_argument("--graine", metavar="N", type=_int_type(0), default=1, help=argparse.SUPPRESS)
    n.add_argument("--affichage-s", dest="affichage_s", metavar="S", type=_float_type(0.0), default=60.0,
                   help="afficher l'état toutes les S secondes (défaut : 60)")
    common_run_options(n, f"dossier où créer le dossier de la session (défaut : {_q('WilsonTrack Résultats')} dans "
                          "Documents)")
    n.set_defaults(func=cmd_nuit)

    # ---- lot ---------------------------------------------------------------
    lo = sub.add_parser(
        "lot", help="analyser une série de vidéos reçues, l'une après l'autre",
        description=_desc("Analyse chaque vidéo (même condition pour toutes), enregistre un dossier de résultats et "
                          "un rapport par vidéo, et tient à jour un tableau récapitulatif resume_lot.csv (+ .xlsx) "
                          f"dans le dossier de sortie. Les jokers sont acceptés (ex. {_q('D:/vidéos/*.mp4')})."))
    lo.add_argument("fichiers", nargs="+", metavar="FICHIER", help="vidéo(s) à analyser")
    common_run_options(lo, f"dossier des résultats et du résumé (défaut : {_q('WilsonTrack Résultats')} dans "
                           "Documents)")
    lo.set_defaults(func=cmd_lot)

    # ---- surveiller --------------------------------------------------------
    sv = sub.add_parser(
        "surveiller", help="analyser automatiquement chaque nouvelle vidéo déposée dans un dossier",
        description=_desc("Surveille un dossier : chaque nouvelle vidéo COMPLÈTE (taille stable, fichier lisible ; "
                          "les fichiers en cours de téléchargement sont ignorés) est analysée automatiquement. Les "
                          f"vidéos déjà traitées sont mémorisées dans {_q('.wilsontrack_traitees.json')} et ne sont "
                          "jamais analysées deux fois. Ctrl+C pour arrêter."))
    sv.add_argument("dossier", metavar="DOSSIER", help="dossier à surveiller")
    sv.add_argument("--intervalle", metavar="S", type=_float_type(0.1), default=5.0,
                    help="vérifier le dossier toutes les S secondes (défaut : 5)")
    sv.add_argument("--stabilite", metavar="S", type=_float_type(0.0), default=10.0,
                    help="une vidéo est complète quand sa taille n'a pas changé depuis S secondes (défaut : 10)")
    sv.add_argument("--une-fois", dest="une_fois", action="store_true",
                    help="analyser les vidéos présentes puis quitter (au lieu d'attendre les suivantes)")
    common_run_options(sv, f"dossier des résultats et du résumé (défaut : {_q('WilsonTrack Résultats')} dans "
                           "Documents)")
    sv.set_defaults(func=cmd_surveiller)
    return p


def main(argv=None) -> int:
    """Point d'entrée : ``main(["analyser", "video.mp4", ...])`` → code de sortie."""
    _setup_stdio()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:                       # --aide, --version ou erreur d'arguments
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _err("\nInterrompu par l'utilisateur.")
        return 130
    except (OSError, ValueError) as e:
        _err(f"Erreur : {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
