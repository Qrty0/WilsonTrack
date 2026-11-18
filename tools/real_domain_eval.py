"""Mesure honnête sur les VRAIES vidéos étiquetées + entraînement d'un « adaptateur réel ».

Données : _training/real/tracks.jsonl (export_real_tracks.py) + labels_visuels.py.
Étiquettes regroupées en 4 familles d'avis : alpha, fine (β ou μ, y compris électrons courbés), artefact.
Validation : « une vidéo laissée de côté » (on entraîne sur 5 vidéos, on teste sur la 6e) → aucune fuite.

    python tools/real_domain_eval.py           # rapport
    python tools/real_domain_eval.py --save    # entraîne sur tout et enregistre l'adaptateur
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_training" / "real"))

import numpy as np  # noqa: E402

from wilsontrack.detection import reference_model as RM  # noqa: E402

FAM = {"a": "alpha", "b": "fine", "f": "fine", "p": "alpha", "x": "artefact"}
CLASSES = ["alpha", "fine", "artefact"]


def load():
    import labels_visuels as L
    lab = L.labels()
    recs = [json.loads(l) for l in open(ROOT / "_training" / "real" / "tracks.jsonl", encoding="utf-8")]
    out = []
    for r in recs:
        c = lab.get(r["k"], "?")
        if c in FAM:
            r["label"] = FAM[c]
            r["label_fin"] = c
            out.append(r)
    return out


def vec(r, model):
    f = dict(r["features"])
    return model._vector(f, r["duration_s"], r["touches_border"], False) if hasattr(model, "_vector") else None


def fam_of(cls):
    return {"alpha": "alpha", "proton": "alpha", "beta": "fine", "muon": "fine", "fine": "fine",
            "artefact": "artefact"}.get(cls, "incertain")


def report(name, y, p):
    y, p = list(y), list(p)
    n = len(y)
    conf = [i for i in range(n) if p[i] != "incertain"]
    acc = sum(y[i] == p[i] for i in conf) / max(1, len(conf))
    lines = [f"{name} : couverture {len(conf) / n:.0%}, exactitude quand il se prononce {acc:.0%} (n={n})"]
    for c in CLASSES:
        tp = sum(1 for i in range(n) if p[i] == c and y[i] == c)
        npred = sum(1 for i in range(n) if p[i] == c)
        ntrue = sum(1 for i in range(n) if y[i] == c)
        lines.append(f"   {c:9s} précision {tp / npred if npred else float('nan'):.0%} ({tp}/{npred})  "
                     f"rappel {tp / ntrue if ntrue else float('nan'):.0%} ({tp}/{ntrue})")
    return "\n".join(lines)


FEATURE_KEYS = None


def sim_and_rule_inputs(r, model):
    """Avis de l'IA de simulation (5 probabilités) + scores des règles (4) : connaissances physiques
    apprises sur 66 000 traces simulées, réutilisées par l'adaptateur (empilement)."""
    p = model.predict_proba(dict(r["features"]), r["duration_s"], r["touches_border"], False) if model else {}
    sim = [float(p.get(c, 0.0)) for c in RM.ALL_CLASSES]
    rs = r.get("rules_scores") or {}
    return sim + [float(rs.get(c, 0.0)) for c in RM.CLASSES]


def features_matrix(data, model=None, keys=None):
    if keys is None:
        keys = sorted({k for r in data for k, v in r["features"].items() if isinstance(v, (int, float))
                       and not isinstance(v, bool) and not k.startswith("ev_")
                       and k not in ("energy_mev", "momentum_mev")})
    rows = []
    for r in data:
        row = [float(r["features"].get(k) or 0.0) for k in keys] + [r["duration_s"], float(r["touches_border"])]
        if model is not None:
            row += sim_and_rule_inputs(r, model)
        rows.append(row)
    X = np.nan_to_num(np.array(rows, float), nan=0.0, posinf=1e6, neginf=-1e6)
    extra = ["duration_s", "touches_border"] + ([f"sim_{c}" for c in RM.ALL_CLASSES] +
                                                  [f"regle_{c}" for c in RM.CLASSES] if model is not None else [])
    return X, keys, keys + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    data = load()
    y = [r["label"] for r in data]
    vids = [r["video"] for r in data]
    print("Étiquettes :", Counter(y), "par vidéo :", Counter(vids))
    # 1) Règles
    print(report("Règles physiques seules", y, [fam_of(r["rules_cls"]) for r in data]))
    # 2) IA de référence (simulation) en hybride
    model = RM.ReferenceModel.load_default()
    from wilsontrack.config import ClassificationParams
    cp = ClassificationParams()
    preds = []
    for r in data:
        out = (r["rules_cls"], r["rules_conf"], r["rules_scores"], "")
        cls, conf, probs, reason, extra = RM.classify_hybrid_v2(model, dict(r["features"]), out, cp, r["duration_s"],
                                                                r["touches_border"], False)
        if cls == "incertain" and extra.get("family") == "fine" and (extra.get("family_confidence") or 0) >= 0.85:
            cls = "fine"
        preds.append(fam_of(cls))
    print(report("Hybride v2 (IA entraînée sur simulation)", y, preds))
    # 3) Adaptateur réel : gradient boosting sur les vraies traces, validation « une vidéo laissée de côté »
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, keys, names = features_matrix(data, model)
    yv = np.array(y)
    pv = np.empty(len(y), dtype=object)
    probs_all = np.zeros((len(y), len(CLASSES)))
    for v in sorted(set(vids)):
        te = np.array([vv == v for vv in vids])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
        clf.fit(X[~te], yv[~te])
        P = clf.predict_proba(X[te])
        cls = list(clf.classes_)
        P2 = np.zeros((te.sum(), len(CLASSES)))
        for j, c in enumerate(cls):
            P2[:, CLASSES.index(c)] = P[:, j]
        probs_all[te] = P2
    for thr in (0.0, 0.6, 0.7, 0.8, 0.85):
        pv = [CLASSES[int(np.argmax(p))] if p.max() >= thr else "incertain" for p in probs_all]
        print(report(f"Adaptateur réel (une vidéo laissée de côté), seuil {thr:.2f}", y, pv))
    for thr in (0.7, 0.8, 0.85):
        comb = []
        for i, base in enumerate(preds):
            p = probs_all[i]
            top = CLASSES[int(np.argmax(p))]
            if p.max() >= thr and top in ("artefact", "fine"):
                comb.append(top)
            elif base != "incertain" and p[CLASSES.index("artefact")] >= 0.5:
                comb.append("incertain")          # désaccord : l'adaptateur doute d'une trace réelle
            else:
                comb.append(base)
        print(report(f"COMBINÉ (hybride v2 + adaptateur réel, seuil {thr:.2f})", y, comb))
    per_vid = defaultdict(list)
    pv = [CLASSES[int(np.argmax(p))] if p.max() >= 0.7 else "incertain" for p in probs_all]
    for i, v in enumerate(vids):
        per_vid[v].append((y[i], pv[i]))
    for v, l in per_vid.items():
        ok = [t == p for t, p in l if p != "incertain"]
        print(f"   {v[:45]:45s} n={len(l):3d} couverture {len(ok) / len(l):.0%} exactitude {np.mean(ok) if ok else float('nan'):.0%}")
    if a.save:
        import joblib
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
        clf.fit(X, yv)
        dest = ROOT / "wilsontrack" / "assets" / "adaptateur_reel.joblib"
        joblib.dump({"model": clf, "feature_keys": keys, "features": names, "classes": list(clf.classes_), "n": len(y),
                     "videos": sorted(set(vids)), "labels": dict(Counter(y)),
                     "note": "Adaptateur entraîné sur 6 vraies vidéos Wikimedia étiquetées visuellement."}, dest)
        print("enregistré :", dest)


if __name__ == "__main__":
    main()
