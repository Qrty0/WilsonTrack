# WilsonTrack 2.0 — analyse automatique des traces d'une chambre de Wilson

**Projet d'intégration en sciences — Cégep Marie-Victorin (2026)**
Équipe : Yanis Amarouche, Yanees Ait Saadi, Amir Badri, Taha Bich — enseignant : M. Simon Langlois.

WilsonTrack filme (ou relit) une chambre de Wilson vue du dessus, **détecte chaque trace de particule
en temps réel**, la **mesure** (longueur, épaisseur, rectitude, densité d'ionisation, courbure, durée), la
**classe** (alpha, bêta/électron, muon, proton, trace fine, artefact) en **expliquant pourquoi**, et **conclut
statistiquement** : taux ± incertitude, caractère aléatoire (loi de Poisson), stabilité heure par heure,
énergie des alphas, et surtout la **comparaison fond / échantillon** (KCl, banane…) avec la signification
statistique de l'excès de radioactivité.

Tout fonctionne **hors ligne**. En option, l'assistant **Claude** (API Anthropic, Claude Opus 5.5) peut donner
un second avis sur chaque trace et rédiger une interprétation scientifique.

---

## Nouveautés de la version 2.0

- **Menu d'accueil** : caméra en direct, analyse d'une vidéo (glisser-déposer), série de photos, démonstration,
  comparaison, ouverture de résultats, session de nuit, vidéos reçues, Centre d'apprentissage de l'IA, assistant Claude.
- **Session de nuit** : durée ou heure de fin, vidéo réduite découpée en morceaux, photos de contrôle,
  sauvegarde automatique toutes les N minutes (résiste à une panne), anti-veille, estimation de l'espace
  disque, alerte « chambre morte », journal `journal.txt`.
- **Vidéos reçues / analyse par lot** : file d'attente, **dossier surveillé** (par exemple OneDrive), tableau
  récapitulatif `resume_lot.csv`.
- **« Pourquoi ? »** pour chaque trace : critères mesurés, pourquoi pas les autres classes, physique, limites,
  sources ; colonne `pourquoi` dans `traces.csv`, section du rapport HTML ; second avis de Claude Opus 5.5.
- **Classification hybride** : règles physiques + **IA de référence** (gradient boosting) entraînée sur des
  traces simulées réalistes passées dans le vrai détecteur ; 5 classes dont **artefact** ; probabilités
  **calibrées** ; une particule n'est nommée que si elle est **sûre à 85 %** (sinon « incertaine ») ; nouvelle
  classe **« Trace fine (β ou μ) »** quand la famille est sûre mais qu'électron et muon sont physiquement
  indiscernables. Les artefacts sont exclus des comptages.
- **Simulateur réaliste** : transport 3D avec les tables NIST ASTAR/ESTAR/PSTAR, diffusion de Highland (PDG),
  énergies NNDC, durées de vie mesurées (Sussman), artefacts réalistes.
- **Analyse approfondie** : taux par tranche horaire, test de stabilité, périodes mortes, comptages corrigés
  (dépliement avec la matrice de confusion du modèle), analyse des confiances, **score de qualité** sur 100.
- **Moteur ≈ 2,5 fois plus rapide** (≈ 45 images/s en 1280 × 720, résultats identiques à la version 1.1) et
  mémoire bornée pour les séances de plusieurs heures.

## Démarrage rapide (Windows)

1. Décompressez `WilsonTrack-2.0-Windows.zip`, ouvrez le dossier `WilsonTrack` et lancez **`WilsonTrack.exe`**.
   (Si Windows SmartScreen avertit : « Informations complémentaires » › « Exécuter quand même ».)
2. Sans matériel : sur l'**accueil**, carte **🧪 Démonstration** (chambre simulée en temps réel), ou carte
   **🎞 Analyser une vidéo** avec l'une des vidéos d'exemple (dossier `exemples/`).
3. Avec la chambre : branchez la webcam ; carte **📷 Caméra en direct**, **↻** pour trouver la caméra, puis
   **▶ Démarrer**.
4. Tracez la **zone analysée**, **calibrez** avec une règle, indiquez la **condition** (Fond, KCl, Banane…)
   dans le panneau **Réglages**.
5. **⏹ Arrêter** à la fin : résultats, tableau Excel, vignettes et **rapport HTML** sont sauvegardés dans
   `Documents\WilsonTrack Résultats\<date> <condition> essai-NN\`.
6. Carte **⚖ Comparer des résultats** : choisissez l'enregistrement KCl et l'enregistrement « Fond » →
   conclusion chiffrée.
7. Pour une mesure précise : carte **🌙 Session de nuit**, puis comparez une nuit avec échantillon à une nuit
   de fond.

Le guide complet est dans [`docs/GUIDE_UTILISATEUR.md`](docs/GUIDE_UTILISATEUR.md) et la méthode scientifique
(à citer dans le rapport) dans [`docs/METHODE.md`](docs/METHODE.md).

## Fonctions principales

| Domaine | Ce que fait WilsonTrack |
|---|---|
| Acquisition | Webcam / caméra USB (réglages du pilote), fichier vidéo (mp4, avi, mov, webm…), dossier de photos (rafale), démonstration simulée ; enregistrement vidéo nommé selon le protocole (`2026-10-01 KCl essai-01.mp4`), photos brutes + annotées |
| Longues séances | Session de nuit (arrêt à l'heure prévue, sauvegardes automatiques, vidéo segmentée, photos, alertes, reconnexion de la caméra, anti-veille, journal) ; analyse par lot et dossier surveillé ; vidéos de plusieurs heures |
| Détection | Fond adaptatif sélectif, filtre de lignes hessien multi-échelle (ignore les gouttelettes isolées et le brouillard), seuillage par hystérésis normalisé au bruit, fusion des traces en pointillés, suivi temporel (chaque trace comptée **une seule fois**), détection des perturbations (lumière allumée, caméra bougée) |
| Mesures | Longueur et épaisseur **à mi-hauteur du profil d'intensité** (±2 % validé), rectitude, flèche, écart RMS à la droite, tortuosité, densité d'ionisation relative, durée de visibilité, orientation, rayon de courbure (→ impulsion si champ magnétique), énergie des alphas par la relation portée-énergie |
| Classification | Hybride : règles physiques + IA de référence calibrée (5 classes dont artefact), **abstention sous 85 %**, classe « trace fine (β ou μ) » ; explication « Pourquoi ? » de chaque trace ; IA locale entraînable à partir de vos corrections ; Centre d'apprentissage (mesure de la précision sur vos traces corrigées) ; second avis optionnel de Claude |
| Statistiques | Taux ± σ (Poisson), intervalles de Garwood, test binomial exact fond/échantillon (p, σ) sur le groupe des traces fines, test de dispersion, loi exponentielle des intervalles, isotropie des muons (Rayleigh), taux par tranche, stabilité, périodes mortes, comptages corrigés (dépliement), score de qualité |
| Résultats | Conclusions rédigées en français, graphiques, rapport HTML autonome (imprimable en PDF), CSV + Excel (avec la colonne « pourquoi »), vignettes, `resume_lot.csv` |
| Chambre à expansion | Fenêtre sensible après chaque expansion (détection automatique, bouton ou Arduino), statistiques par cycle, micrologiciel Arduino fourni (`arduino/`) |

## Validation et honnêteté scientifique

- **Détection** : sur des vidéos simulées avec vérité terrain (brouillard mouvant, pluie de gouttelettes,
  bruit du capteur), rappel ≈ 87 % et précision ≈ 87 à 90 %, longueurs à ± 2 % (simulateur de la version 1 ;
  le moteur 2.0 donne exactement les mêmes détections). *Outils › Valider l'algorithme* refait un test sur
  votre ordinateur.
- **Classification** : l'IA de référence est évaluée sur des enregistrements simulés réalistes qu'elle n'a
  jamais vus ; les chiffres sont dans [`docs/METHODE.md`](docs/METHODE.md), section 9.2, et dans le Centre
  d'apprentissage de l'IA. Ils sont mesurés **sur simulation seulement** : la précision réelle sur votre
  chambre se mesure en corrigeant à la main des traces tirées au hasard (section 9.5).
- **Limite physique assumée** : sans aimant, un muon et un électron cosmique énergétique laissent la même
  trace fine et droite ; les muons sont donc surtout rapportés comme « traces fines (β ou μ) ». La
  comparaison KCl / fond utilise le groupe des traces fines, qui ne dépend pas de cette ambiguïté.
- **Pourquoi pas des « millions de vidéos » d'Internet ?** Aucun grand corpus public de vidéos étiquetées
  n'existe, des vidéos sans étiquettes n'apprennent pas la bonne réponse, et leur téléchargement massif pose
  des problèmes de droits d'auteur. L'IA est donc entraînée sur une simulation fondée sur des données publiées
  (NIST ASTAR/ESTAR/PSTAR, PDG, NNDC, CERN S'Cool LAB, Sussman) : voir [`docs/METHODE.md`](docs/METHODE.md),
  sections 8 et 10.

## Ligne de commande

```
WilsonTrack-console.exe analyser "2026-10-01 KCl essai-01.mp4" --condition KCl --echantillon "KCl (chlorure de potassium)" --masse 100 --px-par-mm 6.2
```

```
WilsonTrack-console.exe comparer "Résultats\2026-10-01 KCl essai-01" "Résultats\2026-10-01 Fond (rayons cosmiques) essai-01"
```

```
WilsonTrack-console.exe nuit --jusqua 07:00 --condition KCl --echantillon KCl --masse 100 --px-par-mm 6.2
```

```
WilsonTrack-console.exe lot "D:\vidéos reçues\*.mp4" --condition "Fond (rayons cosmiques)"
```

```
WilsonTrack-console.exe surveiller "C:\Users\moi\OneDrive\Chambre" --condition KCl --echantillon KCl --masse 100
```

```
WilsonTrack-console.exe valider --duree 60
```

`WilsonTrack-console.exe COMMANDE --aide` affiche les options d'une commande ;
`WilsonTrack-console.exe --autotest` vérifie l'installation de bout en bout.

## Depuis le code source (développeurs)

```
python -m venv .venv
```

```
.venv\Scripts\python -m pip install -r requirements.txt
```

```
.venv\Scripts\python main.py
```

Tests : `.venv\Scripts\python -m pytest -q tests` — construction de l'exécutable : `.\build_exe.ps1`.
Réentraînement de l'IA de référence : `tools\train_reference_model.py` (voir `--help`).

## Structure

```
wilsontrack/
  detection/   pré-traitement, segmentation, mesures, suivi temporel,
               classifier.py (règles), reference_model.py (IA de référence, calibration, abstention),
               explain.py (explication « Pourquoi ? » de chaque trace)
  analysis/    enregistrement, statistiques et analyse approfondie, conclusions, graphiques, rapports
  gui/         interface PySide6 (accueil, fenêtre principale, session de nuit et lot,
               Centre d'apprentissage de l'IA, réglages, fils de traitement)
  assets/      modèle de l'IA de référence et exemples de traces
  capture.py   caméra, fichiers, photos, enregistrement vidéo segmenté, liaison série Arduino
  longrun.py   sessions de nuit (sauvegardes, alertes, journal, anti-veille)
  batch.py     analyse par lot et dossier surveillé
  synth.py     simulateur réaliste de chambre de Wilson (démonstration, validation, entraînement)
  evaluate.py  comparaison avec la vérité terrain
  ai_assistant.py  assistant Claude (optionnel)
  cli.py       ligne de commande (analyser, comparer, nuit, lot, surveiller, valider…)
tools/         entraînement de l'IA de référence, bancs d'essai
tests/         tests automatisés (pytest)
docs/          méthode scientifique et guide d'utilisation
exemples/      vidéos d'exemple simulées avec vérité terrain
arduino/       déclencheur de chambre à expansion
```

## Licence et crédits

Code écrit pour le projet d'intégration ; libre de réutilisation à des fins éducatives et scientifiques
(merci de citer le projet). Bibliothèques : OpenCV, NumPy, SciPy, scikit-image, scikit-learn, PySide6 (Qt),
pyqtgraph, Matplotlib, openpyxl, SDK Anthropic. Données physiques : NIST (ASTAR, ESTAR, PSTAR, XCOM),
Particle Data Group, NNDC (Brookhaven), CERN S'Cool LAB.
