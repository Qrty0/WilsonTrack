# Vidéos d'exemple

Deux vidéos **simulées** de 60 s (chambre de Wilson à diffusion, 960×540, 30 img/s) pour essayer
WilsonTrack 2.0 sans matériel :

| Fichier | Contenu |
|---|---|
| `exemple fond (rayons cosmiques).mp4` | bruit de fond : alphas du radon, muons cosmiques, électrons Compton |
| `exemple KCl (source simulee).mp4` | même fond + ≈ 45 électrons/min émis par un échantillon de KCl simulé (spectre β⁻ du K-40) |

**Échelle exacte : 3,84 px/mm** (champ filmé de 250 mm de large). Quand vous ouvrez ces vidéos depuis
l'accueil, l'échelle est lue automatiquement dans le fichier de vérité terrain ; sinon, entrez-la dans
la boîte « Analyser une vidéo » ou dans *Réglages › Zone analysée et échelle*.

Essai conseillé :
1. Accueil › carte **🎞 Analyser une vidéo** (liste « Vidéos d'exemple », ou glissez-déposez le fichier) →
   `exemple fond…`, condition « Fond (rayons cosmiques) », échelle 3,84 → **▶ Analyser**, attendre la fin.
2. Même chose avec `exemple KCl…`, condition « KCl », échantillon « KCl (chlorure de potassium) », masse 100 g.
3. Accueil › carte **⚖ Comparer des résultats** : échantillon = KCl, référence = Fond → la radioactivité du KCl
   doit être détectée dans le groupe des **traces fines**, sans changement du nombre d'alphas.
4. Onglet **Traces** : cliquez une trace pour lire son explication « Pourquoi ? ».

Variante « par lot » : déposez les **deux** vidéos d'un coup sur l'accueil (ou carte **📥 Vidéos reçues /
analyse par lot**) ; elles sont analysées l'une après l'autre et un tableau `resume_lot.csv` est créé. En
ligne de commande :

```
WilsonTrack-console.exe lot "exemples\*.mp4" --px-par-mm 3.84
```

À savoir :

- Ces deux vidéos ont été produites avec le **générateur simplifié de la version 1**. Elles servent à
  découvrir l'application ; elles sont plus « faciles » que de vraies vidéos (pas d'artefacts réalistes,
  électrons plus faciles à reconnaître). Pour une vidéo produite par le **simulateur réaliste** de la
  version 2.0 : `WilsonTrack-console.exe synthetique "ma vidéo.mp4" --duree 60` (ajoutez `--source 40`
  pour une source β simulée), ou *Outils › Générer une vidéo de démonstration…*.
- En mode hybride (par défaut), une partie des traces fines est classée « **Trace fine (β ou μ)** » ou
  « **incertaine** » : c'est normal (voir `docs/GUIDE_UTILISATEUR.md`, section 11).

Les fichiers `.gt.json` contiennent la **vérité terrain** (classe, position, longueur, instant de chaque
trace simulée) : ils servent à mesurer objectivement la précision de l'algorithme
(`WilsonTrack-console.exe valider`).
