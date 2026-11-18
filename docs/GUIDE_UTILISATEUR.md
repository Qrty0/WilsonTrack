# Guide d'utilisation de WilsonTrack 2.0

Ce guide suit le déroulement d'une séance de mesure, de l'installation de la caméra jusqu'à la
conclusion « le KCl est-il radioactif ? ». Les numéros d'étapes correspondent au protocole de
l'avant-projet (section *Système de captation et d'analyse*). Les explications scientifiques
détaillées (à citer dans le rapport) sont dans [`METHODE.md`](METHODE.md).

**Nouveautés de la version 2.0** : menu d'accueil, **session de nuit**, **vidéos reçues / analyse par lot**
avec dossier surveillé, explication **« Pourquoi ? »** de chaque trace, classification **hybride**
(règles physiques + IA de référence) qui ne nomme une particule que si elle est **sûre à 85 %**, nouvelle
classe **« Trace fine (β ou μ) »**, **Centre d'apprentissage de l'IA**, **analyse approfondie** (taux heure
par heure, stabilité, score de qualité) et moteur environ **2,5 fois plus rapide**.

---

## 1. Installer la caméra et l'éclairage

- **Caméra au-dessus** de la chambre, objectif vers la plaque froide, fixée sur un support rigide
  (trépied, bras) : la moindre vibration crée de fausses différences d'image.
- **Fond noir mat** sous la chambre, **DEL sur les côtés**, orientées vers la fine couche de brouillard
  au-dessus de la plaque, jamais directement vers la caméra. Pièce aussi sombre que possible.
- Résolution **1080p** ou **720p** recommandée, 30 images/s. Désactivez l'exposition et la mise au point
  automatiques si la caméra le permet (bouton *Réglages du pilote* dans *Réglages › Caméra*).
- Téléphone intelligent : une application « webcam » (DroidCam, Iriun, Camo…) le fait apparaître comme
  une caméra dans la liste. Vous pouvez aussi filmer avec le téléphone et analyser le fichier ensuite.

## 2. Premier lancement : le menu d'accueil

Décompressez `WilsonTrack-2.0-Windows.zip`, ouvrez le dossier `WilsonTrack` et lancez **`WilsonTrack.exe`**
(si Windows SmartScreen avertit : « Informations complémentaires » › « Exécuter quand même »).

L'application s'ouvre sur l'onglet **🏠 Accueil**, un menu de cartes :

| Carte | À quoi elle sert |
|---|---|
| 📷 **Caméra en direct** | choisir la caméra (↻ pour la rechercher), puis **▶ Démarrer** : détection en temps réel |
| 🎞 **Analyser une vidéo** | choisir un fichier (ou le **glisser-déposer** n'importe où sur la page), régler condition, échantillon, masse et échelle, puis **▶ Analyser** ; listes des vidéos récentes et des vidéos d'exemple |
| 🌙 **Session de nuit** | filmer et analyser toute une nuit sans surveillance (section 8) |
| 🖼 **Série de photos** | analyser un dossier d'images (rafale, time-lapse) comme une vidéo |
| 📥 **Vidéos reçues / analyse par lot** | analyser plusieurs vidéos à la suite, ou surveiller un dossier (par exemple OneDrive) (section 9) |
| 🧪 **Démonstration** | chambre simulée en temps réel, sans matériel ; bouton *Avec source KCl* pour ajouter une source β |
| ⚖ **Comparer des résultats** | comparer un enregistrement « échantillon » à un « fond » ; *📂 Ouvrir…* pour relire des résultats enregistrés |
| 🧠 **Centre d'apprentissage de l'IA** | comment l'IA de référence a appris, ce que valent ses résultats, et mesure de sa précision sur VOTRE chambre (section 12) |
| 🤖 **Assistant Claude (Opus 5.5)** | second avis sur les traces, interprétation des résultats (facultatif, clé API) |

Le reste de la fenêtre :

- à gauche, le panneau **Réglages** (groupes repliables : caméra, zone analysée et échelle, expérience,
  détection, physique, classification, chambre à expansion…) ;
- en haut, la barre d'outils : **🏠 Accueil**, **📷 Caméra**, **🎞 Vidéo…**, **🖼 Photos…**, **🧪 Démo**,
  **⏯ Pause**, **⏹ Arrêter**, **⏺ Enregistrer la vidéo**, **📸 Photo**, **💥 Expansion**, **🤖 Demander à Claude** ;
- les onglets **Accueil**, **Direct**, **Traces**, **Graphiques**, **Conclusions**, **Comparer**,
  **Assistant IA**, **Lot**, **IA** (Centre d'apprentissage) et **Aide**.

Pour découvrir l'application sans matériel, cliquez la carte **🧪 Démonstration** : une chambre de Wilson
simulée (alphas du radon, muons, électrons, brume…) s'affiche en temps réel et les traces sont détectées et classées.

## 3. Démarrer la caméra

1. Branchez la caméra. Sur la carte **📷 Caméra en direct**, cliquez **↻** pour rechercher les caméras,
   choisissez la bonne, puis **▶ Démarrer** (ou bouton **📷 Caméra** de la barre d'outils, Ctrl+K).
   La résolution se choisit dans *Réglages › Caméra*.
2. L'onglet **Direct** s'ouvre. Pendant les premières images, le logiciel **apprend le fond** (message
   « apprentissage du fond »), puis la détection démarre.
3. Menu *Affichage* au-dessus de l'image :
   - *Traces annotées* (normal) ;
   - *Image brute* ;
   - *Contraste (image − fond)* : ce qui change par rapport au fond ;
   - *Carte de détection (SNR)* : ce que « voit » l'algorithme (utile pour régler la sensibilité) ;
   - *Masque des détections*.
   Le curseur *Contraste* rehausse l'affichage des traces faibles (sans influencer la détection).

## 4. Zone analysée et calibration (important)

- *Réglages › Zone analysée et échelle* › **Rectangle** ou **Polygone** : tracez sur l'image la zone
  utile, en excluant les parois, reflets, vis, électrodes ou tout objet brillant.
- **📏 Calibrer** : posez une règle graduée sur la plaque (dans le plan des traces), cliquez deux
  graduations éloignées (par ex. 0 et 100 mm) et entrez la distance réelle. L'échelle (px/mm) est
  mémorisée. **Sans calibration, les longueurs en mm et l'énergie des alphas sont approximatives.**
- Ces deux réglages s'appliquent au démarrage de l'analyse ; si l'analyse tourne, le logiciel propose
  de redémarrer.

## 5. Renseigner l'expérience

*Réglages › Expérience* : **Condition** (Fond (rayons cosmiques), KCl, Banane…), **Échantillon** et **Masse**
(servent à calculer l'activité attendue du potassium-40), **Essai n°**, opérateur, notes (température de
la plaque, quantité d'alcool…). Ces informations nomment les fichiers :
`2026-10-01 KCl essai-01`. Pour une vidéo, une session de nuit ou un lot, les mêmes champs sont demandés
dans la boîte de dialogue de départ.

## 6. Mesurer

- Les traces sont entourées en direct avec leur classe et la confiance :
  **α alpha** (rouge), **β électron** (bleu), **μ muon** (vert), **p proton** (orange),
  **f trace fine (β ou μ)** (bleu-vert), **? incertaine** (gris). Les **artefacts** (brume, gouttelettes,
  reflets) sont reconnus et **exclus des comptages**. Quand une trace disparaît, son tracé mesuré s'affiche
  avec son numéro et sa longueur.
- À droite : compteurs par classe avec le taux par minute et son incertitude, courbe cumulée, dernières traces.
- **⏺ Enregistrer la vidéo** (touche R) enregistre la vidéo brute dans `Vidéos\` : elle pourra être
  réanalysée plus tard avec d'autres réglages. **📸 Photo** (touche P) enregistre l'image brute et annotée.
- Durée conseillée : **10 à 20 minutes par condition au minimum** (une nuit entière est encore mieux,
  section 8), et **les mêmes réglages** pour toutes les conditions que l'on compare (même glace sèche
  fraîche, même alcool, même éclairage, même zone).
- **⏹ Arrêter** : les références de classification sont recalculées sur l'ensemble de l'enregistrement,
  toutes les traces sont reclassées, puis les résultats sont sauvegardés automatiquement.

**Pourquoi tant de traces « fines » ou « incertaines » ?** C'est voulu. Le logiciel ne nomme une particule
que s'il en est **sûr à au moins 85 %**. Or un électron rapide et un muon laissent la **même** trace fine et
droite sur quelques centimètres (même ionisation, presque aucune déviation) : sans aimant, la physique ne
permet pas de les départager. Le logiciel l'écrit honnêtement : « Trace fine (β ou μ) ». Pour la question du
projet, ce n'est pas un problème : la comparaison KCl / fond utilise le groupe des **traces fines**
(section 15). Explication complète : section 11 et [`METHODE.md`](METHODE.md), sections 5.4 et 5.5.

## 7. Analyser une vidéo, une série de photos, la démonstration

- **Vidéo** : carte **🎞 Analyser une vidéo** › *Choisir une vidéo…* (Ctrl+O), ou glissez-déposez le fichier
  sur l'accueil. La boîte « Analyser une vidéo » montre la première image et les informations du fichier ;
  indiquez la **condition**, l'**échantillon**, la **masse** et l'**échelle** (px/mm ; pour les vidéos
  d'exemple, elle est lue automatiquement), puis **▶ Analyser**. Case *Vidéo à vitesse réelle* (onglet
  Direct) : décochée, la vidéo est analysée aussi vite que possible. Une vidéo illisible est signalée :
  convertissez-la en MP4 (H.264) ou AVI (MJPG) avec VLC ou HandBrake.
- **Plusieurs vidéos** déposées d'un coup (ou un dossier) ouvrent l'analyse par lot (section 9).
- **Série de photos** : carte **🖼 Série de photos** › *Choisir un dossier…* (Ctrl+Maj+O).
- **Démonstration** : carte **🧪 Démonstration** (Ctrl+D), ou *Fichier › Démo avec source KCl simulée*.
  Elle utilise le simulateur réaliste de la version 2.0.

## 8. Session de nuit

Plus on compte de traces, plus le résultat est précis (l'incertitude relative vaut 1/√N). Une nuit de 10 h
donne environ 30 fois plus de traces qu'un essai de 20 min.

1. Préparez la chambre (beaucoup de glace sèche et d'alcool), fixez la caméra, **calibrez** et tracez la
   **zone analysée** (section 4).
2. Carte **🌙 Session de nuit** › *Préparer la nuit…*. Réglez :
   - **Caméra** ;
   - **Durée** : *Pendant* X h, ou *Jusqu'à* HH h MM (défaut : 7 h 30 ; une heure déjà passée = demain) ;
     l'heure de fin prévue s'affiche ;
   - **Dans la chambre** : condition, échantillon, masse, échelle ;
   - **Vidéo** : *Aucune* (vignettes et photos seulement), *Réduite* (½, ⅓ ou ¼ de la résolution,
     10 img/s par défaut ; recommandé) ou *Complète* (lourde). L'analyse utilise toujours toutes les images ;
   - **Fichiers de** N min : la vidéo est découpée en morceaux (30 min par défaut) ; en cas de coupure de
     courant, seul le dernier morceau est perdu ;
   - **Photo toutes les** N min : photo de contrôle (brute + annotée), pour vérifier le lendemain ;
   - **Sauvegarde toutes les** N min (5 par défaut) : les résultats sont écrits sur le disque ; si
     l'ordinateur plante, presque rien n'est perdu ;
   - **Alerte sans trace** après N min (20 par défaut) : la chambre est peut-être « morte » (plus d'alcool
     ou de glace sèche) ;
   - **Empêcher la mise en veille de l'ordinateur** (coché) ;
   - **Dossier** de la session.
   Une **estimation de l'espace disque** nécessaire s'affiche (ordre de grandeur pour 8 h en 1280 × 720 :
   ≈ 0,7 Go sans vidéo, ≈ 3,4 Go en vidéo réduite, ≈ 33 Go en vidéo complète). Si l'espace manque,
   l'avertissement est rouge et le bouton devient « Lancer quand même ».
3. Vérifiez la liste **✅ Avant de partir** : assez d'alcool et de glace sèche ? caméra fixée ? ordinateur
   branché sur le secteur ? Puis lancez.
4. Un **bandeau** en haut de l'onglet Direct affiche le temps écoulé et restant, le nombre de traces, le
   taux par minute, l'espace disque libre, le morceau de vidéo en cours, les dernières et prochaines
   sauvegardes et les **alertes** (chambre sans trace, baisse significative du taux, disque presque plein,
   caméra débranchée). Pour arrêter avant l'heure, cliquez **deux fois** sur le bouton d'arrêt (sécurité
   contre les clics accidentels).
5. Si la caméra se débranche, le logiciel tente de la reconnecter pendant 15 min ; la coupure n'est pas
   comptée dans la durée d'observation. Si le disque se remplit, la vidéo s'arrête mais les mesures continuent.
6. À l'heure prévue, les résultats définitifs et le rapport sont produits automatiquement.

Dossier produit : `Documents\WilsonTrack Résultats\<date> <condition> nuit <HHhMM>\` avec `run.json`,
`traces.csv`, `traces.xlsx`, `rapport.html`, `session.json`, **`journal.txt`** (une ligne d'état par minute et
toutes les alertes), `vignettes\`, `photos\` et `video\` (morceaux `.avi`).

> Les sauvegardes intermédiaires sont marquées **« partiel »** (résultats provisoires). N'utilisez dans le
> rapport que les résultats **finaux**.

En ligne de commande (sans interface) :

```
WilsonTrack-console.exe nuit --jusqua 07:00 --condition KCl --echantillon KCl --masse 100 --px-par-mm 6.2
```

Options utiles : `--duree-h 8`, `--video aucune|reduite|complete`, `--segment-min 30`, `--photo-min 10`,
`--sauvegarde-min 5`, `--alerte-min 20`, `--resolution 1280x720`, `--camera 1`, `--simulation` (pour
essayer la commande avec une chambre simulée). `WilsonTrack-console.exe nuit --aide` donne la liste complète.

## 9. Vidéos reçues, analyse par lot et dossier surveillé

Pratique quand des coéquipiers filment chez eux ou au laboratoire et vous envoient leurs vidéos.

1. Carte **📥 Vidéos reçues / analyse par lot** › *Choisir les vidéos…* (ou déposez plusieurs vidéos ou un
   dossier sur l'accueil). Ajoutez des vidéos ou *Tout un dossier…* ; les fichiers qui ne sont pas des vidéos
   sont ignorés (et comptés).
2. Réglez la condition, l'échantillon, la masse et l'échelle (appliqués à toutes les vidéos), le dossier de
   sortie et l'option *Créer un rapport pour chaque vidéo*.
3. Option **👁 Surveiller un dossier** : choisissez par exemple un dossier **OneDrive** ou **Google Drive**
   partagé. Chaque nouvelle vidéo **complète** qui y arrive est analysée automatiquement (une vidéo encore en
   cours de téléchargement est ignorée jusqu'à ce que sa taille ne change plus). Une vidéo déjà traitée n'est
   **jamais analysée deux fois** (mémoire dans le fichier `.wilsontrack_traitees.json`).
4. L'onglet **📥 Lot** montre la file : état (en attente, en cours, terminé, erreur, annulé), progression,
   temps restant, nombre d'α, β, μ et p, et un bouton pour ouvrir le dossier de résultats (ou le détail de
   l'erreur).

Chaque vidéo produit un dossier `<date> <condition> <nom de la vidéo>` et le dossier de sortie contient un
tableau récapitulatif **`resume_lot.csv`** (+ `.xlsx`). En ligne de commande :

```
WilsonTrack-console.exe lot "D:\vidéos reçues\*.mp4" --condition "Fond (rayons cosmiques)" --px-par-mm 6.2
```

```
WilsonTrack-console.exe surveiller "C:\Users\moi\OneDrive\Chambre" --condition KCl --echantillon KCl --masse 100
```

(`--une-fois` : analyser les vidéos présentes puis quitter ; Ctrl+C pour arrêter la surveillance.)

## 10. Onglet Traces : vérifier, comprendre, corriger

- **Tableau** triable (cliquez un en-tête) et **Galerie** de vignettes, filtre par classe. La dernière colonne,
  **Pourquoi (résumé)**, explique la décision en une ligne.
- Cliquez une trace : vignette agrandie et **explication complète « Pourquoi ? »** :
  - le **verdict** (classe, confiance, méthode) ;
  - **ce qui a été mesuré**, critère par critère, avec la valeur attendue : ✔ pour, ✘ contre, • neutre ;
    les indices qui n'ont pas servi à décider sont marqués « indice complémentaire » ;
  - les **probabilités** de chaque classe ;
  - **pourquoi pas les autres ?** : une phrase chiffrée par classe écartée (par exemple « Pas une alpha :
    165 mm de long alors qu'une alpha naturelle parcourt au plus ≈ 74 mm dans l'air à −25 °C ») ;
  - la **physique** en jeu, les **limites** et les **sources** (NIST, PDG, CERN…).
  Pour une trace incertaine, l'explication nomme les deux classes possibles et dit pourquoi les mesures ne
  peuvent pas les séparer.
- **🤖 Pourquoi ? — 2e avis de Claude Opus 5.5** : Claude examine la vignette et les mesures, juge d'abord
  seul, puis dit s'il est d'accord avec le logiciel et pourquoi (clé API nécessaire, section 16). C'est un
  second avis : il ne change ni la classe ni les comptages.
- **Corriger la classe** si le logiciel s'est trompé (alpha, beta, muon, proton, fine, incertain, ou
  « artefact » pour retirer une fausse détection). Une correction remplace la classe automatique dans les
  comptages.
- **🧠 Entraîner l'IA locale avec mes corrections** : une forêt aléatoire apprend à partir de vos corrections
  (et, si la case est cochée, des traces sûres à ≥ 85 %). Choisissez ensuite *Réglages › Classification ›
  Méthode : IA locale (vos corrections)*. Plus vous corrigez, meilleure elle devient.
- **Exporter CSV / Excel** : fichiers `traces.csv` et `traces.xlsx` (feuille « Résumé » + « Traces »), avec
  les colonnes `classe`, `famille` (fine ou lourde), `methode` et `pourquoi`.

## 11. Comment le logiciel classe les traces (en bref)

*Réglages › Classification › Méthode* :

- **Hybride : règles + IA de référence** (par défaut). Deux « juges » votent :
  1. les **règles physiques** : épaisseur et densité de gouttelettes (pouvoir ionisant), rectitude
     (diffusion multiple), longueur (portée des alphas) ;
  2. l'**IA de référence** : un modèle de *gradient boosting* entraîné sur des **centaines de vidéos
     simulées** de façon réaliste (données du NIST, du PDG et du NNDC), passées dans le vrai détecteur.
     Elle connaît 5 classes : alpha, électron, muon, proton et **artefact**.

  Leurs avis sont combinés en **probabilités calibrées** (une probabilité de 90 % veut dire « juste environ
  9 fois sur 10 »). Une particule n'est nommée que si sa probabilité atteint **85 %**. Sinon :
  - si la trace est sûrement **fine** mais qu'on hésite entre électron et muon → **« Trace fine (β ou μ) »** ;
  - sinon → **« Incertaine »**, avec les deux classes possibles (« muon 62 % ou électron 35 % »).
  Les traces fines et incertaines comptent dans le total et dans leur famille ; les **artefacts** sont exclus.
- **Règles physiques seules** : explicables et identiques d'une séance à l'autre, mais moins fiables (score
  de confiance flou, pas une probabilité ; « incertaine » sous 30 %).
- **IA locale (vos corrections)** : section 10.

**Limites à connaître (et à écrire dans le rapport)** : l'IA de référence n'a été validée que **sur
simulation** ; elle n'a jamais vu votre chambre. Les **muons** sont presque toujours rapportés comme
« traces fines », car à cette échelle ils sont physiquement indiscernables des électrons cosmiques
énergétiques. La précision réelle sur votre montage se mesure dans le Centre d'apprentissage (section 12).

## 12. Centre d'apprentissage de l'IA (onglet 🧠 IA)

Cette page explique, comme pour un élève, comment l'IA de référence a appris :

1. on **simule** des milliers de traces de chaque type, avec la physique réelle (portées, ionisation,
   diffusion, durées de vie mesurées) ;
2. chaque vidéo simulée passe dans le **vrai détecteur** de l'application ;
3. l'IA apprend à relier les **mesures** de chaque trace à sa vraie classe, connue par construction ;
4. on l'**examine** sur des enregistrements simulés qu'elle n'a jamais vus, puis dans des **conditions
   extrêmes**.

Elle montre les familles de traces simulées, des **exemples** de chaque classe, les chiffres clés, les
résultats des deux examens (couverture, exactitude avec son intervalle de confiance à 95 %, précision et
rappel par classe, comparaison règles seules / IA seule / combinaison) et la **matrice de confusion**
(quelle vraie classe devient quelle classe), avec un guide de lecture.

**Résultats de l'IA de référence livrée avec cette version :**

**Résultats de l'entraînement final** (20 octobre 2026) : 66 443 traces simulées étiquetées (1 630 enregistrements simulés réalistes passés dans le vrai détecteur) ; 35 887 pour l'apprentissage, 11 605 pour la validation et la calibration, 12 712 pour le test (enregistrements jamais vus), 6 239 en « conditions extrêmes ».

| Mesure (test, simulations jamais vues) | Résultat |
|---|---|
| Exactitude quand l'IA nomme une particule (seuil 85 %) | **96,2 %** (95,0 % en conditions extrêmes) |
| Précision alpha | 96,5 % |
| Précision électron (β) | 96,8 % |
| Précision proton | 94,8 % |
| Précision artefact (rejet) | 97,3 % |
| Famille fine / lourde | **99,4 %** (99,8 % quand elle est sûre, soit 93 % des traces) |
| Calibration (ECE) | 0,015 (une probabilité annoncée de 90 % est juste ≈ 90 % du temps) |
| Règles physiques seules (version 1) | 48 % |

Le muon n'est jamais affirmé seul : même avec 60 000 exemples, sa précision n'atteint que 77,6 %, car un électron cosmique énergétique laisse exactement la même trace fine et droite. Ces traces sont classées « Trace fine (β ou μ) », dont la famille est sûre à plus de 99 %. Les particules nommées le sont donc toutes à plus de 85 %. Ces chiffres sont mesurés sur des simulations : la précision réelle sur votre chambre se mesure dans le Centre d'apprentissage avec vos traces corrigées.

Ces chiffres sont mesurés **sur simulation**. Pour savoir ce que vaut l'IA **sur VOTRE chambre** (section
« Et sur VOTRE chambre ? ») :

1. ouvrez un enregistrement réel ;
2. dans l'onglet **Traces**, corrigez la classe d'au moins 5 traces **tirées au hasard** (idéalement 30 ou
   plus de chaque type) en vous aidant de la vignette et de l'explication ;
3. revenez dans l'onglet 🧠 IA et lancez la validation : le logiciel compare les réponses de l'IA aux
   vôtres. Avec moins de 20 traces d'une classe, les chiffres de cette classe sont signalés ⚠ (très
   incertains).

Ce résultat est une vraie **validation sur données réelles**, à citer dans le rapport.

## 13. Onglet Graphiques

Longueur–épaisseur (le graphique classique pour séparer les particules), histogrammes de longueur,
taux en fonction du temps (vérifie la stabilité de la chambre), densité–épaisseur, énergie des alphas
(avec les raies du radon et de ses descendants), rose des orientations des muons. Le rapport HTML ajoute,
quand les données existent : taux par tranche, courbe cumulée, bruit au cours du temps, distribution des
confiances et comptages corrigés.

## 14. Onglet Conclusions et analyse approfondie

Taux par classe (± incertitude de Poisson), par cm², puis les conclusions rédigées :

- **score de qualité** de l'enregistrement sur 100 (durée, nombre de traces, stabilité, bruit, temps perdu,
  calibration, proportion d'incertaines, alphas), avec des **conseils** pour améliorer la prochaine mesure ;
  un enregistrement de moins de 10 min ne dépasse pas 59/100 ;
- **caractère aléatoire** : test de dispersion (variance/moyenne ≈ 1) et loi exponentielle des temps
  entre traces — preuve que les traces viennent de processus aléatoires indépendants ;
- **stabilité** : taux par tranche (1 min, 10 min ou 1 h selon la durée), test « le taux est-il constant ? »
  et dérive en % par heure, pour le total, les alphas et les traces fines ; si seuls les alphas varient,
  c'est plutôt le radon de la pièce, si tout baisse, c'est plutôt la chambre qui s'épuise ;
- **périodes mortes ou dégradées** (chambre épuisée, bruit élevé) : signalées et exclues des taux « en temps
  actif », jamais des comptages bruts ;
- **alphas** : énergie estimée par la portée (règle de Geiger corrigée de la température de l'air),
  émetteur le plus proche (Rn-222, Po-218, Po-214…) ;
- **muons** : taux, flux par cm², isotropie des orientations (s'il y a assez de muons nommés) ;
- **ambiguïté β / μ** et **traces fines** ;
- **comptages corrigés** (mode hybride) : estimation des vrais nombres de chaque particule à partir de la
  matrice de confusion de l'IA (mesurée sur simulation) ; toujours présentée **à côté** des comptages
  bruts, jamais à leur place ;
- **fiabilité de la classification** : confiances, accord règles/IA, accord avec vos corrections ;
- avertissements : échelle non calibrée, statistique insuffisante, conditions instables.

Boutons : **💾 Sauvegarder les résultats**, **📄 Rapport HTML** (s'ouvre dans le navigateur ; *Imprimer ›
Enregistrer en PDF* pour l'annexer au rapport ; contient la section « Analyse approfondie », la section
« Pourquoi ces types ? » et, pour une nuit, la section « Session de nuit »), **📂 Ouvrir le dossier**,
**🤖 Interprétation par Claude**.

## 15. Onglet Comparer — la conclusion du projet

1. Les enregistrements sauvegardés apparaissent dans la liste (bouton *Ajouter un dossier de résultats…* pour
   d'autres, par exemple ceux d'un lot).
2. **Échantillon** = l'enregistrement KCl (ou banane) ; **Référence (fond)** = l'enregistrement sans source.
3. **⚖ Comparer** : pour chaque classe, taux avec et sans échantillon, **excès net ± incertitude**,
   **p-valeur** et **signification en σ** (test binomial exact). Conclusions :
   - « Radioactivité β détectée » si l'excès est significatif (p < 0,01) ;
   - le groupe des **traces fines** (bêta + muon + trace fine + incertaines de famille fine) est testé :
     c'est l'indicateur le plus robuste, car il ne dépend pas de la confusion électron / muon ;
   - **contrôle** : le nombre d'alphas ne doit pas changer (le K-40 n'émet pas d'alpha) ;
   - **activité attendue** du K-40 dans l'échantillon (Bq) et **efficacité apparente** de détection ;
   - les **périodes mortes** sont exclues des deux enregistrements, et un avertissement s'affiche si le taux
     d'un enregistrement n'était pas stable.
4. **📄 Rapport de comparaison** : rapport HTML complet à joindre au rapport final.

Pour la banane, l'activité attendue (~17 Bq par banane) est environ 100 fois plus faible que 100 g de
KCl : un excès non significatif est le résultat attendu, et c'est une conclusion scientifique valable.
Une **session de nuit** avec la banane, comparée à une nuit de fond, donne la meilleure chance de voir
un effet.

## 16. Assistant Claude — facultatif

1. Créez une clé API sur le site d'Anthropic (console) et collez-la dans l'onglet **🤖 Assistant IA**
   (cochez *Mémoriser sur cet ordinateur* pour la garder). Choisissez le **modèle** (Claude Opus 5.5
   recommandé, Sonnet 5.5 plus rapide, Haiku 5.5 économique) et l'**effort**, puis *Tester la connexion*.
   Les appels sont facturés par Anthropic.
2. **🖼 Analyser l'image actuelle** (ou 🤖 dans la barre d'outils, idéalement en pause) : Claude examine
   l'image annotée et donne sa classification de chaque trace numérotée, signale les traces manquées,
   évalue la qualité de l'image et propose des améliorations du montage.
3. **2e avis sur une trace** : onglet Traces (section 10).
4. **🧾 Interpréter l'enregistrement / ⚖ Interpréter la comparaison** : texte scientifique structuré (à relire
   et adapter — ne pas copier tel quel dans le rapport).
5. Zone de question : dialogue libre sur vos résultats.

Seules les images et mesures que vous envoyez quittent l'ordinateur. Sans clé ou sans Internet, tout le
reste de l'application fonctionne normalement.

## 17. Si les résultats ne sont pas bons

| Symptôme | Solution |
|---|---|
| Beaucoup de fausses détections (taches, brouillard) | Baisser la **Sensibilité**, augmenter *Contraste min.* ou *Visibilité min.*, réduire la zone analysée, améliorer l'éclairage rasant |
| Traces visibles mais non détectées | Augmenter la **Sensibilité**, vérifier la mise au point, affichage *Carte de détection* |
| Une trace comptée plusieurs fois | Augmenter *Longueur min.* ; vérifier que la caméra ne bouge pas |
| « PERTURBATION » affiché | Lumière allumée, caméra heurtée, ombre : le fond est réappris automatiquement |
| Beaucoup de « traces fines » et d'« incertaines » | **Normal** en mode hybride (section 11). Calibrer l'échelle aide ; la comparaison utilise le groupe des traces fines |
| Tout est classé « incertaine » | Calibrer l'échelle ; vérifier l'éclairage et la mise au point ; corriger quelques traces puis mesurer la précision (section 12) |
| Alerte « chambre sans trace » pendant la nuit | Plus d'alcool ou de glace sèche, givre sur la paroi : la période est signalée comme « morte » dans les conclusions |
| Score de qualité faible | Lire les conseils donnés sous le score (durée trop courte, chambre instable, bruit, calibration…) |
| Vidéo lente | *Largeur de calcul* 640 px, résolution caméra 1280×720 |
| Barre de défilement horizontale dans les Réglages (petit écran) | Élargir le panneau Réglages en tirant sur son bord |

## 18. Chambre à expansion (objectif 6)

*Réglages › Chambre à expansion* : cocher le mode, choisir le déclenchement :
- **auto** : l'expansion est reconnue au saut de luminosité global de l'image ;
- **série (Arduino)** : le micrologiciel `arduino/declencheur_expansion` envoie `EXP` à chaque expansion,
  et le bouton 💥 (touche E) peut commander l'électrovanne ;
- **manuel** : touche E au moment de l'expansion.
Seules les traces apparues dans la **fenêtre sensible** (délai + durée réglables, typiquement 40 ms
puis 0,3–0,6 s) sont comptées ; chaque trace est associée à son cycle d'expansion.

## 19. Fichiers produits

```
Documents\WilsonTrack Résultats\
  2026-10-01 KCl essai-01\
    run.json        paramètres, métadonnées, résumé
    traces.csv      une ligne par trace (toutes les mesures, classe, famille, « pourquoi »)
    traces.xlsx     idem pour Excel (+ feuille de résumé)
    vignettes\      image de chaque trace
    rapport.html    rapport illustré (avec analyse approfondie)
  2026-10-01 KCl nuit 22h15\     session de nuit
    run.json, traces.csv, traces.xlsx, rapport.html
    session.json    réglages et déroulement de la session
    journal.txt     état minute par minute, alertes
    vignettes\  photos\  video\   (morceaux .avi)
  resume_lot.csv / resume_lot.xlsx   tableau récapitulatif d'une analyse par lot
  Vidéos\           vidéos brutes enregistrées
  Photos\           photos brutes et annotées
```

## 20. Ligne de commande (résumé)

`WilsonTrack-console.exe` (dans le dossier de l'application) :

| Commande | Rôle |
|---|---|
| `analyser VIDEO… --condition … --echantillon … --masse … --px-par-mm …` | analyser une ou plusieurs vidéos |
| `comparer DOSSIER_ECHANTILLON DOSSIER_FOND` | comparer deux enregistrements |
| `nuit [--duree-h H \| --jusqua HH:MM] …` | session de nuit sans interface (section 8) |
| `lot FICHIER…` | analyser une série de vidéos (section 9) |
| `surveiller DOSSIER` | analyser chaque nouvelle vidéo d'un dossier (section 9) |
| `synthetique SORTIE.mp4` | générer une vidéo simulée avec vérité terrain |
| `valider --duree 60` | mesurer la fiabilité de la détection sur une vidéo simulée |
| `cameras` | lister les caméras détectées |

`--aide` après une commande affiche toutes ses options ; `--autotest` vérifie l'installation.

## 21. Pourquoi l'IA n'a-t-elle pas appris sur des millions de vidéos d'Internet ?

Question fréquente, et bonne question. Trois raisons :

1. **Il n'existe pas de grande collection publique de vidéos étiquetées** de chambres à brouillard. Les jeux
   trouvés comptent de quelques centaines à quelques milliers d'images, avec des étiquettes d'amateurs
   incohérentes, presque pas de muons, ou sont payants.
2. **Une vidéo sans étiquettes n'apprend pas la bonne réponse** : personne ne sait quelle trace d'une vidéo
   YouTube est un électron et laquelle est un muon, et pour une trace fine et droite, même un expert ne peut
   pas le savoir. Les conditions (échelle, température, éclairage) sont aussi inconnues.
3. **Droits d'auteur** : télécharger massivement ces vidéos pour entraîner un modèle pose des problèmes de
   droits et contrevient aux conditions d'utilisation des plateformes.

La solution retenue est la **simulation fondée sur la physique** (NIST, PDG, NNDC, CERN) : la vraie classe de
chaque trace est connue par construction. Détails et liste des jeux de données examinés :
[`METHODE.md`](METHODE.md), section 10.

## 22. Raccourcis clavier

Ctrl+H : accueil · Ctrl+K : caméra · Ctrl+O : vidéo · Ctrl+Maj+O : photos · Ctrl+D : démo ·
Espace : pause · Ctrl+. : arrêter · R : enregistrer la vidéo · P : photo · E : expansion ·
Ctrl+I : Claude · F1 : aide · Échap : annuler un tracé · Ctrl+Q : quitter.
