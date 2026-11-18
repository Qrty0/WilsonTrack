# WilsonTrack : cadre théorique et méthodologie

*Détection, classification et analyse statistique des traces de particules dans une chambre de Wilson à diffusion*

Projet de physique, Cégep Marie-Victorin. Ce document décrit **ce que fait réellement le logiciel** (version **2.0**), étape par étape, et **pourquoi**. Il peut servir de base aux sections « Cadre théorique » et « Méthodologie » du rapport final. Les valeurs numériques indiquées sont les **réglages par défaut** du logiciel (fichier `wilsontrack/config.py`). Elles se modifient dans le panneau **Réglages**.

> **Nouveautés de la version 2.0 décrites ici** : classification **hybride** (règles physiques + IA de référence calibrée) avec **abstention sous 85 %** et nouvelle classe « **Trace fine (β ou μ)** » (section 5) ; **explication « Pourquoi ? »** de chaque trace (5.6) ; **simulateur réaliste** fondé sur les données du NIST, du PDG et du NNDC (section 8) ; **analyse approfondie** : taux par tranche horaire, stabilité, périodes mortes, comptages corrigés, score de qualité (7.8 à 7.11) ; **sessions de nuit** et **analyse par lot** (2.1) ; moteur environ **2,5 fois plus rapide** (3.13).

Les formules sont écrites en texte simple, lisibles dans n'importe quel visualiseur. Notations : `·` pour la multiplication, `√` pour la racine carrée, `ln` pour le logarithme naturel.

---

## Table des matières

1. [Physique de la formation des traces](#1-physique-de-la-formation-des-traces)
2. [Montage et acquisition des images](#2-montage-et-acquisition-des-images)
3. [Chaîne de traitement d'image](#3-chaîne-de-traitement-dimage)
4. [Mesures effectuées sur chaque trace](#4-mesures-effectuées-sur-chaque-trace)
5. [Classification des traces](#5-classification-des-traces)
6. [Mode chambre à expansion](#6-mode-chambre-à-expansion)
7. [Analyse statistique](#7-analyse-statistique)
8. [Le simulateur réaliste](#8-le-simulateur-réaliste)
9. [Validation de l'algorithme](#9-validation-de-lalgorithme)
10. [Pourquoi ne pas apprendre sur des « millions de vidéos » en ligne ?](#10-pourquoi-ne-pas-apprendre-sur-des--millions-de-vidéos--en-ligne-)
11. [Limites connues et précautions](#11-limites-connues-et-précautions)
12. [Bibliographie](#12-bibliographie)
13. [Annexe : réglages par défaut](#annexe--réglages-par-défaut)

---

## 1. Physique de la formation des traces

### 1.1 Sursaturation et condensation sur les ions

Une vapeur est **sursaturée** quand sa pression partielle `p` dépasse la pression de vapeur saturante `p_sat(T)` à la même température. Le rapport de sursaturation est :

```
S = p / p_sat(T)          (S > 1 : vapeur sursaturée)
```

Même si `S > 1`, une gouttelette ne peut croître que si son rayon dépasse un **rayon critique**. Une gouttelette plus petite s'évapore, car la tension de surface augmente la pression de vapeur au-dessus d'une surface très courbée. C'est l'équation de Kelvin :

```
r* = 2·γ·v / (k_B·T·ln S)
```

avec `γ` la tension de surface du liquide, `v` le volume d'une molécule dans le liquide, `k_B` la constante de Boltzmann et `T` la température absolue.

Sans germe, la condensation spontanée exige donc une très forte sursaturation. **Un ion abaisse cette barrière** : l'attraction électrostatique entre la charge et les molécules polaires de la vapeur stabilise les très petites gouttelettes (effet Thomson). C. T. R. Wilson a montré, pour l'eau, que les ions négatifs servent de germes vers `S ≈ 4`, les ions positifs vers `S ≈ 6`, alors que l'air sans ions ne condense qu'au-delà de `S ≈ 8`.

Il existe donc une plage de sursaturation où **seuls les ions deviennent des gouttelettes**. Une particule chargée qui traverse ce milieu laisse derrière elle une traînée d'ions. En quelques dizaines de millisecondes, ces ions grossissent en gouttelettes visibles (de l'ordre de 10 µm) qui diffusent la lumière. **La trace visible est donc la trajectoire de la particule, rendue visible par des milliers de gouttelettes.**

### 1.2 Chambre à diffusion et chambre à expansion

- **Chambre à diffusion** (notre montage, Langsdorf 1939). Un feutre imbibé d'isopropanol, en haut, est à la température ambiante. La plaque du fond est refroidie par de la glace sèche (−78,5 °C sous la plaque, environ −30 à −40 °C à sa surface). La vapeur d'alcool, plus dense que l'air, descend en se refroidissant. Juste au-dessus de la plaque, sur **quelques millimètres à environ 1 cm**, elle devient sursaturée : c'est la **couche sensible**. Cette chambre est **sensible en continu**. Chaque trace reste visible **0,5 à 3 s**, puis les gouttelettes tombent, dérivent ou s'évaporent.
- **Chambre à expansion** (Wilson 1911-1912). Une détente brusque et adiabatique du gaz le refroidit et rend la vapeur sursaturée dans tout le volume, mais seulement pendant **une fraction de seconde** après chaque expansion. WilsonTrack gère ce mode (section 6) avec le programme Arduino fourni dans `arduino/`.

### 1.3 Ionisation : la formule de Bethe

Une particule chargée perd son énergie surtout en ionisant les molécules du gaz. La perte d'énergie moyenne par unité de longueur, ou **pouvoir d'arrêt**, est donnée par la formule de Bethe :

```
−dE/dx = K · z² · (Z/A) · ρ · (1/β²) · [ ½·ln(2·mₑc²·β²γ²·T_max / I²) − β² − δ(βγ)/2 ]

K = 4π·N_A·rₑ²·mₑc² = 0,307 MeV·cm²/mol
```

- `z` : charge de la particule (en unités de `e`) ;
- `Z/A` : rapport numéro atomique / masse molaire du milieu (air : 0,499) ;
- `ρ` : masse volumique du milieu ;
- `β = v/c` et `γ = 1/√(1 − β²)` ;
- `I` : énergie moyenne d'excitation (air : 85,7 eV) ;
- `T_max` : énergie maximale transférable à un électron ;
- `δ` : correction d'effet de densité (négligeable ici).

Trois conséquences expliquent **l'aspect des traces** :

1. **Le facteur `z²`**. Une alpha (`z = 2`) ionise 4 fois plus qu'une particule de charge 1 à la même vitesse.
2. **Le facteur `1/β²`**. Une particule **lente** ionise beaucoup plus qu'une particule rapide. Une alpha de 5 MeV a `β ≈ 0,05`, donc `1/β² ≈ 400`. Un muon cosmique de 4 GeV a `β ≈ 1`.
3. **Le minimum d'ionisation**. Toutes les particules de charge 1 très énergétiques (`βγ ≈ 3` à `4`) perdent à peu près la même énergie, soit environ **1,8 MeV·cm²/g**. Dans l'air, cela donne environ **2,2 keV/cm à 15 °C** et **2,6 keV/cm à −25 °C** (l'air froid est plus dense). C'est le cas des **muons** et des **électrons de plus de ≈ 1 MeV**. On parle de **particules au minimum d'ionisation**.

Il faut en moyenne `W ≈ 34 eV` pour créer une paire d'ions dans l'air. On obtient donc les ordres de grandeur suivants :

| Particule | dE/dx dans l'air | Paires d'ions par cm | Aspect de la trace |
|---|---|---|---|
| Muon, électron rapide (≥ 1 MeV) | ≈ 2 à 3 keV/cm | ≈ 60 à 80 | **fine**, parfois en pointillés |
| Électron lent (≤ 100 keV) | 5 à 30 keV/cm, croissant en fin de parcours | quelques centaines | fine, **très sinueuse** |
| Alpha de 5 à 8 MeV | ≈ 1,3 MeV/cm en moyenne (pic de Bragg en fin de parcours) | ≈ 40 000 | **épaisse, brillante, continue** |

Pour une alpha, la densité d'ionisation est **de l'ordre de 500 fois** celle d'un muon. Les gouttelettes sont si serrées qu'elles forment un trait plein. Elles se disputent aussi la vapeur disponible et les ions diffusent latéralement avant de condenser : la trace est donc **plus large**. Une trace de muon ne compte que quelques dizaines de gouttelettes par centimètre : elle est **fine**, peu lumineuse, et peut paraître en pointillés.

> **Idée clé pour le classement** : l'**épaisseur** et la **densité lumineuse par unité de longueur** d'une trace mesurent son pouvoir ionisant (`dE/dx`). Elles distinguent les particules **lourdes et lentes** (alphas, protons) des particules **au minimum d'ionisation** (muons, électrons).

### 1.4 Diffusion multiple : pourquoi les électrons zigzaguent

En traversant le gaz, une particule chargée subit de nombreuses petites déviations sur les noyaux (diffusion coulombienne multiple). L'écart-type de l'angle de déviation projeté sur un plan, après une épaisseur `x`, est donné par la formule de Highland (PDG) :

```
θ₀ = (13,6 MeV / (β·c·p)) · z · √(x/X₀) · [1 + 0,038 · ln(x·z² / (X₀·β²))]
```

`p` est la quantité de mouvement et `X₀` la longueur de radiation du milieu. Pour l'air, `X₀ = 36,62 g/cm²`, soit environ **304 m à 20 °C** et **257 m à −25 °C**.

La déviation varie comme `1/(β·p)`. Elle est donc **énorme pour les électrons de faible énergie** (masse 0,511 MeV/c²) et **négligeable pour les muons** (masse 105,7 MeV/c², énergie de plusieurs GeV). Valeurs calculées pour l'air à −25 °C :

| Particule | p (MeV/c) | θ₀ après 1 cm | θ₀ après 10 cm | Portée dans l'air |
|---|---|---|---|---|
| e⁻ de 50 keV | 0,23 | ≈ 35° | direction perdue | ≈ 4 cm |
| e⁻ de 200 keV | 0,49 | ≈ 9° | ≈ 33° | ≈ 40 cm |
| e⁻ de 560 keV (moyenne du K-40) | 0,94 | ≈ 3,7° | ≈ 13° | ≈ 1,5 m |
| e⁻ de 1,31 MeV (maximum du K-40) | 1,75 | ≈ 1,8° | ≈ 6,5° | ≈ 4 à 5 m |
| μ de 4 GeV | 4100 | ≈ 0,0006° | ≈ 0,003° | plusieurs km |

(La formule de Highland n'est précise qu'à environ 11 % près et perd sa validité quand `θ₀` dépasse quelques dizaines de degrés. Les valeurs servent d'ordres de grandeur. Portées : ordre de grandeur de la portée CSDA des tables NIST ESTAR.)

Conséquences :

- les **électrons lents** (fin de parcours, électrons Compton de faible énergie) laissent des traces **courtes, sinueuses, en tire-bouchon** ;
- les **muons** laissent des traces **parfaitement rectilignes** ;
- les **électrons énergétiques** du K-40 (au-delà de ≈ 0,5 MeV) ne dévient que de quelques degrés sur quelques centimètres. Leur trace **ressemble alors à celle d'un muon**, un point important pour l'interprétation (section 9.3).

La formule de Highland est établie pour des épaisseurs bien plus grandes que celles d'une chambre (ici `x/X₀ ≈ 10⁻⁵` à `10⁻⁴`) : ces valeurs sont **indicatives à ± 30 à 50 %**. Elles suffisent pour la conclusion qualitative : sur 1 à 2 cm, un électron de plus de ≈ 1 MeV est presque aussi droit qu'un muon.

### 1.5 Particules alpha : relation portée-énergie

Une alpha perd son énergie de façon quasi continue et s'arrête après une **portée** `R` bien définie : toutes les alphas de même énergie ont presque la même longueur. Dans l'air, à 15 °C et 1 atm, la **règle de Geiger** relie la portée à l'énergie cinétique `E` (valable pour ≈ 4 à 8 MeV) :

```
R(15 °C, 1 atm) ≈ 0,318 · E^1,5        (R en cm, E en MeV)
```

La portée est inversement proportionnelle à la masse volumique de l'air. Or la couche sensible est **froide**, donc l'air y est plus dense et les alphas y sont **plus courtes**. WilsonTrack applique la correction de densité d'un gaz parfait :

```
facteur de densité  f = (288,15 / (T + 273,15)) · (P / 101,325)      (T en °C, P en kPa)
R(T, P) = R(15 °C, 1 atm) / f
E = (R_mesurée · f / 0,318)^(2/3)
```

À −25 °C (valeur par défaut, réglable), `f = 1,161`, donc les portées sont réduites de 14 %. Les alphas visibles dans une chambre viennent surtout du **radon-222 de l'air** et de ses descendants. Le thoron (Rn-220) et le Po-210 sont moins fréquents :

| Émetteur | E (MeV) | Portée à 15 °C | Portée à −25 °C |
|---|---|---|---|
| Po-210 | 5,30 | 3,9 cm | 3,4 cm |
| Rn-222 | 5,49 | 4,1 cm | 3,5 cm |
| Po-218 | 6,00 | 4,7 cm | 4,0 cm |
| Rn-220 | 6,29 | 5,0 cm | 4,3 cm |
| Po-216 | 6,78 | 5,6 cm | 4,8 cm |
| Po-214 | 7,69 | 6,8 cm | 5,8 cm |
| Po-212 | 8,79 | 8,3 cm | 7,1 cm |

Pour chaque alpha qui ne touche pas le bord de la zone analysée, le logiciel calcule l'énergie correspondant à la longueur mesurée (colonne `energie_MeV`). **Attention : c'est une énergie minimale.** Vue de dessus, on ne mesure que la **projection horizontale** de la partie de la trace située **dans la couche sensible** (≈ 8 mm d'épaisseur, alors qu'une alpha parcourt 35 à 60 mm) : une simulation Monte-Carlo (émission isotrope dans une couche de 8 mm) montre que seules 10 à 30 % des alphas détectées sont entièrement contenues, et que l'énergie **moyenne** ainsi calculée sous-estime l'énergie réelle d'environ 30 % (≈ 3,8 MeV pour le Rn-222 à 5,49 MeV). L'émetteur n'est donc **pas** identifié à partir de la moyenne, mais à partir du **bord supérieur** de la distribution : le 90ᵉ centile des énergies (au moins 10 alphas), qui correspond aux traces entièrement contenues et presque horizontales. La même simulation montre que ce 90ᵉ centile retrouve l'énergie vraie à quelques % près.

**Vérification avec les tables du NIST.** La base ASTAR du NIST donne les portées CSDA des alphas dans l'air ; ramenées à l'air froid par la loi des gaz parfaits (la portée en g/cm² ne dépend que de la masse d'air traversée), elles valent à −25 °C : 3,4 cm (Po-210), 3,5 cm (Rn-222), 4,1 cm (Po-218), 6,0 cm (Po-214) et 7,4 cm (Po-212, 8,785 MeV, l'alpha naturelle la plus énergétique). La règle de Geiger reproduit ASTAR à moins de 3 % entre 5,3 et 7,7 MeV et sous-estime d'environ 4 % la portée à 8,8 MeV. Le calcul d'énergie utilise toujours la règle de Geiger ; le **simulateur** (section 8), l'**explication** des traces (5.6) et la **longueur maximale d'une alpha** (80 mm ≈ 1,05 × la portée ASTAR du Po-212 à −20 °C environ) utilisent ASTAR.

### 1.6 Le potassium-40 : source des échantillons KCl et banane

Le potassium naturel contient **0,0117 %** de K-40 (demi-vie `T½ = 1,248 × 10⁹ ans`). Données de désintégration (NNDC/DDEP) :

| Mode | Rapport d'embranchement | Produit | Rayonnement émis |
|---|---|---|---|
| β⁻ | **89,3 %** | Ca-40 | électron d'énergie **maximale 1,311 MeV**, moyenne 0,560 MeV, et un antineutrino |
| Capture électronique | **10,7 %** | Ar-40 | photon γ de **1,461 MeV** (10,66 % des désintégrations) |

**Activité massique du potassium naturel** :

```
A_K = (N_A / M_K) · a · ln 2 / T½
    = (6,022×10²³ / 39,098 g/mol) · 1,17×10⁻⁴ · 0,6931 / (1,248×10⁹ × 3,156×10⁷ s)
    ≈ 31,7 Bq par gramme de potassium
```

**Chlorure de potassium** (KCl, M = 74,55 g/mol) : la fraction massique de K vaut 39,098 / 74,551 = 0,5245, donc :

```
A_KCl ≈ 0,5245 × 31,7 ≈ 16,6 Bq/g      dont 89,3 % de β⁻, soit ≈ 14,9 électrons par seconde et par gramme
```

Exemples calculés par le logiciel :

- **10 g de KCl** → 5,24 g de K → **≈ 166 Bq** (≈ 149 β⁻/s, ≈ 18 γ/s) ;
- **une banane** (≈ 150 g de chair, ≈ 3,6 g de K par kg) → 0,54 g de K → **≈ 17 Bq** (≈ 15 β⁻/s).

Seule une **petite fraction** de ces électrons produit une trace visible :

- la moitié part vers le bas ;
- les électrons sont absorbés dans l'échantillon lui-même : la portée d'un électron de 0,56 MeV est d'environ 0,2 g/cm², soit **≈ 1 mm de KCl**. Seule une couche superficielle contribue, d'où l'intérêt d'**étaler l'échantillon en couche mince** ;
- beaucoup traversent la couche sensible trop verticalement pour laisser une longue trace.

Le photon γ de 1,461 MeV n'interagit presque jamais dans l'air de la chambre. Quand il le fait, il produit un électron Compton. **Le K-40 n'émet aucune particule alpha** : le taux d'alphas ne doit donc pas changer avec l'échantillon. C'est un **contrôle interne** de l'expérience (section 7.3).

### 1.7 Muons cosmiques

Les rayons cosmiques primaires (surtout des protons) interagissent dans la haute atmosphère et produisent des pions, qui se désintègrent en **muons**. Au niveau de la mer (Particle Data Group) :

- le **flux** traversant une surface horizontale est d'environ **1 muon par cm² par minute** ;
- l'intensité verticale est d'environ **70 m⁻²·s⁻¹·sr⁻¹** ;
- la distribution angulaire suit environ `I(θ) ∝ cos²θ`, où `θ` est l'angle par rapport à la verticale ;
- l'énergie moyenne est d'environ **4 GeV**. Ce sont des particules au minimum d'ionisation, presque pas déviées : leurs traces sont **fines et parfaitement droites**.

**Pourquoi voit-on beaucoup moins de muons que 1 /cm²/min ?** La couche sensible est mince (épaisseur `h` de l'ordre de 5 à 15 mm). Un muon incliné d'un angle `θ` qui la traverse laisse une trace de longueur projetée :

```
ℓ = h · tan θ
```

Un muon presque vertical ne laisse donc qu'un **point** ou un **court trait**. Pour obtenir une trace longue (`ℓ ≥ L`), il faut `tan θ ≥ L/h`, donc un muon **presque horizontal**. En intégrant la loi `cos²θ` sur le flux à travers une surface horizontale (`dN ∝ cos²θ · cos θ · dΩ`), on obtient la fraction des muons qui laissent une trace de longueur au moins `L` :

```
fraction(ℓ ≥ L) = cos⁴θ_min = 1 / (1 + (L/h)²)²
```

| Épaisseur sensible h | L ≥ 30 mm | L ≥ 45 mm | Taux attendu (L ≥ 45 mm) sur 400 cm² |
|---|---|---|---|
| 5 mm | 7 × 10⁻⁴ | 1,5 × 10⁻⁴ | ≈ 0,06 /min |
| 8 mm | 4 × 10⁻³ | 9 × 10⁻⁴ | ≈ 0,4 /min |
| 10 mm | 1 × 10⁻² | 2 × 10⁻³ | ≈ 0,9 /min |
| 15 mm | 4 × 10⁻² | 1 × 10⁻² | ≈ 4 /min |

Il s'agit d'une **estimation d'ordre de grandeur**. La loi `cos²θ` sous-estime le flux près de l'horizontale, et les effets de bord sont négligés. Elle montre cependant que **le taux de longues traces droites dépend très fortement de l'épaisseur de la couche sensible**, donc de la qualité de la sursaturation. D'où l'importance de comparer des enregistrements faits **dans les mêmes conditions** (section 11).

### 1.8 Autres traces possibles

- **Électrons Compton** : les rayons γ ambiants (sol, béton, K-40 du corps humain) arrachent des électrons, qui laissent des traces fines et sinueuses. Ils forment une partie importante des « bêta » du bruit de fond.
- **Électrons delta** : un électron éjecté avec assez d'énergie par une particule rapide forme une petite branche sur la trace principale (colonne `embranchements`).
- **Protons** : traces épaisses comme celles des alphas, mais plus longues et plus fines. Elles sont rares (protons de recul, secondaires cosmiques). Un proton lent de 3 MeV ionise environ 56 fois plus qu'un muon (environ le septième d'une alpha) et parcourt ≈ 12 cm d'air froid (NIST PSTAR) ; au niveau de la mer, les protons cosmiques ne représentent qu'environ 1 % des muons (PDG).
- **Électrons cosmiques** : au niveau de la mer, il existe aussi des électrons et positons cosmiques énergétiques (intensité verticale ≈ 30 m⁻²·s⁻¹·sr⁻¹ au-delà de 10 MeV, en plein air, PDG), surtout issus de la désintégration des muons. Ils sont au minimum d'ionisation et presque pas déviés : **une longue trace fine et droite n'est donc pas toujours un muon**. À l'intérieur d'un bâtiment, leur proportion est mal connue.
- **Paires en V du thoron** : le Rn-220 (6,29 MeV) se désintègre en Po-216, qui émet une seconde alpha (6,78 MeV) après une demi-vie de 0,14 s : deux alphas partent presque du même point en moins d'une seconde (NNDC).
- **Coudes et Y** : une trace droite qui change brusquement de direction peut être un muon qui se désintègre en électron ; un Y, un rayon delta (CERN S'Cool LAB).
- **« Pluie » de fond et artefacts** : gouttelettes isolées qui condensent sur des poussières ou tombent du feutre, bouffées de brume, reflets près des parois. Ce sont surtout des **points** ou des **taches**, pas des lignes : l'algorithme est conçu pour les ignorer (section 3.5), et l'IA de référence sait reconnaître ceux qui passent malgré tout (classe « artefact », section 5.4).

### 1.9 Résumé des signatures

| Classe | Épaisseur / densité | Forme | Longueur typique | Origine principale |
|---|---|---|---|---|
| **Alpha (α)** | épaisse, très dense | droite | 2 à 7 cm | radon de l'air et descendants |
| **Bêta (β⁻)** | fine | sinueuse ou en zigzag, souvent courte | quelques mm à plusieurs cm | K-40 (échantillon), électrons Compton |
| **Muon (μ)** | fine | parfaitement droite | longue, traverse souvent le champ | rayons cosmiques |
| **Proton (p)** | épaisse | droite | plus longue qu'une alpha | rare |
| **Trace fine (β ou μ)** | fine | droite ou presque | variable | électron rapide **ou** muon : famille sûre, particule impossible à départager (section 5.4) |
| **Incertaine** | — | — | — | forme ambiguë : aucune classe n'atteint le seuil de certitude (85 % en mode hybride) |
| **Artefact** | tache peu allongée ou peu contrastée | — | — | brume, gouttelettes, reflet : **exclue des comptages** |

---

## 2. Montage et acquisition des images

- **Caméra au-dessus de la chambre**, objectif dirigé vers le bas, **fond noir** mat et **éclairage latéral** par des DEL, rasant à la couche sensible. Les gouttelettes diffusent la lumière vers la caméra ; le fond reste sombre.
- **Réglages de la caméra fixes** : exposition, gain, mise au point et balance des blancs en mode **manuel**. Un réglage automatique qui varie pendant l'enregistrement modifie le contraste des traces et fausse les comparaisons.
- **Calibration spatiale** (`px_per_mm`) : on filme une règle posée à la hauteur de la couche sensible et on indique la conversion pixels → millimètres. Sans calibration, le logiciel suppose une largeur de champ de 250 mm : les longueurs en mm et les énergies des alphas sont alors **approximatives**.
- **Zone d'intérêt** (ROI) : un polygone qui exclut les bords de la chambre, les reflets et les zones sans sursaturation. **La surface de la ROI sert à calculer les taux par cm².**
- **Température de la couche sensible** et **pression** : elles servent à la correction de densité des portées des alphas (section 1.5).

### 2.1 Sessions de longue durée et analyse par lot

La précision d'un taux ne dépend que du nombre de traces comptées (section 7.1). Pour mesurer un faible excès (une banane, par exemple), il faut donc filmer **longtemps**. La version 2.0 permet de laisser la chambre tourner toute une nuit (`wilsontrack/longrun.py`) :

- **arrêt automatique** après une durée choisie ou à une heure de fin (une heure déjà passée désigne le lendemain) ;
- **points de sauvegarde** toutes les N minutes (5 par défaut) : `run.json` (marqué `"partiel": true`), `traces.csv` et vignettes sont écrits dans un fil d'exécution séparé, sans interrompre l'analyse. Après une panne, au plus N minutes sont perdues. Ces résultats intermédiaires sont **provisoires** : les traces encore suivies ne sont pas comptées et les classes ne sont pas encore réévaluées ;
- **vidéo découpée en morceaux** (30 min par défaut), en AVI : un fichier AVI interrompu brutalement reste lisible, contrairement à un MP4. La vidéo peut être réduite (½ résolution, 10 img/s par défaut), complète ou absente ; l'**analyse** utilise toujours toutes les images de la caméra ;
- **photo de contrôle** (brute + annotée) toutes les N minutes et **journal** (`journal.txt`) avec une ligne d'état par minute ;
- **alertes** : chambre « morte » (aucune trace valide pendant 20 min d'analyse), baisse **statistiquement significative** du taux (test de Poisson, p < 0,001), disque presque plein (sous 500 Mo, la vidéo s'arrête mais les mesures continuent), caméra perdue (reconnexion automatique pendant 15 min ; la coupure **n'est pas comptée** dans la durée d'observation) ;
- l'ordinateur est **empêché de se mettre en veille** pendant la session ;
- **mémoire bornée** : seules les 300 vignettes les plus récentes restent en mémoire, les autres sont écrites sur le disque ; les détections rejetées ne sont plus conservées. La mémoire ne croît qu'avec le nombre de traces valides (≈ 15 Ko par trace), pas avec le nombre d'images.

L'**analyse par lot** (`wilsontrack/batch.py`) traite à la suite plusieurs vidéos reçues (clé USB, courriel, OneDrive…), avec un dossier de résultats et un rapport par vidéo, et un tableau récapitulatif `resume_lot.csv` (+ `.xlsx`). Un **dossier surveillé** est vérifié toutes les 5 s : une vidéo n'est analysée que lorsqu'elle est **complète** (taille stable depuis 10 s et fichier lisible), et une vidéo déjà traitée ne l'est jamais deux fois (fichier `.wilsontrack_traitees.json`). Les vidéos de plus de 20 min sont analysées avec la même gestion de mémoire que les sessions de nuit.

---

## 3. Chaîne de traitement d'image

Chaque image de la vidéo (ou de la caméra en direct) passe par les étapes ci-dessous (`wilsontrack/detection/`). L'objectif est double :

1. **ne détecter que de vraies traces**, pas le bruit du capteur, le brouillard, la « pluie » de gouttelettes ni les variations d'éclairage ;
2. **compter chaque trace physique une seule fois**, même si elle reste visible pendant des dizaines d'images.

```
image couleur
   │  1. niveaux de gris, mise à l'échelle (960 px), zone d'intérêt, stabilisation (option)
   │  2. moyenne temporelle glissante (2 images) + lissage gaussien (σ = 1 px)
   ▼
image de travail I
   │  3. soustraction du fond adaptatif B (mise à jour sélective)
   │  4. passe-haut + normalisation par le bruit → image « SNR »
   │  5. filtre de lignes hessien multi-échelle → image « graines »
   ▼
   │  6. seuillage par hystérésis (graines > 4,5 σ, extension > 2,2 σ)
   │  7. fermeture morphologique, composantes connexes
   │  8. fusion des fragments colinéaires
   ▼
détections de l'image
   │  9. rejet des images perturbées
   │ 10. suivi temporel (association, croissance, fantômes, absorption)
   │ 11. choix de l'image la plus nette de chaque trace
   │ 12. mesures (section 4), critères de validation, classification (section 5)
   ▼
une ligne par trace physique dans traces.csv
```

### 3.1 Niveaux de gris, mise à l'échelle et zone d'intérêt

L'image est convertie en niveaux de gris, puis réduite à une **largeur de traitement de 960 px** par moyenne de surfaces (`INTER_AREA`). Cette réduction moyenne aussi le bruit. Toutes les étapes suivantes se font à cette échelle. Les longueurs sont ensuite converties en mm avec la calibration de l'image native.

Un masque binaire limite l'analyse à la **zone d'intérêt**. Si la caméra vibre, une **stabilisation** optionnelle mesure le décalage global de chaque image par **corrélation de phase** (transformée de Fourier) et le compense.

### 3.2 Moyenne temporelle et lissage

Une trace vit environ 1 s, alors que le bruit du capteur change à chaque image. Une **moyenne glissante sur N = 2 images** réduit donc le bruit d'un facteur ≈ √2 sans atténuer les traces. Un **flou gaussien** de σ = 1 px élimine ensuite le bruit pixel par pixel.

### 3.3 Fond adaptatif à mise à jour sélective

Le fond `B` (brouillard, reflets, éclairage non uniforme, feutre visible) est estimé par une **moyenne mobile exponentielle** :

```
B ← (1 − α) · B + α · I          avec  α = 1 / (fps · τ),   τ = 3 s
```

- Pendant l'**apprentissage initial** (12 premières images), on prend `α = max(α, 1/n)`, ce qui revient à une moyenne simple. Aucune trace n'est détectée pendant cette phase.
- **Mise à jour sélective** : les pixels couverts par une trace détectée (masque dilaté de 5 × 5 px) n'apprennent qu'à **5 % de la vitesse normale**. Sans cette précaution, une trace qui dure 2 à 3 s « s'imprimerait » peu à peu dans le fond et perdrait son contraste. Le taux de 5 % n'est pas nul, ce qui évite qu'une zone reste « collée » indéfiniment.

On travaille ensuite sur la **différence** `D = I − B`, c'est-à-dire ce qui est apparu récemment.

### 3.4 Passe-haut et normalisation par le bruit

1. **Passe-haut** : on soustrait de `D` sa version très floue (gaussienne de σ = 12 px, calculée sur une image réduite 4 fois pour la rapidité). Il ne reste que les structures fines. Les nappes de brouillard qui bougent et les gradients d'éclairage disparaissent. Le fond flou est estimé sur `D` **écrêté** à `médiane + 4σ` : sans cet écrêtage, une trace très brillante (alpha) serait étalée par le flou et la soustraction créerait un **halo négatif** de ≈ 25 px autour d'elle, dont les bords sont vus par le filtre de lignes comme de **fausses lignes parallèles** (cause de doublons, identifiée pendant la validation).
2. **Estimation robuste du bruit** : l'écart-type du bruit `σ` est estimé par la **déviation absolue médiane** (MAD) sur environ 60 000 pixels de la ROI :

   ```
   σ = 1,4826 · médiane(|x − médiane(x)|)
   ```

   Le facteur 1,4826 rend la MAD égale à l'écart-type pour un bruit gaussien. Contrairement à l'écart-type ordinaire, la MAD **n'est pas influencée par les traces elles-mêmes**, qui n'occupent qu'une petite fraction des pixels.
3. **Image SNR** : `SNR = (passe-haut − médiane) / σ`. Chaque pixel est exprimé **en nombre d'écarts-types du bruit**. Les seuils restent ainsi valables quels que soient la caméra, le gain ou l'éclairage.

### 3.5 Filtre de lignes hessien multi-échelle (les « graines »)

Le problème principal est de distinguer une **trace** (structure **allongée**) d'une **gouttelette isolée** de la « pluie » (tache **ronde**), qui peut être tout aussi brillante. On utilise la **matrice hessienne** (dérivées secondes) de l'image lissée à l'échelle σ :

```
H = | ∂²I/∂x²   ∂²I/∂x∂y |       valeurs propres  λ₁ ≤ λ₂
    | ∂²I/∂x∂y  ∂²I/∂y²  |
```

| Structure | λ₁ | λ₂ | Réponse du filtre |
|---|---|---|---|
| Ligne brillante sur fond sombre | très négative (courbure en travers) | ≈ 0 (pas de courbure le long) | **forte** |
| Tache ronde (gouttelette) | négative | ≈ λ₁ | **≈ 0** |
| Fond uniforme ou bruit | ≈ 0 | ≈ 0 | ≈ 0 |

La réponse utilisée vaut `R = λ₂ − λ₁` si `λ₂ < 0`, et `R = −λ₁` sinon. Elle est **maximale pour une ligne et quasi nulle pour une tache ronde**. Ce principe est le même que celui du filtre de « vaisselité » de Frangi et al. (1998), utilisé en imagerie médicale pour les vaisseaux sanguins.

- La réponse est normalisée par `σ²` (normalisation d'échelle de Lindeberg), pour que les échelles soient comparables.
- Deux échelles, **σ = 1,5 px et 3 px**, couvrent les traces fines (muons, électrons) et les traces épaisses (alphas). L'échelle 3 px est calculée sur une image réduite de moitié, 4 fois plus vite.
- Chaque échelle est divisée par son propre **niveau de bruit**, soit le 84ᵉ centile de la réponse dans la ROI (≈ +1 σ), lissé d'une image à l'autre.
- L'image des **graines** est le maximum des deux échelles normalisées.

On forme aussi une **image d'extension** `E = max(SNR, graines)`. Elle permet de récupérer toute l'étendue de la trace : ses extrémités, les gouttelettes voisines et le cœur des alphas très épaisses, où la réponse hessienne est plus faible.

### 3.6 Seuillage par hystérésis

On applique le principe du détecteur de contours de Canny :

1. on garde les régions connexes (8-connexité) où `E > k_bas = 2,2 σ`, après une fermeture morphologique (section 3.7) qui relie les gouttelettes voisines ;
2. on ne conserve que celles qui contiennent **au moins un pixel de graine** `> k_haut = 4,5 σ`.

Un seul étiquetage des composantes suffit (fermeture puis étiquetage, puis sélection des composantes amorcées), ce qui divise le temps de calcul de cette étape par ≈ 1,4.

Une région ne peut donc **naître que d'une structure allongée nettement au-dessus du bruit**. Une fois amorcée, elle s'étend à tous les pixels voisins modérément contrastés. Le seuil haut élimine les fausses détections et le seuil bas conserve la trace entière, y compris ses parties pâles.

### 3.7 Fermeture morphologique et composantes connexes

Une **fermeture** morphologique (dilatation puis érosion, élément elliptique de 3 px) relie les gouttelettes d'une trace en pointillés. Les **composantes connexes** sont ensuite étiquetées. Celles de moins de 12 px sont ignorées.

Pour chaque composante, les **moments d'ordre 2** donnent le centre, l'orientation `θ` et les variances principales `λ₁ ≥ λ₂`. On en tire une longueur et une largeur rapides, exactes pour un segment uniforme :

```
L ≈ √(12 · λ₁)          W ≈ √(12 · λ₂)          allongement = L / W
```

### 3.8 Fusion des fragments colinéaires

Une trace de muon ou d'électron rapide, peu dense, peut apparaître comme **plusieurs segments alignés**. Deux composantes sont fusionnées (algorithme *union-find*) si les conditions suivantes sont toutes remplies :

- l'une d'elles sert d'**ancre** : elle est allongée (allongement ≥ 2,5, longueur ≥ 8 px) ;
- l'**écart le long de l'axe** entre les deux fragments est ≤ **8 mm** ;
- le **décalage latéral** par rapport à l'axe de l'ancre est ≤ `max(3 px ; 0,75 · (W_a + W_b))` ;
- si l'autre fragment est lui aussi allongé, les **orientations** diffèrent de moins de **14°** ;
- l'autre fragment n'est pas une grosse tache ronde.

### 3.9 Rejet des images perturbées

Plus de **25 % de la ROI** peut s'allumer d'un coup (main au-dessus de la chambre, ajout de glace sèche, caméra bougée si ce contrôle est activé), ou la **luminosité globale** peut sauter de plus de **6 niveaux de gris** par rapport au fond (lumière de la salle allumée : ce décalage uniforme est invisible après le passe-haut, il est donc mesuré sur le contraste brut). L'image est alors déclarée **perturbée**. Les traces en cours sont closes, le fond est réinitialisé, et l'image **n'est pas comptée dans la durée d'observation**.

> **Durée effective (horloge d'analyse).** La durée utilisée pour calculer les taux est la **somme des intervalles de temps réels** entre images effectivement analysées. Elle **exclut** l'apprentissage initial du fond, les images perturbées et les pauses. Pour une caméra, les intervalles viennent des **horodatages réels** et non de la cadence annoncée : une webcam qui annonce 30 img/s mais n'en livre que 10 dans une pièce sombre ne fausse donc pas les taux (sinon ils seraient 3 fois trop élevés). Les seuils temporels (visibilité minimale, période de croissance, fantômes) utilisent aussi la cadence réelle estimée. Chaque trace reçoit son instant d'apparition sur cette horloge (colonne `t_analyse_s`), utilisé par les tests statistiques.

### 3.10 Suivi temporel : compter chaque trace une seule fois

Une trace reste visible pendant 15 à 90 images. Sans suivi, elle serait comptée à chaque image. Le suivi associe les détections d'une image aux **traces actives** :

1. **Association par recouvrement.** Le masque de chaque trace active est dilaté de 5 px. Chaque détection est associée à la trace avec laquelle elle partage le plus de pixels, de façon gloutonne, en commençant par les plus grands recouvrements.
2. **Période de croissance (≈ 0,3 s).** Une trace qui vient de naître est encore en formation : ses gouttelettes grossissent et de nouveaux segments apparaissent le long de la trajectoire. Pendant cette période, **tout ce qui lui est connexe en fait partie**.
3. **Absorption des fragments.** Une trace peut naître en plusieurs morceaux qui se rejoignent ensuite dans une même détection. Si ces morceaux sont tous « jeunes » (période de croissance), ils sont **fusionnés en une seule trace** au lieu d'être comptés séparément. De même, les détections nouvelles et voisines apparues **à la même image** (séparées de quelques pixels seulement, au plus 10 px à l'échelle de traitement) sont regroupées.
4. **Croisements.** Après la période de croissance, une nouvelle partie peut apparaître dans une détection déjà suivie. Si elle ne prolonge **pas** la trace (angle de plus de 20°, ou décalage latéral de plus de `max(5 px ; 2 × largeur)`) et qu'elle est assez longue, elle est traitée comme une **nouvelle trace** qui croise la première. Quand deux traces déjà suivies se touchent et forment une seule composante, les pixels sont **partagés** selon la carte des traces : chacune continue d'être mise à jour avec sa propre partie (aucune n'est « affamée » puis perdue).
5. **Nouvelles traces.** Une détection non associée crée une nouvelle trace, sauf dans deux cas : elle est couverte à plus de 25 % par une trace existante, ou à plus de 40 % par un **fantôme** (point 6).
6. **Fin d'une trace et fantômes.** Une trace est close après **5 images consécutives sans détection**. Son masque (élargi de 8 px pour tenir compte de la dérive et de la diffusion) reste ensuite en mémoire pendant **1,2 s** sous forme de **fantôme**. Une trace qui s'estompe, se fragmente, disparaît sous le seuil puis réapparaît n'est donc **pas recomptée**. L'association utilise aussi le **masque complet** de la trace (celui du meilleur instant) : si une partie d'une trace pâle disparaît une image puis réapparaît, elle reste rattachée à sa trace.
7. **Durée minimale.** Une vraie trace doit être vue sur au moins `max(3 images ; 0,15 s)`. Les éclairs d'une seule image (bruit, reflet) sont rejetés.
8. **Doublons et fragments.** Si une même composante recouvre plusieurs traces suivies : deux événements dont les masques se recouvrent à plus de 30 % suivent le même objet et sont **fusionnés** ; deux morceaux allongés **alignés bout à bout** (orientation à ±20°, décalage latéral ≤ 1,2 largeur) sont les fragments d'une même trace et sont fusionnés ; sinon il s'agit d'un **croisement** et chaque pixel est attribué à la trace la plus proche (propagation des étiquettes par transformée de distance).
9. **Naissance près d'une gouttelette.** Si une trace apparaît au contact d'une gouttelette déjà suivie (événement jamais allongé), ou si une trace **trois fois plus contrastée** apparaît sur une structure pâle préexistante, un **nouvel événement** commence : la trace n'hérite ni de l'instant d'apparition ni de l'historique de son voisin.

### 3.11 Mesure à l'instant le plus net

Une trace change pendant sa vie : elle se forme, atteint sa pleine longueur, puis ses gouttelettes **diffusent**. Elle devient plus large, plus floue et moins contrastée. Pour mesurer l'épaisseur de façon fiable, il faut choisir **le bon instant**. Pour chaque image, le logiciel calcule un score de qualité :

```
score = longueur rapide × contraste moyen (SNR) / (1 + âge / 0,5 s)
```

Ce score est maximal quand la trace est **complète ET encore nette**, peu après sa formation (0,1 à 0,3 s) ; le facteur d'âge empêche de retenir une image tardive où la trace, diffusée, paraît plus longue et plus large. De plus, après la période de croissance, les dimensions de la trace sont **figées** : une détection plus de 25 % plus longue (ou 80 % plus large) n'est plus retenue, car c'est presque toujours la trace fusionnée avec une structure voisine. L'image de meilleur score est conservée : masque, contraste trace − fond et vignette couleur en pleine résolution. **Toutes les mesures de la section 4 sont faites sur cette image.** La colonne `t_meilleure_s` indique cet instant.

### 3.12 Critères de validation

Une trace suivie n'est comptée que si elle satisfait **tous** les critères suivants (sinon elle est rejetée et le motif est noté) :

| Critère | Seuil | Rejette |
|---|---|---|
| longueur | ≥ 4 mm | points, gouttelettes |
| allongement longueur / largeur | ≥ 2,2 | taches rondes |
| contraste de crête (SNR) | ≥ 6 σ | bruit, brouillard |
| tache floue courte | rejet si allongement < 4 **et** longueur < 10 mm **et** SNR < 12 | bouffées de brume, amas de gouttelettes |

À la fin de l'enregistrement, toutes les traces valides sont **reclassées** avec la référence définitive des traces fines (section 5.1).

En mode hybride, l'IA de référence peut encore déclarer une trace validée « **artefact** » (section 5.4) : elle est alors retirée des comptages, avec le motif « artefact probable », comme une détection rejetée.

### 3.13 Vitesse de calcul et mémoire

Le moteur de la version 2.0 donne **exactement les mêmes résultats** que celui de la version 1.1 (vérifié bit à bit sur toutes les traces, valides et rejetées, de trois vidéos de test), mais il est environ **2,5 fois plus rapide** : réutilisation des zones mémoire, calcul de l'image d'évidence seulement autour des détections, répliques exactes et plus rapides des médianes et centiles de NumPy, mesures différées jusqu'à l'affichage ou la fin de la trace. Sur un ordinateur chargé, une image de 1280 × 720 est traitée en ≈ 22 ms, soit **≈ 45 images/s** (au lieu de ≈ 18 images/s) : l'analyse en direct à 30 images/s est assurée. Pour les longues séances, le nombre de détections rejetées gardées en mémoire est plafonné et les vignettes anciennes sont écrites sur le disque (section 2.1).

---

## 4. Mesures effectuées sur chaque trace

Les mesures portent sur le **masque** de la trace et sur l'image de **contraste** `D = trace − fond` (`wilsontrack/detection/features.py`). Les longueurs et largeurs sont mesurées sur le **profil d'intensité**, et non sur le masque binaire. Elles ne dépendent donc presque pas du seuil de détection. Le contraste utilisé pour les mesures n'est **pas** filtré passe-haut (le filtre, utile pour la détection, rétrécirait les traces épaisses de 15 à 20 %) : on retire plutôt une **ligne de base locale**, la médiane des pixels autour de la trace (brouillard, éclairage).

### 4.1 Chemin central de la trace

0. **Cœur brillant et pontage des pointillés** : le chemin central est cherché sur la partie **brillante** du masque (pixels ≥ 30 % du 90ᵉ centile d'intensité de la trace), afin que de faibles appendices accolés (gouttelettes, brume) ne détournent pas le squelette. Une trace en pointillés (chapelet de gouttelettes, fragments fusionnés en 3.8) est formée de plusieurs morceaux : ils sont ordonnés le long de l'axe principal et reliés par des segments entre extrémités voisines ; sans cela, le squelette ne décrirait qu'un seul fragment.
1. **Squelette** : le masque est aminci à une ligne d'un pixel (algorithme de Zhang-Suen, `skimage`).
2. **Plus long chemin géodésique** : le squelette peut comporter de petites branches. On en extrait le plus long chemin par un **double algorithme de Dijkstra** (voisinage 8, pas de 1 ou √2 px). On part d'un point quelconque, on trouve le point le plus éloigné `a`, puis le point le plus éloigné de `a`.
3. **Lissage sous-pixel** : le chemin est rééchantillonné tous les 1 px, puis lissé par un filtre gaussien (σ = `max(1,5 ; 0,35 × largeur du masque)` px). On obtient une ligne centrale lisse, en coordonnées **sous-pixel**.

### 4.2 Longueur (profil à mi-hauteur)

Le chemin lissé est **prolongé** à ses deux extrémités selon la tangente locale, de `1,5 × largeur + 4` px (entre 5 et 25 px). On échantillonne ensuite le contraste le long de ce chemin prolongé (interpolation bilinéaire, profil lissé par σ = 2 px) :

```
référence  = 70ᵉ centile du profil sur le cœur de la trace   (robuste aux portions pâles)
extrémités = points où le profil croise 50 % de la référence   (interpolation linéaire sous-pixel)
longueur   = distance curviligne entre les deux extrémités
```

C'est une **mesure à mi-hauteur**, comme une largeur à mi-hauteur (FWHM) mais dans le sens de la trace. Elle est indépendante du seuil de détection et de la luminosité. Sur les vidéos de validation, elle reproduit les longueurs vraies à ± 2 % près (section 9). Colonne `longueur_mm`.

### 4.3 Épaisseur (FWHM du profil transversal moyen)

À environ 24 positions régulièrement réparties le long du chemin, on échantillonne le contraste **perpendiculairement** à la trace, par pas de 0,5 px. Ces profils transversaux sont **moyennés** ; la ligne de base (médiane des extrémités du profil) est retirée, puis on mesure la **largeur à mi-hauteur** (FWHM) du profil moyen, avec interpolation linéaire. Colonne `epaisseur_mm`.

Si le profil est inutilisable (trace très courte), on utilise une largeur effective : `signal total / (longueur × intensité de crête)`.

### 4.4 Forme de la trajectoire

| Mesure | Définition | Interprétation | Colonne |
|---|---|---|---|
| **Rectitude** | corde (distance entre extrémités) / longueur du chemin | 1 = parfaitement droite ; < 0,86 = sinueuse | `rectitude` |
| **Flèche relative** | écart maximal au segment qui joint les extrémités / longueur | courbure globale (arc, crochet) | (interne) |
| **Écart RMS à la droite** | écart quadratique moyen des points à la meilleure droite (plus petite valeur singulière / √N), divisé par la longueur | zigzag fin, même si la trace est globalement droite | (interne) |
| **Tortuosité** | somme des changements de direction \|Δθ\| le long du chemin (pas de `max(4 px ; 2 mm ; 1,5 × largeur)`) / longueur | rad/mm ; élevée pour un électron en tire-bouchon | `tortuosite_rad_mm` |
| **Cohérence de courbure** | \|Σ Δθ\| / Σ \|Δθ\| | 1 = tourne toujours dans le même sens (arc de cercle) ; ≈ 0 = zigzag | (interne) |
| **Étalement latéral excédentaire** | `√max(0 ; λ_perp − (FWHM/2,355)²)` : variance perpendiculaire des pixels (ACP pondérée par l'intensité) moins celle due à l'épaisseur | ≈ 0 pour une trace droite, ≈ 0,3 × flèche pour un arc ; **indépendant du squelette**, donc fiable pour les traces courtes (< 2 cm) | (interne) |
| **Embranchements** | nombre de jonctions du squelette | électrons delta, croisements | `embranchements` |
| **Orientation** | angle de la corde, de 0 à 180° (0 = horizontale) | test d'isotropie des muons (section 7.6) | `orientation_deg` |

### 4.5 Courbure et quantité de mouvement (si un aimant est utilisé)

Un **cercle** est ajusté aux points du chemin lissé par la **méthode algébrique de Taubin** (1991), dans la version stable par décomposition en valeurs singulières de N. Chernov. Cette méthode est non itérative, peu biaisée et robuste pour des arcs courts. Le rayon `R` obtenu est dans la colonne `rayon_courbure_mm`.

Dans un champ magnétique `B` perpendiculaire à la trajectoire, une particule de charge `|q| = z·e` décrit un cercle de rayon :

```
p [MeV/c] = 0,2998 · z · B [T] · R [mm]
```

Exemple : un électron dans un champ de 0,1 T qui décrit un arc de 50 mm de rayon a `p ≈ 1,5 MeV/c`. L'énergie cinétique vaut ensuite :

```
T = √(p² + m²) − m      (m = masse de la particule, en MeV/c²)
```

Le logiciel n'utilise la courbure que si :

- un champ `B > 0` est indiqué dans les paramètres ;
- l'arc est **régulier** : cohérence de courbure > 0,6 et flèche relative > 0,02 ;
- le rayon est inférieur à 5 m.

Un zigzag de diffusion multiple ne donne donc pas une fausse quantité de mouvement. Attention : on mesure la **projection** de la trajectoire. Si la particule n'est pas perpendiculaire à `B`, la valeur obtenue est une limite inférieure de `p`.

### 4.6 Densité d'ionisation

```
densité  = (somme du contraste positif dans le masque dilaté) / longueur / σ_bruit
SNR crête = 60ᵉ centile du profil longitudinal / σ_bruit
```

La densité est une **luminosité par unité de longueur**. Elle est proportionnelle au nombre de gouttelettes par mm, donc au **pouvoir ionisant** `dE/dx` (section 1.3). Sa valeur absolue dépend de l'éclairage, de la caméra et de la sursaturation. C'est pourquoi le classement utilise une **densité relative** (`densite_rel`) par rapport aux traces fines du même enregistrement (section 5.1).

On mesure aussi l'**asymétrie** du profil longitudinal : le rapport entre les intensités moyennes du premier et du dernier tiers de la trace. Elle signale un pic de Bragg (alpha plus dense en fin de parcours) ou un électron qui ralentit.

---

## 5. Classification des traces

Trois méthodes existent (*Réglages › Classification › Méthode* ; `wilsontrack/detection/classifier.py` et `reference_model.py`) :

- **« Hybride : règles + IA de référence »** (par défaut depuis la version 2.0) : les règles physiques et une IA entraînée sur des traces simulées réalistes votent ensemble ; une particule n'est **nommée que si la probabilité calibrée atteint 85 %** (section 5.4) ;
- **« Règles physiques seules »** : les règles explicables décrites en 5.1 à 5.3 ;
- **« IA locale (vos corrections) »** : un modèle appris à partir des traces que vous avez corrigées (section 5.8).

Quelle que soit la méthode, les **règles physiques sont toujours calculées** : ce sont elles qui fournissent les mesures relatives et l'**explication « Pourquoi ? »** de chaque trace (section 5.6).

### 5.1 Référence automatique des « traces fines »

L'épaisseur et la densité absolues varient d'une séance à l'autre (éclairage, mise au point, sursaturation). Le logiciel utilise donc des **grandeurs relatives**, mesurées par rapport à une **référence estimée automatiquement** sur les traces du même enregistrement :

```
épaisseur_rel = épaisseur / épaisseur_réf          densité_rel = densité / densité_réf
```

Estimation de la référence :

1. S'il y a **au moins 6 traces** valides, on prend le **35ᵉ centile** de ln(densité) et de l'épaisseur. Les traces fines sont majoritaires.
2. S'il y a **au moins 20 traces**, on ajuste un **mélange de deux gaussiennes** (algorithme EM) aux points (ln densité, ln épaisseur). Si la distribution est nettement **bimodale** (voir les conditions ci-dessous), la référence est le centre de la **composante basse**, celle des particules au minimum d'ionisation.
3. La référence est mise à jour toutes les 10 nouvelles traces. **À la fin de l'enregistrement, toutes les traces sont reclassées** avec la référence définitive.

Conditions de bimodalité :

- écart entre les moyennes des deux composantes > 0,8 en ln, soit un facteur supérieur à 2,2 ;
- écart > 1,2 fois la somme des écarts-types ;
- poids de la composante basse > 10 %.

Sans référence (moins de 6 traces), l'épaisseur relative est estimée à partir d'une largeur typique de trace fine de 0,6 mm.

### 5.2 Règles physiques « floues »

Chaque critère est une **rampe** entre 0 et 1, plutôt qu'un seuil brutal. Une trace à la limite obtient ainsi un score intermédiaire au lieu de basculer arbitrairement :

```
rampe(x, a, b) = 0 si x ≤ a ;  1 si x ≥ b ;  (x − a)/(b − a) entre les deux
```

**Critères** (valeurs par défaut ; `L` = longueur en mm, `S` = rectitude, `T` = tortuosité en rad/mm) :

```
épaisse   = max( rampe(épaisseur_rel, 1,36 ; 2,04) ,  rampe(épaisseur_mm, 0,98 ; 1,50) si calibré )
dense     = rampe(densité_rel, 1,65 ; 2,86)
lourde    = 0,35 · épaisse + 0,65 · dense            légère = 1 − lourde

pondération longue/courte :  h = rampe(L, 12 ; 25 mm)       (lat = étalement latéral excédentaire, mm)

droite_μ  = h · [rampe(S, 0,91 ; 0,96) · (1 − rampe(flèche, 0,035 ; 0,07)) · (1 − rampe(écart_RMS/L, 0,012 ; 0,022))]
          + (1 − h) · [1 − rampe(lat, 0,20 ; 0,45)]
          × [1 − h · rampe(écart_RMS, 0,2 ; 0,32 mm)]          (critère d'écart RMS, nouveau en 2.0)
droite_α  = rampe(S, 0,80 ; 0,90)     (traces courtes : ou 1 − rampe(lat, 0,3 ; 0,6))
longue    = rampe(L, 29 ; 45)
courte_α  = 1 − rampe(L, 80 ; 104)
sinueuse  = h · max( 1 − rampe(S, 0,86 ; 0,95) , rampe(flèche, 0,05 ; 0,10) ,
                     rampe(écart_RMS/L, 0,015 ; 0,03) , rampe(T, 0,10 ; 0,20) )
          + (1 − h) · rampe(lat, 0,30 ; 0,60)
```

**Deux réglages ont changé en 2.0, d'après les tables du NIST et les traces simulées réalistes** : la longueur maximale d'une alpha passe de 75 à **80 mm** (≈ 1,05 × la portée ASTAR du Po-212 dans l'air froid : 78 mm à −25 °C, 86 mm à 0 °C) ; un **écart RMS à la droite** supérieur à 0,2 mm (sur une trace assez longue) exclut le muon. La diffusion multiple écarte un électron du K-40 de 0,5 à 7 mm de la droite sur 2 à 5 cm, un muon de 4 GeV d'environ 0,001 mm seulement (Highland). La longueur minimale d'un muon reste 45 mm : 30 mm n'a pas amélioré les résultats sur les traces simulées réalistes.

Pour une trace courte (< ≈ 2 cm), le squelette dévie aux extrémités d'environ une demi-largeur, ce qui ferait paraître courbée une trace parfaitement droite (flèche relative de 0,1 à 0,3 mesurée sur des muons droits de 5 à 10 mm) : la forme est alors jugée sur l'**étalement latéral**, indépendant du squelette. Sur les vidéos de validation, cela fait passer les muons courts bien classés de 50 % à ≈ 80 %.

**Scores des classes** :

```
alpha  = lourde · droite_α · courte_α
proton = lourde · droite_α · (1 − courte_α)
muon   = légère · droite_μ · (0,3 + 0,7 · longue)
bêta   = légère · max( sinueuse , 0,45 · (1 − longue) · (1 − droite_μ) )
```

La densité pèse plus que l'épaisseur (0,65 contre 0,35), car elle est moins sensible à la mise au point. Les critères de rectitude du muon sont **plus stricts** que ceux de l'alpha : un muon de 4 GeV n'est pas dévié du tout, alors qu'une alpha peut présenter un léger crochet en fin de parcours (diffusion nucléaire).

**Traces fines, courtes et parfaitement droites.** Dans une couche sensible de ≈ 8 mm, la longueur projetée d'un muon vaut `h · tan θ` : avec un flux ∝ cos³θ·sinθ à travers une surface horizontale, environ 64 % des muons laissent une trace ≥ 4 mm, mais seulement 0,4 % dépassent 30 mm. **La plupart des muons visibles sont donc courts.** Une trace fine, courte et droite est classée « muon » (confiance plafonnée à 0,45 et mention « muon raide ou électron énergétique » dans la raison), car un électron énergétique du K-40 peut aussi paraître droit sur quelques mm. Le plancher « court → bêta » ne s'applique qu'aux traces qui ne sont pas nettement droites. Cette ambiguïté est la raison pour laquelle la comparaison utilise aussi le groupe des **traces fines β + μ** (section 7.3).

### 5.3 Confiance et classe « incertaine » (méthode « règles seules »)

Soit `b` le meilleur score et `s₂` le deuxième :

```
confiance = min(1 ; 2 · b² / (b + s₂))
```

La confiance est élevée si le meilleur score est **fort** et **nettement supérieur** au deuxième. Si elle est **inférieure à 0,30**, la trace est classée « **incertaine** » : elle est comptée dans le total, mais pas attribuée à une catégorie. La colonne `raison` résume les mesures qui ont conduit à la décision, par exemple : `L=38.2 mm, épaisseur=1.45 mm (×2.3 réf.), densité ×3.1, rectitude=0.982, tortuosité=0.012 rad/mm → lourde, droite, courte`.

Cette confiance des règles est un **score flou**, pas une probabilité : 0,6 ne veut pas dire « juste 6 fois sur 10 ». C'est l'une des raisons de la méthode hybride, dont les probabilités sont **calibrées** (section 5.4).

### 5.4 Méthode hybride : règles + IA de référence calibrée, avec abstention

**L'IA de référence.** C'est un modèle d'apprentissage automatique de type *gradient boosting* (arbres de décision construits les uns après les autres, chacun corrigeant les erreurs des précédents ; `HistGradientBoostingClassifier` de scikit-learn), livré avec l'application (`wilsontrack/assets/`). Il reçoit **27 mesures** de la trace : les 15 mesures de la section 5.8, plus la durée de visibilité, le contact avec le bord, la présence d'une calibration, le rapport corde/longueur, le taux de remplissage du masque (trace continue ou pointillée), la somme des changements de direction, l'étalement latéral et l'écart RMS relatifs à la longueur, le rayon de courbure relatif, l'écart à l'arc de cercle, le rapport des épaisseurs (masque / profil) et la longueur exprimée en « largeurs de trace fine ». Il prédit **5 classes** : alpha, électron (β), muon, proton et **artefact** (brume, gouttelettes, reflet : ce n'est pas une particule).

**Comment il a appris.** Il n'a **jamais vu de vraie chambre** : il a été entraîné sur des **centaines d'enregistrements simulés** par le simulateur réaliste (section 8), dont les conditions (résolution, bruit, éclairage, épaisseur de la couche, taux de chaque particule…) sont tirées au hasard. Chaque vidéo simulée passe dans le **vrai détecteur** de l'application ; chaque détection reçoit l'étiquette de la trace simulée qu'elle recouvre (ou « artefact » si elle ne correspond à aucune particule). L'IA apprend donc à partir des **mesures telles que le logiciel les fait réellement**, avec leurs imperfections. Les enregistrements sont répartis en trois groupes **séparés** (entraînement, validation, test) : une trace d'un même enregistrement n'apparaît jamais dans deux groupes, et les chiffres de la section 9.2 sont mesurés sur des enregistrements que l'IA n'a jamais vus, plus un jeu « conditions extrêmes ».

**Combinaison avec les règles.** Les scores des règles sont convertis en probabilités, puis combinés avec celles de l'IA :

```
p(c)        = (1 − w) · (1 − p_IA(artefact)) · p_règles(c) + w · p_IA(c)      pour c = alpha, bêta, muon, proton
p(artefact) = p_IA(artefact)
```

Le poids `w` de l'IA est choisi sur le groupe de validation et enregistré dans le modèle (*Réglages* : poids −1 = valeur optimale enregistrée).

**Calibration des probabilités.** Une IA peut être « trop sûre d'elle ». Deux corrections rendent ses probabilités **honnêtes** :

1. **correction des proportions** (*prior shift*) : les proportions de classes de l'entraînement sont ramenées à des proportions réalistes (par exemple, les protons ne représentent qu'environ 1 % des particules, PDG) ;
2. **mise à l'échelle par température** : un seul paramètre `T`, ajusté sur la validation, « adoucit » les probabilités (`p ∝ p_brute^(1/T)`).

Une probabilité est **bien calibrée** si, parmi toutes les traces auxquelles le modèle donne 90 %, environ 90 % sont réellement de cette classe. On le mesure par l'**erreur de calibration attendue** (ECE, moyenne des écarts entre probabilité annoncée et fréquence observée) et par un diagramme de fiabilité (section 9.2).

**Abstention sous 85 %.** Une particule n'est **nommée que si sa probabilité calibrée est d'au moins 0,85** (`ml_threshold`). Sinon, la trace est déclarée « **incertaine** », avec ses **deux classes possibles** et leurs probabilités (par exemple « incertaine : muon 62 % ou électron 35 % ; trace fine sûre à 99 % »). De plus, une classe n'est jamais affirmée si sa précision mesurée sur la validation n'est pas solidement au-dessus de 85 % (borne inférieure de l'intervalle de Wilson à 90 % ≥ 0,85) ; le seuil est aussi relevé à 0,90 pour une trace fine, courte et droite (cas ambigu de la section 5.2). C'est un choix délibéré : **mieux vaut dire « je ne sais pas » que se tromper**. Une trace incertaine est **comptée dans le total** et dans sa **famille** ; elle n'est simplement pas attribuée à une particule.

**Artefacts.** Une détection dont la classe la plus probable est « artefact » (au-dessus du seuil) est **exclue des comptages de particules**, mais reste visible dans le tableau des traces (filtre et motif de rejet).

### 5.5 La classe « Trace fine (β ou μ) »

La première question est celle de la **famille** : trace **fine** (particule au minimum d'ionisation : électron rapide ou muon) ou **lourde** (alpha, proton). Elle est tranchée par `p(β) + p(μ)` contre `p(α) + p(p)` et elle est très fiable, car l'épaisseur et la densité de gouttelettes mesurent directement `dE/dx` (section 1.3).

La seconde question, **électron ou muon ?**, est souvent **physiquement impossible** à trancher :

- les deux sont au **minimum d'ionisation** : même densité de gouttelettes, même épaisseur ;
- un électron de plus de ≈ 1 MeV ne dévie que de quelques degrés sur 1 à 2 cm (Highland, PDG ; section 1.4) : sur la longueur visible dans une couche sensible d'environ 1 cm, il est **presque droit, comme un muon** ;
- les **électrons cosmiques** énergétiques (section 1.8) ont exactement l'aspect d'un muon : sur les traces simulées, l'IA leur donne en moyenne la même probabilité « muon » qu'aux vrais muons.

Quand l'IA hésite **seulement entre électron et muon** et que la famille « fine » est sûre (au moins 85 %), la trace reçoit donc la classe **« Trace fine (β ou μ) »** (symbole `f`, couleur bleu-vert) au lieu de « incertaine ». C'est une réponse **juste mais moins précise**. Conséquences :

- les **muons ne sont presque jamais nommés** : la plupart des traces de muons apparaissent comme « traces fines » ; ce n'est pas un défaut du logiciel, c'est la physique (sans aimant, on ne peut pas distinguer un muon d'un électron cosmique énergétique à cette échelle) ;
- les électrons **lents** (sinueux, en tire-bouchon) restent bien reconnus comme « bêta » ;
- pour la question du projet (le KCl est-il radioactif ?), on n'a pas besoin de séparer électrons et muons : la comparaison utilise le **groupe des traces fines** (bêta + muon + trace fine + incertaines de famille fine ; section 7.3), qui ne dépend pas de cette ambiguïté.

### 5.6 L'explication « Pourquoi ? » de chaque trace

Chaque trace reçoit une explication rédigée (`wilsontrack/detection/explain.py`), visible dans l'onglet **Traces**, résumée dans la colonne `pourquoi` de `traces.csv` et illustrée par des exemples dans le rapport HTML (section « Pourquoi ces types ? »). Elle comprend :

1. **le verdict** : classe, confiance et méthode (règles, hybride, IA locale, ou correction manuelle) ;
2. **ce qui a été mesuré**, critère par critère (épaisseur, densité, forme, longueur, avis de l'IA), avec la valeur mesurée, la valeur attendue et un symbole ✔ (pour), ✘ (contre) ou • (neutre). Les indices qui **n'ont pas servi** à la décision (par exemple la durée de visibilité en mode « règles seules ») sont marqués « indice complémentaire (non utilisé pour décider) » ;
3. **les probabilités** de chaque classe ;
4. **pourquoi pas les autres ?** : une phrase chiffrée par classe écartée, qui nomme le critère décisif (par exemple « Pas une alpha : trace 1,8× trop fine… ; 165 mm de long alors qu'une alpha naturelle parcourt au plus ≈ 74 mm dans l'air à −25 °C (Po-212, NIST ASTAR) ») ;
5. **la physique** : formule de Bethe, portées ASTAR à la température réglée, diffusion multiple de Highland, géométrie des muons, K-40 ;
6. **les limites** : échelle non calibrée, trace qui touche le bord, projection 2D, incertitude sur la température de la couche, désaccord règles/IA, IA validée sur simulation seulement ;
7. **les sources** (NIST, PDG, NNDC, CERN S'Cool LAB, Sussman…).

Pour une trace **incertaine**, l'explication nomme les deux classes en concurrence et dit **pourquoi** les mesures ne peuvent pas les séparer (par exemple : sur 14 mm, un électron de 1,31 MeV ne s'écarte que d'environ 0,3 mm de la droite, à peine un pixel). L'explication n'utilise **que ce que la décision a réellement utilisé** : elle ne justifie pas après coup avec des critères inventés.

### 5.7 Annotations physiques

- **Alphas** qui ne touchent pas le bord (de l'image ou de la zone analysée), de plus de 3 mm : **énergie minimale** par la relation portée-énergie (section 1.5). L'émetteur est identifié globalement à partir du 90ᵉ centile de ces énergies (section 1.5), jamais trace par trace.
- **Champ magnétique** : quantité de mouvement et énergie cinétique à partir de la courbure (section 4.5). La charge 2 est utilisée pour les alphas.

### 5.8 IA locale : apprendre de vos corrections (optionnelle)

L'utilisateur peut **corriger la classe** de traces dans l'interface (colonne `classe_manuelle`). Ces étiquettes servent à entraîner une **forêt aléatoire** (Random Forest, Breiman 2001 ; 300 arbres, poids de classes équilibrés) sur 15 mesures :

> longueur, épaisseur, épaisseur relative, densité relative, rectitude, tortuosité, allongement, flèche relative, asymétrie, coefficient de variation du profil, embranchements, SNR de crête, cohérence de courbure, étalement latéral excédentaire, écart RMS à la droite.

- Il faut au moins **6 traces étiquetées** et **2 classes** différentes.
- L'exactitude est estimée par **validation croisée à k plis** (k = min(5, effectif de la plus petite classe)). Elle est affichée avec les variables les plus utiles (importance de Gini).
- Le modèle est sauvegardé (`.joblib`) et peut être réutilisé pour d'autres enregistrements **faits avec le même montage**.

L'IA apprend **les critères de l'utilisateur**. Elle ne peut pas être meilleure que les étiquettes fournies. Pour le rapport, les règles physiques et l'IA de référence ont l'avantage d'être **identiques pour tous les enregistrements**.

Les traces corrigées à la main servent aussi à **mesurer la précision réelle de l'IA de référence sur votre chambre** (section 9.5).

### 5.9 Second avis de Claude (optionnel)

Avec une clé API Anthropic, le bouton « **🤖 Pourquoi ? — 2e avis de Claude Opus 5.5** » de l'onglet Traces envoie la **vignette** et les **mesures** de la trace au modèle Claude (Opus 5.5 recommandé ; Sonnet 5.5 et Haiku 5.5 au choix). Claude doit d'abord juger **indépendamment**, à partir de ce qu'il voit et de la physique, puis comparer avec la décision du logiciel. Sa réponse est structurée : indices visuels, classe, confiance, raisons, pourquoi pas les autres classes, accord ou désaccord avec le logiciel, limites. C'est un **second avis**, pas une vérité : il ne modifie ni la classe ni les comptages.

---

## 6. Mode chambre à expansion

Dans une chambre à expansion, les traces ne se forment que pendant une courte **fenêtre sensible** après chaque détente. WilsonTrack :

1. repère l'instant de chaque expansion de l'une des trois façons suivantes :
   - **« série »** : la ligne `EXP <millis>` envoyée par l'Arduino (`arduino/declencheur_expansion/`) ;
   - **« auto »** : un saut brusque de la luminosité moyenne de l'image, supérieur à 6 fois l'écart-type robuste des variations d'une image à l'autre ;
   - **« manuel »** : la touche E ;
2. remplace le fond par l'image prise à cet instant ;
3. n'accepte de **nouvelles** traces qu'entre `délai` (40 ms) et `délai + fenêtre` (600 ms) après l'expansion ;
4. note le numéro du cycle d'expansion de chaque trace (colonne `cycle_expansion`).

Les taux peuvent alors être exprimés **par expansion** plutôt que par minute.

---

## 7. Analyse statistique

(`wilsontrack/analysis/stats.py` et `conclusions.py`)

### 7.1 Loi de Poisson

Les désintégrations radioactives et l'arrivée des rayons cosmiques sont des événements **indépendants et aléatoires**, de taux moyen constant. Le nombre `N` de traces observées pendant une durée `T` suit donc une **loi de Poisson** de moyenne `μ = r·T`, dont la variance est égale à la moyenne :

```
taux :                  r = N / T
incertitude (1 σ) :     σ_r = √N / T
incertitude relative :  σ_r / r = 1 / √N
```

| N traces | 10 | 25 | 100 | 400 | 1000 |
|---|---|---|---|---|---|
| Incertitude relative | 32 % | 20 % | 10 % | 5 % | 3,2 % |

**La précision ne dépend que du nombre de traces comptées.** Pour la doubler, il faut compter 4 fois plus longtemps.

### 7.2 Intervalle de confiance exact (Garwood)

Pour les petits comptages, par exemple quelques alphas, l'approximation `N ± √N` est mauvaise et peut même donner un taux négatif. Le logiciel calcule donc l'**intervalle exact de Garwood** (1936), au niveau de confiance 68,27 % (équivalent à ± 1 σ) :

```
borne inférieure = ½ · χ²(α/2 ; 2N) / T
borne supérieure = ½ · χ²(1 − α/2 ; 2N + 2) / T          α = 1 − 0,6827
```

`χ²(q ; k)` est le quantile d'ordre `q` de la loi du khi-deux à `k` degrés de liberté. Pour `N = 0`, la borne inférieure est 0.

### 7.3 Comparaison échantillon / bruit de fond : test binomial conditionnel exact

**Question** : l'échantillon (KCl, banane) produit-il **plus** de traces que le bruit de fond seul ?

On compte `n_s` traces pendant `T_s` sur une surface analysée `A_s` avec l'échantillon, et `n_b` traces pendant `T_b` sur `A_b` sans échantillon. Sous l'hypothèse nulle H₀ (« même taux par unité de surface »), et **sachant le total** `N = n_s + n_b`, chaque trace a une probabilité `p₀` d'appartenir à l'enregistrement « échantillon » (exposition = durée × surface) :

```
sous H₀ :  n_s ~ Binomiale(N ; p₀ = T_s·A_s / (T_s·A_s + T_b·A_b))
p-valeur (unilatérale) = P(X ≥ n_s)   pour X ~ Binomiale(N ; p₀)
```

Ce test (Przyborowski et Wilenski 1940) est **exact** : il ne suppose pas que les comptages sont grands. Il tient compte automatiquement de durées **et de surfaces analysées** différentes (le fond — rayons cosmiques, radon — est uniforme, son taux est ramené à la surface de l'échantillon ; un avertissement s'affiche si les surfaces ou les calibrations diffèrent de plus de 5 %). Le logiciel donne aussi :

- l'**excès net** `Δr = r_s − r_b`, avec son incertitude `σ_Δ = √(n_s/T_s² + n_b/T_b²)` ;
- le **rapport** `r_s / r_b` ;
- l'**équivalent en sigmas** de la p-valeur, `Z = Φ⁻¹(1 − p)`, où `Φ` est la fonction de répartition de la loi normale. Par exemple, `p = 0,00135` correspond à 3 σ.

**Critère retenu** : l'excès est déclaré **significatif** si `p < 0,01` (≈ 2,3 σ) et si `Δr > 0`.

La comparaison est faite **classe par classe** (alpha, bêta, muon, proton, trace fine, incertaine), pour le **total**, et pour le groupe des **traces fines** (section 9.3). Depuis la version 2.0, ce groupe contient les traces **bêta**, **muon**, **« trace fine (β ou μ) »** et les traces **incertaines de famille fine** : il réunit toutes les particules au minimum d'ionisation, que l'on sache ou non les départager. Une classe définitive (corrigée à la main ou affirmée par le logiciel) l'emporte sur la famille. Si l'un des deux enregistrements est ancien (sans colonne `famille`), le groupe revient à bêta + muon pour les deux. Les artefacts ne sont jamais comptés. Par défaut, les **périodes mortes** (section 7.8) sont exclues des deux enregistrements, et un avertissement s'affiche si le taux d'un enregistrement n'est pas stable. Deux résultats sont attendus pour une source β⁻ comme le K-40 :

- un **excès de traces fines** (bêta, ou bêta + muon) ;
- **aucun changement du taux d'alphas**. C'est un **contrôle**, testé de façon **bilatérale** (hausse OU baisse) : un excès d'alphas signalerait une variation du radon ambiant ou une contamination ; une baisse signalerait une chambre moins sensible pendant l'un des enregistrements. Dans les deux cas, la comparaison β est signalée comme « à interpréter avec prudence ».

**Planifier la durée des enregistrements.** Avec des durées égales `T`, un taux de fond `b` et un excès attendu `s` (en traces/min), la signification attendue est `Z ≈ s · √T / √(2b + s)`. Pour atteindre `Z` sigmas, il faut donc :

```
T ≈ Z² · (2b + s) / s²
```

Exemple : avec un fond de 20 traces fines/min, un excès de 5 /min se voit à 3 σ en `T ≈ 9 × 45 / 25 ≈ 16 min` par condition. Un excès de 2 /min demande environ 1 h 30 par condition.

### 7.4 Test de dispersion (le processus est-il poissonnien ?)

L'enregistrement est découpé en `k` intervalles de même durée, choisis pour contenir ≈ 5 traces chacun, avec au moins 8 intervalles. Le test exige au moins 8 traces. Pour un processus de Poisson, la **variance** des comptages est égale à leur **moyenne** :

```
indice de dispersion  D = s² / moyenne       (≈ 1 attendu)
χ² = (k − 1) · s² / moyenne                  suit une loi du χ² à (k − 1) degrés de liberté sous H₀
```

Le test est bilatéral. Si `p > 0,01`, les comptages sont **compatibles avec la loi de Poisson** : les traces arrivent au hasard et la chambre est restée stable. Un indice nettement supérieur à 1 signale des **conditions qui ont changé** pendant l'enregistrement (sursaturation qui baisse, glace sèche épuisée, éclairage). Il faut alors regarder la courbe du taux en fonction du temps.

### 7.5 Temps entre deux traces

Pour un processus de Poisson de taux `r`, le temps entre deux événements successifs suit une **loi exponentielle** de moyenne `1/r`. Le logiciel compare la distribution observée des intervalles à une exponentielle de même moyenne par le **test de Kolmogorov-Smirnov**.

La moyenne étant estimée sur les mêmes données, la p-valeur de la table de KS classique serait **beaucoup trop indulgente** (sous H₀, elle rejette ≈ 0,05 % du temps au lieu de 1 %). Le logiciel utilise donc la version de **Lilliefors** (1969) : la p-valeur est obtenue par **Monte-Carlo paramétrique** (1999 simulations d'échantillons exponentiels de même taille, graine fixe pour un résultat reproductible ; `scipy.stats.goodness_of_fit`). Les intervalles nuls (deux traces apparues à la même image) sont conservés. Les instants utilisés sont ceux de l'**horloge d'analyse** (hors perturbations et pauses), comme pour le test de dispersion.

### 7.6 Isotropie des muons (test de Rayleigh pour des données axiales)

L'orientation d'une trace n'a pas de sens de parcours : 10° et 190° désignent la même direction. On utilise donc des **données axiales** (0 à 180°), dont on **double les angles** avant d'appliquer le test de Rayleigh :

```
R = | moyenne de exp(2iθ) |          (0 = orientations uniformes ; 1 = toutes parallèles)
Z = n · R²
p ≈ exp(−Z) · [1 + (2Z − Z²)/(4n) − (24Z − 132Z² + 76Z³ − 9Z⁴)/(288n²)]
```

Le test est fait sur les traces classées « muon » (au moins 8). En mode hybride, les muons sont rarement nommés (la plupart deviennent des « traces fines », section 5.5) : le test porte alors sur peu de traces, ou n'est pas fait. Dans le plan horizontal, les muons n'ont pas de direction privilégiée : on attend `p > 0,05`. Une orientation privilégiée peut signaler une **anisotropie de la couche sensible** (par exemple un courant d'air) ou un **biais de l'éclairage**, qui détecte mieux les traces perpendiculaires aux DEL.

### 7.7 Taux par unité de surface et efficacité apparente

- **Taux par cm²** : taux divisé par la surface de la ROI (calculée en mm² grâce à la calibration).
- **Efficacité apparente** pour un échantillon de masse et de composition connues : `excès de traces fines (traces/s) / nombre de β⁻ émis par seconde` (section 1.6), avec son incertitude. Elle n'est donnée que si l'excès est **significatif** ; sinon, le logiciel donne une **limite supérieure à 95 %** (`excès + 1,645 σ`). Le groupe « traces fines » est choisi **à l'avance** (et non le plus favorable de plusieurs groupes, ce qui biaiserait l'estimation vers le haut). Elle est **très inférieure à 1** (auto-absorption, géométrie, couche sensible mince) et sert surtout à **comparer** des échantillons mesurés dans la même géométrie.

### 7.8 Analyse en fonction du temps : taux par tranche, stabilité, périodes mortes

(`stats.deep_analysis`, section « Analyse approfondie » du rapport.) Une chambre de Wilson **change** pendant une longue séance : la glace sèche s'épuise, l'alcool aussi, le radon de la pièce varie. Le logiciel le vérifie au lieu de le supposer.

- **Tranches** : l'enregistrement est découpé en tranches de **1 h** (séance de 3 h ou plus), **10 min** (30 min ou plus), **1 min** (5 min ou plus), sinon en environ 6 tranches. Pour chaque tranche : taux par classe et des traces fines, avec l'incertitude de Poisson. Le temps utilisé est celui de l'**horloge d'analyse** (section 3.9) ; avec une session de nuit, les tranches sont aussi affichées à l'**heure réelle**.
- **Test de stabilité** : un test du χ² de Pearson compare les comptages par tranche à un taux constant (p-valeur par Monte-Carlo quand moins de 5 traces sont attendues par tranche). Une **dérive** est aussi ajustée (moindres carrés pondérés, en % par heure). Le test est fait pour le total, les alphas et les traces fines. Si le taux d'**alphas** varie mais pas celui des traces fines, c'est plutôt le **radon** ; si tout baisse ensemble, c'est plutôt la **sensibilité de la chambre**.
- **Périodes mortes** : au moins 2 tranches consécutives (d'environ 30 traces attendues chacune) dont le taux est au plus 25 % de la médiane. **Périodes dégradées** : bruit d'image au moins 2 fois plus élevé que d'habitude **et** taux au plus 60 % de la normale. Dans les deux cas, il faut une probabilité de Poisson p < 0,001 divisée par le nombre de tranches (correction de Bonferroni, pour ne pas en « trouver » par hasard). Ces périodes sont exclues des taux « en temps actif », **jamais des comptages bruts**, et le texte le dit.
- **Bruit d'image** : médiane, dérive au cours de la séance et pointes, fraction d'images perturbées.

### 7.9 Comptages corrigés (dépliement)

Aucune méthode de classification n'est parfaite. La **matrice de confusion** du modèle, mesurée sur le jeu de test simulé (section 9.2), indique pour chaque vraie classe la fraction des traces attribuée à chaque classe (y compris « incertaine »). On peut l'utiliser à l'envers pour estimer les **vrais nombres** d'alphas, d'électrons, de muons et de protons à partir des nombres observés : c'est le **dépliement bayésien itératif** de D'Agostini (1995), avec 4 itérations à partir des proportions observées. Les incertitudes viennent d'un **bootstrap** (rééchantillonnage de Poisson des comptages et multinomial de la matrice ; graine fixe, résultat reproductible). Sur une matrice de test synthétique, le biais est d'environ 0,1 σ.

Ce calcul n'est fait qu'en mode hybride, les corrections manuelles étant comptées telles quelles. Il est toujours présenté comme une **« estimation corrigée (d'après la validation sur simulation) »** : il dépend du modèle et de la ressemblance entre la simulation et votre chambre, et il **ne remplace jamais les comptages bruts**.

### 7.10 Analyse de la confiance

Pour chaque classe : médiane et quartiles de la confiance, et fraction des traces au-dessus du seuil (85 %). Pour les traces incertaines : répartition par famille et par classe concurrente, et raison (cas ambigu « court et droit », désaccord entre règles et IA). Sont aussi donnés le taux d'accord règles/IA et, s'il y a des corrections manuelles, le taux d'accord avec vous. Rappel honnête affiché dans le rapport : les probabilités de l'IA sont **calibrées sur simulation** ; la confiance des règles seules est un score flou ; la fiabilité réelle ne se mesure qu'en **vérifiant à la main un échantillon tiré au hasard**.

### 7.11 Score de qualité de l'enregistrement (0 à 100)

Pour aider à juger un enregistrement avant de l'utiliser dans le rapport, le logiciel calcule un score sur 100 à partir de 8 critères pondérés : **durée** (25 points, maximum à 2 h), **nombre de traces** (15), **stabilité** du taux (15), **bruit** (10), **temps perdu** par perturbations (10), **calibration** (10), proportion d'**incertaines** (10) et **alphas** (5). Un critère sans données n'est pas évalué. Un enregistrement de moins de 10 min est plafonné à 59/100 (le plafond est expliqué). Le score est accompagné de conseils concrets, classés par points perdus, et d'une appréciation : excellente, bonne, passable ou insuffisante.

---

## 8. Le simulateur réaliste

On ne connaît jamais avec certitude les « vraies » traces d'une vidéo réelle. Pour mesurer objectivement la performance, et pour entraîner l'IA de référence, le module `wilsontrack/synth.py` génère des **vidéos simulées** accompagnées de leur **vérité terrain** (fichier `.gt.json` : classe, type de source, position, longueur, énergie, instant de naissance et durée de vie de chaque trace).

La version 2.0 remplace le générateur simplifié de la version 1 (section 9.1) par un **simulateur fondé sur la physique et sur des données publiées**. Une leçon importante : avec le simulateur 1.x, les électrons étaient **trop faciles** à reconnaître, ce qui rendait les chiffres de classification trop optimistes. Les chiffres de la section 9.2 sont mesurés avec le simulateur réaliste.

### 8.1 Ce qui est simulé

**Transport des particules.** Chaque particule chargée est suivie **pas à pas en 3D** dans de l'air froid dont la masse volumique suit la loi des gaz parfaits (1,42 mg/cm³ à −25 °C). À chaque pas :

- la **perte d'énergie** vient des tables du NIST (ESTAR pour les électrons, ASTAR pour les alphas, PSTAR pour les protons, interpolation log-log) ;
- la **diffusion multiple** suit la formule de Highland (PDG), avec `X₀` de l'air ramené à la température de la couche ;
- des **rayons delta** (électrons arrachés assez énergétiques pour laisser une petite branche) sont produits selon le PDG, sauf sur les alphas (énergie transférable maximale d'environ 3 keV seulement).

On ne garde que la partie de la trajectoire située **dans la couche sensible**, puis on la projette sur l'image (vue de dessus).

**Sources.**

- **Alphas** : Rn-222, Po-218, Po-214 et Po-210 (sur les surfaces), **paires en V du thoron** (Rn-220 puis Po-216 après une demi-vie de 0,145 s), Po-212 et Bi-212 rares. Énergies du NNDC.
- **Électrons** : électrons **Compton** produits par le gamma de 1,461 MeV dans la plaque et les parois (section efficace de Klein-Nishina, spectre dégradé, front Compton 1,243 MeV) ; électrons **β⁻ du K-40** d'un échantillon (spectre « troisième interdit unique », énergie maximale 1,311 MeV, moyenne 0,560 MeV, NNDC/ENSDF 2017) ; **électrons cosmiques** énergétiques.
- **Muons** : longueur projetée `L = d · tan θ` avec `θ` tiré selon la loi en cos²θ (tirage inverse exact) ; la plupart sont donc **courts**. Aussi : gerbes cosmiques, coudes de désintégration du muon (fréquence exagérée pour l'apprentissage).
- **Protons lents** (2 à 40 MeV).
- **Artefacts** (classe « artefact » de la vérité terrain) : bouffées de brume, amas de gouttelettes, scintillements, reflets près des parois, éblouissement au bord. Selon le type, 49 à 71 % d'entre eux créent une détection : ce sont des **faux positifs réalistes**, que le détecteur doit apprendre à rejeter.

**Rendu.** Un **seul modèle** pour toutes les particules : le nombre de gouttelettes par mm est **proportionnel au dE/dx local** ; toutes les traces subissent la même diffusion latérale ; la réponse optique sature pour les traces très denses. L'aspect plus épais et plus brillant des alphas, et leur fin plus dense (pic de Bragg), **découlent** de ce modèle : aucun paramètre d'apparence n'est propre à une classe (sinon l'IA apprendrait un « truc » du simulateur au lieu de la physique).

**Durées de vie.** Elles sont réglées pour que les durées **mesurées par le vrai détecteur** reproduisent les mesures de Sussman : traces fines, médiane 0,30 s et moyenne 0,385 s (Sussman : 0,27 et 0,38 s) ; alphas 1,56 ± 0,47 s (Sussman : 1,49 ± 0,43 s). Électrons et muons ont **exactement la même loi** de durée et d'épaisseur : la durée ne peut pas servir de « truc » pour les séparer.

**Image.** Brouillard mouvant, pluie de gouttelettes, vignettage et dégradé d'éclairage, bruit du capteur et bruit de photons, compression JPEG au choix.

**Vérifications.** Les portées ESTAR sont reproduites à 0,2 % près et PSTAR à 1 % près. Les portées des alphas à −25 °C (33,6 / 35,4 / 40,6 / 60,0 / 74,5 mm pour Po-210, Rn-222, Po-218, Po-214, Po-212) coïncident avec ASTAR (33,6 / 35,4 / 40,6 / 60,0 / 74,4 mm). L'écart latéral simulé d'un électron de 0,5 MeV sur 50 mm (4,99 mm) correspond à Highland (4,96 mm).

### 8.2 Tableau de justification

Chaque différence entre les types de traces simulées est **justifiée par une source** ; les deux hypothèses sans mesure publiée sont signalées.

| # | Différence simulée | Justification | Source |
|---|---|---|---|
| 1 | Gouttelettes par mm proportionnelles à l'ionisation locale : alpha ≈ 390 ×, proton de 3 MeV ≈ 56 ×, muon et électron rapide 1 ×, électron lent 2 à 12 × | pouvoirs d'arrêt ; dE/dx minimal de l'air 1,815 MeV·cm²/g | NIST ASTAR, PSTAR, ESTAR ; PDG |
| 2 | Traces lourdes plus épaisses et plus brillantes (même diffusion latérale pour toutes, saturation optique commune) | conséquence du point 1 ; observation « alpha épaisse, muon fin » | CERN S'Cool LAB |
| 3 | Fin des alphas plus dense (× 2,8) | pic de Bragg : pouvoir d'arrêt maximal ≈ 2 021 MeV·cm²/g vers 0,7 MeV | NIST ASTAR |
| 4 | Portée des alphas de 3,4 à 7,4 cm à −25 °C selon l'émetteur, ∝ 1/ρ(T) | portées CSDA, énergies des émetteurs, gaz parfait | NIST ASTAR ; NNDC |
| 5 | Alphas presque rectilignes | facteur de détour 0,9925 ; Highland avec z = 2 | NIST ASTAR ; PDG |
| 6 | Paires en V du thoron (0,145 s) | demi-vie du Po-216 | NNDC ; Nuledo |
| 7 | Électrons sinueux ; lents courts et enroulés, rapides presque droits | diffusion multiple ; portées des électrons | PDG (Highland) ; NIST ESTAR |
| 8 | Électrons Compton dégradés venant de la plaque et des parois (front 1,243 MeV) | libre parcours du gamma de 1,46 MeV dans l'air ≈ 130 m | NIST XCOM ; Klein-Nishina |
| 9 | β⁻ du K-40 (max 1,311 MeV, moyenne 0,560 MeV) visibles sur ≈ 1 cm de couche | données de désintégration ; portée > 1 m | NNDC / ENSDF 2017 ; NIST ESTAR |
| 10 | Muons droits, fraction plus longue que x : `(1 + (x/d)²)⁻²` | loi en cos²θ | PDG ; groupe HEP de Cambridge |
| 11 | Électrons cosmiques ≈ 10 % des muons (tirés entre 5 et 25 % à l'entraînement) | **hypothèse** pour l'intérieur d'un bâtiment ; en plein air, I_v(e > 10 MeV) ≈ 30 m⁻²·s⁻¹·sr⁻¹ | PDG |
| 12 | Rayons delta sur les traces rapides, aucun sur les alphas | énergie maximale transférable | PDG §34.2.7 ; S'Cool LAB (traces en Y) |
| 13 | Protons lents : épaisseur intermédiaire, longs et droits | pouvoir d'arrêt et portée des protons | NIST PSTAR ; Müller et al. 2023 ; Nuledo |
| 14 | Même loi de durée pour toutes les traces fines ; alphas ≈ 1,5 s | durées mesurées sur 731 traces | Sussman |
| 15 | Durée de vie des protons interpolée selon l'ionisation | **hypothèse** (aucune mesure publiée) | — |
| 16 | Coude de désintégration du muon (fréquence exagérée) | observation qualitative | CERN S'Cool LAB |
| 17 | Artefacts (gouttelettes qui tombent, brume, reflets) | observation courante | Toronto PHY326 |

**Domaine aléatoire.** Pour que l'IA ne dépende pas d'un réglage particulier, chaque enregistrement simulé tire au hasard sa résolution (640 × 360 à 1920 × 1080), le bruit, l'éclairage, la compression, l'épaisseur de la couche, la température, la proportion d'électrons cosmiques et les taux de chaque particule. Un jeu « **conditions extrêmes** » utilise des paramètres encore plus difficiles.

---

## 9. Validation de l'algorithme

### 9.1 Validation de la détection (simulateur simplifié de la version 1)

Les résultats ci-dessous ont été obtenus avec le **générateur simplifié de la version 1** et les règles de la version 1. Ils restent valables pour la **détection** et la **mesure des longueurs** (le moteur de détection 2.0 donne exactement les mêmes résultats, section 3.13). Les chiffres de **classification** de ce paragraphe sont en revanche **trop optimistes** : sur des traces simulées réalistes, les règles seules se trompent beaucoup plus souvent, surtout entre électron et muon (section 9.2).

Modèle physique du générateur de la version 1 :

- **alphas** : portée donnée par la règle de Geiger pour les émetteurs de la chaîne du radon, corrigée pour l'air à −25 °C ; émission **isotrope dans une couche sensible de 8 mm**, la trace visible étant la projection horizontale de la partie contenue dans la couche (d'où beaucoup de traces raccourcies, comme en réalité) ; traces épaisses et brillantes ;
- **électrons** (β⁻ du K-40 tirés selon le spectre « troisième interdit unique » avec fonction de Fermi, facteur de forme ajusté pour retrouver l'énergie moyenne mesurée de 0,560 MeV ; électrons Compton) : **marche aléatoire** par pas de 0,6 mm, avec une déviation de diffusion multiple de Rossi-Highland et une perte d'énergie `dE/dx ≈ (dE/dx)_min / β²` (forme de Bethe, qui retrouve les portées CSDA de l'ESTAR à ± 30 % près). On obtient des traces sinueuses en fin de parcours et presque droites à haute énergie ;
- **muons** : droites fines ; 70 % ont une longueur projetée `h · tan θ` avec θ tiré selon le flux cos³θ·sinθ (traces courtes de muons presque verticaux), 30 % sont de longues traces de muons très inclinés ;
- **rendu** : chapelets de gouttelettes, apparition en ≈ 0,1 s, diffusion et estompement en 0,8 à 2,5 s, dérive lente ;
- **défauts réalistes** : brouillard mouvant, « pluie » de gouttelettes isolées (250 par image), éclairage latéral non uniforme (vignettage), bruit du capteur.

Paramètres des vidéos de validation : 1280 × 720 px, 30 images/s, champ de 250 mm, taux de 5 alphas/min, 8 bêtas/min, 12 muons/min et 0,2 proton/min, bruit σ = 2,5 niveaux de gris.

**Règle d'appariement.** Une trace vraie d'au moins 5 mm est **retrouvée** si une trace détectée remplit deux conditions :

- elle naît entre 0,2 s avant et 0,8 s après la trace vraie ;
- au moins 50 % de son chemin se trouve à moins de 3,5 mm de la trajectoire vraie.

Chaque détection ne peut être appariée qu'une fois (`wilsontrack/evaluate.py`, script `tools/validate_synth.py`). On en déduit trois indicateurs :

- **rappel** : fraction des traces vraies retrouvées ;
- **précision** : fraction des détections qui correspondent à une trace vraie ;
- **exactitude de classification** : fraction des traces retrouvées classées dans la bonne catégorie.

Depuis la version 2.0, une détection appariée à un **artefact** de la vérité terrain compte comme une **fausse détection** (signalée à part).

**Résultats (version 1)**

Deux vidéos de **90 s** ont été analysées avec les **réglages par défaut**, sans aucun ajustement :

- une vidéo de **bruit de fond seul** (graine aléatoire 7) ;
- une vidéo **avec une source KCl simulée** (graine 8). La source ajoute 40 électrons β⁻/min, émis depuis une zone de 15 mm de rayon.

Le champ par défaut (250 mm) coïncide avec celui des vidéos : les résultats avec et sans calibration sont identiques.

**Détection et classification**

| Indicateur | Bruit de fond | Fond + source KCl simulée |
|---|---|---|
| Traces vraies (≥ 5 mm) | 46 (4 α, 15 β, 27 μ) | 99 (9 α, 75 β, 15 μ) |
| Traces détectées | 46 | 87 |
| **Rappel** | **87,0 %** | 78,8 % |
| **Précision** | **87,0 %** (6 fausses détections) | 89,7 % (9 fausses détections) |
| **Exactitude de classification** | **87,5 %** | 65,4 % (confusion β → μ, voir 9.3) |
| Longueur mesurée / vraie (médiane) | α 1,02 ; β 0,98 ; μ 1,00 | α 1,02 ; β 1,00 ; μ 1,00 |

Détail par classe pour la vidéo de bruit de fond :

| Classe vraie | Retrouvées | Classées comme… |
|---|---|---|
| Alpha | 4 / 4 (100 %) | 3 alpha, 1 bêta |
| Bêta | 13 / 15 (87 %) | 10 bêta, 3 muon |
| Muon | 23 / 27 (85 %) | 22 muon, 1 bêta |

Pendant ces 90 s, **plus de 20 000 candidats éphémères** (gouttelettes de « pluie », bruit, brouillard) ont été écartés par le suivi temporel et les critères de validation, pour seulement 6 fausses détections. **Les longueurs sont mesurées à ± 2 % près** grâce à la mesure à mi-hauteur (section 4.2).

**Comparaison « KCl simulé » contre « fond »** (test binomial conditionnel exact, section 7.3) :

| Groupe | Fond | Avec source | Excès (/min) | p-valeur | Signification |
|---|---|---|---|---|---|
| Alpha (contrôle) | 3 (2,0 /min) | 6 (4,0 /min) | +2,0 ± 2,0 | 0,25 | 0,7 σ : **non significatif**, comme attendu |
| Bêta | 13 (8,7 /min) | 38 (25,5 /min) | +16,7 ± 4,8 | 3,1 × 10⁻⁴ | **3,4 σ** |
| Muon | 30 (20,1 /min) | 43 (28,8 /min) | +8,7 ± 5,7 | 0,08 | 1,4 σ |
| **Traces fines (β + μ)** | 43 (28,8 /min) | 81 (54,2 /min) | **+25,5 ± 7,5** | 4,1 × 10⁻⁴ | **3,4 σ** |
| Total | 46 (30,8 /min) | 87 (58,3 /min) | +27,5 ± 7,7 | 2,4 × 10⁻⁴ | 3,5 σ |

L'excès dû à la source est **détecté à 3,4 σ** en seulement 90 s par condition, et le taux d'alphas reste inchangé, ce qui valide le contrôle. L'excès mesuré sur les traces fines (+25,5 ± 7,5 /min) représente environ 65 % de l'excès vrai (+40 /min). La différence s'explique par le rappel des électrons (73 %) : les électrons de basse énergie laissent des traces courtes et pâles, plus difficiles à détecter.

Deux remarques :

- Le taux de **muons** augmente aussi (+8,7 /min) avec la source. Ce n'est pas un effet des rayons cosmiques : **25 des 55 électrons retrouvés (fond et source confondus) ont été classés « muon »**. Les deux vidéos utilisent des graines aléatoires différentes : leur partie « fond » diffère donc par des fluctuations poissonniennes, comme deux enregistrements réels.
- Ces vidéos sont courtes. Avec de vraies chambres, dont l'excès est souvent plus faible, il faut prévoir **10 à 20 min ou plus par condition** (formule de la section 7.3).

### 9.2 Validation de l'IA de référence

L'IA de référence (section 5.4) est évaluée sur des enregistrements simulés réalistes (section 8) qu'elle **n'a jamais vus** pendant l'entraînement ni pendant le choix de ses réglages, puis sur le jeu « conditions extrêmes ». Les indicateurs sont :

- **couverture** : fraction des traces auxquelles une particule est attribuée (les autres sont « incertaines » ou « traces fines ») ;
- **exactitude quand elle se prononce** : fraction des traces nommées qui sont de la bonne classe ;
- **précision par classe** : parmi les traces nommées « alpha », fraction de vraies alphas (idem pour chaque classe) ; **rappel par classe** : parmi les vraies alphas, fraction nommées « alpha » ;
- **exactitude de la famille** (fine ou lourde) ;
- **calibration** (ECE et diagramme de fiabilité) ;
- **comparaison** avec les règles seules et avec le modèle pilote de la version 1.x, sur les mêmes traces ;
- **matrice de confusion** (lignes : vraie classe ; colonnes : classe donnée, y compris « incertaine »), utilisée aussi pour les comptages corrigés (section 7.9).

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

**Ce que ces chiffres veulent dire, et ne veulent pas dire.**

- Ils sont mesurés **sur simulation uniquement**. L'IA n'a jamais vu votre chambre : sur de vraies vidéos, ses performances peuvent être moins bonnes. La section 9.5 explique comment mesurer sa précision réelle.
- L'abstention est un **compromis voulu** : en exigeant 85 %, l'IA se prononce sur moins de traces, mais se trompe beaucoup plus rarement quand elle le fait.
- Les **muons** ne sont pratiquement jamais nommés : les électrons cosmiques énergétiques leur sont physiquement identiques à cette échelle (section 5.5). Ce résultat dépend de la proportion supposée d'électrons cosmiques à l'intérieur (hypothèse n° 11 du tableau de la section 8.2).
- Une classe n'est affirmée que si sa précision mesurée sur la validation est solidement au-dessus de 85 % ; une classe pour laquelle les exemples de validation sont trop peu nombreux (les protons, rares) peut rester bloquée par prudence.

### 9.3 Ambiguïté bêta / muon et indicateur « traces fines »

Les électrons du K-40 ont une énergie moyenne de 0,56 MeV et peuvent atteindre 1,31 MeV. Au-delà d'environ 0,5 MeV, ils sont **presque au minimum d'ionisation** et ne dévient que de **quelques degrés par dizaine de centimètres** (tableau de la section 1.4). Sur les quelques centimètres visibles, leur trace est **fine et presque droite, comme celle d'un muon**. Il en va de même des électrons cosmiques énergétiques (section 1.8). Aucune mesure d'image ne permet alors de les distinguer à coup sûr. Seuls un champ magnétique (courbure) ou la direction d'arrivée (depuis l'échantillon) le permettraient.

Par conséquent :

- avec les règles seules, une partie des β⁻ énergétiques de l'échantillon est classée « **muon** » ;
- réciproquement, un muon qui traverse la couche sensible en biais laisse une trace **courte** qui ressemble à celle d'un électron ;
- en mode hybride, ces traces reçoivent honnêtement la classe « **trace fine (β ou μ)** » (section 5.5).

Le groupe des **traces fines** (bêta + muon + trace fine + incertaines de famille fine : toutes les particules au minimum d'ionisation) ne dépend pas de cette confusion. **C'est l'indicateur le plus robuste de la radioactivité β de l'échantillon.** Les conclusions automatiques le testent en plus de la classe bêta seule, et l'alpha sert de contrôle.

### 9.4 Essais sur des vidéos réelles

L'algorithme a aussi été appliqué à des vidéos réelles de chambres à brouillard publiées sous licence Creative Commons : chambres à diffusion artisanales et institutionnelles, avec ou sans source alpha. Sans vérité terrain, ces essais ne donnent pas de chiffres de rappel ou de précision. Ils servent à vérifier **visuellement** trois points :

- la robustesse aux conditions réelles : brouillard, reflets, compression vidéo, éclairages variés ;
- l'absence de comptages multiples d'une même trace ;
- la cohérence du classement : les alphas des sources sont épaisses et courtes, les muons longs et droits.

### 9.5 Mesurer la précision sur VOTRE chambre

La seule façon de connaître la fiabilité réelle sur votre montage est de comparer le logiciel à un **jugement humain** :

1. ouvrez un enregistrement réel et, dans l'onglet **Traces**, choisissez **au hasard** (pas seulement les plus belles) des traces, idéalement **30 ou plus de chaque type** ;
2. corrigez leur classe (« Corriger la classe » : alpha, beta, muon, proton ou artefact) en vous aidant de la vignette et de l'explication ;
3. dans le **Centre d'apprentissage de l'IA** (onglet 🧠 IA), cliquez sur le bouton de validation : le logiciel compare ce que l'IA de référence aurait répondu à vos réponses et affiche la couverture, l'exactitude avec son intervalle de confiance à 95 %, la précision et le rappel par classe, et la matrice de confusion.

Il faut au moins 5 traces corrigées ; avec moins de 20 traces d'une classe, les chiffres de cette classe sont très incertains (signalés par ⚠). Ce résultat peut être cité dans le rapport : c'est une **validation sur données réelles**, à la différence de la section 9.2.

---

## 10. Pourquoi ne pas apprendre sur des « millions de vidéos » en ligne ?

L'idée est naturelle : il existe beaucoup de vidéos de chambres à brouillard sur Internet. Une recherche systématique a montré pourquoi elle ne fonctionne pas.

**1. Il n'existe pas de grand corpus public de vidéos étiquetées.** Les jeux de données trouvés sont petits, incomplets ou inaccessibles :

| Jeu de données | Taille | Étiquettes | Remarques |
|---|---|---|---|
| Müller et al. 2023 (*Rev. Sci. Instrum.* 94, 063304) | inconnue | masques par pixel : alpha, V du thoron, électrons, protons ; muons traités comme du fond | meilleure référence scientifique ; disponibilité publique des images non confirmée |
| Roboflow Universe (projets communautaires « cloud chamber ») | ≈ 110 à 2 500 images par projet | boîtes englobantes ; classes différentes d'un projet à l'autre | annotations d'amateurs ou d'élèves, sans contrôle physique |
| lonlonago, « Cloud Chamber Particle Detection/Classification dataset » | 1 465 images, 2 859 boîtes (1 562 α, 1 241 β, 56 μ) | boîtes | payant, origine des images non précisée, presque aucun muon |
| Hans-Stuffer, *cloud-chamber-ml* (MIT) | ≈ 14,5 min de vidéo réelle ; 910 traces candidates revues | partielles (seuls les muons sont exploités) | sur vidéo réelle, leur détecteur U-Net n'atteint que F1 ≈ 0,28 |
| Wikimedia Commons (catégories « cloud chamber ») | 5 vidéos et 46 images | aucune | utile pour des démonstrations (section 9.4), pas pour l'entraînement |
| AlessandroLovo, *CloudChamber* (Padoue, 2019) | code seulement | non publiées | approche par descripteurs proche de WilsonTrack |
| Sussman (rapport d'élève) | ≈ 5 h filmées, 731 événements analysés | alpha contre « bêta » | seules des statistiques sont publiées (utilisées pour les durées de vie, section 8.1) |
| Schonfeld 2022 (arXiv:2201.13249) | une vidéo publique | aucune classification | illustre que les vidéos publiques sont des données « non contrôlées » |

**2. Des vidéos sans étiquettes n'apprennent pas les bonnes réponses.** Une IA supervisée apprend à reproduire des **réponses connues**. Une vidéo YouTube ne dit pas quelle trace est un électron et laquelle est un muon ; et pour les traces fines droites, **même un expert ne peut pas le savoir** (section 5.5). Étiqueter ces vidéos à la main reviendrait à apprendre les erreurs de l'étiqueteur. De plus, les conditions (résolution, échelle, température, éclairage, compression) sont inconnues et changent d'une vidéo à l'autre.

**3. Droits d'auteur et conditions d'utilisation.** Télécharger massivement des vidéos de YouTube ou d'autres plateformes pour entraîner un modèle pose des problèmes de droits d'auteur et contrevient aux conditions d'utilisation de ces sites.

**La solution retenue** est donc la **simulation fondée sur la physique** (section 8) : on connaît la vraie classe de chaque trace **par construction**, on peut en produire autant qu'on veut, dans des conditions variées, et chaque différence entre particules repose sur des **données publiées** (NIST ASTAR, ESTAR et PSTAR, PDG, NNDC, CERN S'Cool LAB, Sussman). Le revers est que l'IA n'est validée que sur simulation : d'où l'importance de la section 9.5.

---

## 11. Limites connues et précautions

1. **Projection en 2D.** La caméra voit la **projection** de la trajectoire sur le plan horizontal. Une trace inclinée paraît plus courte. Une alpha qui plonge dans la couche sensible ou en sort n'est pas vue sur toute sa portée : son énergie est **sous-estimée**. La courbure mesurée dans un champ `B` donne une **limite inférieure** de la quantité de mouvement. Seule une vue stéréoscopique (deux caméras) donnerait la 3D.
2. **Calibration.** Sans règle filmée, l'échelle est **supposée** (250 mm de largeur de champ). Les longueurs, les épaisseurs en mm, les énergies des alphas et les taux par cm² sont alors approximatifs. La **perspective** compte aussi : il faut calibrer **à la hauteur de la couche sensible**, pas à celle du couvercle.
3. **Épaisseur de la couche sensible.** Elle n'est pas mesurée par le logiciel. Elle dépend de la température de la plaque, de la quantité d'alcool et du temps écoulé, et elle modifie fortement le taux de longues traces (section 1.7). Le paramètre `sensitive_depth_mm` n'est qu'un ordre de grandeur.
4. **Stabilité de la chambre.** Au cours d'une séance, la sursaturation varie : mise en régime lente (10 à 15 min), épuisement de la glace sèche ou de l'alcool, givre sur les parois. Les tests de dispersion (7.4), le test de stabilité par tranche (7.8) et la courbe du taux en fonction du temps permettent de le **vérifier**. Il faut **attendre la mise en régime** avant d'enregistrer. Pendant une session de nuit, une chambre qui s'épuise produit une **période morte** : elle est signalée, et exclue des taux « en temps actif » (7.8).
5. **Comparaisons : réglages identiques obligatoires.** Une comparaison échantillon / fond n'a de sens que si les deux enregistrements sont faits avec :
   - la même chambre, la même quantité de glace sèche et d'alcool, le même temps de mise en régime ;
   - le même éclairage, la même caméra et les mêmes réglages de caméra (fixes) ;
   - la même zone d'intérêt ;
   - **les mêmes paramètres de détection et de classification** (enregistrer un préréglage) ;
   - idéalement, en **alternant** fond, échantillon, fond, échantillon, pour contrôler la dérive.

   La référence des traces fines est estimée **par enregistrement**. Si un enregistrement compte très peu de traces, sa référence est moins fiable.
6. **Fond non constant.** Le radon de l'air varie selon l'heure, la ventilation et la saison. Le taux d'alphas peut donc changer d'un jour à l'autre, indépendamment de l'échantillon.
7. **Traces superposées.** Deux traces qui naissent au même endroit à la même image peuvent être fusionnées. Deux fragments d'une même trace très pâle, séparés de plus de 8 mm, peuvent être comptés deux fois. Ces cas sont rares aux taux habituels.
8. **Traces qui sortent du champ.** Elles sont comptées, mais leur longueur est tronquée (colonne `touche_bord = 1`). Aucune énergie n'est calculée pour une alpha qui touche le bord.
9. **Classification.** Il s'agit d'une **aide à l'identification**, validée sur des données **simulées**, et non d'une identification certaine. En mode hybride, beaucoup de traces sont « incertaines » ou « traces fines » : c'est voulu (section 5.4). Électrons rapides et muons ne peuvent pas être séparés sans aimant (section 5.5) ; la classe « proton » est rare et à interpréter avec prudence. Il est recommandé de **vérifier visuellement** un échantillon de vignettes tiré au hasard (dossier `vignettes/`), de mesurer la précision réelle (section 9.5) et de le mentionner dans le rapport.
10. **Validation sur simulation.** Le simulateur reproduit les principaux phénomènes à partir de données publiées (section 8), mais pas tous : formes de gouttelettes, turbulence réelle, reflets particuliers de votre montage. Deux paramètres reposent sur des **hypothèses** (proportion d'électrons cosmiques à l'intérieur, durée de vie des traces de protons). Les performances sur une vraie chambre peuvent être inférieures. Le réglage de la ROI et des seuils (`k_high`, `min_peak_snr`) sur une courte vidéo de test est recommandé.
11. **Paires en V du thoron.** Les deux bras d'un V, nés au même point à moins d'une seconde d'intervalle, sont souvent fusionnés en une seule trace : sur simulation, la seconde alpha (Po-216) n'est détectée séparément qu'environ une fois sur trois.
12. **Comptages corrigés et probabilités.** Les comptages corrigés (7.9) et les probabilités calibrées (5.4) reposent sur la ressemblance entre la simulation et votre chambre. Ils complètent les comptages bruts, sans jamais les remplacer.
13. **Résultats provisoires d'une session de nuit.** Les sauvegardes intermédiaires (`"partiel": true`) ne comptent pas les traces encore suivies et ne sont pas reclassées : seuls les résultats finaux doivent être utilisés dans le rapport.

---

## 12. Bibliographie

**Chambre de Wilson : articles historiques et guides**

1. C. T. R. Wilson, « On a method of making visible the paths of ionising particles through a gas », *Proceedings of the Royal Society of London A* **85**, 285-288 (1911).
2. C. T. R. Wilson, « On an expansion apparatus for making visible the tracks of ionising particles in gases and some results obtained by its use », *Proceedings of the Royal Society of London A* **87**, 277-292 (1912).
3. A. Langsdorf, « A continuously sensitive diffusion cloud chamber », *Review of Scientific Instruments* **10**, 91-103 (1939).
4. CERN S'Cool LAB, J. Woithe, *Cloud Chamber — Do-It-Yourself Manual* (guide de construction et d'identification des traces ; v8 2024 sur Zenodo, CC BY 4.0), https://scoollab.web.cern.ch/
5. QuarkNet (Fermilab / University of Notre Dame), activité « Cloud Chamber », https://quarknet.org/
5a. University of Cambridge, High Energy Physics group, page « Cloud Chamber » (vulgarisation), https://www.hep.phy.cam.ac.uk/outreach/cloud
5b. K. Clark, « The Cloud Chamber Experiment », manuel de laboratoire PHY326, Université de Toronto (2019), https://www.physics.utoronto.ca/~phy326/cc/cc.pdf
5c. Nuledo, page « Cloud chambers » (fabricant ; source commerciale, utilisée avec prudence), https://nuledo.com/en/cloud-chambers/

**Physique des particules et des rayonnements**

6. Particle Data Group, S. Navas et al., « Review of Particle Physics », *Physical Review D* **110**, 030001 (2024). Chapitres « Passage of particles through matter » (formule de Bethe, formule de Highland, rayons delta, longueurs de radiation), « Atomic and Nuclear Properties : Air (dry, 1 atm) » et « Cosmic rays » (flux et spectre des muons, électrons et protons cosmiques ; RPP 2020, section 30.3). https://pdg.lbl.gov/
7. H. Bethe, « Zur Theorie des Durchgangs schneller Korpuskularstrahlen durch Materie », *Annalen der Physik* **397**, 325-400 (1930).
8. V. L. Highland, « Some practical remarks on multiple scattering », *Nuclear Instruments and Methods* **129**, 497-499 (1975) ; G. R. Lynch et O. I. Dahl, *Nuclear Instruments and Methods B* **58**, 6-10 (1991).
9. H. Geiger, « The ionisation produced by an α-particle », *Proceedings of the Royal Society of London A* **83** (1910). C'est l'origine de la règle de Geiger : la portée varie comme le cube de la vitesse, donc comme `E^1,5`. La forme numérique `R ≈ 0,318·E^1,5` est tirée de R. D. Evans, *The Atomic Nucleus*, McGraw-Hill (1955).
10. M. J. Berger et al., *ESTAR, PSTAR and ASTAR: Computer Programs for Calculating Stopping-Power and Range Tables for Electrons, Protons, and Helium Ions*, NIST Standard Reference Database 124, https://physics.nist.gov/PhysRefData/Star/Text/
11. National Nuclear Data Center (Brookhaven National Laboratory), NuDat 3 (ENSDF), données de désintégration du K-40 (J. Chen, *Nuclear Data Sheets* **140**, 1, 2017), du Rn-222, du Rn-220 et des polonium 210, 212, 214, 216 et 218, https://www.nndc.bnl.gov/nudat3/ ; Laboratoire National Henri Becquerel, tables DDEP du K-40 (révision 2025 : valeurs légèrement différentes, effet < 1 % sur l'expérience ; ce document utilise de façon cohérente ENSDF 2017), http://www.lnhb.fr/nuclides/K-40_tables.pdf
11a. J. H. Hubbell et S. M. Seltzer, *X-Ray Mass Attenuation Coefficients* (NIST), table de l'air, https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html
12. T. K. Gaisser, R. Engel et E. Resconi, *Cosmic Rays and Particle Physics*, 2ᵉ éd., Cambridge University Press (2016).
13. G. F. Knoll, *Radiation Detection and Measurement*, 4ᵉ éd., Wiley (2010). Statistique de comptage, interaction des rayonnements.

**Analyse d'images de chambres à brouillard (travaux antérieurs)**

14. Z. Sussman, « Automatic Particle Detection in Cloud Chambers » (rapport d'élève, non révisé par les pairs ; durées de visibilité des traces), https://zacharysussman.com/pdf/cloud-chamber.pdf
15. Hans-Stuffer, *cloud-chamber-ml* (classification des traces par apprentissage automatique), https://github.com/Hans-Stuffer/cloud-chamber-ml
16. Codeknightjl, *cloudchamber* (détection de traces par traitement d'image), https://github.com/Codeknightjl/cloudchamber
16a. N. J. Müller et al., « Neuro-explicit semantic segmentation of the diffusion cloud chamber », *Review of Scientific Instruments* **94**, 063304 (2023), https://doi.org/10.1063/5.0109284
16b. S. Zeze et al., « A sensitive cloud chamber without radioactive sources », arXiv:1202.2423.
16c. Schonfeld, analyse d'une vidéo publique de chambre à diffusion, arXiv:2201.13249 (2022).
16d. A. Lovo, *CloudChamber* (thèse de licence, Université de Padoue, 2019), https://github.com/AlessandroLovo/CloudChamber
16e. Jeux de données communautaires : Roboflow Universe (« cloud chamber »), https://universe.roboflow.com/ ; lonlonago, « Cloud Chamber Particle Detection/Classification dataset », GitHub ; Wikimedia Commons, Category:Cloud chamber particle tracks.

**Traitement d'image et ajustement**

17. A. F. Frangi, W. J. Niessen, K. L. Vincken et M. A. Viergever, « Multiscale vessel enhancement filtering », *MICCAI 1998*, Lecture Notes in Computer Science **1496**, 130-137 (1998).
18. T. Lindeberg, « Edge detection and ridge detection with automatic scale selection », *International Journal of Computer Vision* **30**, 117-156 (1998).
19. J. Canny, « A computational approach to edge detection », *IEEE Transactions on Pattern Analysis and Machine Intelligence* **8**, 679-698 (1986). Seuillage par hystérésis.
20. T. Y. Zhang et C. Y. Suen, « A fast parallel algorithm for thinning digital patterns », *Communications of the ACM* **27**, 236-239 (1984). Squelettisation.
21. G. Taubin, « Estimation of planar curves, surfaces, and nonplanar space curves defined by implicit equations with applications to edge and range image segmentation », *IEEE Transactions on Pattern Analysis and Machine Intelligence* **13**, 1115-1138 (1991).
22. N. Chernov, *Circular and Linear Regression: Fitting Circles and Lines by Least Squares*, CRC Press (2010).
23. P. J. Rousseeuw et C. Croux, « Alternatives to the median absolute deviation », *Journal of the American Statistical Association* **88**, 1273-1283 (1993).
24. L. Breiman, « Random Forests », *Machine Learning* **45**, 5-32 (2001).
24a. J. H. Friedman, « Greedy function approximation: a gradient boosting machine », *Annals of Statistics* **29**, 1189-1232 (2001). *Gradient boosting* (IA de référence).
24b. C. Guo, G. Pleiss, Y. Sun et K. Q. Weinberger, « On calibration of modern neural networks », *ICML 2017*. Mise à l'échelle par température, erreur de calibration attendue (ECE).
24c. M. Saerens, P. Latinne et C. Decaestecker, « Adjusting the outputs of a classifier to new a priori probabilities », *Neural Computation* **14**, 21-41 (2002). Correction des proportions de classes.

**Statistique**

25. F. Garwood, « Fiducial limits for the Poisson distribution », *Biometrika* **28**, 437-442 (1936).
26. J. Przyborowski et H. Wilenski, « Homogeneity of results in testing samples from Poisson series », *Biometrika* **31**, 313-323 (1940). Test conditionnel de comparaison de deux taux.
27. H. W. Lilliefors, « On the Kolmogorov-Smirnov test for the exponential distribution with mean unknown », *Journal of the American Statistical Association* **64**, 387-389 (1969).
28. K. V. Mardia et P. E. Jupp, *Directional Statistics*, Wiley (2000). Test de Rayleigh, données axiales.
29. G. D'Agostini, « A multidimensional unfolding method based on Bayes' theorem », *Nuclear Instruments and Methods A* **362**, 487-498 (1995). Comptages corrigés.
30. E. B. Wilson, « Probable inference, the law of succession, and statistical inference », *Journal of the American Statistical Association* **22**, 209-212 (1927). Intervalle de Wilson pour une proportion.

**Logiciels utilisés** : Python, NumPy, SciPy, OpenCV, scikit-image, scikit-learn, Matplotlib, PySide6 (Qt).

---

## Annexe : réglages par défaut

Tous ces réglages se trouvent dans le panneau **Réglages** (certains seulement dans le fichier de préréglage) et sont enregistrés dans `run.json` avec chaque enregistrement. Ils sont aussi exportables comme préréglage.

| Groupe | Paramètre (nom interne) | Défaut | Rôle |
|---|---|---|---|
| Pré-traitement | `process_width` | 960 px | largeur de traitement |
| | `temporal_avg` | 2 images | moyenne temporelle |
| | `blur_sigma` | 1,0 px | lissage spatial |
| Fond | `bg_time_s` | 3 s | constante de temps du fond |
| | `bg_warmup_frames` | 12 | apprentissage initial |
| Seuillage | `k_high` / `k_low` | 4,5 σ / 2,2 σ | hystérésis (graine / extension) |
| | `ridge_scales` | 1,5 et 3 px | échelles du filtre hessien |
| | `close_px` | 3 px | fermeture morphologique |
| | `min_area_px` | 12 px | aire minimale |
| | `max_blob_frac` | 25 % | seuil d'image perturbée |
| Fusion | `merge_gap_mm` / `merge_angle_deg` | 8 mm / 14° | fragments colinéaires |
| Suivi | `min_visible_s` / `min_frames` | 0,15 s / 3 | visibilité minimale |
| | `max_missed` | 5 images | fin d'une trace |
| | `match_dilate_px` | 5 px | tolérance d'association |
| Validation | `min_length_mm` | 4 mm | longueur minimale |
| | `min_elongation` | 2,2 | allongement minimal |
| | `min_peak_snr` | 6 σ | contraste minimal |
| Classification | `mode` | hybride | méthode : `hybride`, `regles` ou `ia` (section 5) |
| | `ml_threshold` | 0,85 | seuil de certitude (abstention) en mode hybride |
| | `ref_model_weight` | −1 | poids de l'IA de référence (−1 = valeur enregistrée dans le modèle) |
| | `alpha_width_mm` | 1,3 mm | épaisseur absolue « lourde » (si calibré) |
| | `heavy_width_rel` / `heavy_density_rel` | 1,7 / 2,2 | seuils relatifs « lourde » |
| | `alpha_max_length_mm` | 80 mm | longueur max. d'une alpha (75 mm en 1.x) |
| | `muon_min_length_mm` | 45 mm | longueur min. d'un muon |
| | `muon_max_rms_mm` | 0,2 mm | écart RMS max. à la droite pour un muon (0 = désactivé ; nouveau) |
| | `straight_min` / `curly_max` | 0,95 / 0,86 | rectitude droite / sinueuse |
| | `ref_percentile` | 35 | centile de référence des traces fines |
| | `min_confidence` | 0,30 | seuil de la classe « incertaine » (règles seules) |
| Physique | `chamber_temp_c` | −25 °C | température de la couche sensible |
| | `pressure_kpa` | 101,3 kPa | pression |
| | `b_field_mt` | 0 mT | champ magnétique (0 = sans aimant) |
| | `default_fov_mm` | 250 mm | champ supposé si non calibré |
| | `sensitive_depth_mm` | 8 mm | épaisseur de la couche sensible (ordre de grandeur) |
| Expansion | `delay_ms` / `window_ms` | 40 ms / 600 ms | fenêtre sensible |
| Session de nuit | durée / heure de fin | jusqu'à 7 h 30 | arrêt automatique |
| | morceaux de vidéo / vidéo | 30 min / réduite (½, 10 img/s) | enregistrement |
| | sauvegarde / photo / alerte | 5 / 10 / 20 min | robustesse et surveillance |

## Annexe : validation sur de vraies vidéos (Wikimedia Commons)

**Données.** Six vraies vidéos de chambres à brouillard sous licence Creative Commons (auteurs : Stephen Curry, aza, Mickyfitz13, Michael F. Schönitzer, Bilovitskiy, Tilnishok ; liens et licences dans `exemples/vidéos réelles (Wikimedia)/SOURCES.md`). Les 917 détections ont été extraites (`tools/export_real_tracks.py`) puis **étiquetées à l'œil** sur des vignettes (`_training/real/labels_visuels.py`) : 542 artefacts, 284 traces fines (dont 84 électrons nettement courbés), 30 alphas, 61 cas douteux écartés. Cet étiquetage n'a pas été fait par un physicien : c'est une vérité terrain approximative.

**Méthode de mesure.** Chaque vidéo est testée par un modèle entraîné sur les cinq autres (« une vidéo laissée de côté ») : aucune trace testée n'a servi à l'apprentissage.

| Méthode (vraies vidéos jamais vues) | Se prononce sur | Exactitude quand elle se prononce |
|---|---|---|
| Règles physiques seules | 90 % | 32 % (aucun artefact rejeté) |
| IA de référence entraînée sur simulation | 42 % | 64 % |
| **Combinaison retenue : IA de simulation + adaptateur vidéos réelles (seuil 85 %)** | **79 %** | **80 %** (artefacts : précision 81 % ; traces fines : 75 %) |

**Conclusion honnête.** Les 96 % mesurés sur simulation ne se transposent pas tels quels aux vraies vidéos : sur des vidéos inconnues, l'application atteint environ 80 %. Elle reconnaît bien les artefacts et les traces fines, mais **pas encore les alphas réelles**, faute d'exemples : seulement 30 alphas dans ces vidéos. C'est pourquoi l'adaptateur n'a pas le droit de contredire une décision « alpha ». Pour votre chambre, corrigez une trentaine de traces de chaque type et mesurez la précision dans l'onglet 🧠 IA : c'est la seule mesure qui vaut pour votre montage.
