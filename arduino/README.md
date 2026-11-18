# Déclencheur de chambre à expansion (Arduino)

Ce dossier contient le programme Arduino **optionnel** qui commande une chambre de Wilson **à expansion**. Il ne sert à rien pour une chambre à diffusion (glace sèche), qui est sensible en continu.

```
arduino/
└── declencheur_expansion/
    └── declencheur_expansion.ino   ← à ouvrir avec l'IDE Arduino
```

L'Arduino ouvre brièvement une **électrovanne** qui relie la chambre à un **réservoir sous vide**. La détente brusque du gaz rend la vapeur d'alcool sursaturée pendant une fraction de seconde, et les traces apparaissent. À chaque expansion, l'Arduino envoie `EXP <millis>` à l'ordinateur. WilsonTrack ne compte alors que les traces qui apparaissent dans la **fenêtre sensible** qui suit.

---

## Sécurité (à lire avant tout)

- **Vide = risque d'implosion.** Utilisez un réservoir **conçu pour le vide** : dessiccateur à vide, fiole à vide à paroi épaisse ou réservoir métallique. N'utilisez jamais de bouteille en plastique mince, de bocal ordinaire ni de verrerie fêlée. Entourez le verre de ruban adhésif et travaillez derrière un écran.
- **Lunettes de sécurité** pour toutes les personnes présentes.
- **Jamais sans surveillance**, et toujours avec un enseignant ou un technicien. À la fin, débranchez le 12 V : la vanne *normalement fermée* se referme toute seule.
- **Isopropanol inflammable** : pas de flamme, aérez la pièce, et gardez les relais (qui font des étincelles) loin des vapeurs.
- **Glace sèche** : portez des gants isolants. Ne l'enfermez jamais dans un contenant hermétique. Le CO₂ est asphyxiant dans une petite pièce.
- **Électricité** : n'alimentez jamais la bobine directement par une broche de l'Arduino. Utilisez un MOSFET ou un relais, avec une **diode de roue libre** et une **masse commune**. Mettez un fusible de 1 à 2 A sur le 12 V. Ne faites aucun câblage maison sur le secteur 120 V.

Le programme impose un **intervalle minimal entre deux expansions** (20 s par défaut, jamais moins de 5 s). Cette pause protège la vanne contre l'échauffement et laisse la chambre se rééquilibrer. Ne la contournez pas.

---

## Matériel

| Élément | Exemple | Remarque |
|---|---|---|
| Carte | Arduino Uno ou Nano | 5 V, ATmega328P |
| Électrovanne | 2/2 voies, **normalement fermée**, 12 V CC, compatible vide | ≈ 0,5 à 1 A |
| Commutation | MOSFET « logic level » IRLZ44N (ou module MOSFET, ou module relais 5 V) | + résistances de 220 Ω et 10 kΩ |
| Diode de roue libre | 1N4007 | **obligatoire**, aux bornes de la bobine |
| Alimentation | 12 V CC, 2 A, avec fusible | masse reliée au GND de l'Arduino |
| Bouton poussoir | n'importe lequel | entre D2 et GND |
| Flash (optionnel) | ruban de DEL 12 V + 2ᵉ MOSFET | sur D8 |
| Capteur de pression (optionnel) | MPX4115A (absolu, 15 à 115 kPa) | sur A0 |
| Réservoir sous vide | dessiccateur ou fiole à vide + pompe | **conçu pour le vide** |

## Branchements

| Broche Arduino | Relié à |
|---|---|
| **D7** | grille du MOSFET de la vanne (par 220 Ω ; 10 kΩ entre la grille et GND), ou entrée `IN` d'un module relais |
| **D2** | bouton poussoir, l'autre patte au **GND** (résistance de tirage interne) |
| **D8** | grille du MOSFET du flash (optionnel) |
| **A0** | sortie `Vout` du capteur de pression (optionnel ; capteur alimenté en 5 V) |
| **GND** | GND de l'alimentation 12 V (**masse commune**) |
| D13 (DEL intégrée) | témoin : allumée tant que la vanne est ouverte |

Le schéma détaillé (MOSFET, diode, relais, capteur) est dessiné en ASCII au début du fichier `.ino`. La physique de l'expansion et le traitement des traces par WilsonTrack sont décrits dans `docs/METHODE.md` (sections 1.2 et 6).

> **Module relais « actif à l'état bas »** : c'est le cas de beaucoup de modules bon marché. Dans le `.ino`, mettez `VANNE_ACTIVE_NIVEAU_HAUT = false`. Sinon, la vanne s'ouvrirait au repos. **Vérifiez sans vide** avant le premier essai réel.

## Téléverser le programme

1. Ouvrez `declencheur_expansion/declencheur_expansion.ino` dans l'IDE Arduino (version 1.8 ou 2.x).
2. Choisissez **Outils › Carte** (Arduino Uno ou Nano) et **Outils › Port** (COM3, COM4…).
3. Cliquez sur **Téléverser**. Aucune bibliothèque externe n'est nécessaire.
4. Options en haut du fichier : `UTILISER_CAPTEUR_PRESSION` (0 ou 1), les niveaux actifs, les valeurs par défaut et l'étalonnage du capteur.

## Tester seul avec le moniteur série (sans vide d'abord !)

Ouvrez **Outils › Moniteur série**, à **115200 bauds**, avec la fin de ligne réglée sur **« Nouvelle ligne »**. Environ 3 s après la mise sous tension, l'Arduino affiche `OK pret`.

| Commande | Effet | Réponse |
|---|---|---|
| `E` | déclenche une expansion | `EXP 123456` (millis de l'Arduino), puis `OK vanne fermee apres 150 ms` |
| `D 150` | durée d'ouverture de la vanne (10 à 2000 ms) | `OK D 150 ms` |
| `F 40` | délai du flash après l'ouverture (0 à 1000 ms) | `OK F 40 ms` |
| `L 30` | durée du flash (0 = pas de flash, max 1000 ms) | `OK L 30 ms` |
| `M 20` | intervalle minimal entre expansions (5 à 3600 s) | `OK M 20 s` |
| `C 1` / `C 0` | active ou désactive le capteur de pression | `OK C 1`, puis `P 101.3` deux fois par seconde |
| `A` | **arrêt immédiat** : ferme la vanne, annule le flash | `OK A vanne fermee, flash annule` |
| `S` | état et réglages | lignes `# ...` |
| `?` ou `H` | aide | lignes `# ...` |

Si vous tapez une commande sans valeur (`D`, `M`…), l'Arduino affiche la valeur actuelle. Une demande refusée (trop tôt, valeur hors limites…) donne une ligne `ERR ...`. Le bouton poussoir a le même effet que `E`.

Les réglages ne sont **pas mémorisés** : ils reviennent aux valeurs par défaut à chaque redémarrage. Pour changer une valeur par défaut, modifiez la constante correspondante dans le `.ino`.

**Premier essai conseillé** : sans réservoir sous vide, vérifiez que la vanne claque une seule fois par `E`, qu'elle se referme, et qu'un second `E` envoyé trop tôt est refusé (`ERR trop tot ...`).

## Utilisation avec WilsonTrack

1. **Fermez le moniteur série de l'IDE Arduino.** Sous Windows, un port COM ne peut être ouvert que par un seul programme à la fois.
2. Dans WilsonTrack, ouvrez **Paramètres › Chambre à expansion** :
   - cochez **Mode chambre à expansion** ;
   - pour **Déclenchement**, choisissez **« série (Arduino) »** ;
   - pour **Port série**, choisissez le port de l'Arduino (par ex. `COM4`). La vitesse est de 115200 bauds.
   - **Délai** : temps entre l'expansion et le début du comptage (40 ms par défaut) ;
   - **Fenêtre sensible** : durée pendant laquelle les nouvelles traces sont comptées (600 ms par défaut).
3. Démarrez la caméra. WilsonTrack ouvre alors le port série, ce qui **redémarre l'Arduino** (Uno et Nano). **Attendez environ 3 s** (armement) avant la première expansion.
4. Déclenchez une expansion avec le bouton poussoir, la touche **E** ou le bouton **💥 Expansion**. Dans les trois cas, c'est la ligne `EXP` envoyée par l'Arduino qui marque le début de la fenêtre.

Remarques :

- Si l'Arduino refuse une demande (par exemple parce que l'intervalle minimal n'est pas écoulé), il n'envoie pas `EXP`, et WilsonTrack **n'ouvre pas** de fenêtre sensible. Attendez puis recommencez.
- Réglez le **flash** (`F`) sur la même valeur que le **Délai** de WilsonTrack. Ainsi, l'éclairage et le comptage commencent au même moment.
- La liaison USB ajoute quelques millisecondes de retard, ce qui est négligeable devant une fenêtre de plusieurs centaines de millisecondes.
- WilsonTrack reconnaît une expansion à toute ligne qui **commence par `EXP`**. Le programme n'envoie aucune autre ligne commençant ainsi : les lignes d'information commencent par `#`.
- Sans Arduino, le mode « auto » de WilsonTrack détecte l'expansion par le saut de luminosité de l'image, et le mode « manuel » la marque avec la touche E.

## Étalonnage du capteur de pression (optionnel)

Le programme suppose un **MPX4115A** : `Vout = Vs × (0,009 × P − 0,095)`. Pour un autre capteur, modifiez `CAPTEUR_PENTE` et `CAPTEUR_DECALAGE` d'après sa fiche technique. Pour corriger le zéro, comparez la valeur affichée à la pression atmosphérique du jour (Environnement Canada, en kPa) et ajustez `CAPTEUR_CORRECTION_KPA`. Les lignes `P <kPa>` servent à surveiller le vide du réservoir. WilsonTrack les ignore.
