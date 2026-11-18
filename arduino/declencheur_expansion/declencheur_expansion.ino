/*
 * =============================================================================
 *  WilsonTrack — Déclencheur de chambre à EXPANSION (chambre de Wilson « vraie »)
 *  Fichier  : declencheur_expansion.ino           Version : 1.0
 *  Cartes   : Arduino Uno / Nano (ATmega328P). Fonctionne aussi sur Mega.
 *  Projet   : Cégep Marie-Victorin — détection de particules (fond, KCl, banane)
 * =============================================================================
 *
 *  À QUOI SERT CE PROGRAMME ?
 *  --------------------------
 *  Dans une chambre à EXPANSION, la sursaturation de la vapeur d'alcool est
 *  obtenue par une détente brusque (adiabatique) du gaz : on ouvre pendant un
 *  court instant une électrovanne qui relie la chambre à un RÉSERVOIR SOUS VIDE.
 *  Le gaz se refroidit, la vapeur devient sursaturée et, pendant une fraction de
 *  seconde, les ions laissés par les particules servent de germes à des
 *  gouttelettes : les traces apparaissent.
 *
 *  L'Arduino :
 *    1. ouvre l'électrovanne (broche 7) pendant une durée réglable ;
 *    2. allume optionnellement un flash de DEL (broche 8) après un délai réglable,
 *       pour éclairer les traces au bon moment ;
 *    3. envoie « EXP <millis> » au PC à CHAQUE expansion, pour que WilsonTrack
 *       sache exactement quand la « fenêtre sensible » commence ;
 *    4. impose un intervalle minimal entre deux expansions (sécurité) ;
 *    5. peut lire un capteur de pression analogique (A0) et afficher « P <kPa> ».
 *
 *  Une expansion peut être déclenchée :
 *    - par le bouton poussoir (broche 2) ;
 *    - par le PC : ligne « E » reçue sur le port série (WilsonTrack le fait quand
 *      on appuie sur la touche E ou sur le bouton « Déclencher une expansion »).
 *
 * =============================================================================
 *  !!!  AVERTISSEMENTS DE SÉCURITÉ — À LIRE AVANT TOUT BRANCHEMENT  !!!
 * =============================================================================
 *  1. VIDE = RISQUE D'IMPLOSION. Le réservoir sous vide doit être CONÇU pour le
 *     vide : dessiccateur à vide, fiole à vide à paroi épaisse, réservoir
 *     métallique. JAMAIS de bouteille en plastique mince, de bocal ordinaire ni
 *     de verrerie fêlée ou rayée. Entourez le verre de ruban adhésif ou d'un
 *     grillage et travaillez derrière un écran de protection.
 *  2. LUNETTES DE SÉCURITÉ obligatoires pour toutes les personnes à proximité
 *     (implosion, éclats de verre, projection d'alcool).
 *  3. NE JAMAIS LAISSER LE MONTAGE FONCTIONNER SANS SURVEILLANCE. À la fin,
 *     débranchez l'alimentation 12 V : l'électrovanne « normalement fermée »
 *     (NF) se referme d'elle-même sans courant. Travaillez sous la supervision
 *     d'un enseignant ou d'un technicien.
 *  4. ALCOOL INFLAMMABLE (isopropanol) : aucune flamme ; éloignez les relais et
 *     contacts (qui font des étincelles) des vapeurs d'alcool ; aérez la pièce.
 *  5. GLACE SÈCHE (−78 °C) : gants isolants et pinces ; le CO2 est asphyxiant
 *     dans une petite pièce fermée ; ne JAMAIS l'enfermer dans un contenant
 *     hermétique (surpression → explosion).
 *  6. ÉLECTRICITÉ : ne JAMAIS alimenter la bobine de l'électrovanne directement
 *     par une broche de l'Arduino (40 mA max) : utilisez un MOSFET ou un module
 *     relais. Diode de roue libre OBLIGATOIRE aux bornes de la bobine. Masse
 *     (GND) commune entre l'Arduino et l'alimentation 12 V. Mettez un fusible
 *     (1 à 2 A) sur le +12 V. Pas de câblage « maison » sur le secteur 120 V
 *     (pompe à vide : utilisez-la telle que vendue, avec sa prise d'origine).
 *  7. Ce programme N'EST PAS un dispositif de sécurité certifié. La sécurité
 *     repose d'abord sur le matériel (réservoir adapté, vanne NF, fusible) et
 *     sur la surveillance humaine. L'intervalle minimal entre expansions protège
 *     la vanne (échauffement de la bobine) et laisse à la chambre le temps de se
 *     rééquilibrer : ne le contournez pas.
 *
 * =============================================================================
 *  SCHÉMA DE CÂBLAGE (ASCII)
 * =============================================================================
 *
 *  (A) ÉLECTROVANNE 12 V NF commandée par un MOSFET canal N « logic level »
 *      (IRLZ44N, IRL540N, ou module MOSFET prêt à l'emploi)
 *
 *            +12 V (alimentation de la vanne, avec fusible 1-2 A)
 *              |
 *              +--------------+
 *              |              |
 *          [ BOBINE ]       --+--  diode 1N4007 (roue libre)
 *        électrovanne        /_\   bande (cathode) du côté +12 V
 *         12 V NF             |
 *              |              |
 *              +--------------+
 *              |
 *              D (drain)
 *   D7 --[220 Ω]--+-- G (grille)     MOSFET IRLZ44N
 *                 |   S (source)
 *              [10 kΩ]    |
 *                 |       |
 *   GND ----------+-------+---------- GND de l'alimentation 12 V (MASSE COMMUNE)
 *
 *      Variante : MODULE RELAIS 5 V
 *        IN -> D7,  VCC -> 5V,  GND -> GND ;  contacts COM/NO en série avec
 *        le +12 V de la vanne ; diode 1N4007 aux bornes de la bobine de la vanne.
 *        Beaucoup de modules relais sont « actifs à l'état BAS » : mettez alors
 *        VANNE_ACTIVE_NIVEAU_HAUT à false (voir plus bas).
 *
 *  (B) FLASH DE DEL (optionnel) — même principe avec un 2e MOSFET
 *
 *     +12 V --[ ruban DEL 12 V (ou DEL de puissance + résistance) ]-- D
 *     D8 --[220 Ω]--+-- G      MOSFET #2
 *                [10 kΩ]  S
 *                   |     |
 *     GND ----------+-----+-------- GND commun
 *
 *  (C) BOUTON POUSSOIR (résistance de tirage interne activée : INPUT_PULLUP)
 *
 *     D2 ----o  o---- GND        (appuyé = niveau BAS)
 *              ‾‾
 *  (D) CAPTEUR DE PRESSION ANALOGIQUE (optionnel), ex. MPX4115A (absolu, 15-115 kPa)
 *
 *     broche 1 (Vout) ---> A0      (condensateur 470 pF vers GND conseillé)
 *     broche 2 (GND)  ---> GND
 *     broche 3 (Vs)   ---> 5V      (condensateur 1 µF vers GND conseillé)
 *     Le capteur se raccorde au RÉSERVOIR (ou à la chambre) par un tuyau.
 *     Autre capteur : modifiez CAPTEUR_PENTE et CAPTEUR_DECALAGE (fiche technique).
 *
 *  (E) Témoin : la DEL intégrée (broche 13) est allumée tant que la vanne est ouverte.
 *
 * =============================================================================
 *  PROTOCOLE SÉRIE — 115200 bauds, lignes terminées par \n (ou \r\n)
 * =============================================================================
 *   PC -> Arduino
 *     E          déclenche une expansion
 *     D <ms>     durée d'ouverture de la vanne       (10 à 2000 ms, défaut 150)
 *     F <ms>     délai du flash après l'ouverture    (0 à 1000 ms,  défaut 40)
 *     L <ms>     durée du flash (0 = pas de flash)   (0 à 1000 ms,  défaut 30)
 *     M <s>      intervalle minimal entre expansions (5 à 3600 s,   défaut 20)
 *     C <0|1>    capteur de pression sur A0 : 1 = actif, 0 = inactif
 *     A          ARRÊT immédiat : ferme la vanne et annule le flash
 *     S          affiche l'état et les réglages
 *     ?  (ou H)  affiche l'aide
 *     Sans valeur, D, F, L, M et C affichent la valeur actuelle.
 *     S, ? et H sont refusés (ERR occupe) pendant une expansion, pour ne pas
 *     retarder la fermeture de la vanne par un long affichage.
 *
 *   Arduino -> PC
 *     EXP <millis>   une expansion vient de commencer (vanne ouverte à <millis> ms)
 *     P <kPa>        pression (2 fois par seconde si le capteur est actif)
 *     OK ...         accusé de réception d'une commande
 *     ERR ...        commande refusée (trop tôt, valeur hors limites, etc.)
 *     # ...          ligne d'information (aide, état, démarrage)
 *
 *   IMPORTANT : WilsonTrack considère TOUTE ligne qui commence par « EXP » comme
 *   une expansion. Aucune autre ligne envoyée par ce programme ne commence par
 *   ces trois lettres (les messages d'information commencent par « # »).
 *   Les messages série sont volontairement SANS ACCENTS (compatibles avec tous
 *   les terminaux) ; les commentaires, eux, sont en français accentué.
 *
 *  Toutes les temporisations utilisent la forme (millis() - debut) >= duree avec
 *  des entiers non signés : elle reste correcte quand millis() repasse à zéro
 *  (toutes les 49,7 jours). Aucune fonction delay() : le programme reste réactif
 *  (bouton, port série, arrêt d'urgence) même pendant une expansion.
 * =============================================================================
 */

#include <Arduino.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// OPTIONS DE COMPILATION
// -----------------------------------------------------------------------------
// Capteur de pression actif au démarrage ? (0 = non, 1 = oui). Modifiable ensuite
// par la commande « C 1 » / « C 0 ». Laisser à 0 si aucun capteur n'est branché :
// une entrée analogique « en l'air » donne des valeurs aléatoires.
#define UTILISER_CAPTEUR_PRESSION 0

// Chien de garde matériel (AVR seulement) : si le programme se bloquait, la carte
// redémarrerait en moins d'une seconde et setup() refermerait la vanne.
// Laissé à 0 par défaut : certains vieux Nano (ancien chargeur d'amorçage)
// redémarrent en boucle quand le chien de garde est utilisé.
#define UTILISER_CHIEN_DE_GARDE 0

#if UTILISER_CHIEN_DE_GARDE && defined(__AVR__)
#include <avr/wdt.h>
#endif

// -----------------------------------------------------------------------------
// BROCHES
// -----------------------------------------------------------------------------
const uint8_t BROCHE_VANNE    = 7;            // MOSFET ou relais de l'électrovanne
const uint8_t BROCHE_BOUTON   = 2;            // bouton poussoir vers GND
const uint8_t BROCHE_FLASH    = 8;            // MOSFET du flash de DEL (optionnel)
const uint8_t BROCHE_PRESSION = A0;           // capteur de pression analogique (optionnel)
const uint8_t BROCHE_TEMOIN   = LED_BUILTIN;  // DEL intégrée (13) : vanne ouverte

// Niveau logique qui ACTIVE la sortie.
// true  : MOSFET ou relais « actif à l'état HAUT » (cas du schéma ci-dessus).
// false : module relais « actif à l'état BAS » (IN = LOW -> relais collé).
const bool VANNE_ACTIVE_NIVEAU_HAUT = true;
const bool FLASH_ACTIF_NIVEAU_HAUT  = true;

// -----------------------------------------------------------------------------
// RÉGLAGES PAR DÉFAUT ET LIMITES DE SÉCURITÉ
// -----------------------------------------------------------------------------
const uint32_t DUREE_VANNE_DEFAUT_MS  = 150;   // ouverture typique : 100 à 300 ms
const uint32_t DUREE_VANNE_MIN_MS     = 10;
const uint32_t DUREE_VANNE_MAX_MS     = 2000;  // au-delà : échauffement inutile de la bobine

const uint32_t DELAI_FLASH_DEFAUT_MS  = 40;    // = « Délai » par défaut de WilsonTrack
const uint32_t DELAI_FLASH_MAX_MS     = 1000;  // les traces sont nettes dans la 1re demi-seconde

const uint32_t DUREE_FLASH_DEFAUT_MS  = 30;
const uint32_t DUREE_FLASH_MAX_MS     = 1000;

const uint32_t INTERVALLE_DEFAUT_S    = 20;    // intervalle minimal entre deux expansions
const uint32_t INTERVALLE_MIN_S       = 5;     // plancher non modifiable par le port série
const uint32_t INTERVALLE_MAX_S       = 3600;
// Note : dans le pire cas, une expansion occupe au plus
//   max(DUREE_VANNE_MAX_MS, DELAI_FLASH_MAX_MS + DUREE_FLASH_MAX_MS) = 2 s  <  INTERVALLE_MIN_S = 5 s,
// donc deux expansions ne peuvent jamais se chevaucher.

const uint32_t ANTI_REBOND_MS         = 30;    // stabilité exigée du bouton
const uint32_t DELAI_ARMEMENT_MS      = 3000;  // aucune expansion pendant 3 s après la mise sous tension
const uint32_t PERIODE_PRESSION_MS    = 500;   // « P <kPa> » deux fois par seconde

// -----------------------------------------------------------------------------
// ÉTALONNAGE DU CAPTEUR DE PRESSION
// -----------------------------------------------------------------------------
// Capteur ratiométrique (sortie proportionnelle à l'alimentation Vs).
// MPX4115A / MPXA6115A : Vout = Vs × (0,009 × P − 0,095), P en kPa
//   => P = (Vout/Vs + 0,095) / 0,009
// Comme l'ADC de l'Arduino mesure aussi par rapport à Vs (5 V), le rapport
// Vout/Vs = lecture/1023 ne dépend pas de la tension exacte d'alimentation.
const float   CAPTEUR_PENTE          = 0.009f;  // fraction de Vs par kPa
const float   CAPTEUR_DECALAGE       = 0.095f;  // fraction de Vs
const float   CAPTEUR_CORRECTION_KPA = 0.0f;    // correction du zéro (comparer à la pression atmosphérique)
const uint8_t NB_ECHANTILLONS_ADC    = 8;       // moyenne de 8 lectures (réduit le bruit)

const char VERSION_PROGRAMME[] = "1.0";

// -----------------------------------------------------------------------------
// VARIABLES D'ÉTAT
// -----------------------------------------------------------------------------
// Réglages modifiables par le port série
uint32_t dureeVanneMs   = DUREE_VANNE_DEFAUT_MS;
uint32_t delaiFlashMs   = DELAI_FLASH_DEFAUT_MS;
uint32_t dureeFlashMs   = DUREE_FLASH_DEFAUT_MS;
uint32_t intervalleMinS = INTERVALLE_DEFAUT_S;
bool     capteurActif   = (UTILISER_CAPTEUR_PRESSION != 0);

// Expansion en cours
bool     arme                = false;  // devient vrai DELAI_ARMEMENT_MS après le démarrage
bool     vanneOuverte        = false;
bool     dejaExpanse         = false;  // au moins une expansion depuis le démarrage ?
uint32_t tExpansionMs        = 0;      // instant (millis) de la dernière ouverture de vanne
uint32_t dureeEnCoursMs      = 0;      // durée figée au moment du déclenchement
uint32_t nbExpansions        = 0;

// Flash
bool     flashEnAttente      = false;
bool     flashAllume         = false;
uint32_t delaiFlashEnCoursMs = 0;
uint32_t dureeFlashEnCoursMs = 0;
uint32_t tFlashMs            = 0;

// Bouton (anti-rebond)
int      boutonBrut          = HIGH;
int      boutonStable        = HIGH;
uint32_t tChangementBrutMs   = 0;

// Capteur de pression
uint32_t tDernierePressionMs = 0;

// Réception série (une ligne à la fois)
const uint8_t TAILLE_LIGNE = 32;
char     ligne[TAILLE_LIGNE];
uint8_t  longueurLigne   = 0;
bool     ligneTropLongue = false;

// =============================================================================
// SORTIES
// =============================================================================
void ecrireVanne(bool ouverte) {
  // ouverte == niveau actif -> HIGH ; sinon LOW
  digitalWrite(BROCHE_VANNE, (ouverte == VANNE_ACTIVE_NIVEAU_HAUT) ? HIGH : LOW);
  digitalWrite(BROCHE_TEMOIN, ouverte ? HIGH : LOW);
}

void ecrireFlash(bool allume) {
  digitalWrite(BROCHE_FLASH, (allume == FLASH_ACTIF_NIVEAU_HAUT) ? HIGH : LOW);
}

// Vrai tant que la vanne est ouverte ou que le flash n'est pas terminé.
// Pendant ce temps, on évite les longs affichages (aide, état) : le tampon
// d'émission série ne fait que 64 octets et un long texte bloquerait la boucle
// (~80 ms pour l'aide à 115200 bauds), ce qui retarderait la fermeture de la vanne.
bool expansionEnCours() {
  return vanneOuverte || flashEnAttente || flashAllume;
}

// =============================================================================
// AFFICHAGE : AIDE ET ÉTAT
// =============================================================================
void afficherAide() {
  Serial.println(F("# ---- Declencheur d'expansion WilsonTrack : commandes (terminer par Entree) ----"));
  Serial.println(F("# E         declencher une expansion (reponse : EXP <millis>)"));
  Serial.print  (F("# D <ms>    duree d'ouverture de la vanne (10 a 2000 ms), actuel : "));
  Serial.println(dureeVanneMs);
  Serial.print  (F("# F <ms>    delai du flash apres l'ouverture (0 a 1000 ms), actuel : "));
  Serial.println(delaiFlashMs);
  Serial.print  (F("# L <ms>    duree du flash (0 = pas de flash, max 1000 ms), actuel : "));
  Serial.println(dureeFlashMs);
  Serial.print  (F("# M <s>     intervalle minimal entre deux expansions (5 a 3600 s), actuel : "));
  Serial.println(intervalleMinS);
  Serial.print  (F("# C <0|1>   capteur de pression A0 (affiche P <kPa> 2 fois/s), actuel : "));
  Serial.println(capteurActif ? 1 : 0);
  Serial.println(F("# A         ARRET immediat : ferme la vanne, annule le flash"));
  Serial.println(F("# S         etat et reglages"));
  Serial.println(F("# ? ou H    cette aide"));
  Serial.println(F("# Sans valeur, D F L M C affichent la valeur actuelle. Bouton (broche 2) = commande E."));
  Serial.println(F("# SECURITE : lunettes, reservoir concu pour le vide, jamais sans surveillance."));
}

void afficherEtat() {
  Serial.print(F("# Version "));
  Serial.print(VERSION_PROGRAMME);
  Serial.print(F(" | temps "));
  Serial.print(millis());
  Serial.print(F(" ms | expansions : "));
  Serial.println(nbExpansions);

  Serial.print(F("# Vanne : "));
  if (vanneOuverte) {
    Serial.println(F("OUVERTE"));
  } else {
    Serial.println(F("fermee"));
  }

  Serial.print(F("# Pret : "));
  if (!arme) {
    Serial.println(F("non (demarrage en cours)"));
  } else if (vanneOuverte) {
    Serial.println(F("non (expansion en cours)"));
  } else {
    const uint32_t intervalleMs = intervalleMinS * 1000UL;
    const uint32_t ecoule = millis() - tExpansionMs;          // sûr au débordement
    if (dejaExpanse && ecoule < intervalleMs) {
      Serial.print(F("dans "));
      Serial.print((intervalleMs - ecoule) / 1000.0, 1);
      Serial.println(F(" s"));
    } else {
      Serial.println(F("oui"));
    }
  }

  Serial.print(F("# D = "));  Serial.print(dureeVanneMs);   Serial.println(F(" ms (ouverture vanne)"));
  Serial.print(F("# F = "));  Serial.print(delaiFlashMs);   Serial.println(F(" ms (delai flash)"));
  Serial.print(F("# L = "));  Serial.print(dureeFlashMs);   Serial.println(F(" ms (duree flash, 0 = aucun)"));
  Serial.print(F("# M = "));  Serial.print(intervalleMinS); Serial.println(F(" s (intervalle minimal)"));
  Serial.print(F("# C = "));  Serial.println(capteurActif ? 1 : 0);
}

// =============================================================================
// EXPANSION
// =============================================================================
// Renvoie true si l'expansion a réellement commencé.
bool declencherExpansion(const __FlashStringHelper* origine) {
  if (!arme) {
    Serial.print(F("ERR pas encore pret (demarrage), demande ignoree : "));
    Serial.println(origine);
    return false;
  }
  if (vanneOuverte) {
    Serial.print(F("ERR expansion deja en cours, demande ignoree : "));
    Serial.println(origine);
    return false;
  }

  const uint32_t maintenant   = millis();
  const uint32_t intervalleMs = intervalleMinS * 1000UL;
  if (dejaExpanse) {
    const uint32_t ecoule = maintenant - tExpansionMs;         // sûr au débordement
    if (ecoule < intervalleMs) {
      Serial.print(F("ERR trop tot : attendre encore "));
      Serial.print((intervalleMs - ecoule) / 1000.0, 1);
      Serial.print(F(" s (intervalle minimal "));
      Serial.print(intervalleMinS);
      Serial.print(F(" s), demande ignoree : "));
      Serial.println(origine);
      return false;
    }
  }

  // --- 1) Ouverture de la vanne : c'est l'instant de l'expansion -------------
  ecrireVanne(true);
  vanneOuverte   = true;
  tExpansionMs   = maintenant;
  dureeEnCoursMs = dureeVanneMs;       // un « D » reçu pendant l'expansion n'a pas d'effet sur celle-ci
  dejaExpanse    = true;
  nbExpansions++;

  // --- 2) Programmation du flash (s'il est utilisé) --------------------------
  flashAllume = false;
  ecrireFlash(false);
  if (dureeFlashMs > 0) {
    flashEnAttente      = true;
    delaiFlashEnCoursMs = delaiFlashMs;
    dureeFlashEnCoursMs = dureeFlashMs;
  } else {
    flashEnAttente = false;
  }

  // --- 3) Message au PC, IMMÉDIATEMENT (WilsonTrack démarre sa fenêtre) ------
  Serial.print(F("EXP "));
  Serial.println(maintenant);
  Serial.print(F("# expansion n. "));
  Serial.print(nbExpansions);
  Serial.print(F(" ("));
  Serial.print(origine);
  Serial.println(F(")"));
  return true;
}

void arretUrgence() {
  ecrireVanne(false);
  vanneOuverte = false;
  flashEnAttente = false;
  flashAllume = false;
  ecrireFlash(false);
  Serial.println(F("OK A vanne fermee, flash annule"));
}

// =============================================================================
// COMMANDES SÉRIE
// =============================================================================
// Lit un entier décimal (éventuellement précédé d'espaces, de « = » ou « : »).
// Renvoie false si la chaîne ne contient pas exactement un nombre.
bool lireEntier(const char* s, long* valeur) {
  while (*s == ' ' || *s == '\t' || *s == '=' || *s == ':') {
    s++;
  }
  if (*s == '\0') {
    return false;
  }
  char* fin = NULL;
  const long v = strtol(s, &fin, 10);
  if (fin == s) {
    return false;                       // aucun chiffre lu
  }
  while (*fin == ' ' || *fin == '\t') {
    fin++;
  }
  if (*fin != '\0') {
    return false;                       // caractères parasites après le nombre
  }
  *valeur = v;
  return true;
}

// Sans valeur : affiche la valeur actuelle. Avec valeur : vérifie les limites puis
// l'enregistre. Renvoie true seulement si une nouvelle valeur a été enregistrée.
bool reglerParametre(char nom, const char* arg, uint32_t* cible, long vmin, long vmax,
                     const __FlashStringHelper* unite) {
  while (*arg == ' ' || *arg == '\t') {
    arg++;
  }
  if (*arg == '\0') {                   // simple lecture
    Serial.print(F("OK "));
    Serial.print(nom);
    Serial.print(' ');
    Serial.print(*cible);
    if (unite != NULL) {
      Serial.print(' ');
      Serial.print(unite);
    }
    Serial.println();
    return false;
  }
  long v = 0;
  if (!lireEntier(arg, &v)) {
    Serial.print(F("ERR valeur invalide pour "));
    Serial.print(nom);
    Serial.println(F(" (nombre entier attendu)"));
    return false;
  }
  if (v < vmin || v > vmax) {
    Serial.print(F("ERR "));
    Serial.print(nom);
    Serial.print(F(" doit etre entre "));
    Serial.print(vmin);
    Serial.print(F(" et "));
    Serial.print(vmax);
    if (unite != NULL) {
      Serial.print(' ');
      Serial.print(unite);
    }
    Serial.println();
    return false;
  }
  *cible = (uint32_t)v;
  Serial.print(F("OK "));
  Serial.print(nom);
  Serial.print(' ');
  Serial.print(*cible);
  if (unite != NULL) {
    Serial.print(' ');
    Serial.print(unite);
  }
  Serial.println();
  return true;
}

void traiterLigne(char* l) {
  // Retire les espaces au début et à la fin
  while (*l == ' ' || *l == '\t') {
    l++;
  }
  size_t n = strlen(l);
  while (n > 0 && (l[n - 1] == ' ' || l[n - 1] == '\t')) {
    l[--n] = '\0';
  }
  if (n == 0) {
    return;
  }

  const char  cmd = (char)toupper((unsigned char)l[0]);
  const char* arg = l + 1;

  switch (cmd) {
    case 'E': {
      while (*arg == ' ' || *arg == '\t') {
        arg++;
      }
      if (*arg != '\0') {               // ex. un écho « EXP ... » renvoyé par erreur
        Serial.println(F("ERR la commande E ne prend pas d'argument"));
        break;
      }
      declencherExpansion(F("serie"));
      break;
    }
    case 'D':
      reglerParametre('D', arg, &dureeVanneMs, (long)DUREE_VANNE_MIN_MS, (long)DUREE_VANNE_MAX_MS, F("ms"));
      break;
    case 'F':
      reglerParametre('F', arg, &delaiFlashMs, 0L, (long)DELAI_FLASH_MAX_MS, F("ms"));
      break;
    case 'L':
      reglerParametre('L', arg, &dureeFlashMs, 0L, (long)DUREE_FLASH_MAX_MS, F("ms"));
      break;
    case 'M':
      reglerParametre('M', arg, &intervalleMinS, (long)INTERVALLE_MIN_S, (long)INTERVALLE_MAX_S, F("s"));
      break;
    case 'C': {
      uint32_t v = capteurActif ? 1UL : 0UL;
      if (reglerParametre('C', arg, &v, 0L, 1L, NULL)) {
        capteurActif = (v != 0);
        if (capteurActif) {
          tDernierePressionMs = millis() - PERIODE_PRESSION_MS;   // première mesure immédiate
        }
      }
      break;
    }
    case 'A':
      arretUrgence();
      break;
    case 'S':
    case '?':
    case 'H':
      if (expansionEnCours()) {
        Serial.println(F("ERR occupe (expansion en cours), reessayez dans 1 s"));
      } else if (cmd == 'S') {
        afficherEtat();
      } else {
        afficherAide();
      }
      break;
    default:
      Serial.print(F("ERR commande inconnue : "));
      Serial.print(l);
      Serial.println(F(" (tapez ? pour l'aide)"));
      break;
  }
}

void lireSerie() {
  while (Serial.available() > 0) {
    const int c = Serial.read();
    if (c < 0) {
      break;
    }
    if (c == '\n' || c == '\r') {
      if (ligneTropLongue) {
        Serial.println(F("ERR ligne trop longue, ignoree"));
      } else if (longueurLigne > 0) {
        ligne[longueurLigne] = '\0';
        traiterLigne(ligne);
      }
      longueurLigne = 0;                 // un \r\n donne une 2e ligne vide, ignorée
      ligneTropLongue = false;
    } else if (longueurLigne < TAILLE_LIGNE - 1) {
      ligne[longueurLigne++] = (char)c;
    } else {
      ligneTropLongue = true;            // on continue de lire jusqu'à la fin de ligne
    }
  }
}

// =============================================================================
// BOUTON (anti-rebond logiciel)
// =============================================================================
void lireBouton() {
  const int brut = digitalRead(BROCHE_BOUTON);
  const uint32_t maintenant = millis();
  if (brut != boutonBrut) {              // le signal bouge : on relance le chronomètre
    boutonBrut = brut;
    tChangementBrutMs = maintenant;
  }
  if ((uint32_t)(maintenant - tChangementBrutMs) >= ANTI_REBOND_MS && brut != boutonStable) {
    boutonStable = brut;                 // stable depuis ANTI_REBOND_MS : nouvel état validé
    if (boutonStable == LOW) {           // front descendant = appui
      declencherExpansion(F("bouton"));
    }
  }
}

// =============================================================================
// GESTION DU TEMPS : fermeture de la vanne, flash
// =============================================================================
void gererSorties() {
  const uint32_t maintenant = millis();

  // Fermeture de la vanne à la fin de la durée programmée
  if (vanneOuverte && (uint32_t)(maintenant - tExpansionMs) >= dureeEnCoursMs) {
    ecrireVanne(false);
    vanneOuverte = false;
    Serial.print(F("OK vanne fermee apres "));
    Serial.print((uint32_t)(maintenant - tExpansionMs));
    Serial.println(F(" ms"));
  }

  // Allumage du flash après son délai (compté depuis l'ouverture de la vanne)
  if (flashEnAttente && (uint32_t)(maintenant - tExpansionMs) >= delaiFlashEnCoursMs) {
    flashEnAttente = false;
    flashAllume = true;
    tFlashMs = maintenant;
    ecrireFlash(true);
  }

  // Extinction du flash
  if (flashAllume && (uint32_t)(maintenant - tFlashMs) >= dureeFlashEnCoursMs) {
    flashAllume = false;
    ecrireFlash(false);
  }
}

// =============================================================================
// CAPTEUR DE PRESSION
// =============================================================================
void gererPression() {
  if (!capteurActif) {
    return;
  }
  const uint32_t maintenant = millis();
  if ((uint32_t)(maintenant - tDernierePressionMs) < PERIODE_PRESSION_MS) {
    return;
  }
  tDernierePressionMs = maintenant;

  uint32_t somme = 0;
  for (uint8_t i = 0; i < NB_ECHANTILLONS_ADC; i++) {
    somme += (uint32_t)analogRead(BROCHE_PRESSION);
  }
  const float rapport = ((float)somme / (float)NB_ECHANTILLONS_ADC) / 1023.0f;   // Vout / Vs
  const float kpa = (rapport + CAPTEUR_DECALAGE) / CAPTEUR_PENTE + CAPTEUR_CORRECTION_KPA;
  Serial.print(F("P "));
  Serial.println(kpa, 1);
}

// =============================================================================
// INITIALISATION
// =============================================================================
void setup() {
  // 1) SÉCURITÉ D'ABORD : on écrit le niveau « repos » AVANT de passer la broche
  //    en sortie, pour que la vanne et le flash ne s'activent jamais, même un instant.
  digitalWrite(BROCHE_VANNE, VANNE_ACTIVE_NIVEAU_HAUT ? LOW : HIGH);
  pinMode(BROCHE_VANNE, OUTPUT);
  digitalWrite(BROCHE_FLASH, FLASH_ACTIF_NIVEAU_HAUT ? LOW : HIGH);
  pinMode(BROCHE_FLASH, OUTPUT);
  pinMode(BROCHE_TEMOIN, OUTPUT);
  digitalWrite(BROCHE_TEMOIN, LOW);

  // 2) Bouton : un bouton déjà enfoncé au démarrage ne déclenche rien ;
  //    il faudra le relâcher puis appuyer de nouveau.
  pinMode(BROCHE_BOUTON, INPUT_PULLUP);
  boutonBrut = digitalRead(BROCHE_BOUTON);
  boutonStable = boutonBrut;
  tChangementBrutMs = millis();

  // 3) Port série
  Serial.begin(115200);
  Serial.print(F("# WilsonTrack - declencheur d'expansion v"));
  Serial.println(VERSION_PROGRAMME);
  Serial.println(F("# SECURITE : lunettes obligatoires, reservoir concu pour le vide, jamais sans surveillance."));
  Serial.println(F("# Tapez ? pour l'aide. Armement dans 3 s..."));

  tDernierePressionMs = millis();

#if UTILISER_CHIEN_DE_GARDE && defined(__AVR__)
  wdt_enable(WDTO_1S);
#endif
}

// =============================================================================
// BOUCLE PRINCIPALE (non bloquante : quelques microsecondes par tour)
// =============================================================================
void loop() {
#if UTILISER_CHIEN_DE_GARDE && defined(__AVR__)
  wdt_reset();
#endif

  // Armement unique après le démarrage (laisse passer le chargeur d'amorçage et
  // les parasites de mise sous tension). Une fois armé, on le reste.
  if (!arme && millis() >= DELAI_ARMEMENT_MS) {
    arme = true;
    Serial.println(F("OK pret"));
  }

  lireSerie();       // commandes du PC
  lireBouton();      // bouton poussoir
  gererSorties();    // fermeture de la vanne, flash
  gererPression();   // « P <kPa> » deux fois par seconde (si actif)
}
