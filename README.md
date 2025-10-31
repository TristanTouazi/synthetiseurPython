# Synthesiseur Python

Application graphique permettant de generer une onde sinusoidale (ou carre/triangle), de visualiser son enveloppe, d'ajouter une reverb par convolution et d'exporter les signaux en CSV ou en WAV.

- Interface interactive creee avec `matplotlib`
- Reglages en temps reel des parametres (amplitude, frequence, phase, offset, amortissement, bruit, etc.)
- Choix independant de la forme d'onde principale et de l'impulsion de reverb
- Lecture audio optionnelle directement depuis l'application
- Export des courbes vers un CSV et du signal final vers un fichier WAV

## Prerequis

- Python 3.10 ou plus recent (recommande)
- Bibliotheques principales : `matplotlib`, `numpy`
- Lecture audio (facultatif) :
  - Option 1 : `sounddevice`
  - Option 2 : `simpleaudio` en definissant `USE_SIMPLEAUDIO=1`

Installez les dependances principales :

```bash
python -m pip install matplotlib numpy
```

Pour activer la lecture audio, ajoutez :

```bash
python -m pip install sounddevice simpleaudio
```

> Remarque : `sounddevice` s'appuie sur PortAudio. Selon votre plateforme, il peut etre necessaire d'installer la bibliotheque native correspondante (ex : `sudo apt install libportaudio2` sous Debian/Ubuntu).

## Installation rapide

1. (Optionnel) creer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows : venv\Scripts\activate
   ```
2. Installer les dependances (voir ci-dessus).

## Lancer l'application

```bash
python sinusoide.py
```

Une fenetre matplotlib s'ouvre avec :

- Les courbes du signal principal, de l'impulsion de reverb et du signal mixe.
- Des curseurs pour ajuster amplitude, frequence, phase, offset, amortissement et bruit.
- Des boutons radio pour choisir la forme d'onde (sinus, carre, triangle) pour le signal principal et la reverb.
- Des cases a cocher pour afficher/masquer chaque courbe.
- Des boutons pour lire le signal, exporter en CSV, exporter en WAV et arreter la lecture.

## Lecture et export

- `Lire` : lance la lecture du signal. Necessite `sounddevice` ou `simpleaudio` (avec `USE_SIMPLEAUDIO=1`).
- `Exporter CSV` : sauvegarde un fichier `sinusoide_YYYYMMDD_HHMMSS.csv` contenant le temps et les trois courbes.
- `Exporter WAV` : sauvegarde le signal mixe normalise dans `sinusoide_YYYYMMDD_HHMMSS.wav`.

## Configuration optionnelle

- `MPLBACKEND` : variable d'environnement permettant de choisir le backend matplotlib (par defaut `TkAgg` pour assurer la compatibilite multi-plateforme).
- `USE_SIMPLEAUDIO=1` : force l'utilisation de `simpleaudio` pour la lecture si `sounddevice` n'est pas disponible.

## Tests rapides

- Lancer `python sinusoide.py`, ajuster les curseurs et verifier que les courbes se mettent a jour.
- Cliquer sur `Exporter CSV` et `Exporter WAV`, puis confirmer la creation des fichiers dans le dossier du projet.
- (Optionnel) Tester la lecture audio si les dependances sont installees.
