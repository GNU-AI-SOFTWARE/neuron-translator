<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 Claire Ivanenka <claire@gnu-ai.org>
-->

# Versionning du projet (GNU AI)

## Règles

- Un seul fichier **`VERSION`** à la racine de chaque dépôt : la version
  semver `X.Y.Z` du paquet. C'est la **source de vérité**.
- `configure.ac` la lit : `AC_INIT([pkg], m4_esyscmd([cat VERSION | tr -d ' \n']), ...)`.
  Ne jamais éditer la version à la main dans `configure.ac` : le code
  fait `#define VERSION PACKAGE_VERSION` (venant de `config.h`).
- Les cinq translators forment une **pile** : une release de la pile
  bump *tous* les dépôts de l'org avec la même version :
  `scripts/release.sh 1.0.0 --all /chemin/des/clones`.
- Chaque release = commit `release: vX.Y.Z` + **tag annoté `vX.Y.Z`**
  (immuable, jamais réécrit, jamais forcé) + section
  `## [X.Y.Z] — YYYY-MM-DD` dans `CHANGELOG.md` (format *Keep a
  Changelog*), posée au-dessus d'un `## [Unreleased]` neuf.
- `scripts/check-version.sh` (porte CI) vérifie : semver strict,
  cohérence `CHANGELOG.md`, cohérence du dernier tag `v*`.

## Utilisation

```sh
scripts/release.sh patch                 # 0.1.0 -> 0.1.1, ce dépôt
scripts/release.sh minor                 # 0.1.0 -> 0.2.0
scripts/release.sh 1.0.0                 # version explicite
scripts/release.sh 1.0.0 --all ~/gnu-ai  # toute la pile de l'org
sh scripts/check-version.sh              # vérif locale (aussi en CI)
git push origin main v1.0.0              # après une release
```

## Pourquoi un fichier VERSION

Avant, la version était écrite à la main dans `AC_INIT` *et* en dur
dans le code (`#define VERSION "0.1.0"`) : les deux divergeaient à
chaque release oubliée. Désormais `autogen.sh` régénère `configure`
depuis `VERSION`, `config.h` porte `PACKAGE_VERSION`, et le binaire
affiche la bonne version via `--version`. Un seul endroit à éditer,
zéro divergence possible.
