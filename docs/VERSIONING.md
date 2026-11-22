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
  Changelog*), insérée comme **première** section `## [X.Y.Z]` du
  fichier (c'est celle que lit la porte CI).
- `scripts/check-version.sh` (porte CI) vérifie : semver strict,
  cohérence `CHANGELOG.md`, cohérence du dernier tag `v*`.

## Utilisation

```sh
scripts/release.sh patch                 # 0.1.0 -> 0.1.1, ce dépôt
scripts/release.sh minor                 # 0.1.0 -> 0.2.0
scripts/release.sh iterate               # itération auto : 0.1.0 -> 0.1.1 -> 0.1.2 ...
scripts/release.sh 1.0.0                 # version explicite
scripts/release.sh 1.0.0 --all ~/gnu-ai  # toute la pile de l'org
scripts/version.sh                       # version courante
scripts/version.sh --next                # prochaine version itérative
sh scripts/check-version.sh              # vérif locale (aussi en CI)
git push origin main v1.0.0              # après une release
```

## Versionnage itératif

La version de départ est **0.1.0**. À chaque itération de développement
(un cycle « corriger → tester → livrer »), on incrémente le patch
automatiquement :

```sh
scripts/version.sh --next                          # prochaine version : 0.1.0 -> 0.1.1
scripts/release.sh iterate                         # coupe l'itération : 0.1.0 -> 0.1.1
scripts/release.sh iterate --all ~/clones-org      # toute la pile d'un coup
git push origin main v0.1.1                        # pousser branche + tag
```

Séquence : 0.1.0 → 0.1.1 → 0.1.2 → … Les jalons plus larges restent
disponibles : `minor` (0.1.x → 0.2.0), `major` (→ 1.0.0) ou une version
explicite `X.Y.Z`. `iterate` vaut sémantiquement `patch`, mais le nom
rend le cycle itératif explicite dans l'historique (`release: v0.1.1`,
`v0.1.2`, …). À chaque itération, la porte CI `check-version.sh` garde
`VERSION`, `CHANGELOG.md` et le dernier tag `v*` cohérents, et le
binaire affiche la nouvelle version via `PACKAGE_VERSION` après
recompilation (ex. bannière « inference (GNU AI) 0.1.1 »).

## Pourquoi un fichier VERSION

Avant, la version était écrite à la main dans `AC_INIT` *et* en dur
dans le code (`#define VERSION "0.1.0"`) : les deux divergeaient à
chaque release oubliée. Désormais `autogen.sh` régénère `configure`
depuis `VERSION`, `config.h` porte `PACKAGE_VERSION`, et le binaire
affiche la bonne version via `--version`. Un seul endroit à éditer,
zéro divergence possible.
