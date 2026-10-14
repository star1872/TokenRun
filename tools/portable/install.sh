#!/bin/sh
# TokenRun for Linux in one command (no sudo, nothing outside your home folder):
#   curl -fsSL https://www.modelscope.cn/models/mymodel3861/TokenRun/resolve/master/install.sh | sh
# It downloads TokenRun-linux-x64.tar.gz (ModelScope first - mainland China reaches it - then GitHub), puts it in
# ~/.local/share/tokenrun (or $TOKENRUN_HOME) and links the `tokenrun` command into ~/.local/bin.  Running it again
# updates the program; the models (Strata-data) and the engines already downloaded are kept.
set -eu
NAME=TokenRun-linux-x64.tar.gz
URLS="${TOKENRUN_URL:-}
https://www.modelscope.cn/models/mymodel3861/TokenRun/resolve/master/$NAME
https://github.com/star1872/TokenRun/releases/download/tokenrun-latest/$NAME"
DEST="${TOKENRUN_HOME:-$HOME/.local/share/tokenrun}"
BIN="$HOME/.local/bin"

say() { printf '%s\n' "$*"; }
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    say "TokenRun's ready-made package is for Linux on x86_64 (this is $(uname -s) $(uname -m))."
    say "From the source instead: git clone https://github.com/star1872/TokenRun && cd TokenRun && ./setup.sh"
    exit 1
fi
if command -v curl >/dev/null 2>&1; then get() { curl -fL --retry 2 --connect-timeout 15 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then get() { wget -O "$2" "$1"; }
else say "Needs curl or wget (e.g. sudo apt install curl)."; exit 1; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ok=
for url in $URLS; do
    say "Downloading $url"
    if get "$url" "$tmp/$NAME"; then ok=1; break; fi
    say "  could not download it from there"
done
[ -n "$ok" ] || { say "Could not download TokenRun (neither ModelScope nor GitHub answered)."; exit 1; }

mkdir -p "$tmp/new" "$DEST" "$BIN"
tar -xzf "$tmp/$NAME" -C "$tmp/new"
# the program (python/, app/, tokenrun) is replaced.  What the old app/ has that the package does not - the engines,
# the installed models' run configs and start scripts - moves into the new one; Strata-data beside it stays
if [ -d "$DEST/app" ]; then
    for e in "$DEST"/app/* "$DEST"/app/.[!.]*; do
        [ -e "$e" ] || continue
        [ -e "$tmp/new/TokenRun/app/$(basename "$e")" ] || mv "$e" "$tmp/new/TokenRun/app/"
    done
fi
rm -rf "$DEST/python" "$DEST/app" "$DEST/tokenrun"
mv "$tmp/new/TokenRun/python" "$tmp/new/TokenRun/app" "$tmp/new/TokenRun/tokenrun" "$DEST/"
for f in "$tmp/new/TokenRun/"*; do
    [ -e "$f" ] && cp -r "$f" "$DEST/"
done
ln -sf "$DEST/tokenrun" "$BIN/tokenrun"

say ""
say "TokenRun is installed in $DEST"
case ":$PATH:" in
    *":$BIN:"*) say "Start it with:  tokenrun" ;;
    *) say "Start it with:  $BIN/tokenrun"
       say "(or add ~/.local/bin to PATH - log in again, or: export PATH=\"\$HOME/.local/bin:\$PATH\" - then just: tokenrun)" ;;
esac
say "On a server without a screen: tokenrun --host 0.0.0.0, then open http://<this server>:8090/ from another PC."
