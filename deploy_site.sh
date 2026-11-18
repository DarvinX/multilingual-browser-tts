#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p artifacts/logs artifacts/published
exec 9>artifacts/deploy.lock
flock -x 9

base=${BENGALI_BROWSER_DIR:-$HOME/tiny-voca/runs/bengali-browser/dist}
host=${TTS_DEPLOY_HOST:-ubuntu@140.245.219.255}
key=${TTS_DEPLOY_KEY:-$HOME/.ssh/id_ed25519_oracle}
remote=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -i "$key" "$host")
site=${TTS_SITE_URL:-https://tts.hansda.in}
python prepare_site.py "$base" artifacts/site
manifest_hash=$(sha256sum artifacts/site/languages.json | cut -d ' ' -f 1)
fingerprint=$(sha256sum artifacts/site/languages.json web/* | sha256sum | cut -d ' ' -f 1)
marker=artifacts/published/site-manifest.txt
if [[ -s "$marker" ]] && [[ $(cat "$marker") == "$fingerprint" ]]; then exit 0; fi

python -u check_site.py artifacts/site
previous=$("${remote[@]}" 'readlink -f /var/www/bengali-tts/current')
release=$("${remote[@]}" 'sudo -n mktemp -d /var/www/bengali-tts/releases/multilingual.XXXXXXXX')
if [[ ! "$release" =~ ^/var/www/bengali-tts/releases/multilingual\.[A-Za-z0-9]+$ ]]; then
  echo "Unexpected site release path" >&2
  exit 1
fi
if [[ ! "$previous" =~ ^/var/www/bengali-tts/releases/[A-Za-z0-9.]+$ ]]; then
  echo "Unexpected current site path" >&2
  exit 1
fi
tar -C artifacts/site -czf - . | "${remote[@]}" "sudo -n tar --no-same-owner -xzf - -C '$release'"
"${remote[@]}" "test -s '$release/languages.json' && sudo -n chmod 755 '$release' && sudo -n ln -s '$release' '$release.link' && sudo -n mv -Tf '$release.link' /var/www/bengali-tts/current"

if ! live=$(curl -fsSL --retry 3 --max-time 22 "$site/languages.json" | sha256sum | cut -d ' ' -f 1); then live=''; fi
if [[ "$live" != "$manifest_hash" ]] || ! python -u check_site.py "$site"; then
  "${remote[@]}" "if [ \"\$(readlink -f /var/www/bengali-tts/current)\" = '$release' ]; then sudo -n ln -s '$previous' '$release.rollback' && sudo -n mv -Tf '$release.rollback' /var/www/bengali-tts/current; fi"
  echo "Live verification failed and the preceding release was restored" >&2
  exit 1
fi
printf '%s\n' "$fingerprint" > "$marker.tmp"
mv -T "$marker.tmp" "$marker"
printf '%s\n' "$release"
