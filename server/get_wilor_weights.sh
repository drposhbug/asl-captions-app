#!/bin/sh
# Downloads WiLoR-mini's weights into the conda env. HuggingFace's CDN stalls mid-transfer on some networks,
# so each attempt aborts when it drops under 20 KB/s for 30 s, resumes if the server allows it, and retries.
D="${WILOR_DIR:-C:/Users/thorn/miniconda3/envs/wilor/Lib/site-packages/wilor_mini/pretrained_models}"
B="https://huggingface.co/warmshao/WiLoR-mini/resolve/main/pretrained_models"
for f in detector.pt wilor_final.ckpt; do
  want=$(curl -sIL --max-time 30 "$B/$f" | grep -i "^content-length" | tail -1 | tr -dc 0-9)
  [ -z "$want" ] && { echo "$f: could not read size"; exit 1; }
  part="${D:?}/${f:?}.part"
  i=0
  while [ "$(stat -c %s "${D:?}/${f:?}" 2>/dev/null || echo 0)" != "$want" ]; do
    i=$((i + 1)); [ "$i" -gt 40 ] && { echo "$f: giving up after 40 attempts"; exit 1; }
    have=$(stat -c %s "$part" 2>/dev/null || echo 0)
    echo "$f: attempt $i, resuming from $((have / 1048576)) MB of $((want / 1048576)) MB"
    curl -L -s -S -C - --speed-limit 20000 --speed-time 30 -o "$part" "$B/$f"
    rc=$?
    if [ "$rc" = 33 ]; then echo "$f: server refused resume, restarting"; rm -f "$part"; fi
    [ "$(stat -c %s "$part" 2>/dev/null || echo 0)" = "$want" ] && mv -f "$part" "${D:?}/${f:?}"
    sleep 3
  done
  echo "$f: complete ($want bytes)"
done
