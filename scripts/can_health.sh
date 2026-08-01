#!/usr/bin/env bash
# CAN bus health check — run with the arms POWERED and the adapters plugged
# in (ideally right after a run that threw `loss communication`).
#
# `loss communication` (motor error 0xD) is the MOTOR's own watchdog: it
# fired because a command didn't reach it in time. That can be the wire or
# the host, and those need opposite fixes. This tells you which:
#
#   bus-error / error-warning / error-passive / bus-off / restarts NONZERO
#       -> PHYSICAL layer. The frames themselves are being corrupted or
#          nacked. Check termination first (see below), then connectors,
#          cable, stub length, ground. Retries also drag the control loop
#          rate down, which is a second-order cause of watchdog timeouts.
#
#   those counters all ZERO, but motors still report loss communication
#       -> HOST timing. The frames that went out were fine; there just
#          weren't enough of them in time. That's GIL/CPU starvation on
#          i2rt's polling thread (inference, checkpoint load, camera
#          enumeration), not an electrical problem.
set -uo pipefail

for i in can_left can_right; do
    echo "════════════════════ $i ════════════════════"
    if ! ip link show "$i" &>/dev/null; then
        echo "  NOT PRESENT (adapter unplugged or arm unpowered)"
        continue
    fi
    ip -details -statistics link show "$i" | sed 's/^/  /'
    echo
done

echo "════════════════════ USB link ════════════════════"
lsusb | grep -i "1d50:606f" || echo "  no gs_usb/canable2 adapters enumerated"
echo
for d in /sys/bus/usb/devices/*; do
    [ -f "$d/product" ] || continue
    if grep -qi "canable\|gs_usb" "$d/product" 2>/dev/null; then
        echo "  $(basename "$d"): $(cat "$d/product") @ $(cat "$d/speed")Mbps"
    fi
done

echo
echo "════════════════════ kernel log ════════════════════"
dmesg 2>/dev/null | grep -iE "can[0-9_]|gs_usb|bus-off" | tail -15 \
    || echo "  (needs sudo: sudo dmesg | grep -iE 'can|gs_usb|bus-off')"

cat <<'EOF'

──────────────────────── what to check on the wire ────────────────────────
Termination is the single most common cause of flaky CAN. The bus needs
exactly 120 ohm at EACH END and nothing in between — two resistors total.

  With everything POWERED OFF, measure across CANH/CANL:
      ~60 ohm  -> correct (two 120s in parallel)
      ~120 ohm -> only one terminator; the far end is missing
      <50 ohm  -> too many terminators (a motor with its jumper still set)
      open     -> no termination / broken conductor

Also: keep stubs off the daisy-chain under ~30 cm, reseat every connector
(especially the arm that errors more), and verify both adapters run the
same bitrate as the motors (`bitrate` field above).

Worth setting so the controller self-heals instead of latching bus-off:
    sudo ip link set can_right type can restart-ms 100
EOF
