#!/usr/bin/env bash
# CAN bus health check — run with the arms POWERED and the adapters plugged in.
#
# `loss communication` (motor error 0xD) is the motor's own watchdog, and can be
# the wire or the host. This says which:
#
#   error counters NONZERO   -> physical layer. Check termination, connectors,
#                               cable, stub length, ground.
#   counters ZERO            -> host timing. GIL/CPU starvation on i2rt's
#                               polling thread, not an electrical problem.
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
