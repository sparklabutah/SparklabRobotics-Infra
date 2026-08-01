# Arm servers must already be running (they own the CAN loop and the
# torque):  ./scripts/start_arm_servers.sh
# Auto-recovery is now a SERVER flag (arm_server --enable-auto-recovery),
# not a --robot.* one. Leaving it off is the better default now: it
# re-asserts the pre-error target at full gains from inside i2rt's
# driver, which the follower's velocity clamp cannot bound.
# allenai/MolmoAct2-BimanualYAM (the LoRA base model) is already fully
# cached locally (~21GB) -- without this, huggingface_hub still does a
# network round-trip per file to check freshness/etags on every run before
# falling through to the cache, which is most of what "Fetching N files /
# Reconstruction complete: 0.00B/0.00B" is actually spending time on (the
# 0.00B confirms nothing new is being downloaded). Skipping that check
# also shrinks the window where i2rt's CAN polling threads sit starved
# right after connect() -- the same window that produced the
# 'loss communication' failures on both arms during earlier runs.
export HF_HUB_OFFLINE=1

# Reserve cores 0-5 for this process (24 logical CPUs total on this
# machine) so unrelated system load (browser, IDE, apport, etc.) can't
# preempt it. This does NOT separate our own inference thread from i2rt's
# CAN polling threads (Thread-1/Thread-2, _set_torques_and_update_state) --
# they're all threads inside this one Python process sharing one GIL, and
# CPU affinity doesn't change that: only one thread executes Python
# bytecode at a time regardless of how many cores are reserved. This is
# isolation from OTHER processes, not a fix for intra-process GIL
# contention between inference and the CAN threads -- that would need
# real-time thread priority (chrt -f), which needs rtprio capability this
# user account doesn't currently have (needs a one-time root-level change
# to /etc/security/limits.conf), or moving inference to a separate process.
exec taskset -c 0-5 lerobot-rollout \
    --strategy.type=base \
    --policy.path=MolmoAct2/008000/pretrained_model \
    --policy.inference_action_mode=continuous \
    --inference.type=sync \
    --policy.device=cuda \
    --policy.model_dtype=bfloat16 \
    --policy.use_amp=true \
    --policy.enable_inference_cuda_graph=true \
    --policy.n_action_steps=30 --policy.chunk_size=30 \
    --robot.type=yam_ultra_bimanual \
    --robot.left_channel=can_left --robot.right_channel=can_right \
    --robot.max_tick_s=0.25 \
    --task="put the marker into the cardboard box" \
    --fps 30 --display_data true \
    --duration=2000
