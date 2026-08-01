# Arm servers must already be running (they own the CAN loop and the
# torque):  ./scripts/start_arm_servers.sh
# Auto-recovery is now a SERVER flag (arm_server --enable-auto-recovery),
# not a --robot.* one. Leaving it off is the better default now: it
# re-asserts the pre-error target at full gains from inside i2rt's
# driver, which the follower's velocity clamp cannot bound.
# DAgger (human-in-the-loop) data collection: run the current fine-tuned
# checkpoint autonomously, take over via Quest teleop when it starts to
# drift, and record those correction windows as new training data --
# specifically targeting the "policy doesn't know how to recover once it's
# off-distribution" gap, which more steps on the same 49-episode put-box
# data can't fix on its own.
#
# Controls (keyboard only right now -- no foot pedal attached, see
# `evdev.list_devices()`; a USB pedal would make this hands-free instead of
# needing a second person at the keyboard while you're in the headset):
#   p  = pause/resume autonomous policy execution
#   c  = toggle correction recording (only while paused)
#   u  = push dataset to hub on demand
#   ESC = stop session
#
# --strategy.record_autonomous=false (corrections-only mode): only the
# correction windows get recorded, one episode per correction -- the most
# information-dense choice for filling the specific "recovery" gap. Flip to
# true for sentry-like continuous recording (autonomous + corrections both
# tagged, `intervention=True/False`) if you want broader volume instead of
# targeted correction data -- bigger dataset, less concentrated signal.
#
# New repo_id, NOT put-box: DAgger tags every frame with an `intervention`
# bool column that put-box's schema doesn't have, so this can't be appended
# to put-box directly without a schema migration first.

export HF_HUB_OFFLINE=1

exec taskset -c 0-5 lerobot-rollout \
    --strategy.type=dagger \
    --strategy.input_device=keyboard \
    --strategy.record_autonomous=false \
    --policy.path=MolmoAct2/008000/pretrained_model \
    --policy.inference_action_mode=continuous \
    --policy.device=cuda \
    --policy.model_dtype=bfloat16 \
    --policy.use_amp=true \
    --policy.enable_inference_cuda_graph=true \
    --policy.n_action_steps=30 --policy.chunk_size=30 \
    --robot.type=yam_ultra_bimanual \
    --robot.left_channel=can_left --robot.right_channel=can_right \
    --teleop.type=bi_quest_teleop \
    --teleop.ws_url=wss://127.0.0.1:8443/ws \
    --dataset.repo_id=minhphd/put-box-dagger \
    --dataset.single_task="put the tape into the cardboard box" \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --strategy.num_episodes=20
