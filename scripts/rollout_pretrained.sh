# Arm servers must already be running (they own the CAN loop and the
# torque):  ./scripts/start_arm_servers.sh
# Auto-recovery is now a SERVER flag (arm_server --enable-auto-recovery),
# not a --robot.* one. Leaving it off is the better default now: it
# re-asserts the pre-error target at full gains from inside i2rt's
# driver, which the follower's velocity clamp cannot bound.
# Baseline rollout: the stock lerobot/MolmoAct2-BimanualYAM-LeRobot checkpoint,
# NOT fine-tuned on put-box -- useful for comparing against rollout.sh's LoRA
# fine-tune to see how much of the observed behavior (e.g. the generic
# up-and-down motion) is pretrained-prior vs fine-tune-specific.
#
# --task should describe something this checkpoint's own pretraining data
# would recognize -- it was NOT trained on put-box's task phrasing/objects,
# so reusing one of those exact strings isn't meaningful here. Adjust to
# whatever the base checkpoint's own model card documents as example tasks.
#
# First run downloads the checkpoint from the Hub (large -- expect it to take
# a while) -- don't set HF_HUB_OFFLINE=1 until AFTER that first run has fully
# cached it locally, or lerobot-rollout will fail immediately instead of
# fetching it. Once cached, add the same `export HF_HUB_OFFLINE=1` rollout.sh
# uses, for the same speed/reliability reasons documented there.

lerobot-rollout \
    --strategy.type=base \
    --policy.path=lerobot/MolmoAct2-BimanualYAM-LeRobot \
    --policy.inference_action_mode=continuous \
    --policy.device=cuda \
    --policy.model_dtype=bfloat16 \
    --policy.use_amp=true \
    --policy.enable_inference_cuda_graph=true \
    --policy.n_action_steps=30 --policy.chunk_size=30 \
    --robot.type=yam_ultra_bimanual \
    --robot.left_channel=can_left --robot.right_channel=can_right \
    --task="put the tape into the cardboard box" \
    --rename_map='{"observation.images.top": "observation.images.top", "observation.images.right": "observation.images.right_wrist", "observation.images.left": "observation.images.left_wrist"}' \
    --duration=60
