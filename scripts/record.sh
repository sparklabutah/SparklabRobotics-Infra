#!/usr/bin/env bash
set -euo pipefail
# Teleop dataset recording. Needs the arm servers AND the relay already up:
#     ./scripts/start_arm_servers.sh
#     sparklab-relay --host 0.0.0.0 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
# Anchor to the repo root rather than the caller's cwd.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

lerobot-record \
    --robot.type=yam_ultra_bimanual \
    --robot.left_channel=can_left --robot.right_channel=can_right \
    --teleop.type=bi_quest_teleop \
    --teleop.ws_url=wss://127.0.0.1:8443/ws \
    --dataset.repo_id=minhphd/test \
    --dataset.single_task="put stuff in a box" \
    --dataset.num_episodes=1 --dataset.fps=30 \
    --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false