# Switch to the transformers/huggingface_hub versions lerobot-record needs.
# lerobot==0.4.4's own bundled GR00T policy module only imports cleanly
# under transformers<5.0.0 -- and it gets imported as a side effect of
# lerobot-record's plugin-registry scan at startup, even though record.sh
# uses no policy at all. Run this before record.sh / any lerobot-* CLI.
set -e
pip install "transformers>=4.57.1,<4.58" "huggingface_hub[cli,hf-transfer]>=0.34.2,<0.36.0"
