# Switch to the transformers version the MolmoAct2-BimanualYAM checkpoint
# actually needs (confirmed: 4.57.x fails on its tokenizer_config.json --
# extra_special_tokens is a list, but 4.57.x's tokenizer init code requires
# a dict; 5.14.1 handles it correctly). Run this before deploy/molmoact2.py.
# This deliberately does NOT satisfy lerobot-record's own transformers<5.0.0
# constraint -- don't run lerobot-record / record.sh right after this
# without switching back first.
set -e
pip install "transformers==5.14.1" "huggingface_hub[cli,hf-transfer]==1.25.1"
