# KenLM
## Setup Environment
```
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install sentencepiece
```

## Download the KenLM model and score the dataset
``` bash
python data_scoring/kenlm/entry.py --input-data-path $INPUT_DATA_PATH --output-data-path $OUTPUT_DATA_PATH --config-path $CONFIG_PATH

# e.g. python data_scoring/kenlm/entry.py --input-data-path data/source-cc-1b.jsonl --output-data-path data/source-cc-1b_scored-kenlm.jsonl --config-path data_scoring/config/kenlm.yaml
```
