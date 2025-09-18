# Relative Time Intervals Representation for Word-Level Timestamping with Masked Training

This repository is the office implements for the paper: 
Relative Time Intervals Representation for Word-Level Timestamping with Masked Training
![](img/intro.png)

## File Structure
```
/.
│  index.html
│  README.md
│
├─conf
│      ds_config.json
│      multiprompt.jsonl
│
├─data
│      sample_data.jsonl
│
├─dataset
│  │  dataset.py
│  │
│  └─__pycache__
│          asr_feat.cpython-310.pyc
│          speech_dataset_large.cpython-310.pyc
│
├─img
│      intro.png
│
├─model
│  │  config.py
│  │  loss_fn.py
│  │  model.py
│  │
│  └─__pycache__
│          adapter.cpython-310.pyc
│          aispeech_asr.cpython-310.pyc
│          aispeech_asr_config.cpython-310.pyc
│          asr_feat.cpython-310.pyc
│          conformer_encoder.cpython-310.pyc
│
└─script
        add_new_token.py
```
## The rest code will be updated after the paper is accepted.

## [Github Pages Here](https://quanwei.fun/Timestamp-Aware-Speech-LLM/)
