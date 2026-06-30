# TACL MT Model Merging

This repository contains code and analysis artifacts for our multilingual machine translation model-merging experiments.

We used [d-gurgurov/Language-Neurons-Manipulation](https://github.com/d-gurgurov/Language-Neurons-Manipulation.git) as a starting point for parts of our codebase, then adapted and extended it for span-conditioned activation analysis, neuron usage alignment, masked CKA, and representation-geometry analyses for multilingual MT checkpoints.

The fine-tuned and merged model checkpoints are not included at this time to preserve submission anonymity. We plan to release the checkpoints in the future after the anonymity requirement no longer applies.

## Repository Layout

- `merging/`: data preparation, full fine-tuning, LoRA fine-tuning, and evaluation scripts for MT models.
- `merging/yamls_qwen/`: LLaMA-Factory configs for English-to-X directions and multilingual variants.
- `merging/yamls_indic_en/`: LLaMA-Factory configs for Indic-to-English directions and multilingual variants.
- `analysis/`: span-conditioned activation extraction and mechanistic analysis scripts.
- `analysis/src/`: helper scripts used by the activation-analysis pipeline.

## Environment

Install the core packages:

```bash
pip install torch transformers datasets vllm numpy pandas matplotlib seaborn scikit-learn tqdm sacrebleu huggingface_hub safetensors
```

Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The training scripts clone and install it automatically, but you can also install it manually:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Data Preparation

The `merging/` scripts create LLaMA-Factory JSON datasets from Samanantar and FLORES.

English to Indic:

```bash
cd merging
python download_data.py --output-dir ./data
```

Indic to English:

```bash
cd merging
python download_data_indic_en.py --output-dir ./data
```

English to German/French/Arabic/Ukrainian:

```bash
cd merging
python download_data_en_others.py --output-dir ./data
```

These scripts write files such as `MT_En_Hindi.json`, `MT_En_Hindi_valid.json`, `MT_En_Hindi_test.json`, and update `dataset_info.json`.

## Fine-Tuning

Full fine-tuning is launched through LLaMA-Factory YAML files.

English to Indic:

```bash
cd merging
bash train.sh hin.yaml
bash train.sh ben.yaml
bash train.sh tamil.yaml
bash train.sh telugu.yaml
```

Indic to English:

```bash
cd merging
bash train_indic_en.sh hin.yaml
bash train_indic_en.sh ben.yaml
bash train_indic_en.sh tamil.yaml
bash train_indic_en.sh telugu.yaml
```

Multilingual configs are also provided:

```bash
cd merging
bash train.sh multilingual.yaml
bash train_indic_en.sh multilingual.yaml
```


```bash
cd merging
bash train_eval_lora.sh
```

Edit `LANGUAGES`, `LORA_CONFIGS`, and the `BASE_DIR`/`DATA_DIR` variables inside the script before running large sweeps.

## Evaluation

The evaluation scripts use vLLM and report BLEU/CHRF.

English to Indic:

```bash
cd merging
python eval_en_indic.py --model <MODEL_OR_CHECKPOINT> --data_dir ./data
```

Indic to English:

```bash
cd merging
python eval_indic_en.py --model <MODEL_OR_CHECKPOINT> --data_dir ./data
```

Bidirectional models:

```bash
cd merging
python eval_bidirectional.py --model <MODEL_OR_CHECKPOINT> --data_dir ./data
```

Multilingual checkpoints:

```bash
cd merging
python eval_multilingual.py --model <MODEL_OR_CHECKPOINT> --data_dir ./data
```

Layer replacement / patching experiments:

```bash
cd merging
python eval_en_indic_layer_patch.py --help
```

## Model Merging

The paper uses external merging tools, primarily MergeKit, for Task Arithmetic, TIES, DARE, and SCE-Merging. This repository keeps the training, evaluation, and analysis scripts, but merged checkpoints and their MergeKit run outputs are not included for anonymity.

After producing merged checkpoints with MergeKit, evaluate them with the scripts in `merging/`.

## Activation Analysis

The `analysis/` directory contains the span-conditioned neuron and representation-analysis pipeline.

First build masked teacher-forced inputs:

```bash
cd analysis
bash 0-data.sh
```

`0-data.sh` currently builds English-to-Indic masked data. Uncomment the Indic-to-English block inside the script if needed.

Then collect MLP gate activations and identify language-specific neurons.

Indic to English:

```bash
cd analysis
bash 000-consolidated.sh
```

English to Indic:

```bash
cd analysis
bash 000-consolidated-en-indic.sh
```

Before running, edit the `model=` line in each script to point to the model/checkpoint you want to analyze. These scripts write:

- `data_masked_qwen/`
- `acts_<MODEL_SAFE_NAME>/`
- `activation_mask/`

## Neuron Count Figures

Export per-layer neuron-count JSON:

```bash
cd analysis
python plot_counts_json.py \
  --root ./activation_mask/indic_en \
  --models Qwen_Qwen2.5-3B-Instruct anonymous_AnonMT_Hindi_English anonymous_AnonMT_Bengali_English anonymous_AnonMT_Tamil_English anonymous_AnonMT_Telugu_English \
  --out_dir ./matrix_json_indic_en
```

Plot layerwise count matrices:

```bash
cd analysis
python plot_counts_matrix_with_numbers.py \
  --pt ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/tgt.pt \
  --langs "hi bn ta te" \
  --out_dir ./matrix_plots
```

## Neuron Usage Alignment

Indic to English:

```bash
cd analysis
python nua.py \
  --acts_root ./ \
  --models Qwen_Qwen2.5-3B-Instruct anonymous_AnonMT_Hindi_English anonymous_AnonMT_Bengali_English anonymous_AnonMT_Tamil_English anonymous_AnonMT_Telugu_English \
  --langs hi bn ta te \
  --span tgt \
  --normalize rate \
  --out_dir nua_outputs \
  --save_plots
```

English to Indic:

```bash
cd analysis
python nua_en_indic.py \
  --acts_root ./ \
  --models Qwen_Qwen2.5-3B-Instruct anonymous_AnonMT_English_Hindi anonymous_AnonMT_English_Bengali anonymous_AnonMT_English_Tamil anonymous_AnonMT_English_Telugu \
  --langs hi bn ta te \
  --span tgt \
  --normalize rate \
  --out_dir nua_outputs_en_indic \
  --save_plots
```

Draw the summary plots:

```bash
cd analysis
python draw_nua.py
```

## Masked CKA

Run masked CKA from teacher-forced masked inputs:

```bash
cd analysis
python cka_masked.py \
  --lang hi \
  --data_dir data_masked_qwen/en_indic \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --models Qwen/Qwen2.5-3B-Instruct anonymous/AnonMT_English_Hindi anonymous/AnonMT_English_Bengali anonymous/AnonMT_English_Tamil anonymous/AnonMT_English_Telugu \
  --N 128 \
  --max_length 512 \
  --span both \
  --out_dir cka_outputs_masked_en_indic
```

Use `last_six_layers.py` to summarize late-layer CKA values from the generated JSON files.

## Principal-Angle Geometry

Per-language geometry:

```bash
cd analysis
python mds.py \
  --lang hi \
  --data_dir data_masked_qwen/en_indic \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --N 256 \
  --max_length 512 \
  --batch_size 8 \
  --plot_heatmaps \
  --k 64 \
  --layers_to_map 0 12 24 34 35 36 \
  --out_dir geom_outputs_en_indic
```

Combined across languages:

```bash
cd analysis
python mds_full_data.py \
  --langs hi bn ta te \
  --data_dir data_masked_qwen/en_indic \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --N_per_lang 128 \
  --max_length 512 \
  --batch_size 8 \
  --plot_heatmaps \
  --k 64 \
  --layers_to_map 0 12 24 34 35 36 \
  --out_dir geom_outputs_en_indic_all
```

Plot selected principal-angle curves:

```bash
cd analysis
python calc_angle.py
```

## Notes

- The scripts contain hard-coded defaults from our experiment environment. Check `BASE_DIR`, `DATA_DIR`, `model=...`, output paths, and GPU settings before running.
- Activation extraction and evaluation require vLLM and a CUDA GPU.
- Some generated outputs are included for reproducibility of analysis figures, but the model checkpoints are not included during anonymity.
