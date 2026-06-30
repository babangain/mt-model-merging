# Compute late-layer averages (31–36)
# For each file:
#   - Only include pairs containing the language-specific fine-tuned model
#   - Exclude any pair involving Qwen/Qwen2.5-3B-Instruct
#   - Exclude self-comparisons
#   - Average remaining comparisons

import json
import pandas as pd
import os
import numpy as np

files = {
    # En -> Indic
    "en_indic_hi": "en_indic.hi.masked_cka.json",
    "en_indic_bn": "en_indic.bn.masked_cka.json",
    "en_indic_ta": "en_indic.ta.masked_cka.json",
    "en_indic_te": "en_indic.te.masked_cka.json",
    # Indic -> En
    "indic_en_hi": "indic_en.hi.masked_cka.json",
    "indic_en_bn": "indic_en.bn.masked_cka.json",
    "indic_en_ta": "indic_en.ta.masked_cka.json",
    "indic_en_te": "indic_en.te.masked_cka.json",
}

layers = [f"layer_{i:02d}" for i in range(31, 37)]

lang_map = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu"
}

rows = []

for key, path in files.items():
    if not os.path.exists(path):
        continue
        
    with open(path, "r") as f:
        data = json.load(f)
    
    direction = "En→Indic" if key.startswith("en_indic") else "Indic→En"
    lang_code = key.split("_")[-1]
    lang_full = lang_map[lang_code]
    
    # Determine model name to filter
    if direction == "En→Indic":
        target_model = f"anonymous/AnonMT_English_{lang_full}"
    else:
        target_model = f"anonymous/AnonMT_{lang_full}_English"
    
    for span in ["src", "tgt"]:
        pairs = data["spans"][span]["pairs"]
        
        layer_values_accum = {layer: [] for layer in layers}
        
        for pair_name, layer_dict in pairs.items():
            model_a, model_b = pair_name.split("|||")
            
            # skip base model comparisons
            if "Qwen/Qwen2.5-3B-Instruct" in pair_name:
                continue
            
            # skip self comparisons
            if model_a == model_b:
                continue
            
            # keep only pairs containing the specific language model
            if target_model not in pair_name:
                continue
            
            for layer in layers:
                if layer in layer_dict:
                    layer_values_accum[layer].append(layer_dict[layer])
        
        row = {
            "Direction": direction,
            "Language": lang_full,
            "Span": span
        }
        
        for layer in layers:
            if len(layer_values_accum[layer]) > 0:
                row[layer] = float(np.mean(layer_values_accum[layer]))
            else:
                row[layer] = None
        
        rows.append(row)

df_lang_specific = pd.DataFrame(rows)
print(df_lang_specific)
