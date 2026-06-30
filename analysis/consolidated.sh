# MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
# # MODEL_NAME=anonymous/AnonMT_Telugu_English
# mkdir activation_mask
# bash 1-activation.sh $MODEL_NAME
# bash 2-identify.sh
# # python draw_graph.py



MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
mkdir activation_mask
bash 1-activation.sh $MODEL_NAME
# python draw_graph.py


MODEL_NAME=anonymous/AnonMT_Hindi_English
mkdir activation_mask
bash 1-activation.sh $MODEL_NAME

MODEL_NAME=anonymous/AnonMT_Bengali_English
mkdir activation_mask
bash 1-activation.sh $MODEL_NAME