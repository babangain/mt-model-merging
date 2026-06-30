# python compare_masks.py \
#     --base_dir ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct \
#     --ft_root ./activation_mask/indic_en \
#     --ft_template "baban_QwenTranslate_{name}_English" \
#     --langs "hi bn ta te" \
#     --names "Hindi Bengali Tamil Telugu" \
#     --out_dir ./mask_change_reports \
#     --plot



  # python compare_hin_vs_instruct.py \
  #   --base_src_pt ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/tgt.pt \
  #   --hi_ft_src_pt ./activation_mask/indic_en/baban_QwenTranslate_Hindi_English/tgt.pt \
  #   --langs "hi bn ta te" \
  #   --out_dir ./hi_ft_vs_base_report \
  #   --plot \
  #   --save_json

python compare_base_vs_ft.py \
    --base_pt ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/tgt.pt \
    --ft_pt   ./activation_mask/indic_en/baban_QwenTranslate_Hindi_English/tgt.pt \
    --span tgt \
    --langs "hi bn ta te" \
    --out_dir ./compare_reports \
    --plot \
    --save_json