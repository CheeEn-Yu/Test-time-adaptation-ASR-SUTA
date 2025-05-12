run_asr_pipeline() {
    local asr_lang="$1"
    local lang="$2"
    local dataset_dir="./fleurs_with_translation"

    for asr_model in "openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large-v3"; do
    # for asr_model in "openai/whisper-tiny"; do
        echo "run ASR model: $asr_model"
        torchrun --nnodes=1 --nproc_per_node=8 main_multi_gpus.py \
            task="transcribe" \
            noise_dir="../res" \
            asr="$asr_model" \
            dataset_name="fleurs" \
            dataset_dir="$dataset_dir" \
            lang="$lang" \
            asr_lang="$asr_lang" \
            num_data=false \
            batch_size=1
    done
}

# run_asr_pipeline "hi" "hi_in"
# run_asr_pipeline "ta" "ta_in"
# run_asr_pipeline "ar" "ar_eg"
# run_asr_pipeline "de" "de_de"
# run_asr_pipeline "es" "es_419"
# run_asr_pipeline "sv" "sv_se"
# run_asr_pipeline "tr" "tr_tr"
# run_asr_pipeline "ru" "ru_ru"
# run_asr_pipeline "sl" "sl_si"
# run_asr_pipeline "sw" "sw_ke"
# run_asr_pipeline "ms" "ms_my"
run_asr_pipeline "vi" "vi_vn"
run_asr_pipeline "zh" "cmn_hans_cn"
run_asr_pipeline "ja" "ja_jp"