run_asr_pipeline() {
    local asr_lang="$1"
    local lang="$2"
    local dataset_dir="../TTA_LAS/covost2_$lang"

    for asr_model in "openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large-v3"; do
        echo "run ASR model: $asr_model"
        torchrun --nnodes=1 --nproc_per_node=8 main_multi_gpus.py \
            noise_dir="../res" \
            asr="$asr_model" \
            dataset_dir="$dataset_dir" \
            lang="$lang" \
            asr_lang="$asr_lang" \
            num_data=false \
            batch_size=1
    done
}

run_asr_pipeline "ar" "ar"
run_asr_pipeline "ca" "ca"
run_asr_pipeline "cy" "cy"
run_asr_pipeline "de" "de"
run_asr_pipeline "es" "es"
run_asr_pipeline "et" "et"
run_asr_pipeline "fa" "fa"
run_asr_pipeline "fr" "fr"
run_asr_pipeline "id" "id"
run_asr_pipeline "ja" "ja"
run_asr_pipeline "lv" "lv"
run_asr_pipeline "pt" "pt"
run_asr_pipeline "ru" "ru"
run_asr_pipeline "sl" "sl"
run_asr_pipeline "sv" "sv-SE"
run_asr_pipeline "ta" "ta"
run_asr_pipeline "tr" "tr"
run_asr_pipeline "mn" "mn"
run_asr_pipeline "zh" "zh-CN"