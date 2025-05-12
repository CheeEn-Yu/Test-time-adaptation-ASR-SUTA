run_asr_pipeline() {

    declare -A lang_map=(
        ["dutch"]="nl"
        ["french"]="fr"
        ["german"]="de"
        ["italian"]="it"
        ["polish"]="pl"
        ["portuguese"]="pt"
        ["spanish"]="es"
    )

    for dataset_dir in "spanish" "italian" "french" "german"; do
        local lang="${lang_map[$dataset_dir]}"

        for asr_model in "openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large-v3"; do
            echo "run ASR model: $asr_model on dataset: $dataset_dir with lang: $lang"
            torchrun --nnodes=1 --nproc_per_node=8 test.py \
                task="transcribe" \
                dataset_name="multilibri" \
                asr="$asr_model" \
                dataset_dir="$dataset_dir" \
                lang="$lang" \
                asr_lang="$lang" \
                num_data=false \
                noise_dir="../res" \
                batch_size=1
        done
    done
}

run_asr_pipeline