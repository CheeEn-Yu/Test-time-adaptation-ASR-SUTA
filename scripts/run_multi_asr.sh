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

    for dataset_dir in "dutch" "french" "german" "italian" "polish" "portuguese" "spanish"; do
        local lang="${lang_map[$dataset_dir]}"

        for asr_model in "openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large-v3"; do
            echo "run ASR model: $asr_model on dataset: $dataset_dir with lang: $lang"
            python main.py \
                noise_dir="../res" \
                asr="$asr_model" \
                dataset_dir="$dataset_dir" \
                lang="$lang" \
                asr_lang="$lang" \
                num_data=false \
                batch_size=1
        done
    done
}

run_asr_pipeline