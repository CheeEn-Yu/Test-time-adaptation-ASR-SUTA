
run_asr_pipeline() {
    local asr_lang="$1"
    local lang="$2"
    local dataset_dir="../TTA_LAS/test"

    for asr_model in "openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small"; do
        echo "run ASR model: $asr_model"
        # python main.py \
        torchrun --nnodes=1 --nproc_per_node=8 main_multi_gpus.py \
            task="transcribe" \
            asr="$asr_model" \
            dataset_name="fleurs" \
            dataset_dir="$dataset_dir" \
            lang="$lang" \
            asr_lang="$asr_lang" \
            temp=4.0 \
            noise_dir="Gaussian" \
            batch_size=1
            # exp_name="ex_data/english_analysis" \
            # num_data=false \
        
    done
}

run_asr_pipeline "en" "en"


# python main.py \
#     noise_dir='../res' \
#     exp_name='./ex_data/fleurs_hi' \
#     dataset_name='fleurs' \
#     asr='openai/whisper-small' \
#     dataset_dir='./fleurs_with_translation' \
#     lang=hi_in \
#     asr_lang=hi \
#     num_data=100 \
#     batch_size=1

# python main.py \
#     noise_dir='../res' \
#     asr='openai/whisper-tiny' \
#     dataset_dir=../TTA_LAS/covost2_it \
#     lang=it \
#     asr_lang=it \
#     num_data=false \
#     batch_size=1

# python main.py \
#     noise_dir='../res' \
#     asr='openai/whisper-base' \
#     dataset_dir=../TTA_LAS/covost2_it \
#     lang=it \
#     asr_lang=it \
#     num_data=false \
#     batch_size=1

# python main.py \
#     noise_dir='../res' \
#     asr='openai/whisper-small' \
#     dataset_dir=../TTA_LAS/covost2_it \
#     lang=it \
#     asr_lang=it \
#     num_data=false \
#     batch_size=1


# python hf_main.py \
#     exp_name='ex_data/small_it_noise' \
#     noise_dir='../res' \
#     asr='openai/whisper-small' \
#     dataset_dir=../covost2_it \
#     lang=it \
#     asr_lang=it \
#     num_data=false \
#     batch_size=8

# python hf_main.py \
#     exp_name='ex_data/large_it_noise' \
#     noise_dir='../res' \
#     asr='openai/whisper-large-v3' \
#     dataset_dir=../covost2_it \
#     lang=it \
#     asr_lang=it \
#     num_data=false \
#     batch_size=8

# python hf_main.py \
#     exp_name='ex_data/large_it_noise' \
#     noise_dir='../res' \
#     asr='openai/whisper-large-v3' \
#     dataset_dir=../covost2_it \
#     lang=it \
#     tta=true \
#     asr_lang=it \
#     num_data=false \
#     batch_size=1

# python hf_main.py \
#     exp_name='ex_data/small_it_noise' \
#     noise_dir='../res' \
#     asr='openai/whisper-small' \
#     dataset_dir=../covost2_it \
#     lang=it \
#     asr_lang=it \
#     num_data=false


# python hf_main.py \
#     exp_name='ex_data/tiny_zh_noise' \
#     noise_dir='../res' \
#     asr='openai/whisper-tiny' \
#     dataset_dir=../TTA_LAS/covost2_zh \
#     lang=zh-CN \
#     asr_lang=zh \
#     num_data=false

# python hf_main.py \
#     exp_name='ex_data/base_zh_noise' \
#     noise_dir='../res' \
#     asr='openai/whisper-base' \
#     dataset_dir=../TTA_LAS/covost2_zh \
#     lang=zh \
#     num_data=false



# python hf_main.py \
#     exp_name='ex_data/0118_it_suta_all' \
#     objective_f='["e_loss"]' \
#     dataset_dir=../covost2_it \
#     lang=it \
#     num_data=false

# python hf_main.py \
#     exp_name='ex_data/0118_nl_suta_all' \
#     objective_f='["e_loss"]' \
#     dataset_dir=../TTA_LAS/covost2_nl \
#     lang=nl \
#     num_data=false


# python hf_main.py \
#     exp_name='ex_data/0117_nl_suta_debug' \
#     objective_f='["e_loss"]' \
#     dataset_dir=../TTA_LAS/covost2_nl \
#     lang=nl \
#     num_data=5

# python hf_main.py \
#     exp_name='ex_data/0117_nl_p_loss_debug' \
#     objective_f='["p_loss"]' \
#     dataset_dir=../TTA_LAS/covost2_nl \
#     lang=nl \
#     num_data=10



# python hf_main.py \
#     exp_name='ex_data/0117_it_p_loss_debug' \
#     objective_f='["p_loss"]' \
#     dataset_dir=../covost2_it \
#     lang=it


# transcribe script
# python hf_main.py \
#     --exp_name "ex_data/1112_no_cnn" \
#     --task "transcribe" \
#     --topk 30 \
#     --alpha 10 \
#     --train_params "LN" \
#     --objective_f "e_loss c_loss weighted" \
#     --steps 10 \
#     --asr "openai/whisper-tiny" \
#     --train_feature true \
#     --encoderLN true \
#     --decoderLN true \
#     --allEncoder false \
#     --repeat_penal false \
#     --max_decoder_step 128 \
#     --num_data 300 \
#     --opt "AdamW" \
#     --episodic true \
#     --temp 4.5 \
#     --em_coef 0.5 \
#     --p_ratio 1.0 \
#     --lr 1e-3 \
#     --scheduler "null" \
#     --t_max 10 \
#     --lr_min 1e-3 \
#     --noise_dir "./res" \
#     --snr 10 \
#     --dataset_name "noisylibri" \
#     --dataset_dir "./data/libri_test_noise_10.0" \
#     --lang "en" \
#     --batch_size 1 \
#     --seed 42