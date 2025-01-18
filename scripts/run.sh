python hf_main.py \
    exp_name='ex_data/0118_it_suta_all' \
    objective_f='["e_loss"]' \
    dataset_dir=../covost2_it \
    lang=it \
    num_data=false

python hf_main.py \
    exp_name='ex_data/0118_nl_suta_all' \
    objective_f='["e_loss"]' \
    dataset_dir=../TTA_LAS/covost2_nl \
    lang=nl \
    num_data=false


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