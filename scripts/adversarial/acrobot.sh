for i in 1 2 3 4 5; do
    experiment_name="AIL_1_Acrobot_00$i"
    echo ">>>>> STARTING: ${experiment_name}"
    clear && python train_adversarial.py \
    --gpu -1 \
    --pretrained \
    --encoder vector \
    --run_name $experiment_name \
    --data_path ./dataset/acrobot/IDM_VECTOR/ \
    --expert_path ./dataset/acrobot/Acrobot-v1.npz \
    --alpha ./dataset/acrobot/ALPHA/ \
    --domain vector \
    --choice weighted \
    --lr 5e-3 \
    --lr_decay_rate 1 \
    --batch_size 128 \
    --idm_epochs 100 \
    --policy_lr 5e-4 \
    --policy_lr_decay_rate 1 \
    --policy_batch_size 128 \
    --expert_amount 1 \
    --adv_batch_size 8 \
    --env_name acrobot \
    --verbose
done