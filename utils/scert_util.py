import os

import joblib
import numpy as np
import torch
from tqdm import tqdm
from utils.drs import gen_ablation_set_column_fix, gen_ablation_set_row_fix
from utils.topkcert_util import certified_drs_two_delta

device = 'cuda'
def find_cert_in_training_dataset(training_dataset, model, epoch, args):
    ablation_type = args.ablation_type
    ablation_size = args.ablation_size
    patch_size = args.patch_size
    DATASET = args.dataset

    prediction_map_list=[]
    label_list=[]
    DUMP_DIR=args.dump_dir
    MODEL_DIR = os.path.join('.', args.model_dir)
    DATA_DIR = os.path.join(args.data_dir, DATASET)
    DUMP_DIR = os.path.join('.', args.dump_dir)

    if not os.path.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)
    # if DATASET == 'imagenette':
    #     DATA_DIR = os.path.join(DATA_DIR, DATASET)
    MODEL_NAME = args.model
    NUM_IMG = args.num_img

    not_cert_list=[]
    if args.dataset == 'imagenet':
        print(256)
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True, num_workers=4)

    else:
        print(128)
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True, num_workers=4)

    if ablation_type == "column":
        ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_column_fix(ablation_size)
    elif ablation_type == "row":
        ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_row_fix(ablation_size)
    else:
        assert  1==0
    SUFFIX = '_two_mask_{}_{}_m{}_s{}_{}_ep{}.z'.format(DATASET, MODEL_NAME, MASK_SIZE, MASK_STRIDE, NUM_IMG,epoch)

    # elif ablation_type=="block":
    #     ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_block(ablation_size)


    for data, labels in tqdm(train_loader):
        data = data.to(device)
        labels = labels.numpy()
        num_img = data.shape[0]
        num_mask = len(ablation_list)
        # two-mask predictions
        prediction_map = np.zeros([num_img, num_mask], dtype=int)
        confidence_map = np.zeros([num_img, num_mask])
        for i, mask in enumerate(ablation_list):
            masked_output = model(torch.where(mask, data, torch.tensor(0.).cuda()))
            # plt.imshow(
            #     (torch.where(mask, data, torch.tensor(0.).cuda()).cpu()[0]).permute(1, 2, 0))
            # plt.show()
            masked_output = torch.nn.functional.softmax(masked_output, dim=1)
            masked_conf, masked_pred = masked_output.max(1)
            masked_conf = masked_conf.detach().cpu().numpy()
            confidence_map[:, i] = masked_conf
            masked_pred = masked_pred.detach().cpu().numpy()
            prediction_map[:, i] = masked_pred

        # vanilla predictions
        clean_output = model(data)
        clean_conf, clean_pred = clean_output.max(1)
        clean_pred = clean_pred.detach().cpu().numpy()
        prediction_map_list.append(prediction_map)
        label_list.append(labels)

    # joblib.dump(prediction_map_list, os.path.join(DUMP_DIR, 'prediction_map_list_drs' + SUFFIX))
    # joblib.dump(label_list, os.path.join(DUMP_DIR, 'label_list_{}_{}_{}_drs.z'.format(DATASET, MODEL_NAME, NUM_IMG)))

    for i, (label, prediction_map_drs) in enumerate(
            zip(label_list, prediction_map_list)):
        _,cert=certified_drs_two_delta(prediction_map_drs, ablation_size, patch_size)
        if not cert:
            not_cert_list.append(i)
    print(not_cert_list)
    dataset = torch.utils.data.Subset(training_dataset, not_cert_list)
    print(dataset)
    return dataset
