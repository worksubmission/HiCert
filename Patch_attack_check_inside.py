import sys

import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time

from matplotlib import pyplot as plt
# sys.path.append('./imagenet_patch')
# from matplotlib import pyplot as plt
from tqdm import tqdm
import joblib
from Patch_attacker_check_inside import PatchAttacker
from utils.pd import double_masking_detection_confidence_more_warn_original, \
    double_masking_detection_confidence_less_warn_original

from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed
import torch.multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, \
    Normalize
from torchvision.utils import save_image, make_grid

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='./checkpoints', type=str, help="directory of checkpoints")

# parser.add_argument('--data_dir', default='data', type=str,help="directory of data")

parser.add_argument("--data_dir", default='/public/qilinzhou', type=str)
# parser.add_argument("--data_dir",default='../',type=str)



parser.add_argument('--dataset', default='imagenet', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102', 'gtsrb'),
                    help="dataset")
parser.add_argument("--model", default='mae_vit_base', type=str, help="model name")
# vit_base_patch16_224 mae_vit_base
parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=32, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='./dump_attack', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
# parser.add_argument("--size_of_attack", default=32, type=int,
#                     help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument('--step_size', default=0.05, type=float, help='Attack step size')
parser.add_argument('--steps', default=1, type=int, help='Attack steps')
parser.add_argument('--randomizations', default=1, type=int, help='Number of random restarts')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--stop_idx', default=500, type=int, help='stop_idx')


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

print(args)
DATASET = args.dataset
print(args.model)
MODEL_DIR = os.path.join('', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
# if DATASET=='imagenette':
#     DATA_DIR = os.path.join(DATA_DIR, DATASET)
MODEL_NAME = args.model
NUM_IMG = args.num_img
STOP_IDX = args.stop_idx
# get model and data loader
model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=1, num_img=NUM_IMG, train=False)
# 获取参数字典
# params1 = model.state_dict()
max_value = float('-inf')
min_value = float('inf')
for data, target in val_loader:
    max_value = max(max_value, torch.max(data).item())
    min_value = min(min_value, torch.min(data).item())
print(max_value)
print(min_value)

device = 'cuda'
model = model.to(device)
model.eval()
cudnn.benchmark = True
size_of_attack=args.patch_size
attacker = PatchAttacker(model, [0.,0.,0.],[1.,1.,1.], ub=[max_value, max_value, max_value], lb=[min_value, min_value, min_value],kwargs={
    'epsilon':1.0,
    'random_start':True,
    'steps':args.steps,
    'step_size':args.step_size,
    'patch_l':size_of_attack,
    'patch_w':size_of_attack
})

# generate the mask set
mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)

# STOP_IDX=1
# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
SUFFIX = '_{}_{}_m{}_s{}_{}.z'.format(DATASET, MODEL_NAME, MASK_SIZE, MASK_STRIDE, STOP_IDX)
print("MASK_SIZE" + str(MASK_SIZE) + "MASK_STRIDE" + str(MASK_STRIDE))

if not not args.override and os.path.exists(os.path.join(DUMP_DIR, 'prediction_map_list' + SUFFIX)):
    print('loading two-mask predictions')
    prediction_map_list = joblib.load(os.path.join(DUMP_DIR, 'prediction_map_list' + SUFFIX))
    orig_prediction_list = joblib.load(
        os.path.join(DUMP_DIR, 'orig_prediction_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, SUFFIX)))
    label_list = joblib.load(os.path.join(DUMP_DIR, 'label_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, SUFFIX)))
else:
    print('computing two-mask predictions')
    prediction_map_before_attack_list = []
    confidence_map_before_attack_list = []
    confidence_before_attack_list= []
    orig_prediction_before_attack_list = []

    label_list = []

    prediction_map_after_attack_list = []
    confidence_map_after_attack_list = []
    confidence_after_attack_list= []
    orig_prediction_after_attack_list=[]
    random_counter_list=[]
    step_list=[]


    counter=0
    # joblib.dump(cert_incorrect_sample_list,
    #             os.path.join(DUMP_DIR, f'OMA_wrong_{args.patch_size}_{DATASET}_{MODEL_NAME}_{NUM_IMG}.z'))
    # cert_incorrect_sample_list = joblib.load(os.path.join(DUMP_DIR,  f'OMA_wrong_32_{DATASET}_{MODEL_NAME}_{NUM_IMG}.z'))

    for idx,(data, labels) in enumerate(tqdm(val_loader)):
        print("\n")
        print("This is number "+str(idx), flush=True)
        if idx==STOP_IDX:
            break
        # if idx not in cert_incorrect_sample_list:
        #     continue
        data = data.to(device)
        labels =labels.to(device)
        num_img = data.shape[0]
        num_mask = len(mask_list)
        #
        clean_output = model(data)
        confidence = torch.nn.functional.softmax(clean_output, dim=1)

        confidence = confidence.detach().cpu().numpy()
        # prediction_map_list.append(prediction_map)
        confidence_before_attack_list.append(confidence)
        clean_conf, clean_pred_old = clean_output.max(1)
        # print("clean_pred old "+str(clean_pred_old))
        clean_pred_old = clean_pred_old.detach().cpu().numpy()
        #
        # print("\n")
        print("clean_pred_old " + str(clean_pred_old))
        print("labels " + str(labels.cpu().numpy()))
        if not clean_pred_old==labels.cpu().numpy():
            print("prediction label incorrect")




        # before attack
        list_of_mutants=[]
        prediction_map = np.zeros([num_img, num_mask], dtype=int)
        confidence_map = np.zeros([num_img, num_mask])
        for i, mask in enumerate(mask_list):
            # for j in range(i, num_mask):
            #     mask2 = mask_list[j]
            masked_output = model(torch.where(mask, data, torch.tensor(0.).cuda()))
            list_of_mutants.append(torch.where(mask, data, torch.tensor(0.).cuda()))
            # plt.imshow(data.cpu()[0].permute(1, 2, 0), cmap='gray')
            # plt.imshow((torch.where(mask,data,torch.tensor(0.).cuda()).cpu()[0]).permute(1, 2, 0))
            # plt.show()
            masked_output = torch.nn.functional.softmax(masked_output, dim=1)
            masked_conf, masked_pred = masked_output.max(1)
            masked_conf = masked_conf.detach().cpu().numpy()
            confidence_map[:, i] = masked_conf
            masked_pred = masked_pred.detach().cpu().numpy()
            prediction_map[:, i] = masked_pred
            # save_image(make_grid(torch.where(mask, data, torch.tensor(0.).cuda()), nrow=1),
            #            "./img/"+str(i)+".jpg")
        prediction_map_before_attack_list.append(prediction_map)
        confidence_map_before_attack_list.append(confidence_map)
        print("before attack")
        prediction_map=prediction_map.reshape(-1)
        confidence_map=confidence_map.reshape(-1)


        result_dict_ADC_09 = double_masking_detection_confidence_more_warn_original(prediction_map, confidence_map, bear=0.9,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_ADC_08 = double_masking_detection_confidence_more_warn_original(prediction_map, confidence_map, bear=0.8,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_ADC_07 = double_masking_detection_confidence_more_warn_original(prediction_map, confidence_map, bear=0.7,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_ADC_06 = double_masking_detection_confidence_more_warn_original(prediction_map, confidence_map, bear=0.6,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_ADC_05 = double_masking_detection_confidence_more_warn_original(prediction_map, confidence_map, bear=0.5,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_PG_09 = double_masking_detection_confidence_less_warn_original(prediction_map, confidence_map, bear=0.9,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_PG_08 = double_masking_detection_confidence_less_warn_original(prediction_map, confidence_map, bear=0.8,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_PG_07 = double_masking_detection_confidence_less_warn_original(prediction_map, confidence_map, bear=0.7,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_PG_06 = double_masking_detection_confidence_less_warn_original(prediction_map, confidence_map, bear=0.6,
                                                                             orig_pred=clean_pred_old, label=labels)
        result_dict_PG_05 = double_masking_detection_confidence_less_warn_original(prediction_map, confidence_map, bear=0.5,
                                                                             orig_pred=clean_pred_old, label=labels)

        print("result_dict_ADC_09 "+str(result_dict_ADC_09))
        print("result_dict_ADC_08 "+str(result_dict_ADC_08))
        print("result_dict_ADC_07 "+str(result_dict_ADC_07))
        print("result_dict_ADC_06 "+str(result_dict_ADC_06))
        print("result_dict_ADC_05 "+str(result_dict_ADC_05))

        print("result_dict_PG_09 "+str(result_dict_PG_09))
        print("result_dict_PG_08 "+str(result_dict_PG_08))
        print("result_dict_PG_07 "+str(result_dict_PG_07))
        print("result_dict_PG_06 "+str(result_dict_PG_06))
        print("result_dict_PG_05 "+str(result_dict_PG_05))

        if certify_precomputed(prediction_map, clean_pred_old):
            print("OMA "+"{'warn': False, 'cert': True}")
        else:
            print("OMA "+"{'warn': True, 'cert': False}")




        print(prediction_map)
        # print(confidence_map)
        print("start attack")

        # before attack
        attacked, random_counter, step = attacker.perturb(data,labels,float('inf'),mask_list,random_count=args.randomizations)
        print("random_counter")
        print(random_counter)
        print(step)

        params2 = model.state_dict()

        labels = labels.cpu().numpy()
        data_old=data
        data=attacked


        # vanilla predictions
        clean_output = model(data)
        confidence = torch.nn.functional.softmax(clean_output, dim=1)

        confidence = confidence.detach().cpu().numpy()
        # prediction_map_list.append(prediction_map)
        confidence_after_attack_list.append(confidence)
        clean_conf, clean_pred = clean_output.max(1)
        # print("clean_pred new"+str(clean_pred))
        clean_pred = clean_pred.detach().cpu().numpy()
        # if not clean_pred==labels:
        print("clean_pred old " + str(clean_pred_old))
        print("clean_pred new " + str(clean_pred))
        orig_prediction_before_attack_list.append(clean_pred_old)
        orig_prediction_after_attack_list.append(clean_pred)
        prediction_map = np.zeros([num_img, num_mask], dtype=int)
        confidence_map = np.zeros([num_img, num_mask])
        list_of_mutants_attacked=[]
        for i, mask in enumerate(mask_list):
            # for j in range(i, num_mask):
            #     mask2 = mask_list[j]
                masked_output = model(torch.where(mask, data, torch.tensor(0.).cuda()))
                list_of_mutants_attacked.append(torch.where(mask, data, torch.tensor(0.).cuda()))

                # plt.imshow(data.cpu()[0].permute(1, 2, 0), cmap='gray')
                # plt.imshow((torch.where(torch.logical_and(mask,mask2),data,torch.tensor(0.).cuda()).cpu()[0]).permute(1, 2, 0))
                # plt.show()
                masked_output = torch.nn.functional.softmax(masked_output, dim=1)
                masked_conf, masked_pred = masked_output.max(1)
                masked_conf = masked_conf.detach().cpu().numpy()
                confidence_map[:, i] = masked_conf
                masked_pred = masked_pred.detach().cpu().numpy()
                prediction_map[:, i] = masked_pred

        counter=0
        for i in range(len(list_of_mutants_attacked)):
            if (list_of_mutants[i]-list_of_mutants_attacked[i]).sum()==0:
                counter=counter+1
        print(counter)
        print("after attack")
        print(prediction_map)
        # orig_prediction_list.append(clean_pred)
        prediction_map_after_attack_list.append(prediction_map)
        confidence_map_after_attack_list.append(confidence_map)
        label_list.append(labels)
        random_counter_list.append(random_counter)
        step_list.append(step)

        if certify_precomputed(prediction_map, clean_pred):

            print("!!!!!!!!!!!!!!!!")
            # save_image(make_grid(data_old, nrow=1),
            #            "./img/inside_real_" + str(DATASET) + "_" + str(size_of_attack) + "_" + str(idx) +"_clean_pred_old_"+str(clean_pred_old) + ".jpg")
            # plt.imshow(data_old.cpu()[0].permute(1, 2, 0), cmap='gray')
            # plt.show()
            # save_image(make_grid(attacked, nrow=1),
            #            "./img/inside_real_attacks_" + str(DATASET) + "_" + str(size_of_attack) + "_" + str(idx)+"_clean_pred_new_"+str(clean_pred) + ".jpg")
            # plt.imshow(attacked.cpu()[0].permute(1, 2, 0), cmap='gray')
            # plt.show()
            counter=counter+1

    print(counter)
    confidence_before_attack_list=np.concatenate(confidence_before_attack_list)
    prediction_map_before_attack_list = np.concatenate(prediction_map_before_attack_list)
    confidence_map_before_attack_list = np.concatenate(confidence_map_before_attack_list)

    confidence_after_attack_list=np.concatenate(confidence_after_attack_list)
    prediction_map_after_attack_list = np.concatenate(prediction_map_after_attack_list)
    confidence_map_after_attack_list = np.concatenate(confidence_map_after_attack_list)

    orig_prediction_after_attack_list = np.concatenate(orig_prediction_after_attack_list)
    orig_prediction_before_attack_list = np.concatenate(orig_prediction_before_attack_list)
    label_list = np.concatenate(label_list)

    # random_counter_list = np.concatenate(random_counter_list)
    # step_list = np.concatenate(step_list)


    # joblib.dump(confidence_map_list,os.path.join(DUMP_DIR,'attack_confidence_map_list'+SUFFIX))
    # joblib.dump(prediction_map_list,os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX))
    #
    # joblib.dump(orig_prediction_list,os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    # joblib.dump(label_list,os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))

    joblib.dump(confidence_before_attack_list, os.path.join(DUMP_DIR, 'confidence_before_attack_list' + SUFFIX))
    joblib.dump(confidence_map_before_attack_list, os.path.join(DUMP_DIR, 'confidence_map_before_attack_list' + SUFFIX))
    joblib.dump(prediction_map_before_attack_list, os.path.join(DUMP_DIR, 'prediction_map_before_attack_list' + SUFFIX))

    joblib.dump(confidence_after_attack_list, os.path.join(DUMP_DIR, 'confidence_after_attack_list' + SUFFIX))
    joblib.dump(confidence_map_after_attack_list, os.path.join(DUMP_DIR, 'confidence_map_after_attack_list' + SUFFIX))
    joblib.dump(prediction_map_after_attack_list, os.path.join(DUMP_DIR, 'prediction_map_after_attack_list' + SUFFIX))


    # joblib.dump(orig_prediction_after_attack_list,
    #             os.path.join(DUMP_DIR, 'attack_orig_after_attack_prediction_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, STOP_IDX)))
    # joblib.dump(orig_prediction_before_attack_list,
    #             os.path.join(DUMP_DIR, 'attack_orig_before_attack_prediction_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, STOP_IDX)))
    joblib.dump(label_list, os.path.join(DUMP_DIR, 'attack_label_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, STOP_IDX)))
    joblib.dump(random_counter_list, os.path.join(DUMP_DIR, 'random_counter_list'+SUFFIX))
    joblib.dump(step_list, os.path.join(DUMP_DIR, 'step_list'+SUFFIX))

clean_corr = 0
robust = 0
orig_corr = 0
for i, (prediction_map, label, orig_pred) in enumerate(zip(prediction_map_before_attack_list, label_list, orig_prediction_before_attack_list)):
    prediction_map = prediction_map + prediction_map.T - np.diag(
        np.diag(prediction_map))  # generate a symmetric matrix from a triangle matrix
    robust += certify_precomputed(prediction_map, label)
    clean_corr += double_masking_precomputed(prediction_map) == label
    orig_corr += orig_pred == label

print("------------------------------")
print("Certified robust accuracy:", robust / len(prediction_map_before_attack_list))
print("Clean accuracy with defense:", clean_corr / len(prediction_map_before_attack_list))
print("Clean accuracy without defense:", orig_corr / len(prediction_map_before_attack_list))

