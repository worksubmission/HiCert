import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time
from tqdm import tqdm
import joblib

from utils.pd import one_masking_statistic, double_masking_detection_nolemma1, \
    double_masking_detection_context_less_warn, \
     double_masking_detection_context_more_warn, \
     double_masking_detection_confidence_more_warn_original, \
    double_masking_detection_confidence_less_warn_original
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed

#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='checkpoints', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='/public/qilinzhou', type=str, help="directory of data")
parser.add_argument('--dataset', default='imagenet', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102', 'gtsrb'), help="dataset")
parser.add_argument("--model", default='mae_vit_base', type=str, help="model name")
# mae_vit_base resnetv2_50x1_bit_distilled vit_base_patch16_224
parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=64, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump_20241205', type=str, help='directory to dump two-mask predictions')
# dump_2024127 dump_20241205
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--safer", default=1, type=int, help='safer mode')
parser.add_argument("--confidence", default=0, type=int, help='confidence or context')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
print(args)
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

MODEL_NAME = args.model
NUM_IMG = args.num_img
SAFER_MODE=args.safer
CONFIDENCE=args.confidence

# get model and data loader
model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=1, num_img=NUM_IMG, train=False)

device = 'cuda'
model = model.to(device)
model.eval()
cudnn.benchmark = True

# generate the mask set
mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)

# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
SUFFIX = '_two_mask_{}_{}_m{}_s{}_{}.z'.format(DATASET, MODEL_NAME, MASK_SIZE, MASK_STRIDE, NUM_IMG)

clean_corr = 0
robust = 0
orig_corr = 0
statistics_dict = {}

prediction_map_list = joblib.load(os.path.join(DUMP_DIR,f"prediction_map_list_two_mask_{DATASET}_{MODEL_NAME}_m{MASK_SIZE}_s{MASK_STRIDE}_{NUM_IMG}.z"))
orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,f"orig_prediction_list_{DATASET}_{MODEL_NAME}_{NUM_IMG}.z"))
label_list = joblib.load(os.path.join(DUMP_DIR,f"label_list_{DATASET}_{MODEL_NAME}_{NUM_IMG}.z"))
confidence_map_list = joblib.load(os.path.join(DUMP_DIR,f"confidence_map_list_two_mask_{DATASET}_{MODEL_NAME}_m{MASK_SIZE}_s{MASK_STRIDE}_{NUM_IMG}.z"))
# attack_confidence_map_list_two_mask_imagenet_vit_base_patch16_224_m(130, 130)_s(19, 19)_1000
# prediction_map_list_two_mask_imagenet_vit_base_patch16_224_m(50, 50)_s(35, 35)_50000.z
# prediction_map_list_two_mask_imagenet_vit_base_patch16_224_m(50, 50)_s(35, 35)_50000.z
# prediction_map_list_two_mask_imagenet_vit_base_patch16_224_m(50, 50)_s(35, 35)_50000.z
# statistic for two-mask detection
correct_sample = 0
correct_warning_cert = 0
correct_warning_notcert = 0
correct_cert_nowarning = 0
correct_nocert_nowarning = 0

incorrect_sample = 0
incorrect_warning_cert = 0
incorrect_warning_notcert = 0
incorrect_cert_nowarning = 0
incorrect_nocert_nowarning = 0

cert_incorrect_sample_list=[]

start_time = time.time()
for j in [0]:
# for j in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

    for i, (prediction_map, label, orig_pred, confidence_map) in enumerate(zip(prediction_map_list, label_list, orig_prediction_list,confidence_map_list)):
        prediction_map = prediction_map + prediction_map.T - np.diag(
            np.diag(prediction_map))  # generate a symmetric matrix from a triangle matrix
        if SAFER_MODE==1:
            result_dict = double_masking_detection_confidence_more_warn_original(prediction_map, confidence_map, bear=j, orig_pred=orig_pred, label=label)
        else:
            result_dict = double_masking_detection_confidence_less_warn_original(prediction_map, confidence_map, bear=j, orig_pred=orig_pred, label=label)
        cert=result_dict['cert']
        warn=result_dict['warn']

        if orig_pred == label:
            correct_sample += 1
            if cert and warn:
                correct_warning_cert += 1
            elif warn and not cert:
                correct_warning_notcert += 1
            elif cert and not warn:
                correct_cert_nowarning += 1
            else:
                correct_nocert_nowarning += 1
        else:
            incorrect_sample+=1
            if cert and warn:
                incorrect_warning_cert += 1
                print(i)
                print(prediction_map)
                print(orig_pred)
                print(label)

                print("incorrect predicted cert")
            elif warn and not cert:
                incorrect_warning_notcert += 1

            elif cert and not warn:
                incorrect_cert_nowarning += 1
            else:
                incorrect_nocert_nowarning += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n")
    print("tau "+ str(j))
    print("correct_warning_cert " + str(correct_warning_cert)+" "+str(correct_warning_cert*100/NUM_IMG))
    print("correct_cert_nowarning " + str(correct_cert_nowarning)+" "+str(correct_cert_nowarning*100/NUM_IMG))
    print("correct_warning_notcert " + str(correct_warning_notcert)+" "+str(correct_warning_notcert*100/NUM_IMG))
    print("correct_nocert_nowarning " + str(correct_nocert_nowarning)+" "+str(correct_nocert_nowarning*100/NUM_IMG))

    print("incorrect_warning_cert " + str(incorrect_warning_cert)+" "+str(incorrect_warning_cert*100/NUM_IMG))
    print("incorrect_cert_nowarning " + str(incorrect_cert_nowarning)+" "+str(incorrect_cert_nowarning*100/NUM_IMG))
    print("incorrect_warning_notcert " + str(incorrect_warning_notcert)+" "+str(incorrect_warning_notcert*100/NUM_IMG))
    print("incorrect_nocert_nowarning " + str(incorrect_nocert_nowarning)+" "+str(incorrect_nocert_nowarning*100/NUM_IMG))


    print("\n")
    correct_sample = 0
    correct_warning_cert = 0
    correct_warning_notcert = 0
    correct_cert_nowarning = 0
    correct_nocert_nowarning = 0

    incorrect_sample = 0
    incorrect_warning_cert = 0
    incorrect_warning_notcert = 0
    incorrect_cert_nowarning = 0
    incorrect_nocert_nowarning = 0
