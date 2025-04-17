import numpy as np
import torch


def one_masking_statistic(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    total=36
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)
    if len(pred) == 1: # unanimous agreement in the first-round masking
        return 0,pred # Case I: agreed prediction
    else:
        sorted_cnt = np.sort(cnt)
        sorted_idx = np.argsort(cnt)
        majority_pred = pred[sorted_idx][-1]
        return total-sorted_cnt[-1],majority_pred

def double_masking_detection_for_one_mask_agree(pred_one_mask,prediction_map,bear):
    dict={'warn':False,'cert':True}
    # warn=False
    # cert=True
    for i in range(len(pred_one_mask)):
        first_label = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] == first_label
        agreer_second_count = np.sum(tmp == 1)
        if agreer_second_count < bear:
            dict['warn']=True
            return dict
        else:
            continue
    return dict

def double_masking_detection_for_one_mask_agree_confidence(pred_one_mask, confidence_map, bear):
    dict={'warn':False,'cert':True}
    # warn=False
    # cert=True
    conf_one_mask = np.diag(confidence_map)
    # conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        first_label = pred_one_mask[i]
        confidence=conf_one_mask[i]
        if confidence < bear:
            dict['warn']=True
            return dict
        else:
            continue
    return dict

def double_masking_detection_for_one_mask_agree_less_warn(pred_one_mask,prediction_map,bear):
    dict={'warn':False,'cert':True}
    # warn=False
    # cert=True
    for i in range(len(pred_one_mask)):
        first_label = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] == first_label
        agreer_second_count = np.sum(tmp == 1)
        if agreer_second_count <= bear:
            dict['cert']=False
            return dict
        else:
            continue
    return dict

def double_masking_detection_for_one_mask_agree_confidence_less_warn(pred_one_mask, confidence_map, bear):
    dict={'warn':False,'cert':True}
    # warn=False
    # cert=True
    conf_one_mask = np.diag(confidence_map)
    for i in range(len(pred_one_mask)):
        first_label = pred_one_mask[i]
        confidence=conf_one_mask[i]
        if confidence <= bear:
            dict['cert']=False
            return dict
        else:
            continue
    return dict

def double_masking_detection_context_less_warn(prediction_map,bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': False}
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if (len(pred) == 1 and pred[0]==orig_pred): # unanimous agreement in the first-round masking
        dict=double_masking_detection_for_one_mask_agree_less_warn(pred_one_mask, prediction_map, bear)
        return dict
    dict['cert']=False

    # for pred_label in pred:
    #     if not pred_label==orig_pred:
    #         # second-round masking
    #         # get index list of the disagreer mask

    tmp = np.zeros_like(pred_one_mask,dtype=bool)
    for label in pred:
        if label!=orig_pred:
            tmp = np.logical_or(tmp,pred_one_mask==label)
    disagreer_pred_mask_idx = np.where(tmp)[0]
    for i in disagreer_pred_mask_idx:
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i]==dis
        agreer_second_count=np.sum(tmp==1)
        if agreer_second_count>bear:
            dict['cert'] = False
            dict['warn'] = True
            return dict  # -2 for not cert and warn
        else:
            continue
    dict['cert']= False
    dict['warn'] = False
    return dict #-1 for cert and warn

def double_masking_detection_confidence_less_warn_original(prediction_map,confidence_map, bear, orig_pred, label):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]

        if label==this_mask_pred and this_mask_conf > bear:
            pass
        else:
            dict['cert']=False
        if (not orig_pred==this_mask_pred) and this_mask_conf > bear:
            dict['warn'] = True
    return dict

def data_analsys_not_oma_low_confidence(prediction_map,confidence_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    OMA_check=True
    not_OMA_low_confidence_check=True
    OMA_high_confidence_check=True
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if not orig_pred==this_mask_pred:
            OMA_check=False
            if not this_mask_conf < bear:
                not_OMA_low_confidence_check=False
    if OMA_check==True:
        for i in range(len(pred_one_mask)):
            this_mask_conf = conf_one_mask[i]
            if this_mask_conf < bear:
                OMA_high_confidence_check=False

    return OMA_check, not_OMA_low_confidence_check, OMA_high_confidence_check

def data_analsys_not_oma_low_confidence_in_term_of_all_mutants(prediction_map,confidence_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    OMA_check=True
    OMA_num=0
    not_OMA_low_confidence_num=0
    OMA_high_confidence_num=0
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if not orig_pred==this_mask_pred:
            OMA_check=False
            if this_mask_conf < bear:
                not_OMA_low_confidence_num=not_OMA_low_confidence_num+1
        else:
            if not this_mask_conf < bear:
                OMA_high_confidence_num=OMA_high_confidence_num+1
    if OMA_check==True:
        OMA_num=OMA_num+1
    return OMA_num, not_OMA_low_confidence_num, OMA_high_confidence_num

def data_analsys_not_oma_low_confidence_in_term_of_all_mutants_collection(prediction_map,confidence_map, bear, orig_pred):

    pred_one_mask = prediction_map
    conf_one_mask = confidence_map
    conf_list_oma=[]
    conf_list_not_oma=[]
    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if orig_pred==this_mask_pred:
            conf_list_oma.append(this_mask_conf)
        else:
            conf_list_not_oma.append(this_mask_conf)
    return conf_list_oma,conf_list_not_oma

def data_analsys_consistent_or_not_confidence_in_term_of_all_mutants_collection_all(prediction_map,confidence_map, bear, orig_pred):

    pred_one_mask = prediction_map
    conf_one_mask = confidence_map
    conf_list_oma=[]
    conf_list_not_oma=[]
    OMA=True
    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if orig_pred==this_mask_pred:
            continue
        else:
            OMA=False

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if OMA==True:
            conf_list_oma.append(this_mask_conf)
        else:
            conf_list_not_oma.append(this_mask_conf)
            # if not this_mask_pred==orig_pred:
            #     conf_list_not_oma.append(this_mask_conf)
    return conf_list_oma,conf_list_not_oma

def data_analsys_consistent_or_not_confidence_in_term_of_all_mutants_collection_hicert(prediction_map,confidence_map, bear, orig_pred):

    pred_one_mask = prediction_map
    conf_one_mask = confidence_map
    conf_list_oma=[]
    conf_list_not_oma=[]
    OMA=True
    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if orig_pred==this_mask_pred:
            continue
        else:
            OMA=False

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if OMA==True:
            conf_list_oma.append(this_mask_conf)
        else:
            # conf_list_not_oma.append(this_mask_conf)
            if not this_mask_pred==orig_pred:
                conf_list_not_oma.append(this_mask_conf)
    return conf_list_oma,conf_list_not_oma

def data_analysis_confidence(prediction_map,confidence_map, orig_pred, label):
    pred_one_mask = prediction_map
    conf_one_mask = confidence_map
    # confidence_all_mutants=[]
    # confidence_only_inconsistent_mutants=[]
    # confidence_only_consistent_mutants=[]
    # incorrect_confidence_but_oma=[]
    # incorrect_confidence_inconsistent_mutants=[]
    confidence_not_warn_oma_correct=[]
    confidence_not_warn_oma_incorrect=[]
    confidence_inconsistent_mutants_correct=[]
    confidence_inconsistent_mutants_incorrect=[]
    # PG++
    confidence_warn_oma_correct=[]
    confidence_consistent_mutants_correct=[]

    confidence_warn_oma_incorrect=[]
    # no Thm2
    confidence_inconsistent_sample_all_mutants_correct=[]
    confidence_inconsistent_sample_all_mutants_incorrect=[]

    # 0227
    confidence_inconsistent_mutants=[]
    confidence_consistent_mutants=[]
    confidence_inconsistent_samples_all_mutants=[]
    "first, checking OMA y_0, OMA OMA orig_pred or not"
    OMA_y0=True
    OMA_pred=True
    correct_prediction=True
    if not orig_pred==label:
        correct_prediction=False

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        if not this_mask_pred==label:
            OMA_y0=False
        if not this_mask_pred==orig_pred:
            OMA_pred=False

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]
        # now check the warning
        # confidence_all_mutants.append(this_mask_conf)
        # now check the certification
        # if not this_mask_pred==label:
        #     confidence_only_inconsistent_mutants.append(this_mask_conf)
        # if OMA_y0==True and this_mask_pred==label:
        #     confidence_only_consistent_mutants.append(this_mask_conf)
        # if OMA_pred==True and not label==orig_pred:
        #     incorrect_confidence_but_oma.append(this_mask_conf)
        # if not label==orig_pred and not this_mask_pred==label:
        #     incorrect_confidence_inconsistent_mutants.append(this_mask_conf)

        if correct_prediction and OMA_pred:
            confidence_not_warn_oma_correct.append(this_mask_conf)
        if correct_prediction and not OMA_y0:
            if not this_mask_pred==label:
                confidence_inconsistent_mutants_correct.append(this_mask_conf)
        if not correct_prediction and OMA_pred:
            confidence_not_warn_oma_incorrect.append(this_mask_conf)
        if not correct_prediction and not OMA_y0:
            if not this_mask_pred==label:
                confidence_inconsistent_mutants_incorrect.append(this_mask_conf)

        # PG++
        if correct_prediction and not OMA_pred:
            confidence_warn_oma_correct.append(this_mask_conf)
        if correct_prediction and OMA_y0:
            confidence_consistent_mutants_correct.append(this_mask_conf)

        # PG++ new
        if not correct_prediction and not OMA_pred:
            confidence_warn_oma_incorrect.append(this_mask_conf)

        #  no Thm2
        if correct_prediction and not OMA_y0:
            confidence_inconsistent_sample_all_mutants_correct.append(this_mask_conf)
        if not correct_prediction and not OMA_y0:
            confidence_inconsistent_sample_all_mutants_incorrect.append(this_mask_conf)

        # new 0227
        if OMA_y0:
            confidence_consistent_mutants.append(this_mask_conf)
        if not OMA_y0:
            if not this_mask_pred==label:
                confidence_inconsistent_mutants.append(this_mask_conf)

        # sample level HiCErt
        if not OMA_y0:
            # if not this_mask_pred==label:
            confidence_inconsistent_samples_all_mutants.append(this_mask_conf)


        # if  and not this_mask_pred==label
    # return correct_prediction, OMA_y0, OMA_pred, confidence_all_mutants, confidence_only_inconsistent_mutants, confidence_only_consistent_mutants, incorrect_confidence_but_oma, incorrect_confidence_inconsistent_mutants
    return correct_prediction, OMA_y0, OMA_pred, confidence_not_warn_oma_correct, confidence_inconsistent_mutants_correct, confidence_not_warn_oma_incorrect, confidence_inconsistent_mutants_incorrect,\
        confidence_warn_oma_correct, confidence_consistent_mutants_correct, confidence_inconsistent_sample_all_mutants_correct, confidence_inconsistent_sample_all_mutants_incorrect, confidence_warn_oma_incorrect,\
        confidence_consistent_mutants,confidence_inconsistent_mutants, confidence_inconsistent_samples_all_mutants



def data_analysis_the_number_of_mutant(prediction_map,confidence_map, orig_pred, label):
    pred_one_mask = prediction_map
    counter_inconsistent_mutant=0
    counter_diversity_inconsistent_mutant=0
    label_inconsistent=[]
    "first, checking OMA y_0, OMA OMA orig_pred or not"

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        if not this_mask_pred==label:
            if this_mask_pred in label_inconsistent:
                continue
            else:
                label_inconsistent.append(this_mask_pred)
                counter_diversity_inconsistent_mutant=counter_diversity_inconsistent_mutant+1
            counter_inconsistent_mutant=counter_inconsistent_mutant+1
    return counter_inconsistent_mutant, counter_diversity_inconsistent_mutant

# def data_analsys_not_oma_low_confidence_in_term_of_all_mutants_collection_for_different_sample(prediction_map,confidence_map, bear, orig_pred):
#
#     pred_one_mask = prediction_map
#     conf_one_mask = confidence_map
#     conf_list_oma=[]
#     conf_list_not_oma=[]
#     for i in range(len(pred_one_mask)):
#         this_mask_pred=pred_one_mask[i]
#         this_mask_conf=conf_one_mask[i]
#         if orig_pred==this_mask_pred:
#             conf_list_oma.append(this_mask_conf)
#         else:
#             conf_list_not_oma.append(this_mask_conf)
#     return conf_list_oma,conf_list_not_oma

def double_masking_detection_confidence_less_warn_original_tighter(prediction_map,confidence_full_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask_all_label = confidence_full_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf_all_label  = conf_one_mask_all_label[i]

        if orig_pred==this_mask_pred and this_mask_conf_all_label[orig_pred] > bear:
            pass
        else:
            dict['cert']=False
        if (not orig_pred==this_mask_pred) and this_mask_conf_all_label[orig_pred] > bear:
            dict['warn'] = True
    return dict


def double_masking_detection_confidence_less_warn_inverse(prediction_map, confidence_full_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask_full = confidence_full_map

    for i in range(len(pred_one_mask)):
        this_mask_pred = pred_one_mask[i]
        this_mask_conf_full = conf_one_mask_full[i]
        this_mask_orig_pred_confidence=this_mask_conf_full[orig_pred]
        this_mask_conf_full=np.delete(this_mask_conf_full,orig_pred)
        if orig_pred == this_mask_pred and np.max(this_mask_conf_full) < bear:
            pass
        else:
            dict['cert'] = False
        if (not orig_pred == this_mask_pred) and this_mask_orig_pred_confidence < bear:
            dict['warn'] = True
    return dict

    # pred,cnt = np.unique(pred_one_mask,return_counts=True)
    #
    # if (len(pred) == 1 and pred[0]==orig_pred): # unanimous agreement in the first-round masking
    #     dict=double_masking_detection_for_one_mask_agree_confidence_less_warn(pred_one_mask, confidence_map, bear)
    #     return dict
    # dict['cert']=False
    #
    # # for pred_label in pred:
    # #     if not pred_label==orig_pred:
    # #         # second-round masking
    # #         # get index list of the disagreer mask
    #
    # conf_one_mask = np.diag(confidence_map)
    # tmp = np.zeros_like(pred_one_mask,dtype=bool)
    # for label in pred:
    #     if label!=orig_pred:
    #         tmp = np.logical_or(tmp,pred_one_mask==label)
    # disagreer_pred_mask_idx = np.where(tmp)[0]
    # for i in disagreer_pred_mask_idx:
    #     dis = pred_one_mask[i]
    #     conf = conf_one_mask[i]
    #     # check all two-mask predictions
    #     # tmp = prediction_map[i]==dis
    #     # agreer_second_count=np.sum(tmp==1)
    #     if conf>bear:
    #         dict['cert'] = False
    #         dict['warn'] = True
    #         return dict  # -2 for not cert and warn
    #     else:
    #         continue
    # dict['cert'] = False
    # dict['warn'] = False
    # return dict #-1 for cert and warn


def double_masking_detection_context_more_warn(prediction_map,bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': False}
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if len(pred) == 1 and pred[0]==orig_pred: # unanimous agreement in the first-round masking
        dict=double_masking_detection_for_one_mask_agree(pred_one_mask, prediction_map, bear)
        return dict
    dict['warn']=True

    # for pred_label in pred:
    #     if not pred_label==orig_pred:
    #         # second-round masking
    #         # get index list of the disagreer mask

    tmp = np.zeros_like(pred_one_mask,dtype=bool)
    for label in pred:
        if label!=orig_pred:
            tmp = np.logical_or(tmp,pred_one_mask==label)
    disagreer_pred_mask_idx = np.where(tmp)[0]
    for i in disagreer_pred_mask_idx:
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i]==dis
        agreer_second_count=np.sum(tmp==1)
        if agreer_second_count<bear:
            continue
        else:
            dict['cert'] = False
            dict['warn'] = True
            return dict #-2 for not cert and warn
    dict['cert'] = True
    dict['warn'] = True
    return dict #-1 for cert and warn


def double_masking_detection_only_confidence_truelabelshouldlarge(prediction_map, confidence_full_map, bear, orig_pred):
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask_full = confidence_full_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf_full=conf_one_mask_full[i]

        if this_mask_conf_full[orig_pred] > bear:
            pass
        else:
            dict['cert']=False
        this_mask_conf_full=np.delete(this_mask_conf_full,orig_pred)
        if np.max(this_mask_conf_full)>bear:
            dict['warn'] =True
        # for i in range(len(this_mask_conf_full)):
        #     if i == orig_pred:
        #         continue
        #     else:
        #         conf_one_mask_single=this_mask_conf_full[i]
        #         if conf_one_mask_single > bear:
        #             dict['warn'] = True
    return dict

def double_masking_detection_only_confidence_otherlabelshouldless(prediction_map, confidence_full_map, bear, orig_pred):
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    pred_one_mask = prediction_map
    conf_one_mask_full = confidence_full_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf_full=conf_one_mask_full[i]
        if this_mask_conf_full[orig_pred] < bear:
            dict['warn']=True
        this_mask_conf_full=np.delete(this_mask_conf_full,orig_pred)
        if np.max(this_mask_conf_full)<bear:
            pass
        else:
            dict['cert'] = False

        # for i in range(len(this_mask_conf_full)):
        #     if i == orig_pred:
        #         continue
        #     else:
        #         conf_one_mask_single=this_mask_conf_full[i]
        #         if conf_one_mask_single< bear:
        #             pass
        #         else:
        #             dict['cert'] = False

        # if this_mask_conf_full[orig_pred] < bear:
        #     dict['cert']=False
        # for i in range(len(this_mask_conf_full)):
        #     if i == orig_pred:
        #         continue
        #     else:
        #         conf_one_mask_single=this_mask_conf_full[i]
        #         if conf_one_mask_single > bear:
        #             dict['warn'] = True
    return dict

# def double_masking_detection_confidence_more_warn_inverse(prediction_map, confidence_full_map, bear, orig_pred):
#     # perform double masking inference with the pre-computed two-mask predictions
#     '''
#     INPUT:
#     prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point
#
#     OUTPUT:         int, the prediction label
#     '''
#     dict = {'warn': False, 'cert': True}
#     # first-round masking
#     # pred_one_mask = np.diag(prediction_map)
#     # conf_one_mask = np.diag(confidence_map)
#
#     pred_one_mask = prediction_map
#     conf_one_mask_all_label = confidence_full_map
#
#     for i in range(len(pred_one_mask)):
#         this_mask_pred = pred_one_mask[i]
#         this_mask_conf_all_label  = conf_one_mask_all_label[i]
#         if orig_pred == this_mask_pred or this_mask_conf_all_label[orig_pred] > bear:
#             pass
#         else:
#             dict['cert'] = False
#         this_mask_conf_all_label=np.delete(this_mask_conf_all_label, orig_pred)
#         if (not orig_pred == this_mask_pred) or np.max(this_mask_conf_all_label) > bear:
#             dict['warn'] = True
#     return dict


def double_masking_detection_confidence_more_warn_original(prediction_map,confidence_map, bear, orig_pred, label):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    #
    pred_one_mask = prediction_map
    conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]

        if label==this_mask_pred or this_mask_conf < bear:
            pass
        else:
            dict['cert']=False
        if (not orig_pred==this_mask_pred) or this_mask_conf < bear:
            dict['warn'] = True
    return dict

def double_masking_detection_confidence_more_warn_ablation_max_min_only_confidence(prediction_map,confidence_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)
    #
    pred_one_mask = prediction_map
    conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]

        if this_mask_conf < bear:
            pass
        else:
            dict['cert']=False
        if this_mask_conf < bear:
            dict['warn'] = True
    return dict


def how_many_inconsistent_mutants(prediction_map,confidence_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    if prediction_map.ndim == 2:
        prediction_map = np.diag(prediction_map)
    count = 0
    for pred in prediction_map:
        if pred != orig_pred:
            count += 1
    return count




def double_masking_detection_confidence_more_warn_original_onlyconfidence(prediction_map,confidence_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)

    pred_one_mask = prediction_map
    conf_one_mask = confidence_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf=conf_one_mask[i]

        if this_mask_conf < bear:
            pass
        else:
            dict['cert']=False
        if this_mask_conf < bear:
            dict['warn'] = True
    return dict

def double_masking_detection_confidence_more_warn_original_tighter(prediction_map,confidence_full_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)

    pred_one_mask = prediction_map
    conf_one_mask_all_label = confidence_full_map

    for i in range(len(pred_one_mask)):
        this_mask_pred=pred_one_mask[i]
        this_mask_conf_all_label  = conf_one_mask_all_label[i]

        if orig_pred==this_mask_pred or this_mask_conf_all_label[orig_pred] < bear:
            pass
        else:
            dict['cert']=False
        if (not orig_pred==this_mask_pred) or this_mask_conf_all_label[orig_pred] < bear:
            dict['warn'] = True
    return dict

def double_masking_detection_confidence_more_warn_inverse(prediction_map, confidence_full_map, bear, orig_pred):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    dict = {'warn': False, 'cert': True}
    # first-round masking
    # pred_one_mask = np.diag(prediction_map)
    # conf_one_mask = np.diag(confidence_map)

    pred_one_mask = prediction_map
    conf_one_mask_all_label = confidence_full_map

    for i in range(len(pred_one_mask)):
        this_mask_pred = pred_one_mask[i]
        this_mask_conf_all_label  = conf_one_mask_all_label[i]
        if orig_pred == this_mask_pred or this_mask_conf_all_label[orig_pred] > bear:
            pass
        else:
            dict['cert'] = False
        this_mask_conf_all_label=np.delete(this_mask_conf_all_label, orig_pred)
        if (not orig_pred == this_mask_pred) or np.max(this_mask_conf_all_label) > bear:
            dict['warn'] = True
    return dict



    # old
    # pred,cnt = np.unique(pred_one_mask,return_counts=True)

    # if len(pred) == 1 and pred[0]==orig_pred: # unanimous agreement in the first-round masking
    #     dict=double_masking_detection_for_one_mask_agree_confidence(pred_one_mask, confidence_map, bear)
    #     return dict
    # dict['warn']=True
    #
    # # for pred_label in pred:
    # #     if not pred_label==orig_pred:
    # #         # second-round masking
    # #         # get index list of the disagreer mask
    #
    # conf_one_mask = np.diag(confidence_map)
    # # conf_one_mask = confidence_map
    #
    # tmp = np.zeros_like(pred_one_mask,dtype=bool)
    # for label in pred:
    #     if label!=orig_pred:
    #         tmp = np.logical_or(tmp,pred_one_mask==label)
    # disagreer_pred_mask_idx = np.where(tmp)[0]
    # for i in disagreer_pred_mask_idx:
    #     dis = pred_one_mask[i]
    #     # check disagreer's confidence
    #     conf = conf_one_mask[i]
    #     if conf<bear:
    #         continue
    #     else:
    #         dict['cert'] = False
    #         dict['warn'] = True
    #         return dict #-2 for not cert and warn
    # dict['cert'] = True
    # dict['warn'] = True
    # return dict #-1 for cert and warn

def double_masking_detection_nolemma1(prediction_map,bear):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if len(pred) == 1: # unanimous agreement in the first-round masking
        return pred[0],double_masking_detection_for_one_mask_agree(pred_one_mask,prediction_map,bear) # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask,dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp,pred_one_mask==dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in range(len(tmp)):
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i]!=dis
        disagreer_second_count=np.sum(tmp==1)
        if disagreer_second_count>bear:
            continue
        else:
            return majority_pred, -2

    return majority_pred,-1 # Case III: majority prediction