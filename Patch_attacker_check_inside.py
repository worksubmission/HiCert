import torch
import numpy as np
import pdb

from torchvision.utils import save_image, make_grid

from utils.defense import certify_precomputed
from utils.pd import double_masking_detection_confidence_more_warn_original, \
    double_masking_detection_confidence_less_warn_original


class PatchAttacker:
    def __init__(self, model, mean, std, ub, lb, kwargs):
        std = torch.tensor(std)
        mean = torch.tensor(mean)
        self.epsilon = kwargs["epsilon"] / std
        self.steps = kwargs["steps"]
        self.step_size = kwargs["step_size"] / std
        self.step_size.cuda()
        self.model = model
        self.mean = mean
        self.std = std
        self.random_start = kwargs["random_start"]

        # self.lb = (-mean / std)
        self.lb=torch.tensor(lb)
        self.lb.to('cuda')
        # self.ub = (1 - mean) / std
        self.ub=torch.tensor(ub)
        self.ub.to('cuda')
        self.patch_w = kwargs["patch_w"]
        self.patch_l = kwargs["patch_l"]

        self.criterion = torch.nn.CrossEntropyLoss()

    def perturb(self, inputs, labels, norm, mask_list,random_count=1):
        ADC_09_ATTACK_SUCCESS=False
        ADC_08_ATTACK_SUCCESS=False
        ADC_07_ATTACK_SUCCESS=False
        ADC_06_ATTACK_SUCCESS=False
        ADC_05_ATTACK_SUCCESS=False

        PG_plus_09_ATTACK_SUCCESS=False
        PG_plus_08_ATTACK_SUCCESS=False
        PG_plus_07_ATTACK_SUCCESS=False
        PG_plus_06_ATTACK_SUCCESS=False
        PG_plus_05_ATTACK_SUCCESS=False

        OMA_ATTACK_SUCCESS=False

        # with torch.no_grad():
        for param in self.model.parameters():
            param.requires_grad = False
        worst_x = None
        worst_loss = None
        for random_counter in range(random_count):
            # generate random patch center for each image
            idx = torch.arange(inputs.shape[0])[:, None]
            zero_idx = torch.zeros((inputs.shape[0],1), dtype=torch.long)
            w_idx = torch.randint(0, inputs.shape[2]-self.patch_w+1, (inputs.shape[0],1))
            l_idx = torch.randint(0, inputs.shape[3]-self.patch_l+1, (inputs.shape[0],1))
            idx = torch.cat([idx,zero_idx, w_idx, l_idx], dim=1)
            idx_list = [idx]
            for w in range(self.patch_w):
                for l in range(self.patch_l):
                    idx_list.append(idx + torch.tensor([0,0,w,l]))
            idx_list = torch.cat(idx_list, dim =0)

            # create mask
            mask = torch.zeros([inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]],
                               dtype=torch.bool).cuda()
            mask[idx_list[:,0],idx_list[:,1],idx_list[:,2],idx_list[:,3]] = True

            if self.random_start:
                init_delta = np.random.uniform(-self.epsilon, self.epsilon,
                                               [inputs.shape[0]*inputs.shape[2]*inputs.shape[3], inputs.shape[1]])
                init_delta = init_delta.reshape(inputs.shape[0],inputs.shape[2],inputs.shape[3], inputs.shape[1])
                init_delta = init_delta.swapaxes(1,3).swapaxes(2,3)
                x = inputs + torch.where(mask, torch.Tensor(init_delta).to('cuda'), torch.tensor(0.).cuda())

                x = torch.min(torch.max(x, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda()).detach()  # ensure valid pixel range
            else:
                x = inputs.data.detach().clone()

            x_init = inputs.data.detach().clone()

            x.requires_grad_()

            for step in range(self.steps):
                output = self.model(torch.where(mask, x, x_init))
                loss_ind = torch.nn.CrossEntropyLoss(reduction='none')(output, labels)
                # loss_ind = -torch.nn.CrossEntropyLoss(reduction='none')(output, labels)

                if worst_loss is None:
                    worst_loss = loss_ind.data.detach()
                    worst_x = x.data.detach()
                else:
                    worst_x = torch.where(worst_loss.ge(loss_ind.detach())[:, None, None, None], worst_x, x.data.detach())
                    worst_loss = torch.where(worst_loss.ge(loss_ind.detach()), worst_loss, loss_ind.data.detach())
                    # worst_x = torch.where(loss_ind.ge(worst_loss.detach())[:, None, None, None], worst_x, x.data.detach())
                    # worst_loss = torch.where(loss_ind.ge(worst_loss.detach()), worst_loss, loss_ind.data.detach())
                loss = loss_ind.sum()
                # # new
                # old
                grads = torch.autograd.grad(loss, [x])[0]

                if norm == float('inf'):
                    signed_grad_x = torch.sign(grads).detach()
                    delta = signed_grad_x * self.step_size[None, :, None, None].cuda()
                elif norm == 'l2':
                    delta = grads * self.step_size / grads.view(x.shape[0], -1).norm(2, dim=-1).view(-1, 1, 1, 1)

                x.data = delta + x.data.detach()

                # Project back into constraints ball and correct range
                x.data = self.project(x_init, x.data, norm, self.epsilon.cuda())
                x.data = x = torch.min(torch.max(x, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda())

                # check OMA
                clean_output = self.model(x)
                confidence = torch.nn.functional.softmax(clean_output, dim=1)

                confidence = confidence.detach().cpu().numpy()
                # prediction_map_list.append(prediction_map)
                clean_conf, clean_pred = clean_output.max(1)
                # print("clean_pred new"+str(clean_pred))
                clean_pred = clean_pred.detach().cpu().numpy()

                if not clean_pred== labels.cpu().numpy():
                    x_=x.data.detach()
                    prediction_map = np.zeros([1, 36], dtype=int)
                    confidence_map = np.zeros([1, 36])

                    ADC_09_warn = False
                    ADC_08_warn = False
                    PG_09_warn = False
                    PG_08_warn = False
                    for i, mask_ in enumerate(mask_list):

                        # for j in range(i, num_mask):
                        #     mask2 = mask_list[j]
                        masked_output = self.model(torch.where(mask_, x_, torch.tensor(0.).cuda()))
                        # plt.imshow(data.cpu()[0].permute(1, 2, 0), cmap='gray')
                        # plt.imshow((torch.where(torch.logical_and(mask,mask2),data,torch.tensor(0.).cuda()).cpu()[0]).permute(1, 2, 0))
                        # plt.show()
                        masked_output = torch.nn.functional.softmax(masked_output, dim=1)
                        masked_conf, masked_pred = masked_output.max(1)
                        masked_conf = masked_conf.detach().cpu().numpy()
                        confidence_map[:, i] = masked_conf
                        masked_pred = masked_pred.detach().cpu().numpy()
                        prediction_map[:, i] = masked_pred

                    prediction_map = prediction_map.reshape(-1)
                    confidence_map = confidence_map.reshape(-1)

                    if ADC_09_ATTACK_SUCCESS==False:
                        result_dict_ADC_09 = double_masking_detection_confidence_more_warn_original(prediction_map,
                                                                                                confidence_map,
                                                                                                bear=0.9,
                                                                                                orig_pred=clean_pred, label=labels)
                        if result_dict_ADC_09['warn']==False:
                            ADC_09_ATTACK_SUCCESS = True
                            print("ADC_09_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if ADC_08_ATTACK_SUCCESS==False:
                        result_dict_ADC_08 = double_masking_detection_confidence_more_warn_original(prediction_map,
                                                                                                    confidence_map,
                                                                                                    bear=0.8,
                                                                                                    orig_pred=clean_pred, label=labels)
                        if result_dict_ADC_08['warn'] == False:
                            ADC_08_ATTACK_SUCCESS = True
                            print("ADC_08_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if ADC_07_ATTACK_SUCCESS==False:
                        result_dict_ADC_07 = double_masking_detection_confidence_more_warn_original(prediction_map,
                                                                                                    confidence_map,
                                                                                                    bear=0.7,
                                                                                                    orig_pred=clean_pred, label=labels)
                        if result_dict_ADC_07['warn'] == False:
                            ADC_07_ATTACK_SUCCESS = True
                            print("ADC_07_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if ADC_06_ATTACK_SUCCESS==False:
                        result_dict_ADC_06 = double_masking_detection_confidence_more_warn_original(prediction_map,
                                                                                                    confidence_map,
                                                                                                    bear=0.6,
                                                                                                    orig_pred=clean_pred, label=labels)
                        if result_dict_ADC_06['warn'] == False:
                            ADC_06_ATTACK_SUCCESS = True
                            print("ADC_06_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if ADC_05_ATTACK_SUCCESS==False:
                        result_dict_ADC_05 = double_masking_detection_confidence_more_warn_original(prediction_map,
                                                                                                    confidence_map,
                                                                                                    bear=0.5,
                                                                                                    orig_pred=clean_pred, label=labels)
                        if result_dict_ADC_05['warn'] == False:
                            ADC_05_ATTACK_SUCCESS = True
                            print("ADC_05_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if PG_plus_09_ATTACK_SUCCESS == False:
                        result_dict_PG_09 = double_masking_detection_confidence_less_warn_original(prediction_map,
                                                                                                   confidence_map, bear=0.9,
                                                                                                   orig_pred=clean_pred, label=labels)
                        if result_dict_PG_09['warn']==False:
                            PG_plus_09_ATTACK_SUCCESS=True
                            print("PG_plus_09_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if PG_plus_08_ATTACK_SUCCESS == False:
                        result_dict_PG_08 = double_masking_detection_confidence_less_warn_original(prediction_map,
                                                                                               confidence_map, bear=0.8,
                                                                                               orig_pred=clean_pred, label=labels)
                        if result_dict_PG_08['warn']==False:
                            PG_plus_08_ATTACK_SUCCESS=True
                            print("PG_plus_08_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)


                    if PG_plus_07_ATTACK_SUCCESS == False:
                        result_dict_PG_07 = double_masking_detection_confidence_less_warn_original(prediction_map,
                                                                                               confidence_map, bear=0.7,
                                                                                               orig_pred=clean_pred, label=labels)
                        if result_dict_PG_07['warn']==False:
                            PG_plus_07_ATTACK_SUCCESS=True
                            print("PG_plus_07_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if PG_plus_06_ATTACK_SUCCESS == False:
                        result_dict_PG_06 = double_masking_detection_confidence_less_warn_original(prediction_map,
                                                                                               confidence_map, bear=0.6,
                                                                                               orig_pred=clean_pred, label=labels)
                        if result_dict_PG_06['warn']==False:
                            PG_plus_06_ATTACK_SUCCESS=True
                            print("PG_plus_06_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)

                    if PG_plus_05_ATTACK_SUCCESS == False:
                        result_dict_PG_05 = double_masking_detection_confidence_less_warn_original(prediction_map,
                                                                                               confidence_map, bear=0.5,
                                                                                               orig_pred=clean_pred, label=labels)
                        if result_dict_PG_05['warn']==False:
                            PG_plus_05_ATTACK_SUCCESS=True
                            print("PG_plus_05_ATTACK_SUCCESS!")
                            print("random_counter " + str(random_counter))
                            print("step " + str(step))
                            print(prediction_map)
                            print(confidence_map)
                    # print("result_dict_ADC_09 " + str(result_dict_ADC_09))
                    # print("result_dict_ADC_08 " + str(result_dict_ADC_08))
                    # print("result_dict_PG_09 " + str(result_dict_PG_09))
                    # print("result_dict_PG_08 " + str(result_dict_PG_08))
                    if certify_precomputed(prediction_map, clean_pred) and OMA_ATTACK_SUCCESS == False:
                        OMA_ATTACK_SUCCESS=True
                        # print("inside")
                        print("The attack break OMA!")
                        print("random_counter "+str(random_counter))
                        print("step "+str(step))
                        print(prediction_map)
                        print(confidence_map)

                    #     # ADCert
                    #     if masked_conf < 0.9:
                    #         ADC_09_warn = True
                    #     if masked_conf < 0.8:
                    #         ADC_08_warn =True
                    #     # PG++
                    #     if not masked_pred==clean_pred:
                    #         if masked_conf > 0.9:
                    #             PG_09_warn=True
                    #         if masked_conf > 0.8:
                    #             PG_08_warn=True
                    # if PG_09_warn==False and PG_plus_09_ATTACK_SUCCESS==False:
                    #     PG_plus_09_ATTACK_SUCCESS=True
                    #     print("The attack break the PG_plus_09_ATTACK_SUCCESS!")
                    #     print("random_counter "+str(random_counter))
                    #     print("step "+str(step))
                    #     print(prediction_map)
                    #     print(confidence_map)
                    #
                    # if PG_08_warn==False and PG_plus_08_ATTACK_SUCCESS==False:
                    #     PG_plus_08_ATTACK_SUCCESS=True
                    #     print("The attack break the PG_plus_08_ATTACK_SUCCESS!")
                    #     print("random_counter "+str(random_counter))
                    #     print("step "+str(step))
                    #     print(prediction_map)
                    #     print(confidence_map)
                    #
                    # if certify_precomputed(prediction_map, clean_pred):
                    #     if not ADC_09_warn and ADC_09_ATTACK_SUCCESS == False:
                    #         print("The attack break the ADC_09_ATTACK_SUCCESS!")
                    #         print("random_counter " + str(random_counter))
                    #         print("step " + str(step))
                    #         print(prediction_map)
                    #         print(confidence_map)
                    #         ADC_09_ATTACK_SUCCESS = True
                    #     if not ADC_08_warn and ADC_08_ATTACK_SUCCESS == False:
                    #         print("The attack break the ADC_08_ATTACK_SUCCESS!")
                    #         print("random_counter " + str(random_counter))
                    #         print("step " + str(step))
                    #         print(prediction_map)
                    #         print(confidence_map)
                    #         ADC_08_ATTACK_SUCCESS = True
                    # if certify_precomputed(prediction_map, clean_pred) and OMA_ATTACK_SUCCESS == False:
                    #     OMA_ATTACK_SUCCESS=True
                    #     print("inside")
                    #     print("The attack break OMA!")
                    #     print("random_counter "+str(random_counter))
                    #     print("step "+str(step))
                    #     print(prediction_map)
                    #     print(confidence_map)


                        # print(prediction_map)
                        # for i, mask_ in enumerate(mask_list):
                        #     # for j in range(i, num_mask):
                        #     #     mask2 = mask_list[j]
                        #     masked_output = self.model(torch.where(mask_, x_, torch.tensor(0.).cuda()))
                        #     # plt.imshow(data.cpu()[0].permute(1, 2, 0), cmap='gray')
                        #     # plt.imshow((torch.where(torch.logical_and(mask,mask2),data,torch.tensor(0.).cuda()).cpu()[0]).permute(1, 2, 0))
                        #     # plt.show()
                        #     masked_output = torch.nn.functional.softmax(masked_output, dim=1)
                        #     masked_conf, masked_pred = masked_output.max(1)
                        #     masked_conf = masked_conf.detach().cpu().numpy()
                        #     confidence_map[:, i] = masked_conf
                        #     masked_pred = masked_pred.detach().cpu().numpy()
                        #     prediction_map[:, i] = masked_pred
                            # save_image(make_grid(torch.where(mask_, x_, torch.tensor(0.).cuda()), nrow=1),
                            #            "./img/inside_real_attacks" + "_" + str(
                            #                self.patch_w)  +"_"+"_"+str(i)+"_mutant_pre" + str(
                            #                masked_pred) + ".jpg")
                        # return x.data.detach(), random_counter, step
        random_counter=random_count
        step=self.steps
        return worst_x, random_counter, step

    def project(self, x, x_adv, norm, eps, random=1):
        if norm == float('inf'):
            x_adv = torch.max(torch.min(x_adv, x + eps[None, :, None,None]), x - eps[None, :, None,None])
        elif norm == 'l2':
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta

        return x_adv
