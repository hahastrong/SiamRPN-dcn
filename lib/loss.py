import torch, pdb
import torch.nn
from IPython import embed
import torch.nn.functional as F
import pdb
import random
import numpy as np
import time
import functools

from lib.utils import box_transform, compute_iou, box_transform_inv
from .utils import nms, add_box_img, nms_worker
from torch.multiprocessing import Pool, Manager


def rpn_cross_entropy(input, target):
    r"""
    :param input: (15x15x5,2)
    :param target: (15x15x5,)
    :return:
    """
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu])
    return loss


# def rpn_cross_entropy_balance_worker(num_pos, num_neg, anchors, ohem, x):
#     input, target = x
#     min_pos = min(len(np.where(target.cpu() == 1)[0]), num_pos)
#     min_neg = int(min(len(np.where(target.cpu() == 1)[0]) * num_neg / num_pos, num_neg))
#     if not ohem:
#         pos_index = random.sample(np.where(target.cpu() == 1)[0].tolist(), min_pos)
#         neg_index = random.sample(np.where(target.cpu() == 0)[0].tolist(), min_neg)
#         if len(pos_index) > 0:
#             pos_loss_bid = F.cross_entropy(input=input[pos_index],
#                                            target=target[pos_index], reduction='none')
#             neg_loss_bid = F.cross_entropy(input=input[neg_index],
#                                            target=target[neg_index], reduction='none')
#         else:
#             pos_loss_bid = torch.FloatTensor([0]).cuda()
#             neg_index = random.sample(np.where(target.cpu() == 0)[0].tolist(), num_neg)
#             neg_loss_bid = F.cross_entropy(input=input[neg_index],
#                                            target=target[neg_index], reduction='none')
#         loss_bid = (pos_loss_bid.mean() + neg_loss_bid.mean()) / 2
#     else:
#         pos_index = np.where(target.cpu() == 1)[0].tolist()
#         neg_index = np.where(target.cpu() == 0)[0].tolist()
#         if len(pos_index) > 0:
#             pos_loss_bid = F.cross_entropy(input=input[pos_index],
#                                            target=target[pos_index], reduction='none')
#             selected_pos_index = nms(anchors[pos_index], pos_loss_bid.cpu().detach().numpy(), min_pos)
#             pos_loss_bid_ohem = pos_loss_bid[selected_pos_index]
#             neg_loss_bid = F.cross_entropy(input=input[neg_index],
#                                            target=target[neg_index], reduction='none')
#             selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), min_neg)
#             neg_loss_bid_ohem = neg_loss_bid[selected_neg_index]
#         else:
#             pos_loss_bid = torch.FloatTensor([0]).cuda()
#             pos_loss_bid_ohem = pos_loss_bid
#             neg_loss_bid = F.cross_entropy(input=input[neg_index],
#                                            target=target[neg_index], reduction='none')
#             selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), num_neg)
#             neg_loss_bid_ohem = neg_loss_bid[selected_neg_index]
#         loss_bid = (pos_loss_bid_ohem.mean() + neg_loss_bid_ohem.mean()) / 2
#     return loss_bid
#
#
# def rpn_cross_entropy_balance_parallel(input, target, num_pos, num_neg, anchors, ohem=None, num_threads=4):
#     loss_all = []
#     input, target = input.cpu(), target.cpu()
#     x = [[input[i], target[i]] for i in range(target.shape[0])]
#     functools.partial(rpn_cross_entropy_balance_worker, num_pos, num_neg, anchors, ohem)([input[0], target[0]])
#     with Pool(processes=num_threads) as pool:
#         for loss_bid in pool.imap_unordered(
#                 functools.partial(rpn_cross_entropy_balance_worker, num_pos, num_neg, anchors, ohem), x):
#             loss_all.append(loss_bid)
#     final_loss = torch.stack(loss_all).mean()
#     return final_loss

# def rpn_cross_entropy_balance_worker(num_pos, num_neg, anchors, ohem, x):
#     batch_id, input, target = x
#     min_pos = min(len(np.where(target.cpu() == 1)[0]), num_pos)
#     min_neg = int(min(len(np.where(target.cpu() == 1)[0]) * num_neg / num_pos, num_neg))
#     pos_index = np.where(target.cpu() == 1)[0].tolist()
#     neg_index = np.where(target.cpu() == 0)[0].tolist()
#     if len(pos_index) > 0:
#         pos_loss_bid = F.cross_entropy(input=input[pos_index],
#                                        target=target[pos_index], reduction='none')
#         selected_pos_index = nms(anchors[pos_index], pos_loss_bid.cpu().detach().numpy(), min_pos)
#
#         neg_loss_bid = F.cross_entropy(input=input[neg_index],
#                                        target=target[neg_index], reduction='none')
#         selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), min_neg)
#     else:
#         selected_pos_index = [0]
#
#         neg_loss_bid = F.cross_entropy(input=input[neg_index],
#                                        target=target[neg_index], reduction='none')
#         selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), num_neg)
#
#     return batch_id, selected_pos_index, selected_neg_index
#
#
# def rpn_cross_entropy_balance_parallel(input, target, num_pos, num_neg, anchors, ohem=True, num_threads=4):
#     selected_pos_index_all = {}
#     selected_neg_index_all = {}
#     # input, target = input.cpu().detach(), target.cpu().detach()
#     x = [[i, input.cpu().detach()[i], target.cpu().detach()[i]] for i in range(target.shape[0])]
#     # functools.partial(rpn_cross_entropy_balance_worker, num_pos, num_neg, anchors, ohem)(x[0])
#     with Pool(processes=num_threads) as pool:
#         for ret in pool.imap_unordered(
#                 functools.partial(rpn_cross_entropy_balance_worker, num_pos, num_neg, anchors, ohem), x):
#             selected_pos_index_all[ret[0]] = ret[1]
#             selected_neg_index_all[ret[0]] = ret[2]
#
#     loss_all = []
#     for batch_id in range(target.shape[0]):
#         pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
#         neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist()
#         if len(pos_index) > 0:
#             pos_loss_bid = F.cross_entropy(input=input[batch_id][pos_index],
#                                            target=target[batch_id][pos_index], reduction='none')
#             selected_pos_index = selected_pos_index_all[batch_id]
#             pos_loss_bid_ohem = pos_loss_bid[selected_pos_index]
#             neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
#                                            target=target[batch_id][neg_index], reduction='none')
#             selected_neg_index = selected_neg_index_all[batch_id]
#             neg_loss_bid_ohem = neg_loss_bid[selected_neg_index]
#         else:
#             pos_loss_bid = torch.FloatTensor([0]).cuda()
#             pos_loss_bid_ohem = pos_loss_bid
#             neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
#                                            target=target[batch_id][neg_index], reduction='none')
#             selected_neg_index = selected_neg_index_all[batch_id]
#             neg_loss_bid_ohem = neg_loss_bid[selected_neg_index]
#         loss_bid = (pos_loss_bid_ohem.mean() + neg_loss_bid_ohem.mean()) / 2
#         loss_all.append(loss_bid)
#     final_loss = torch.stack(loss_all).mean()
#     return final_loss


# def rpn_cross_entropy_balance_ohem_parallel(input, target, num_pos, num_neg, anchors, ohem=True, num_threads=4):
#     r"""
#     :param input: (N,1125,2)
#     :param target: (15x15x5,)
#     :return:
#     """
#     loss_all = []
#     loss_all_pos = []
#     loss_all_neg = []
#     pos_index_all = []
#     neg_index_all = []
#     selected_pos_index_all = []
#     selected_neg_index_all = []
#     min_pos_all = []
#     min_neg_all = []
#
#     for batch_id in range(target.shape[0]):
#         min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
#         min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg))
#         pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
#         neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist()
#         if len(pos_index) > 0:
#             pos_loss_bid = F.cross_entropy(input=input[batch_id][pos_index],
#                                            target=target[batch_id][pos_index], reduction='none')
#             loss_all_pos.append(pos_loss_bid)
#             pos_index_all.append(pos_index)
#             min_pos_all.append(min_pos)
#
#             neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
#                                            target=target[batch_id][neg_index], reduction='none')
#             loss_all_neg.append(neg_loss_bid)
#             neg_index_all.append(neg_index)
#             min_neg_all.append(min_neg)
#         else:
#             pos_loss_bid = torch.FloatTensor([0]).cuda()
#             loss_all_pos.append(pos_loss_bid)
#             pos_index_all.append(pos_index)
#             min_pos_all.append(0)
#
#             neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
#                                            target=target[batch_id][neg_index], reduction='none')
#             loss_all_neg.append(neg_loss_bid)
#             neg_index_all.append(neg_index)
#             min_neg_all.append(num_neg)
#
#     functools.partial(nms_worker)([anchors[pos_index_all[0]], loss_all_pos[0].cpu().detach().numpy(), min_pos_all[0]])
#
#     x_pos = [[anchors[pos_index_all[i]], loss_all_pos[i].cpu().detach().numpy(), min_pos_all[i]] for i in
#              range(target.shape[0])]
#     x_neg = [[anchors[neg_index_all[i]], loss_all_neg[i].cpu().detach().numpy(), min_neg_all[i]] for i in
#              range(target.shape[0])]
#     with Pool(processes=num_threads) as pool:
#         for selected_pos_index in pool.imap(functools.partial(nms_worker), x_pos):
#             selected_pos_index_all.append(selected_pos_index)
#     with Pool(processes=num_threads) as pool:
#         for selected_neg_index in pool.imap(functools.partial(nms_worker), x_neg):
#             selected_neg_index_all.append(selected_neg_index)
#     for batch_id in range(target.shape[0]):
#         pos_loss_bid_ohem = loss_all_pos[batch_id][selected_pos_index_all[batch_id]]
#         neg_loss_bid_ohem = loss_all_neg[batch_id][selected_neg_index_all[batch_id]]
#         loss_bid = (pos_loss_bid_ohem.mean() + neg_loss_bid_ohem.mean()) / 2
#         loss_all.append(loss_bid)
#     final_loss = torch.stack(loss_all).mean()
#     return final_loss


def rpn_cross_entropy_balance(input, target, num_pos, num_neg, anchors, ohem_pos=None, ohem_neg=None):
    r"""
    :param input: (N,1125,2)
    :param target: (15x15x5,)
    :return:
    """
    # if ohem:
    #     final_loss = rpn_cross_entropy_balance_parallel(input, target, num_pos, num_neg, anchors, ohem=True,
    #                                                     num_threads=4)
    # else:
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
        min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg))
        pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
        neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist()

        if ohem_pos:
            if len(pos_index) > 0:
                pos_loss_bid = F.cross_entropy(input=input[batch_id][pos_index],
                                               target=target[batch_id][pos_index], reduction='none')
                selected_pos_index = nms(anchors[pos_index], pos_loss_bid.cpu().detach().numpy(), min_pos)
                pos_loss_bid_final = pos_loss_bid[selected_pos_index]
            else:
                pos_loss_bid = torch.FloatTensor([0]).cuda()
                pos_loss_bid_final = pos_loss_bid
        else:
            pos_index_random = random.sample(pos_index, min_pos)
            if len(pos_index) > 0:
                pos_loss_bid_final = F.cross_entropy(input=input[batch_id][pos_index_random],
                                                     target=target[batch_id][pos_index_random], reduction='none')
            else:
                pos_loss_bid_final = torch.FloatTensor([0]).cuda()

        if ohem_neg:
            if len(pos_index) > 0:
                neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
                                               target=target[batch_id][neg_index], reduction='none')
                selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), min_neg)
                neg_loss_bid_final = neg_loss_bid[selected_neg_index]
            else:
                neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
                                               target=target[batch_id][neg_index], reduction='none')
                selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), num_neg)
                neg_loss_bid_final = neg_loss_bid[selected_neg_index]
        else:
            if len(pos_index) > 0:
                neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), min_neg)
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduction='none')
            else:
                neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), num_neg)
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduction='none')
        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2
        loss_all.append(loss_bid)
    final_loss = torch.stack(loss_all).mean()
    return final_loss


def rpn_smoothL1(input, target, label, num_pos=16, ohem=None):
    r'''
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    '''


    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos)
        if ohem:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index], reduction='none')
                sort_index = torch.argsort(loss_bid.mean(1))
                loss_bid_ohem = loss_bid[sort_index[-num_pos:]]
            else:
                loss_bid_ohem = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_bid_ohem.mean())
        else:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index])
            else:
                loss_bid = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss


def cal_loss(input_cls, target_cls, input_reg, target_reg, ground_truth, anchors, num_pos=16, num_neg=48):
    """
    calculate the cls loss and regression loss using IoU-balanced Loss
    """
    loss_all_cls = []
    loss_all_bbox = []
    box_pred = box_transform_inv(anchors, input_reg)
    iou = compute_iou(box_pred, ground_truth)


    for batch_id in range(target_cls.shape[0]):
        # pos loss calculate
        min_pos = min(len(np.where(target_cls[batch_id].cpu() == 1)[0]), num_pos)
        min_neg = int(min(len(np.where(target_cls[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg))
        pos_index = np.where(target_cls[batch_id].cpu() == 1)[0].tolist()
        neg_index = np.where(target_cls[batch_id].cpu() == 0)[0].tolist()
    
        if len(pos_index) > 0:
            pos_index_random = random.sample(pos_index, min_pos)
            CE_pos = F.cross_entropy(input=input_cls[batch_id][pos_index_random],
                                                target=target_cls[batch_id][pos_index_random], reduction='none')

            
            # lamda need t modify
            iou_cls = torch.pow(iou[pos_index_random], 1.5)
            w_iou_cls_mul = torch.mul(iou_cls, CE_pos)
            w_iou_cls = w_iou_cls_mul / torch.sum(w_iou_cls_mul)
            pos_loss_bid_final = torch.mul(w_iou_cls, CE_pos)

            # bbox
            reg_bid = F.smooth_l1_loss(input_reg[batch_id][pos_index], target_reg[batch_id][pos_index], reduction='none')

            # lamda need t modify
            iou_bbox = torch.pow(iou[pos_index_random], 1.5)
            w_iou_bbox_mul = torch.mul(iou_bbox, CE_pos)
            w_iou_bbox = w_iou_bbox_mul / torch.sum(w_iou_bbox_mul)
            bbox_bid_final = torch.mul(w_iou_bbox, reg_bid)

        else:
            pos_loss_bid_final = torch.FloatTensor([0]).cuda()
        


        # negative calculate
        if len(pos_index) > 0:
            neg_index_random = random.sample(neg_index, min_neg)
            neg_loss_bid_final = F.cross_entropy(input=input_cls[batch_id][neg_index_random],
                                                    target=target_cls[batch_id][neg_index_random], reduction='none')
        else:
            neg_index_random = random.sample(neg_index, num_neg)
            neg_loss_bid_final = F.cross_entropy(input=input_cls[batch_id][neg_index_random],
                                                    target=target_cls[batch_id][neg_index_random], reduction='none')
        loss_bid_cls = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2
        loss_all_cls.append(loss_bid_cls)
        loss_all_bbox.append(bbox_bid_final.mean())

    final_loss_cls = torch.stack(loss_all_cls).mean()
    final_loss_bbox = torch.stack(loss_all_bbox).mean()
    return final_loss_cls, final_loss_bbox












