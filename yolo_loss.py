import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def make_targets(pred_boxes, target, anchor, best_ns, s, num_anchors, num_classes, nH, nW, sil_thresh):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = int(len(anchor) / num_anchors)

    mask = torch.zeros([nB, nA, nH, nW], dtype=torch.uint8)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW, nC)
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 3] + target[b][t * 5 + 4] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            best_n = int(best_ns[b][s][t].item())

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b, best_n, gj, gi]

            mask[b][best_n][gj][gi] = 1
            tx[b][best_n][gj][gi] = gx - gi
            ty[b][best_n][gj][gi] = gy - gj
            tw[b][best_n][gj][gi] = math.log(gw / (anchor[anchor_step * best_n] + 1e-12))
            th[b][best_n][gj][gi] = math.log(gh / (anchor[anchor_step * best_n + 1] + 1e-12))
            iou = bbox_iou(gt_box, pred_box.data, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi][int(target[b][t * 5])] = 1

            if iou > 0.5 and pred_box[4] > sil_thresh and target[b][t * 5].item() == pred_box[5].item():
                nCorrect = nCorrect + 1

    return nCorrect, mask, tx, ty, tw, th, tconf, tcls

def YoloLoss(outputs, target, num_classes, anchors, anchor_step, seen, device):
    coord_scale = 5
    noobject_scale = 0.5
    object_scale = 1
    class_scale = 1
    thresh = 0.5

    nB = outputs[0].size(0)
    nA = int(len(anchors[0]) / anchor_step)
    nC = num_classes

    nGT = 0
    nCorrect = [0, 0, 0]
    nProposals = [0, 0, 0]
    # best_n = get_best(target, anchors, anchor_step, len(outputs), nB)
    best_n = torch.zeros([nB, len(outputs), 50], dtype=torch.uint8)
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 3] == 0:
                break
            nGT += 1
            for j in range(len(outputs)):
                nH = outputs[j].size(2)
                nW = outputs[j].size(3)
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                gt_box = [0, 0, gw, gh]
                best_iou = 0.0
                for n in range(nA):
                    aw = anchors[j][anchor_step * n]
                    ah = anchors[j][anchor_step * n + 1]
                    anchor_box = [0, 0, aw, ah]
                    iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                    if iou > best_iou:
                        best_iou = iou
                        best_n[b][j][t] = n

    loss = []
    for s in range(len(outputs)):
        nH = outputs[s].size(2)
        nW = outputs[s].size(3)
        output = outputs[s].view(nB, nA, (5 + nC), nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        anchor = anchors[s]
        # Get outputs
        x = torch.sigmoid(output[..., 0])  # Center x
        y = torch.sigmoid(output[..., 1])  # Center y
        w = output[..., 2]  # Width
        h = output[..., 3]  # Height
        conf = torch.sigmoid(output[..., 4])  # Conf
        cls = torch.sigmoid(output[..., 5:])  # Cls pred.

        pred_boxes = torch.FloatTensor(6, nB, nA, nH, nW).to(device)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB, nA, nH, nW).to(device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB, nA, nH, nW).to(device)

        anchor_w = torch.Tensor(anchor).view(nA, anchor_step).index_select(1, torch.LongTensor([0])).to(device)
        anchor_h = torch.Tensor(anchor).view(nA, anchor_step).index_select(1, torch.LongTensor([1])).to(device)
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB, nA, nH, nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB, nA, nH, nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes[4] = conf.data
        _, cls_max_ids = torch.max(cls, 4)
        pred_boxes[5] = cls_max_ids
        pred_boxes = pred_boxes.permute(1, 2, 3, 4, 0).contiguous().cpu()

        nCorrect[s], mask, tx, ty, tw, th, tconf, tcls = \
            make_targets(pred_boxes, target.data, anchor, best_n, s, nA, nC, nH, nW, thresh)

        nProposals[s] = (conf > 0.5).sum().item()

        tx = tx.to(device)
        ty = ty.to(device)
        tw = tw.to(device)
        th = th.to(device)
        tconf = tconf.to(device)
        tcls = tcls.to(device)
        mask = mask.to(device)
        if len(tconf[mask == 1]) != 0:
            loss_x = coord_scale * nn.MSELoss(reduction='sum')(x[mask == 1], tx[mask == 1])
            loss_y = coord_scale * nn.MSELoss(reduction='sum')(y[mask == 1], ty[mask == 1])
            loss_w = coord_scale * nn.MSELoss(reduction='sum')(w[mask == 1], tw[mask == 1])
            loss_h = coord_scale * nn.MSELoss(reduction='sum')(h[mask == 1], th[mask == 1])
            loss_conf = object_scale * nn.MSELoss(reduction='sum')(conf[mask == 1], tconf[mask == 1]) + \
                        noobject_scale * nn.MSELoss(reduction='sum')(conf[mask == 0], tconf[mask == 0])
            loss_cls = class_scale * nn.BCELoss(reduction='sum')(cls[mask == 1], tcls[mask == 1])
            loss.append(loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls)
            print('%d: nGT: %d, recall: %d, proposals: %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
                seen, nGT, nCorrect[s], nProposals[s], loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
                loss_conf.item(), loss_cls.item(), loss[s].item()))
        else:
            loss.append(noobject_scale * nn.MSELoss(reduction='sum')(conf[mask == 0], tconf[mask == 0]))
            print('%d: nGT: %d, recall: %d, proposals: %d, loss: noobj %f' % (
                seen, nGT, nCorrect[s], nProposals[s], loss[s].item()))

    return loss, nGT, nCorrect, nProposals