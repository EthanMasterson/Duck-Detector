import matplotlib.pyplot as plt
from collections import Counter
import torch
import numpy as np
import cv2


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xywh2xyxy(c):
    '''
24.5, 143.0, 49.0, 136.0
562.5, 374.5, 273.0, 157.0

    :param c: takes a list of float coordinates in the form [xywh]
    :return: outputs a list in terms of top lft and btm rght corners [xyxy]

    '''

    coord = [0, 0, 0, 0]
    coord[0] = c[0] - c[2] / 2
    coord[1] = c[1] - c[3] / 2
    coord[2] = c[0] + c[2] / 2

    coord[3] = c[1] + c[3] / 2
    return coord




def updateMatrix(allpred,allgnd,confusion_matrix,frame_idx,count):
    # get boxes for the current frame
    # cv2.imshow("hmmm", frame)
    '''
    confusion matrix indexed in form (predicted class, actual class)
    '''

    pred = [xywh2xyxy(bbox[2:6]) for bbox in allpred if int(bbox[0]) == frame_idx and (bbox[-1] > 0.25)]

    gnd = [xywh2xyxy(bbox[2:6]) for bbox in allgnd if int(bbox[0]) == frame_idx]

    gt_classes = torch.tensor([int(bbox[1]) for bbox in allgnd if int(bbox[0]) == frame_idx])
    detection_classes = torch.tensor(
        [int(bbox[1]) for bbox in allpred if int(bbox[0]) == frame_idx and (bbox[-1] > 0.25)])

    if pred and gnd:
        pred = torch.tensor(pred)
        gnd = torch.tensor(gnd)
        print(pred)
        print(gnd)
        # print("\n")
        iou = box_iou(pred, gnd)
        #print(f"total preds:{len(pred)} total gnd:{len(gnd)} ious: {iou}")
        count += 1

        x = torch.where(iou > 0.45)  # create boolean tuple of valid ious
        # print(x)
        if x[0].shape[0]:

            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            # print(f"matches found at {matches[:,0]} row, {matches[:,1]} column of the predictions:gnds table")
            if x[0].shape[0] > 1:  # if theres more than one match
                matches = matches[matches[:, 2].argsort()[::-1]]  # sort in descending IoU
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # filter out duplicate detections
                matches = matches[matches[:, 2].argsort()[::-1]]  # sort in descending IoU again
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # filter duplicate ground truths
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0  # bool
        # print(matches)
        #print(matches.transpose().astype(int))
        m0, m1, _ = matches.transpose().astype(int)
        #print(f"actual classes {gt_classes}")
        #print(f"classes of guessed TP's {detection_classes}")

        for i, gc in enumerate(gt_classes):  # for each gt in frame
            j = m1 == i  # create a boolean mask where the gnd idx == i
            # print(j)
            if n and sum(j) == 1:  # if theres exaclty one match for the given ground truth
                # print(m1[j])
                idx = detection_classes[m0[j]]
                confusion_matrix[idx, gc] += 1  # correct
            else:
                confusion_matrix[3, gc] += 1  # false negative ( no predictions were made)

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m0 == i):
                    confusion_matrix[dc, 3] += 1  # predicted background

       # print(confusion_matrix)

    elif pred is None:
        for gc in gt_classes:
            confusion_matrix[3, gc] += 1 # false negative (no predictions were made)
    elif gnd is None:
        for det in detection_classes:
            confusion_matrix[det, 3] += 1 # false positive on background


    return confusion_matrix,count


def initData(gtlist, predlist):
    allgnd=[]
    allpred=[]

    summary=open(gtlist,"r")
    for line in summary:
        line=line.strip()
        line= line.split(" ")
        line = [float(line[0])] + [float(item) for item in line[1:]]
        allgnd.append(line)
        #gts ends in format[[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h]]
    summary.close()
    summary = open(predlist,"r")
    for line in summary:
        line = line.strip()
        line= line.split(" ")
        line = [float(line[0])] + [float(item) for item in line[1:]]
        allpred.append(line)
        # gts ends in format[[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h]]
    summary.close()

    return allgnd,allpred

def plot(confusion_matrix,video,m):
    import seaborn as sn
    cols=confusion_matrix.sum(0).reshape(1, -1).astype(int)
    rows = confusion_matrix.sum(1).reshape(1, -1).astype(int)
    print(rows)
    #array=confusion_matrix

    #array = confusion_matrix / ((confusion_matrix.sum(0).reshape(1, -1) + 1E-9))  # normalize columns
    array = confusion_matrix / (confusion_matrix.flatten().sum() + 1E-9)  # normalize whole thing
    array[array < 0.005] = 0.00  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    ticklabelsx = [f'Mallard ({cols[0][0]})', f'Pigeon ({cols[0][1]})', f'Runner Duck ({cols[0][2]})', f'Background ({cols[0][3]})']
    ticklabelsy = [f'Mallard ({rows[0][0]})', f'Pigeon ({rows[0][1]})', f'Runner Duck ({rows[0][2]})', f'Background ({rows[0][3]})']

    sn.heatmap(array,
               ax=ax,
               annot=True,
               annot_kws={
                   'size': 25},
               cmap='Blues',
               fmt='.2f',
               square=True,
               vmin=0.0,
               xticklabels=ticklabelsx,
               yticklabels=ticklabelsy).set_facecolor((1, 1, 1))

    ax.set_xlabel('True Label')
    ax.set_ylabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    plt.show()
    fig.savefig(f'runs/summaries/Confs/{video}_{m}.png', dpi=250)
    plt.close(fig)

if __name__ =="__main__":
    video = "D2"
    m="v5l"
    classes = 3
    cap = cv2.VideoCapture("Videos/D2 - HS.mp4")
    extra=""
    #extra="_fixed"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # load in bboxes
    allgnd ,allpred=initData(f"runs/summaries/{video}/Summary.txt",f"runs/summaries/{video}/{m}summary{extra}.txt")
    #print(allgnd,allpred)
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(allgnd[0][0]) -1)

    count=0
    confusion_matrix=np.zeros((4,4))

    while frame_idx<1000:
        # Capture frame-by-frame
        frame_idx+=1

        print(frame_idx)

        confusion_matrix,count = updateMatrix(allpred,allgnd,confusion_matrix,frame_idx,count)


    print(confusion_matrix)
    plot(confusion_matrix,video,m)

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()




