import matplotlib.pyplot as plt
from collections import Counter
import torch
import numpy as np
import cv2


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


def IoU(c1, c2):
    x1 = max(c1[0] - c1[2] / 2, c2[0] - c2[2] / 2)  # intersect left x
    x2 = min(c1[0] + c1[2] / 2, c2[0] + c2[2] / 2)  # intersect right x
    y1 = max(c1[1] - c1[3] / 2, c2[1] - c2[3] / 2)  # intersect top y
    y2 = min(c1[1] + c1[3] / 2, c2[1] + c2[3] / 2)  # intersect bottom y

    intersect = max(0, (x2 - x1) * (y2 - y1))  # intersect w *h , if intersection is invalid,make area 0
    union = (c1[2] * c1[3] + c2[2] * c2[3]) - intersect  # intersect area of c1 +area of c2 - intersect
    result = intersect / union
    #
    if (x2 - x1)<0 or (y2 - y1)<0:
        result= 0

    return result


def scale_bbox(bbox):
    frame, cls, x, y, w, h = bbox
    x, y, w, h = x * width, y * height, w * width, h * height
    return [frame, cls, x, y, w, h]


def draw_rects(bboxes,frame,frame_num,video,width,height):
    """'
    needs an input of normalised bounding boxes, video name, frame number, and the frame itself.
    """



    colours=[[255,0,0,],[0,255,0],[0,0,255],[0,0,0]]
    for idx,bbox in enumerate(bboxes):
        c,x, y, w, h ,= int(bbox[1]),int(bbox[2]*width), int(bbox[3]*height), int(bbox[4]*width), int(bbox[5]*height)

        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), colours[c], 2)



    #cv2.imshow("hmm",frame)
    location=f"Weaknesses/{video}_{frame_num}.jpg"
    cv2.imwrite( location,frame)


def match_bboxes(pred_boxes, true_boxes,classes, threshold=0.5,conf=0.25):
    '''
    :function : takes lists of predicted boxes for a frame, and list of true boxes, and returns the
    accumulations of the true and false positives
    :param pred_boxes: list of predicted bboxes in form of [image,class,x,y,w,h,conf] (2d list)
    :param true_boxes: list of ground truths in form of [image,class,x,y,w,h] (2d list)
    :param threshold: Threshold to use for the call to the IoU function (float)
    '''
    ground_unfound=[]
    test=[]
    # find detections and ground truths for given class, create a list of the
    for c in range(0,classes):

        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

        # sort detections by confidence and filter out confidecnes lower than 0.5
        detections =[detection for detection in detections if detection[6]>conf]
        detections.sort(key=lambda x: x[6], reverse=True)
        #
        #
        gnd_fnd= np.zeros((len(ground_truths)))  # list as long as the amount of detections on the val set

        # find if a detection is a true or false positive)
        fnd=0
        if ground_truths:
            for detection_idx, detection in enumerate(detections):
                best_gt_idx=10000
                best_iou = 0

                # find the best ground truth for it, if its high enough overlap to be a TP, then add it.
                for idx, gt in enumerate(ground_truths):
                    intersect = IoU(detection[2:6], gt[2:])

                    if intersect > best_iou:
                        best_iou = intersect
                        best_gt_idx = idx

                if best_iou > threshold:
                    if best_iou>1:
                        print(best_iou)
                    # if the ground truth hasn't yet been assigned a prediction,assign current prediction to it
                    if gnd_fnd[best_gt_idx] == 0:
                        gnd_fnd[best_gt_idx] = 1
                        fnd+=1

            #for the missing ground truths, append their boxes to the unfound boxes in the frame

            for idx, gt in enumerate(ground_truths):

                if gnd_fnd[idx]== 0:

                    ground_unfound.append(gt)
                    #save a drawing of the missed ground truths to a file
        # for det in detections:
        #     ground_unfound.append(det)
    print(len(ground_unfound))

    if len(ground_unfound) != len(ground_truths):
        print(f"{fnd}/{len(ground_truths)}")
    return ground_unfound


def initData(gtlist, predlist):
    allgnd=[]
    allpred=[]

    summary=open(gtlist,"r")
    for line in summary:
        line=line.strip()
        line= line.split(" ")
        line = [line[0]] + [float(item) for item in line[1:]]
        allgnd.append(line)
        #gts ends in format[[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h]]
    summary.close()
    summary = open(predlist,"r")
    for line in summary:
        line = line.strip()
        line= line.split(" ")
        line = [line[0]] + [float(item) for item in line[1:]]
        allpred.append(line)
        # gts ends in format[[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h]]
    summary.close()

    return allgnd,allpred


if __name__ =="__main__":
    video = "D1"
    classes = 3
    cap = cv2.VideoCapture("Videos/D1.mp4")


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))




    # load in bboxes
    allgnd ,allpred=initData(f"runs/summaries/{video}/Summary.txt",f"runs/summaries/{video}/v5lsummary.txt")

    frame_idx = int(allgnd[0][0]) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(allgnd[0][0]) -1)
    #for each frame
    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_idx+=1


        if ret == True:
            #get boxes for the current frame
            #cv2.imshow("hmmm", frame)



            pred= [bbox for bbox in allpred if int(bbox[0]) == frame_idx]
            gnd = [bbox for bbox in allgnd if int(bbox[0]) == frame_idx]


            #match the boxes up
            ground_unfound= match_bboxes(pred,gnd,classes)

            if ground_unfound:
                draw_rects(ground_unfound, frame, frame_idx, video,width, height)

            print(f"completed frame {frame_idx}")
            # # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()




