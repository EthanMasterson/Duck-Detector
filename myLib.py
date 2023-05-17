import matplotlib.pyplot as plt
from collections import Counter
import torch
import numpy as np


def xywh2xyxy(c):
	'''
	:Desription:
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


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def match_bboxes(pred_boxes, true_boxes, threshold,c):
	'''
	:function : takes lists of predicted boxes for a set of images, and list of true boxes, and returns the
	accumulations of the true and false positives
	:param pred_boxes: list of predicted bboxes in form of [image,class,x,y,w,h,conf] (2d list)
	:param true_boxes: list of ground truths in form of [image,class,x,y,w,h] (2d list)
	:param threshold: Threshold to use for the call to the IoU function (float)
	:param c: class id being evaluated, (int)
	:return: TP_cumsum (list of accumulated True postives), FP_cumsum (list of accumulated false positives)
	'''

	# find detections and ground truths for given class
	detections = [detection for detection in pred_boxes if detection[1] == c]

	ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

	# create dictionary of ground truths per img e.g {0:3,0:5}
	amount_bboxes = Counter([gt[0] for gt in ground_truths])

	for key, val in amount_bboxes.items():
		amount_bboxes[key] = torch.zeros(val)

	# sort detections by confidence
	detections.sort(key=lambda x: x[6], reverse=True)

	TP = np.zeros((len(detections)))  # list as long as the amount of detections on the val set

	total_true_bboxes = len(ground_truths)  # total number of true positives

	# find if a detection is a true or false positive)
	for detection_idx, detection in enumerate(detections):
		ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
		# only compare gts in the same image as the prediction

		num_gts = len(ground_truth_img)
		best_iou = 0
		# find the best ground truth for it, if its high enough overlap to be a TP, then add it.

		for idx, gt in enumerate(ground_truth_img):
			intersect = IoU(detection[2:6], gt[2:])

			if intersect > best_iou:
				best_iou = intersect
				best_gt_idx = idx

		if best_iou > threshold:
			if best_iou>1:

				print(best_iou)
			# if the ground truth hasn't yet been assigned a prediction,assign current prediction to it
			if amount_bboxes[detection[0]][best_gt_idx] == 0:
				TP[detection_idx] = 1
				amount_bboxes[detection[0]][best_gt_idx] = 1
	#print(TP)
	TP_cumsum = TP.cumsum(0)
	FP_cumsum=(1 - TP).cumsum(0)
	return TP_cumsum,FP_cumsum,total_true_bboxes


def mean_average_precision(pred_boxes,true_boxes,iou_threshold=0.5,num_classes=3,plot=False):
	names=["Mallard","Pigeon","Runner Duck"]
	average_precisions=[]
	eps=1e-6
	results=[]
	#for each class
	for c in range(num_classes):
		class_aps=[0] # empty first item to later become the AP across the IoU range
		px, py = np.linspace(0, 1, 1000), []  # for plotting

		#for each iou threshold
		for i in range(0,10):
			threshold=0.5 + 0.05*i

			TP_cumsum, FP_cumsum , total_true_bboxes= match_bboxes(pred_boxes,true_boxes,threshold,c)
			recall=TP_cumsum/(total_true_bboxes+eps)
			precision = TP_cumsum/(TP_cumsum+FP_cumsum)

			ap, mpre, mrec = compute_ap(recall, precision)

			class_aps.append(ap)

			if plot==True and threshold==0.5:
				py.append(np.interp(px, mrec, mpre))
				plt.plot(mpre,mrec)
				plt.xlim(0, 1)
				plt.ylim(0, 1)
				plt.xlabel("Recall")
				plt.ylabel("Precision")
				plt.title(names[c])
				plt.show()
		average_precisions.append(class_aps)
		average_precisions[c][0]=sum(class_aps)/(len(class_aps)-1)


	#Print AP scores for each class
	for idx,ap in enumerate(average_precisions):
		results.append(str("class: "+str(names[idx])+" AP @0.5:0.95:0.05: "+str(average_precisions[idx][0])+"    AP @0.5: "+str(average_precisions[idx][1])))
		print("class: ",names[idx] ," AP @0.5:0.95:0.05: ",average_precisions[idx][0],"    AP @0.5: ",average_precisions[idx][1])

	# Calculate overall mAP score
	total=0
	totalHalf=0
	for i in range(0,len(average_precisions)):
		total+=average_precisions[i][0]
		totalHalf+=average_precisions[i][1]
	# Print mAP scores
	results.append("mAP: "+str(total/num_classes)+"mAP @0.5: "+str(totalHalf/num_classes))
	print("mAP: ",total/num_classes,"mAP @0.5: ",totalHalf/num_classes)

	return results


def mAP_run(gtlist, predlist, model, w, cls=3, plot=True):
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
	results = open(f"runs/summaries/D1/{model}{w}results.txt", "w")
	#results=open("runs/summaries/test/" +model+w+"results.txt","w")
	ans=mean_average_precision(allpred,allgnd,plot=plot)
	for line in ans:
		results.write(line +"\n")

	results.close()

#mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v5ssummary.txt",model="v5",w="m",plot=True)
# mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v5msummary.txt",model="v5",w="m",plot=False)
# mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v5lsummary.txt",model="v5",w="m",plot=False)
#
# mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v7msummary.txt",model="v5",w="m",plot=False)
#
# mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v8ssummary.txt",model="v5",w="m",plot=False)
# mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v8msummary.txt",model="v5",w="m",plot=False)
# mAP_run("runs/summaries/D1/Summary.txt","runs/summaries/D1/v8lsummary.txt",model="v5",w="m",plot=False)




