import numpy as np
from myLib import IoU,mAP_run
def init_data(gtlist,predlist):
    allgnd=[]
    allpred=[]

    summary=open(gtlist,"r")
    for line in summary:
        line=line.strip()
        line= line.split(" ")
        line = [float(item) for item in line]
        allgnd.append(line)
        #gts ends in format[[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h]]
    summary.close()
    summary = open(predlist,"r")
    for line in summary:
        line = line.strip()
        line= line.split(" ")
        line =  [float(item) for item in line]
        allpred.append(line)
        # gts ends in format[[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h],[cls,x,y,w,h]]

    summary.close()
    return allpred,allgnd

def compute_delay(preds,gts,instance_array,ratio,W=60):
    frames_considered=np.array([np.arange(instance[0],instance[0]+W+1,1) for instance in instance_array]).flatten()
    no_gts=len([gt for gt in gts if gt[0] in frames_considered])

    fp_num = int(no_gts * ratio)
    conf=[0 for instances in instance_array]
    preds=[pred for pred in preds if pred[0] in frames_considered]
    preds = preds[0:f_num]
    for detection in preds:
        gts_frame=[ground for ground in gts if ground[0]==detection[0] and ground[1]==detection[1]]

        best_iou=0

        for idx, gt in enumerate(gts_frame):
            intersect = IoU(detection[2:6], gt[2:])

            if intersect > best_iou:
                best_iou = intersect
                best_gt_idx = idx

        if best_iou > 0.5:
            gt=gts_frame[best_gt_idx]
            delay=detection[0]-instance_array[int(gt[6])-1][0]

            if delay<instance_array[int(gt[6])-1][2]:
                instance_array[int(gt[6])-1][2]=delay

                conf[int(gt[6])-1]=detection[6]



    delays=[instance[2] for instance in instance_array]
    #print(np.mean(delays))
    #print(f"confidences at {ratio} first detections: {conf}, delays is {delays}")
    return np.mean(delays)


def average_delay(preds,gts,):
    fp_ratios=[0.1,0.2,0.4,0.8,1.6,3.2]
    instance_array=[]
    count = 0

    for row in gts:
        if row[6] > count:
            instance_array.append([row[0],row[6],61])
            count+=1

    sum_delay=[]


    preds.sort(key= lambda x:x[6],reverse=True)
    for ratio in fp_ratios:
        sum_delay.append(1/(compute_delay(preds,gts,instance_array,ratio)+1))

    average_p=np.mean(sum_delay)
    ad= 1/(average_p) -1

    return ad


if __name__=="__main__":
    m="background_"
    v="D4"
    w=""
    preds, gts = init_data(f"runs/summaries/{v}/summary_instances_added.txt", f"runs/summaries/{v}/"+m+w+"summary_fixed.txt")

    mAP_run("runs/summaries/D4/Summary.txt", "runs/summaries/D4/background_summary_fixed.txt", model="background", w="", plot=False)
    print(f"average delay for {m}{w} is {average_delay(preds,gts)} \n")
