#from ultralytics import YOLO as V8
#do not load in v5 and v7 at the same time as they have conflicting imports and will read the wrong files.
from Yolo_V5_Custom import train as v5Train,detect as v5Detect,val as val5
#from Yolo_V7_Custom import train as v7Train,detect as v7Detect
import cv2
import itertools
import os
from myLib import mAP_run as mAP


# Load a model
def trainer(d="custom.yaml",i=640,w="best",b=8,e=50,modelli="v5"):
    #Train YOLO V5
    print("Training :",modelli)
    if modelli=="v5":
        v5Train.run(data=d, imgsz=i, weights="weights/v5/"+w +".pt",batch_size=b, epochs=e)

    elif modelli=="v7":
        v7Train.run(data=d, epochs=e, batch_size=b ,weights="weights/v7/"+w +".pt")

    elif modelli=="v8":
        modelv8 = V8("weights/v8/"+w +".pt")
        v8Train = modelv8.train(data=d, epochs=e, batch=b, imgsz = i)  # train the model

def detector(src,d="custom.yaml",conf=0.5,modelli="v5",w="best.pt",iou=0.5):
    # Train YOLO V5
    dest="/runs/Detect/"
    if modelli=="v5":
        boxes,dest =v5Detect.run(source=src,weights="weights/v5/"+w,name="v5",conf_thres=conf,save_txt=True,save_conf=True,iou_thres=iou,ElChange=False)
        print("boxes are :" + str(boxes))

    elif modelli=="v7":
        boxes,dest =v7Detect.run(source=src,name="v7", weights="weights/v7/"+w ,save_txt=True,conf_thres=conf,save_conf=True,view_img=True,iou_thres=iou)
        print("boxes are :"+str(boxes))

    elif modelli=="v8":
        v8Train = modelv8.predict(source=src, show=True,save_txt=True,name="v8",save_conf=True,conf=conf,iou=iou)  # train the model
        boxes = v8Train[0].boxes
        line=[]
        for i in range(0,len(boxes)):
            box = boxes[i]  # returns one box
            box=sum([[int(box.cls)],box.xywhn.view(-1).tolist()],[])
            line.append(box)
        print(line)
    return dest

def validator(model,folder_path,w,custom=""):
    input_folder = folder_path
    output_file = "runs/summaries/" +model+w+"_"+custom+"summary.txt"
    f=open(output_file,"w+")
    f.close()

    with open(output_file, "a") as out_file:
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(input_folder, filename), "r") as in_file:
                    new_line=[]
                    for line in in_file:
                        out_file.write(str(filename[8:-4])+" "+line)




if __name__ == "__main__":
    #m = "v7"
    #trainer(w="m",modelli="v5")
    #
    # modelv8 = V8("weights/v8/" + "s-mine.pt")
    # dir=detector(src="Dataset/50pc/Val",modelli=m,w="s-mine.pt")
    # validator(m, "runs/detect/v83/labels/","s")
    #
    # modelv8 = V8("weights/v8/" + "m-mine.pt")
    # dir = detector(src="Dataset/50pc/Val", modelli=m, w="s-mine.pt")
    # validator(m, "runs/detect/v84/labels/", "m")
    #
    # modelv8 = V8("weights/v8/" + "l-mine.pt")
    #dir = detector(src="Dataset/50pc/Val", modelli=m, w="m-mine.pt",conf=0.001,iou=0.6)
    # validator(m, "runs/detect/v85/labels/", "l")
    #validator(m, "runs/detect/v53"+"/labels/","m")
    #validator(m,"runs/detect/v82/labels/")
    #validator("val", "Dataset/50pc/Val/")
    w="l-mine.pt"
    model="v5"
    dir= detector("Dataset/50pc/Val_test",conf=0.001,iou=0.6, modelli=model, w=w)
    val5.run(weights= "weights/v5/l-mine.pt",data="custom.yaml")
    #validator(model, "runs/detect/v529/labels/", "l",custom="background")

    # v="D4"
    #
    #
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v5ssummary.txt", "v5","s",plot=False)
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v5msummary.txt", "v5", "m", plot=False)
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v5lsummary.txt", "v5", "l", plot=False)
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v7msummary.txt", "v7", "m", plot=False)
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v8ssummary.txt", "v8", "s", plot=False)
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v8msummary.txt", "v8", "m", plot=False)
    # mAP(f"runs/summaries/{v}/summary.txt", f"runs/summaries/{v}/v8lsummary.txt", "v8", "l", plot=False)

