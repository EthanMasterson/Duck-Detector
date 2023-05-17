import numpy as np
import cv2 as cv
from AverageDelay import init_data
import os
from myLib import IoU
# Your 2D list array
trust=False
def scale_bbox(bbox):
    '''
    :description: Denormalises  given bbox
    :param bbox: bbox in YOLO format
    :return: denormalised bbox
    '''
    frame, cls, x, y, w, h = bbox
    x, y, w, h = x * width, y * height, w * width, h * height
    return [frame, cls, x, y, w, h]

def normalise(bbox):
    '''
    :description: Normalise bounding boxes
    :param bbox: bbox in YOLO format
    :return:Normalised BBox
    '''
    frame,cls,x,y,w,h,instance,id= bbox
    x,y,w,h=x / width, y / height, w / width, h / height
    return [frame, cls, x, y, w, h, instance]

def find_sibling(bboxes,instances,type="iou"):
    '''
    :Description: find the nearest matches from the prior frames instances for a set of bboxes
    :param bboxes: ground truths of current frame
    :param instances:ground truths of previous frame, with their instance label
    :param type: type of matching method (iou or euclidean0
    :return:guesses of bbox assignment for the given frame
    '''
    guess=0
    id=99
    type=type
    bbox_guesses=bboxes


    if len(instances)<len(bboxes):
        for instance in instances:
            best_distance=100000
            best_iou=0
            if type=="euclidean":
                a=np.array([instance[0],instance[1]])
                for idx,bbox in enumerate(bboxes):
                    b=np.array([bbox[2],bbox[3]])
                    distance=np.linalg.norm(a-b)
                    if distance<best_distance:
                        best_distance=distance
                        best_idx=idx
            if type=="iou":
                gts=instance[1:]
                for idx,bbox in enumerate(bboxes):
                    b=bbox[2:6]
                    overlap=IoU(gts,b)
                    if overlap>best_iou and len(bbox) <7:
                        best_iou=overlap
                        best_idx=idx

            bbox_guesses[best_idx].append(instance[0])
    elif len(instances)>= len(bboxes):

        for num,bbox in enumerate(bboxes):

            best_distance = 100000
            best_iou = 0
            if type == "euclidean":
                b = np.array([bbox[2], bbox[3]])
                for idx, instance in enumerate(instances):
                    a = np.array([instance[0], instance[1]])
                    distance = np.linalg.norm(a - b)
                    if distance < best_distance:
                        best_distance = distance
                        best_id = instance[0]
            if type == "iou":
                b = bbox[2:6]
                for idx, instance in enumerate(instances):
                    gts = instance[1:]
                    overlap = IoU(gts, b)
                    if overlap > best_iou:
                        best_iou = overlap

                        best_id = instance[0]

            bbox_guesses[num].append(best_id)

    #append empty guess to those that werent matched

    #if there are no instances in the previous frame, just guess 0 as a placeholder
    if not instances:
        bbox_guesses=[[*bbox,0] for bbox in bboxes]
    # [frame,cls,x,y,w,h,instance,id]
    for bbox in bbox_guesses:

        if len(bbox) <7:
            bbox.append(0)
    return bbox_guesses


def draw_rects(bboxes,frame):
    '''
    :Description: draws bboxes with guesses
    :param bboxes: bboxes, possibly with guesses
    :param frame:image frame
    :return: bboxes with temporary id
    '''
    temp_rects=[]
    for idx,bbox in enumerate(bboxes):
        x, y, w, h ,guess= int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]),int(bbox[6])
        colour = np.random.randint(0, 256, 3).tolist()
        cv.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), colour, 2)
        cv.putText(frame, str(idx), (x, y),cv.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv.LINE_AA)

        if guess!=0:
            cv.putText(frame, "id:"+ str(guess), (x, y+20),cv.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv.LINE_AA)
        temp_rects.append([*bbox,idx])
    cv.waitKey(25)
    cv.imshow("Video", frame)
    cv.imwrite("Instances/previous.jpg", frame)
    return temp_rects


def userinput(bboxes,Accepted=False):
    '''
    :Description: Allows the user to check and if neccessray correct/state the currently guessed instances
    :param bboxes: array of Bounding boxes for a given frame
    :param guess: array of suggested instances for a given frame
    :return: correct instances
    '''
    instances=[]

    for bbox in bboxes:
        if Accepted==False:
            print(f"\n Suggested instance for bbox id {bbox[7]}: {bbox[6]}")
            while True:
                user_input = input("Press ENTER to accept suggestion, or enter a number for a custom instance ID: ")
                if not user_input:
                    bbox_descaled = normalise(bbox)
                    new_data.append([*bbox_descaled[0:7]])
                    instances.append([bbox[6],bbox[2],bbox[3],bbox[4],bbox[5]])
                    break
                elif user_input.isdigit():
                    bbox_descaled = normalise(bbox)
                    new_data.append([*bbox_descaled[0:6], int(user_input)])
                    instances.append([int(user_input), bbox[2], bbox[3],bbox[4],bbox[5]])
                    break
                else:
                    print("Invalid input. Please try again.")
        else:
            print(f"\n Suggested instance for bbox id {bbox[7]}: {bbox[6]}")
            bbox_descaled=normalise(bbox)
            new_data.append([*bbox_descaled[0:7]])
            instances.append([bbox[6],bbox[2],bbox[3],bbox[4],bbox[5]])



    return instances



data, preds = init_data("Videos/D3 Labels/Summary.txt", "Videos/D3 Labels/Summary.txt")
playback_start = data[0][0]
playback_end = data[-1][0]

video = "Videos/D3 -D.mp4"
cap = cv.VideoCapture(video)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

frame_no = playback_start
cap.set(cv.CAP_PROP_POS_FRAMES, playback_start)

instances = []
new_data = []
4
file = open("runs/summaries/example.txt", "w")
file.close()

instances=[]

while frame_no<=playback_end:

    new_data = []
    bboxes=[bbox for bbox in data if bbox[0]==frame_no]
    bboxes = [scale_bbox(bbox) for bbox in bboxes]

    ret, frame = cap.read()
    bboxes_guessed=find_sibling(bboxes,instances)
    identifiers=draw_rects(bboxes_guessed,frame)
    print(f"Identifiers: {identifiers}")


    if not ret:
        break

    while True:
        if trust==False:
            key = cv.waitKey(0)
        else:
            key =cv.waitKey(1)

        if key == ord('w'):

            #cv.destroyAllWindows()  # Close the video window before user input
            print(f"\n frame number:{frame_no }")
            instances=userinput(identifiers)
            break
        if key == ord('q'):
            #cv.destroyAllWindows()  # Close the video window before user input
            print(f"\n frame number:{frame_no }")
            if len(bboxes) <= len(instances):
                instances=userinput(identifiers,True)
            else:
                instances = userinput(identifiers)
                trust=False
            break

        if trust==True:
            print(f"\n frame number:{frame_no}")
            #force the user to manually enter input if there is are new unguessed bboxes
            if len(bboxes) <= len(instances):
                instances=userinput(identifiers,True)
            else:
                #cv.destroyAllWindows()
                instances = userinput(identifiers)
                trust=False
            break
        if key == ord('t'):
            trust=True

    # if cv.waitKey(30) == 27:
    #     cv.destroyAllWindows()
    frame_no+=1

    file = open("runs/summaries/D3/summary_instances_added.txt", "a")
    lines=[" ".join(str(item) for item in row) for row in new_data]
    lines=[line+"\n" for line in lines]
    #print(lines)
    file.writelines(lines)
    file.close()





cap.release()
cv.destroyAllWindows()



