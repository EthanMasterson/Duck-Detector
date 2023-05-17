This folder contains all of the relevant work created in this project, and can be navigated as follows

The runs/summaries folder contains the results of all detections for each video and image set for every model. the Confs sub folder includes the confusion matrix plots generated
The v5 ,7 and 8 custom folders MUST be used if any code is attempted to be run, as otherwise there will be import errors and incorrect programs
if attempting to run v7 programming from the master trainer, the file must be saved and ran in the command prompt due to library issues i was unable to solve
It should also be noted that files simply named valsummary or summary indicate the ground truth files

Weights/ contains all of the model weights, with -mine denoted results from training

Dataset/50pc/ contains all of the data used in training along with the individual image labels

Videos/ contains the clipped videos D1-4 used

3D Printing/ contains the stl files designed, including failed iterations

Weaknesses/ stores the outputs from the frame weaknesses program
Custom.yaml is the yaml file needed for training

the py files "Frame extractor", "Frame label combiner" , "text fixer" and "predetections renamer" are utility files not mentioned in the report. 
They are only used to save time when reformatting data and are auxillary to the report.

The code can be ran if desired, however the commenting is poor and some of the file references are explicit to my computer, so it is not reccommended.


