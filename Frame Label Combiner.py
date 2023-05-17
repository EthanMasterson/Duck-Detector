import os
'''
Takes a folder of individually labeled frames for a video and combines them into one file, giving them frame numbers 
so they can be used for the average delay metric

'''
folder_path ="Videos/D4 Labels" # input( path/to/your/folder")
summary_filename ="Summary.txt" #input("summary file name: Vidsummary.txt")

# Scanning through the folder
files = os.listdir(folder_path)
# Filtering only the .txt files
txt_files = [file.strip(".txt").split("_") for file in files if file.endswith(".txt")]
txt_files.sort(key=lambda x:int(x[2]))


# Creating the summary file
with open(folder_path+"/"+summary_filename, "w") as summary_file:
    for frame,txt_file in enumerate(txt_files):

        with open(os.path.join(folder_path, txt_file[0]+"_"+txt_file[1]+"_"+txt_file[2] + ".txt"), "r") as file:
            contents = file.readlines()
            write=[]
            for line in contents:
                #if class id's need replacing
                line=line.split(" ")
                line[0]=2
                line = " ".join(str(item) for item in line)
                line=str(frame)+" "+line
                write.append(line)
            #print(write)
            summary_file.writelines(write)


