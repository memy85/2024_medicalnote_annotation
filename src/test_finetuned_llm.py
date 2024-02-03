#%%
import pandas as pd
from utils import *

config = load_config()
projectPath = config.project_path
dataPath, annotationPath = config.get_data_path('not mimic')

#%%
os.listdir(annotationPath)

#%%
annotations['note'].unique()




#%%
directories = os.listdir(dataPath)
texts = []
for directory in directories : 
    dpath = dataPath.joinpath(directory)
    # notes = os.listdir(dpath)
    directory = directory.split("_")[0]

    annotations = pd.read_csv(annotationPath.joinpath(f"rankings_300notes_20151217_{directory}.txt"),\
                sep="\t")
    
    notes = annotations['note'].unique()

    for note in notes :
        with open(dpath.joinpath(note +".txt"), 'r') as f :
            text = f.read()

        textdict = {"note_type" : directory, 
                    "note_id" : note,
                    "text" : text}


        texts.append(textdict)

df = pd.DataFrame(texts)

#%%
annotations[annotations.note == notename]
#%%
notename






