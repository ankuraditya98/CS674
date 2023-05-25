from tqdm import tqdm

from torch_geometric.io import read_obj
from torch_geometric.data import Data


from os import walk
from os.path import join
import pickle


# Loads the template faces from the dataset
def load_template_dataset(directory):
    temp_files = []
    for dir in tqdm(walk(directory)):
        for file in dir[2]:
                if file.endswith('.obj'):
                    loaded_file = read_obj(join(dir[0], file))
                    temp_files.append(Data(pos=loaded_file.pos, face=loaded_file.face))

    return temp_files


with open('train_data.pickle', 'rb') as f:
            # use the pickle.load() function to load the pickled object from the file
            train_data = pickle.load(f)

# load template data
# train_data['template'] = load_template_dataset('./custom_dataset/face_templates/train')



with open('val_data.pickle', 'rb') as f:
            # use the pickle.load() function to load the pickled object from the file
            val_data = pickle.load(f)

# load template data
# val_data['template'] = load_template_dataset('./custom_dataset/face_templates/val')


print(len(train_data['geom']), len(train_data['geom']), len(train_data['template']))
print(len(val_data['geom']), len(val_data['geom']), len(val_data['template']))


# with open('train_data.pickle', 'wb') as f:
#     # use the pickle.dump() function to pickle the object and write it to the file
#     pickle.dump(train_data, f)




# with open('val_data.pickle', 'wb') as f:
#     # use the pickle.dump() function to pickle the object and write it to the file
#     pickle.dump(val_data, f)