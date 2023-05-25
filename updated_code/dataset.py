import torch
from data_utils import load_data

# Reads the data from the data files
class DataReader:
    def __init__(
            self, mode='train',
            segment_length: int = 32,
            n_vertices: int = 6172,
            audio_length: int = 16000
    ):
        self.segment_length = segment_length
        self.n_vertices = n_vertices
        self.audio_length = audio_length
        self.dataset_size = 100  # placeholder for dataset size
        self.current_idx = 0
        
        if mode=='train':
            self.data = load_data('train')
        else:
            self.data = load_data('val')

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx: pointer to dataset element to be read
        :return: template: V x 3 tensor containing neutral face template
                 geom: segment_length x V x 3 tensor containing animated segment for same subject as template mesh
                 audio: segment_length x audio_length tensor containing the input audio for each frame of the segment
        """
        template = self.data['template']
        geom = self.data['geom']
        audio = self.data['audio']
        return {
            "template": template,
            "geom": geom,
            "audio": audio
        }
