import torch
import numpy as np

def get_chunks(self, flen):
    x_index = torch.arange(flen)

    # The first chunk has a random length
    rand_first = np.random.randint(0, self.chunk_size) + 1
    chunks = x_index[rand_first:].split(self.chunk_size, dim=0)
    chunks = [x_index[:rand_first]] + list(chunks) if len(chunks[0]) > 0 else [x_index[:rand_first]]
    if np.random.rand() > 0.5:
        chunks = chunks[::-1]
    
    # Chunk order only matter when we do global token merging
    if self.merge_global == False:
        return chunks

    # Chunk order. "seq": sequential order. "rand": full permutation. "mix": partial permutation.
    if self.chunk_ord == "rand":
        order = torch.randperm(len(chunks))
    elif self.chunk_ord == "mix":
        randord = torch.randperm(len(chunks)).tolist()
        rand_len = int(len(randord) / self.perm_div)
        seqord = sorted(randord[rand_len:])
        if rand_len > 0:
            randord = randord[:rand_len]
            if abs(seqord[-1] - randord[-1]) < abs(seqord[0] - randord[-1]):
                seqord = seqord[::-1]
            order = randord + seqord
        else:
            order = seqord
    else:
        order = torch.arange(len(chunks))
    chunks = [chunks[i] for i in order]
    return chunks