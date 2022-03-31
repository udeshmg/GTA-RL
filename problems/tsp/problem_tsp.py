from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        dataset = dataset['data']

        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        if len(dataset.size()) == 4:
            d = dataset.gather(2, pi.unsqueeze(-1).unsqueeze(1).expand_as(dataset)).diagonal(dim1=1, dim2=2)
            d = d.permute(0,2,1)
            #dataset = dataset[:,0,:,:]
            #d1 = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        else:
            d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096, dynamic=False):

        assert model is not None, "Provide model"

        if dynamic:
            def propose_expansions(beam, fixed):
                return model.propose_expansions(
                    beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
                )

            return beam_search(dynamic, TSP, input, model, beam_size, propose_expansions)
        else:

            state = TSP.make_state(
                input, visited_dtype=torch.int64 if compress_mask else torch.uint8
            )
            fixed = model.precompute_fixed(input)
            def propose_expansions(beam):
                return model.propose_expansions(
                    beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
                )

            return beam_search(dynamic, state, beam_size, propose_expansions)



class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, is_dynamic=False):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            if is_dynamic:
                self.data = [{
                    'data': self.get_dynamic_data(size, 0.1),
                    'adj': torch.ones((size,size), dtype=torch.bool)
                } for i in range(num_samples) ]
            else:
                self.data = [{'data': torch.FloatTensor(size, 2).uniform_(0, 1),
                              'adj': torch.ones((size,size), dtype=torch.bool)
                } for i in range(num_samples)]


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def as_tensor(self):
        return torch.stack(self.data, dim=0)

    def get_dynamic_data(self, size, strength=0.01):
        total_nodes = []
        next = torch.FloatTensor(size, 2).uniform_(0, 1) # Create initial coordinates
        for i in range(size):
            total_nodes.append(next)
            next = torch.clip(torch.add(next, torch.FloatTensor(size, 2).uniform_(-strength, strength))
                              , 0, 1) # Change the previous coordinates between 0 and 1
        return torch.stack(total_nodes, dim=0)

class DTSP(TSP):
    @staticmethod
    def make_dataset(*args, **kwargs):
        kwargs['is_dynamic'] = True
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)




