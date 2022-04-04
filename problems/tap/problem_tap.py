import random

from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tap.state_tap import StateTAP
from utils.beam_search import beam_search
class TAP(object):

    NAME = 'tap'

    @staticmethod
    def get_costs(dataset, pi, indexes=None):

        data = dataset['data']
        if len(dataset['data'].size()) == 4:
            batch, _, n_loc, _ = data.size()
            index_col = 3
        else:
            batch, n_loc, _ = data.size()
            index_col = 2

        # add source to path
        pi_init = torch.cat((torch.zeros_like(pi[:,0])[:,None], pi), 1)

        # dynamic case needs indexes
        if indexes is not None:
            pi_tuples = torch.cat((pi_init[:, :-1][:, :, None], pi_init[:, 1:][:, :, None], indexes.unsqueeze(-1)), dim=-1)
        else:
            pi_tuples = torch.cat((pi_init[:,:-1][:,:,None], pi_init[:,1:][:,:,None]), dim=-1)

        # Find the same combination
        # Current torch.unique does not support row based unique values. Using a loop and padding
        unique_vals = []
        for t in pi_tuples:
            u, c = torch.unique(t, return_counts=True, dim=0)
            unc = torch.cat((u,c[:,None]), dim=1)
            unique_vals.append(unc)

        # Create padding
        padded_unique_vals = torch.zeros(batch, n_loc*n_loc*2, index_col+1, dtype=torch.int64, device=pi.get_device())
        for index, t in enumerate(unique_vals):
            index_removed = torch.cat((unique_vals[index][:,:index_col], unique_vals[index][:,-1].unsqueeze(-1)), dim=1)
            padded_unique_vals[index, :index_removed.size(0),:] = index_removed
            padded_unique_vals = padded_unique_vals.clip(0, n_loc - 1)


        if len(data.size()) == 4:
            expanded_vals = padded_unique_vals[:, :, 2][:, :, None, None].expand(*padded_unique_vals.size()[0:2],
                                                                                 *data.size()[-2:])
            time_aligned_data = data.gather(1, expanded_vals)

            x = time_aligned_data.gather(2, padded_unique_vals[:, :, 0].unsqueeze(-1).unsqueeze(-1).
                        expand(*padded_unique_vals.size()[0:2],  *data.size()[-2:]))[:, :, 0, 0:2]
            y = time_aligned_data.gather(2, padded_unique_vals[:, :, 1].unsqueeze(-1).unsqueeze(-1).
                        expand(*padded_unique_vals.size()[0:2],  *data.size()[-2:]))[:, :, 0, 0:2]

        else:
            x = data.gather(1, padded_unique_vals[:, :, 0].unsqueeze(-1).
                        expand(*padded_unique_vals.size()[0:2], 2))
            y = data.gather(1, padded_unique_vals[:, :, 1].unsqueeze(-1).
                        expand(*padded_unique_vals.size()[0:2], 2))

        dist = (x-y).norm(p=2, dim=2)
        path_cost = torch.where(torch.logical_and(padded_unique_vals[:, :, 0] == n_loc-1,
                                             padded_unique_vals[:, :, 1] == 0), dist - dist, dist)

        multiplyer = torch.where(padded_unique_vals[:, :, 0] - padded_unique_vals[:, :, 1] == 0, 0,
                                 padded_unique_vals[:, :, index_col])
        multiplyer = torch.where(torch.logical_and(padded_unique_vals[:, :, 0] == n_loc-1,
                                                   padded_unique_vals[:, :, 1] == 0), 0, multiplyer)

        cost = ((path_cost + 0.2*multiplyer)*multiplyer).sum(1) #link performance function

        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TAPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTAP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096, dynamic=False):

        assert model is not None, "Provide model"

        if dynamic:
            def propose_expansions(beam, fixed):
                return model.propose_expansions(
                    beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
                )

            return beam_search(dynamic, TAP, input, model, beam_size, propose_expansions)
        else:

            state = TAP.make_state(
                input, visited_dtype=torch.int64 if compress_mask else torch.uint8
            )
            fixed = model.precompute_fixed(input)
            def propose_expansions(beam):
                return model.propose_expansions(
                    beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
                )

            return beam_search(dynamic, state, beam_size, propose_expansions)

def make_instance(args):
    data, adj, *args = args
    return {
        'data': torch.tensor(data, dtype=torch.float),
        'adj': torch.tensor(adj, dtype=torch.bool)
    }

class TAPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, is_dynamic=False):
        super(TAPDataset, self).__init__()

        TAP.DEMAND = 8

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
        else:
            # Sample points randomly in [0, 1] square
            if is_dynamic:
                self.data = [{'data': self.create_dynamic_instance(size, TAP.DEMAND),
                              'adj': self.create_adj(size)
                } for i in range(num_samples)]
            else:
                self.data = [{'data': self.create_instance(size, TAP.DEMAND),
                              'adj': self.create_adj(size)
                } for i in range(num_samples)]


        self.size = len(self.data)

    def create_instance(self, size, demand_max):
        instance = torch.FloatTensor(size, 2).uniform_(0, 1)
        demand = random.randint(2, demand_max)
        demand_t = torch.zeros(size, 1)
        demand_t[0, 0] = -demand
        demand_t[-1, 0] = demand

        instance = torch.cat((instance, demand_t), dim=1)

        return instance

    def create_adj(self, size):
        adj = torch.zeros(size, size, dtype=torch.bool)
        coords = torch.randint(1, size-1, (size, size-5))
        adj = adj.scatter(1, coords, True)
        adj[size-1, 0] = False # there should always be a connection from destination to source for resetting the path
        return adj

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def as_tensor(self):
        return torch.stack(self.data, dim=0)

    def create_dynamic_instance(self, size, demand_max=8, strength=0.01):
        total_nodes = []
        next = torch.FloatTensor(size, 2).uniform_(0, 1) # Create initial coordinates
        for i in range(size):
            total_nodes.append(next)
            next = torch.clip(torch.add(next, torch.FloatTensor(size, 2).uniform_(-strength, strength))
                              , 0, 1) # Change the previous coordinates between 0 and 1
        data = torch.stack(total_nodes, dim=0)

        demand = random.randint(2, demand_max)
        demand_t = torch.zeros(size, 1)
        demand_t[0, 0] = -demand
        demand_t[-1, 0] = demand

        return torch.cat((data, demand_t[None, :].expand(size,-1,-1)), dim=-1)

class DTAP(TAP):
    @staticmethod
    def make_dataset(*args, **kwargs):
        kwargs['is_dynamic'] = True
        return TAPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTAP.initialize(*args, **kwargs)




