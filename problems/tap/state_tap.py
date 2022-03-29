import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateTAP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    last_a:torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    edge: torch.Tensor
    demand: torch.Tensor
    node_usage: torch.Tensor


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            last_a=self.last_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8, index=-1):

        if index != -1:
            loc, mask = input['data'][:, index, :, :], input['adj']
        else:
            loc, mask = input['data'], input['adj']

        batch_size, n_loc, _ = loc.size()
        demand = input['data'][:, -1, 2][:,None]

        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)

        visited = torch.zeros(
            batch_size, 1, n_loc,
            dtype=torch.uint8, device=loc.device
        )
        visited[:,:,0] = 1 #always starts with the first node

        return StateTAP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=visited,
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=prev_a,
            last_a=torch.zeros(batch_size,1, device=loc.device, dtype=torch.int64)+n_loc-1,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            edge=mask,
            demand=demand,
            node_usage=torch.zeros(batch_size, 1, n_loc, dtype=torch.float, device=loc.device)
        )

    def update_state(self, input, index=-1):
        if index != -1:
            loc = input[:, index, :, :]
        else:
            loc = input

        return self._replace(loc=loc,
                             dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1))


    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        n_loc = self.edge.size(-1)

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update

        #if self.visited_.dtype == torch.uint8:
        #    # Add one dimension since we write a single value
        #    visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        #else:
        #    visited_ = mask_long_scatter(self.visited_, prev_a)

        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        demand_ = torch.where(prev_a == n_loc-1, self.demand-1, self.demand)

        node_usage = self.node_usage.scatter(-1, prev_a[:, :, None],
                                             self.node_usage.gather(-1, prev_a[:, :, None])+1)
        node_usage[:, 0, 0] = 0
        node_usage[:, 0, -1] = 0 #Reset source and destination because these two nodes are used anyway

        # after every round, source node should be selected
        round_complete = torch.ones_like(visited_[0,:,:], device=visited_.get_device())
        round_complete[:,0] = 0
        visited_ = torch.where(prev_a == n_loc-1, round_complete, visited_).diagonal(dim1=0,dim2=1).T[:,None,:]

        # If the demand is not fully satisfied then reset visit nodes
        reset_round = torch.zeros_like(visited_[0,:,:], device=visited_.get_device())
        reset_round[:,0] = 1
        visited_ = torch.where(torch.logical_and(prev_a == 0, demand_ != 0), reset_round,
                               visited_).diagonal(dim1=0,dim2=1).T[:,None,:]

        # Stay at source if demand is zero: Waiting for other instance to finish
        completed = torch.ones_like(visited_[0,:,:], device=visited_.get_device())
        completed[:,0] = 0
        visited_ = torch.where(demand_ == 0, completed, visited_).diagonal(dim1=0,dim2=1).T[:,None,:]

        return self._replace(prev_a=prev_a, visited_=visited_, node_usage=node_usage,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1, demand=demand_)

    def all_finished(self):

        return torch.all(self.demand <= 0) or self.i.item() >= self.loc.size(-2)*10

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        a = torch.logical_or(self.edge[:,self.prev_a][0], self.visited_ > 0)
        return a

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
