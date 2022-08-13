##
## Wenqiao Zhu (zhuwnq@outlook.com)
##
import time
import networkx as nx
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from cogdl.utils import alias_draw, alias_setup
from cogdl.models import BaseModel, register_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from cogdl import experiment

from cogdl.options import get_default_args

class Qalign(nn.Module):
    def __init__(self, size, emb_dim=128, t=2.0, alpha = 4, beta = 2, form="base"):
        super(Qalign, self).__init__()
        self.emb_dim = emb_dim
        self.embs = nn.Embedding(size, emb_dim)
        self.embs.weight.data = self.embs.weight.data.uniform_(-0.5, 0.5)/emb_dim
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.form = form

    def forward(self, u, v):
        src_emb = self.embs(u)
        pos_emb = self.embs(v)

        src_emb = torch.nn.functional.normalize(src_emb)
        pos_emb = torch.nn.functional.normalize(pos_emb)
        
        def align_log(_src, _pos):
            multi = torch.sum(_src * _pos, axis=1)
            return torch.mean(self.t - self.t * multi)

        def uniform_log(_emb):
            prod = _emb @ _emb.T
            coe = self.t * prod - self.t
            exp = torch.log(torch.sum(torch.exp(coe), axis=1))
            return torch.mean(exp)

        loss = 0.0

        if self.form == "log":
            align_loss = align_log(src_emb, pos_emb)        
            uniform_loss_1 = uniform_log(src_emb)
            uniform_loss_2 = uniform_log(pos_emb)            
            loss = self.alpha*align_loss + self.beta*uniform_loss_1 + self.beta*uniform_loss_2
        else:
            print("only log is tested")
            exit(0)        
        return loss

@register_model("qalign")
class QalignTask(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--walk-num', type=int, default=20,
                            help='Number of walks per source. Default is 20.')
        parser.add_argument('--batch-size', type=int, default=64,
                            help='Batch size in SGD training process. Default is 1000.')
        parser.add_argument('--learning-rate', type=float, default=0.025,
                            help='Initial learning rate of SGD. Default is 0.025.')
        parser.add_argument('--t', type=float, default=2.0,
                            help='t.')
        parser.add_argument('--weight_alpha', type=float, default=20,
                            help='alpha.')
        parser.add_argument('--weight_beta', type=float, default=10,
                            help='beta.')
        parser.add_argument('--form', type=str, default="base",
                            help="form of align and uniform")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.batch_size,
            args.learning_rate,
            args.t,
            args.weight_alpha,
            args.weight_beta,
            args.form
        )

    def __init__(self, dimension, walk_length, walk_num, batch_size, learning_rate, t, weight_alpha, weight_beta, form):
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.batch_size = batch_size
        self.init_alpha = learning_rate
        self.t = t
        self.weight_alpha = weight_alpha
        self.weight_beta = weight_beta
        self.form = form


    def train(self, G):
        # run Qalign algorithm
        self.G = G
        self.is_directed = nx.is_directed(self.G)
        self.num_node = G.number_of_nodes()
        self.num_edge = G.number_of_edges()
        print("nodes and edge number: ", self.num_node, self.num_edge)
        self.num_sampling_edge = self.walk_length * self.walk_num * self.num_node

        node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])

        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]
        self.edges_prob = np.asarray([G[u][v].get("weight", 1.0) for u, v in G.edges()])
        print("edges and edge_prob.shape: ", len(self.edges), self.edges_prob.shape)
        print("self.edges_prob: ", self.edges_prob)
        self.edges_prob /= np.sum(self.edges_prob)
        self.edges_table, self.edges_prob = alias_setup(self.edges_prob)

        # train:
        self.learning_rate = self.init_alpha
        batch_size = self.batch_size
        t0 = time.time()
        num_batch = int(self.num_sampling_edge / batch_size)
        epoch_iter = tqdm(range(num_batch))
        
        # begin my code:
        qalign = Qalign(self.num_node, self.dimension, self.t, self.weight_alpha, self.weight_beta, self.form)
        opt = optim.SGD(qalign.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
        lambda_b = lambda b: self.init_alpha * max((1 - b * 1.0 / num_batch), 0.0001)
        #scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_b)
        #scheduler  = lr_scheduler.ExponentialLR(opt, gamma=0.1)
        print("num_batch: ", num_batch)
        #print(self.init_alpha, scheduler.get_lr())
        for b in epoch_iter:
            curr_lr = self.init_alpha * max((1 - b * 1.0 / num_batch), 0.0001)
            if b % 100 == 0:
                epoch_iter.set_description(
                    f"Progress: {b *1.0/num_batch * 100:.4f}%, lr: {curr_lr:.6f}, time: {time.time() - t0:.4f}"
                )
            u, v = [0] * batch_size, [0] * batch_size
            opt.param_groups[0]['lr'] = curr_lr
            #exit(0)
            for i in range(batch_size):
                edge_id = alias_draw(self.edges_table, self.edges_prob)
                u[i], v[i] = self.edges[edge_id]
                if not self.is_directed and np.random.rand() > 0.5: 
                    v[i], u[i] = self.edges[edge_id]

            qalign.zero_grad()            
            u_tensor = torch.LongTensor(u)
            v_tensor = torch.LongTensor(v)

            loss = qalign(u_tensor, v_tensor)
            
            loss.backward()
            opt.step()
            #scheduler.step()

        res = qalign.embs.weight.detach().numpy()
        self.embeddings = preprocessing.normalize(res, "l2")
        return self.embeddings

if __name__ == '__main__':
    print("test code for qalign")
    task="unsupervised_node_classification"
    dataset=["flickr-ne"]
    model="qalign"
    args = get_default_args(task=task, dataset=dataset, model=model)
    args.form = "log"
    args.seed = [0,1,2,3,4]
    args.weight_alpha = 0.5
    args.weight_beta = 1.5
    args.training_percents = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]        
    args.t = 2.0
    args.walk_num = 30
    args.batch_size = 512
    experiment(task=task, dataset=dataset, model=model, args=args)
