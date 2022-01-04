# -*- coding: utf-8 -*-
"""
Ammended for bipartite graph with fixed embeds for one entity
"""
import torch
from torch import empty, matmul, tensor
from torch.cuda import empty_cache
from torch.nn import Parameter
from torch.nn.functional import normalize
import torch.nn as nn

from torchkge.models import TranslationModel
from torchkge.utils import init_embedding
from torchkge.utils.operations import get_bernoulli_probs

from tqdm import tqdm

# Helpers =============================================

import os, json, codecs

import torch
from torch.utils.data import Dataset

class Bipartite_Data(Dataset):
    def __init__(self, df, head_col, tail_col, rel_col=None, head2ix=None, tail2ix=None, 
        rel2ix=None):

        self.head_idx = torch.tensor(df[head_col].values, dtype=torch.long)
        self.tail_idx = torch.tensor(df[tail_col].values, dtype=torch.long)
        self.n_facts = len(df)
        self.n_heads = len(head2ix)
        self.n_tails = len(tail2ix)

        # if rel2ix is None, assume only one relationship kind

        self.n_rel = len(rel2ix) if not rel2ix is None else 1

        # There should only be one relation in a typical bipartite. The following allows the model
        # to accomodate more than one kind of relationships

        self.relations = torch.tensor(df[rel_col].values, dtype=torch.long) if rel_col is not None \
            else torch.zeros_like(self.head_idx)

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        return (self.head_idx[item].item(),
                self.tail_idx[item].item(),
                self.relations[item].item())


class BernoulliNegSampler:
    def __init__(self, kgb, n_neg=1):
        self.kg = kgb
        self.n_heads = kgb.n_heads
        self.n_tails = kgb.n_tails
        self.n_neg = n_neg
        self.bern_probs = self.evaluate_probabilities()

    def evaluate_probabilities(self):
        """Evaluate the Bernoulli probabilities for negative sampling as in the
        TransH original paper by Wang et al. (2014).
        """
        bern_probs = get_bernoulli_probs(self.kg)

        tmp = []
        for i in range(self.kg.n_rel):
            if i in bern_probs.keys():
                tmp.append(bern_probs[i])
            else:
                tmp.append(0.5)

        return torch.tensor(tmp).float()

    def corrupt_batch(self, heads, tails, relations, n_neg):
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted
        mask = torch.bernoulli(self.bern_probs[relations].repeat(n_neg)).double()
        n_h_cor = int(mask.sum().item())
        neg_heads[mask == 1] = torch.randint(1, self.n_heads,
                                       (n_h_cor,),
                                       device=device)
        neg_tails[mask == 0] = torch.randint(1, self.n_tails,
                                       (batch_size * n_neg - n_h_cor,),
                                       device=device)

        return neg_heads.long(), neg_tails.long()

def readjson(path):
    with codecs. open(path) as f:
        data = json.loads(f.read())
    return data

def savejson(path, data):
    with codecs.open(path, 'w', 'utf-8') as f:
        f.write(json.dumps(data))

#======================================================


class Bipartite_Model:
    """
    Bundling of extra variables and functions to support Bipartite execution with one anchor entity
    """

    def __init__(self, *args, **kwargs):
        pass

    def init_tail_emb(self, tail_emb):
        if not isinstance(tail_emb, torch.Tensor):
            tail_emb = torch.tensor(tail_emb)
        tail_emb = normalize(tail_emb, p=2, dim=1)
        self.tail_emb = nn.Embedding.from_pretrained(tail_emb, freeze=True)

    def save_config(self, **kwargs):
        self.config = {k:v for k,v in kwargs.items()}

    @classmethod
    def load_pretrained(self, folderpath, modelname=None, fh=None):
        join = lambda x: os.path.join(folderpath, x)
        folderpath = join(modelname) if not modelname is None else folderpath
        if fh is None:
            config = readjson(join('config.json'))
            head2ix = readjson(join('head2ix.json'))
            model = TransRBipartiteModel(**config)
            missing_unexpected_keys = model.load_state_dict(torch.load(join('model.pt')), strict=False)
        else:
            with fh.open(join('config.json')) as f:
                config = json.loads(f.read())
            with fh.open(join('head2ix.json')) as f:
                head2ix = json.loads(f.read())
            model = TransRBipartiteModel(**config)
            with fh.open(join('model.pt')) as f:
                missing_unexpected_keys = model.load_state_dict(torch.load(f), strict=False)
        print(missing_unexpected_keys)

        return model, head2ix

    def save_pretrained(self, head2ix, folderpath, modelname=None):
        join = lambda x: os.path.join(folderpath, x)
        folderpath = join(modelname) if not modelname is None else folderpath
        if not os.path.isdir(folderpath):
            os.mkdir(folderpath)
        
        try:
            del self.tail_emb
        except:
            pass

        savejson(join('config.json'), self.config)
        savejson(join('head2ix.json'), head2ix)
        self.cpu()
        torch.save(self.state_dict(), join('model.pt'))



class TransRBipartiteModel(TranslationModel, Bipartite_Model):
    """Implementation of TransR model detailed in 2015 paper by Lin et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu.
      `Learning Entity and Relation Embeddings for Knowledge Graph Completion.
      <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523>`_
      In Twenty-Ninth AAAI Conference on Artificial Intelligence, February 2015

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    n_head: int
        Number of head entities in the current data set.
    n_tail: int
        Number of tail entities in the current data set.
    n_rel: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension nof the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    head_emb: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Embeddings of the head entities, initialized with Xavier uniform
        distribution and then normalized.
    tail_emb: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Embeddings of the tail entities, initialized with a numpy matrix.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.
    proj_mat: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim x ent_emb_dim)
        Relation-specific projection matrices. See paper for more details.
    projected_entities: `torch.nn.Parameter`, \
        shape: (n_rel, n_ent, rel_emb_dim)
        Contains the projection of each entity in each relation-specific
        sub-space.
    evaluated_projections: bool
        Indicates whether `projected_entities` has been computed. This should
        be set to true every time a backward pass is done in train mode.
    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_heads, n_tails, n_relations, tail_emb=None, 
        custom_loss=None):

        loss = 'L2' if custom_loss is None else custom_loss

        super().__init__(n_heads + n_tails, n_relations, loss)


        self.n_heads = n_heads
        self.n_tails = n_tails
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim
        self.head_emb = init_embedding(self.n_heads, self.ent_emb_dim)

        self.save_config(
            ent_emb_dim=ent_emb_dim, rel_emb_dim=rel_emb_dim, n_heads=n_heads, n_tails=0, 
            n_relations=n_relations, custom_loss=custom_loss)

        # import weights
        if not tail_emb is None:
            self.init_tail_emb(tail_emb)
        
        self.rel_emb = init_embedding(self.n_rel, self.rel_emb_dim)
        self.proj_mat = init_embedding(self.n_rel, self.rel_emb_dim * self.ent_emb_dim)

        self.normalize_parameters()

        self.evaluated_projections = False
        
    def scoring_function(self, h_idx, t_idx, r_idx=None, new_tails=None):
        """Compute the scoring function for the triplets given as argument:
        :math:`||p_r(h) + r - p_r(t)||_2^2`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        self.evaluated_projections = False

        b_size = h_idx.shape[0]
        h = normalize(self.head_emb(h_idx), p=2, dim=1)
        
        # use embeddings provided in new tails if present
        t = normalize(self.tail_emb(t_idx), p=2, dim=1) if new_tails is None \
            else normalize(new_tails, p=2, dim=1)

        r = self.rel_emb(r_idx) if r_idx is not None else \
            self.rel_emb(torch.zeros_like(h_idx, dtype=torch.long))
            
        proj_mat = self.proj_mat(r_idx).view(b_size,
                                             self.rel_emb_dim,
                                             self.ent_emb_dim)
        return - self.dissimilarity(self.project(h, proj_mat) + r,
                                    self.project(t, proj_mat))

    def reverse_score(self, tail):
        device= next(self.parameters()).device
        h_idx = torch.tensor(list(range(self.n_heads)), device=device, dtype=torch.long)
        h = normalize(self.head_emb(h_idx), p=2, dim=1)
        r_idx = torch.zeros_like(h_idx, dtype=torch.long)

        t = normalize(torch.vstack([tail]*self.n_heads), p=2, dim=1)

        r = self.rel_emb(r_idx)

        proj_mat = self.proj_mat(r_idx).view(self.n_heads,
                                             self.rel_emb_dim,
                                             self.ent_emb_dim)

        return - self.dissimilarity(self.project(h, proj_mat) + r,
                                    self.project(t, proj_mat))



    def project(self, ent, proj_mat):
        proj_e = matmul(proj_mat, ent.view(-1, self.ent_emb_dim, 1))
        return proj_e.view(-1, self.rel_emb_dim)

    def normalize_parameters(self):
        """Normalize the entity and relation embeddings, as explained in
        original paper. This methods should be called at the end of each
        training epoch and at the end of training as well.

        """
        self.head_emb.weight.data = normalize(self.head_emb.weight.data,
                                             p=2, dim=1)
                                             
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with their
        projection matrices.

        Returns
        -------
        head_emb: torch.Tensor, shape: (n_ent, ent_emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, rel_emb_dim), dtype: torch.float
            Embeddings of relations.
        proj_mat: torch.Tensor, shape: (n_rel, rel_emb_dim, ent_emb_dim),
        dtype: torch.float
            Relation-specific projection matrices.
        """
        self.normalize_parameters()
        return self.head_emb.weight.data, self.rel_emb.weight.data, \
            self.proj_mat.weight.data.view(-1,
                                           self.rel_emb_dim,
                                           self.ent_emb_dim)

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `lp_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """

        raise NotImplementedError()

    def lp_evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        self.projected_head_entities = torch.zeros((self.n_rel, self.n_heads, self.rel_emb_dim))

        for i in tqdm(range(self.n_heads), unit='heads',
                      desc='Projecting head entities'):
            projection_matrices = self.proj_mat.weight.data
            projection_matrices = projection_matrices.view(self.n_rel,
                                                           self.rel_emb_dim,
                                                           self.ent_emb_dim)

            mask = tensor([i], device=projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            ent = self.head_emb(mask)
            proj_ent = matmul(projection_matrices, ent.view(self.ent_emb_dim))
            proj_ent = proj_ent.view(self.n_rel, self.rel_emb_dim, 1)
            self.projected_head_entities[:, i, :] = proj_ent.view(self.n_rel,
                                                             self.rel_emb_dim)

            del proj_ent
        """
        if not self.tail_emb is None:
            for i in tqdm(range(self.n_tails), unit='tails',
                        desc='Projecting tail entities'):
                projection_matrices = self.proj_mat.weight.data
                projection_matrices = projection_matrices.view(self.n_rel,
                                                            self.rel_emb_dim,
                                                            self.ent_emb_dim)

                mask = tensor([i], device=projection_matrices.device).long()

                if projection_matrices.is_cuda:
                    empty_cache()

                ent = self.tail_emb(mask)
                proj_ent = matmul(projection_matrices, ent.view(self.ent_emb_dim))
                proj_ent = proj_ent.view(self.n_rel, self.rel_emb_dim, 1)
                self.projected_tail_entities[:, i, :] = proj_ent.view(self.n_rel,
                                                                self.rel_emb_dim)

                del proj_ent"""

        self.evaluated_projections = True
