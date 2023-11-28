import torch
import torch.nn
import torch.nn.functional as F

from federatedml.evaluation.metrics.ranking_metric import torch_dcg_at_k


def get_delta_ndcg(batch_ideal_rankings, batch_predict_rankings, device='cpu'):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_ideal_rankings: the standard labels sorted in a descending order
    :param batch_predict_rankings: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    # ideal discount cumulative gains
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, device=device)

    batch_gains = torch.pow(2.0, batch_predict_rankings) - 1.0


    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_predict_rankings.size(1), dtype=torch.float, device=device)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg


def get_pairwise_comp_probs(batch_preds, batch_std_labels, sigma=None):
    '''
    Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
    @param batch_preds:
    @param batch_std_labels:
    @param sigma:
    @return:
    '''
    # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij



def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
    '''
    @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
    @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
    @param kwargs:
    @return:
    '''


    # sort documents according to the predicted relevance
    batch_descending_preds, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
    # reorder batch_stds correspondingly so as to make it consistent.
    # BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor
    batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

    batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=batch_descending_preds,
                                                         batch_std_labels=batch_predict_rankings,
                                                         sigma=self.sigma)

    batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_std_labels,
                                      batch_predict_rankings=batch_predict_rankings, device=self.device)

    _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                         target=torch.triu(batch_std_p_ij, diagonal=1),
                                         weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='none')

    batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

    self.optimizer.zero_grad()
    batch_loss.backward()
    self.optimizer.step()

    return batch_loss
class LambdaLoss(torch.nn.Module):
    def __init__(self):
        super(LambdaLoss,self).__init__()

    def forward(self ,batch_preds, batch_std_labels):
        batch_loss = custom_loss_function(batch_preds, batch_std_labels)


        return batch_loss