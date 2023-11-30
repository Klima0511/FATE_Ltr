import numpy as np
import torch


def ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], presort=False, device='cpu'):
    '''
    Compute nDCG with multiple cutoff values with the given data
    An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
    '''
    self.eval_mode()  # switch evaluation mode

    num_queries = 0
    sum_ndcg_at_ks = torch.zeros(len(ks))
    for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
        if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
        batch_preds = self.predict(batch_q_doc_vectors)
        if self.gpu: batch_preds = batch_preds.cpu()

        _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
        batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
        if presort:
            batch_ideal_rankings = batch_std_labels
        else:
            batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

        batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                            batch_ideal_rankings=batch_ideal_rankings,
                                            ks=ks, device=device)
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
        num_queries += len(batch_ids)

    avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
    return avg_ndcg_at_ks


def torch_dcg_at_k(batch_rankings, cutoff=None,  device='cpu'):
    '''
    ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    :param batch_rankings: [batch_size, ranking_size] rankings of labels (either standard or predicted by a system)
    :param cutoff: the cutoff position
    :param label_type: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: [batch_size, 1] cumulative gains for each rank position
    '''
    if cutoff is None:  # using whole list
        cutoff = batch_rankings.size(1)


    batch_numerators = torch.pow(2.0, batch_rankings[:, 0:cutoff]) - 1.0

    # no expanding should also be OK due to the default broadcasting
    batch_discounts = torch.log2(
        torch.arange(cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators / batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k


def torch_dcg_at_ks(batch_rankings, max_cutoff,  device='cpu'):
    '''
    :param batch_rankings: [batch_size, ranking_size] rankings of labels (either standard or predicted by a system)
    :param max_cutoff: the maximum cutoff value
    :param label_type: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: [batch_size, max_cutoff] cumulative gains for each rank position
    '''

    batch_numerators = torch.pow(2.0, batch_rankings[:, 0:max_cutoff]) - 1.0


    batch_discounts = torch.log2(
        torch.arange(max_cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_ks = torch.cumsum(batch_numerators / batch_discounts, dim=1)  # dcg w.r.t. each position
    return batch_dcg_at_ks


def torch_ndcg_at_k(batch_predict_rankings, batch_ideal_rankings, k=None, device='cpu'):
    batch_sys_dcg_at_k = torch_dcg_at_k(batch_predict_rankings, cutoff=k,
                                        device=device)  # only using the cumulative gain at the final rank position
    batch_ideal_dcg_at_k = torch_dcg_at_k(batch_ideal_rankings, cutoff=k, device=device)
    batch_ndcg_at_k = batch_sys_dcg_at_k / batch_ideal_dcg_at_k
    return batch_ndcg_at_k


def torch_ndcg_at_ks(batch_predict_rankings, batch_ideal_rankings, ks=None, device='cpu'):
    valid_max_cutoff = batch_predict_rankings.size(1)
    used_ks = [k for k in ks if k <= valid_max_cutoff] if valid_max_cutoff < max(ks) else ks

    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    inds = inds.long()

    batch_sys_dcgs = torch_dcg_at_ks(batch_predict_rankings, max_cutoff=max(used_ks),
                                     device=device)
    batch_sys_dcg_at_ks = batch_sys_dcgs[:, inds]  # get cumulative gains at specified rank positions
    batch_ideal_dcgs = torch_dcg_at_ks(batch_ideal_rankings, max_cutoff=max(used_ks),
                                       device=device)
    batch_ideal_dcg_at_ks = batch_ideal_dcgs[:, inds]

    batch_ndcg_at_ks = batch_sys_dcg_at_ks / batch_ideal_dcg_at_ks

    if valid_max_cutoff < max(ks):
        padded_ndcg_at_ks = torch.zeros(batch_predict_rankings.size(0), len(ks))
        padded_ndcg_at_ks[:, 0:len(used_ks)] = batch_ndcg_at_ks
        return padded_ndcg_at_ks
    else:
        return batch_ndcg_at_ks
class NDCGatkIndex(object):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for ranking tasks in FATE.
    """
    def coumpute(self, test_data=None, k=10, presort=False, device='cpu'):
        '''
        Compute nDCG@k with the given data
        An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
        '''
        self.eval_mode() # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_k = torch.zeros(1)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)

            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_k = torch_ndcg_at_k(batch_predict_rankings=batch_predict_rankings,
                                              batch_ideal_rankings=batch_ideal_rankings,
                                              k=k,device=device)

            sum_ndcg_at_k += torch.sum(batch_ndcg_at_k) # due to batch processing

        avg_ndcg_at_k = sum_ndcg_at_k / num_queries
        return avg_ndcg_at_k



class NDCGatksIndex(object):
    def compute(self, test_data=None, ks=[1, 5, 10], presort=False, device='cpu'):
        '''
        Compute nDCG with multiple cutoff values with the given data
        An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks,  device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        return avg_ndcg_at_ks
