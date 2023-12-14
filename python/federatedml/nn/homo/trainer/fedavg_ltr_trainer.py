

import numpy as np
import torch
import torch as t
from torch.utils.data import DataLoader

from federatedml.evaluation.metrics.ranking_metric import torch_ndcg_at_k, torch_ndcg_at_ks
from federatedml.framework.homo.aggregator import SecureAggregatorServer, SecureAggregatorClient
from federatedml.nn.dataset.table_LTR import LTRDataset, SPLIT_TYPE, LETORSampler
from federatedml.nn.homo.trainer.trainer_base import TrainerBase
from federatedml.util import LOGGER


def ndcg_at_k(self, test_data=None, k=10, presort=False, device='cpu'):
    '''
    Compute nDCG@k with the given data
    An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
    '''
    self.eval_mode()  # switch evaluation mode

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
                                          k=k, device=device)

        sum_ndcg_at_k += torch.sum(batch_ndcg_at_k)  # due to batch processing

    avg_ndcg_at_k = sum_ndcg_at_k / num_queries
    return avg_ndcg_at_k
"""
def load_data(self, eval_dict, data_dict, fold_k):
   
    Load the dataset correspondingly.
    :param eval_dict:
    :param data_dict:
    :param fold_k:
    :param model_para_dict:
    :return:

    file_train, file_vali, file_test = self.determine_files(data_dict, fold_k=fold_k)

    input_eval_dict = eval_dict if eval_dict['mask_label'] else None  # required when enabling masking data

    _train_data = LTRDataset(file=file_train, split_type=SPLIT_TYPE.Train, presort=data_dict['train_presort'],
                             data_dict=data_dict, eval_dict=input_eval_dict)
    train_letor_sampler = LETORSampler(data_source=_train_data, rough_batch_size=data_dict['train_rough_batch_size'])
    train_loader = torch.utils.data.DataLoader(_train_data, batch_sampler=train_letor_sampler, num_workers=0)

    _test_data = LTRDataset(file=file_test, split_type=SPLIT_TYPE.Test, data_dict=data_dict,
                            presort=data_dict['test_presort'])
    test_letor_sampler = LETORSampler(data_source=_test_data, rough_batch_size=data_dict['test_rough_batch_size'])
    test_loader = torch.utils.data.DataLoader(_test_data, batch_sampler=test_letor_sampler, num_workers=0)

    if eval_dict['do_validation'] or eval_dict['do_summary']:  # vali_data is required
        _vali_data = LTRDataset(file=file_vali, split_type=SPLIT_TYPE.Validation, data_dict=data_dict,
                                presort=data_dict['validation_presort'])
        vali_letor_sampler = LETORSampler(data_source=_vali_data,
                                          rough_batch_size=data_dict['validation_rough_batch_size'])
        vali_loader = torch.utils.data.DataLoader(_vali_data, batch_sampler=vali_letor_sampler, num_workers=0)
    else:
        vali_loader = None

    return train_loader, test_loader, vali_loader
"""
class LTRTrainer(TrainerBase):
    def __init__(self, epochs, batch_size, scheduler=None, eval_dict=None, data_dict=None):
        super(LTRTrainer, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.data_dict = data_dict
        self.eval_dict = eval_dict
        self.device=None
        self.gpu = None

    def train_op(self,_optimizer,_loss_fn, batch_q_doc_vectors, batch_std_labels, **kwargs):
        '''
        The training operation over a batch of queries.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs: optional arguments
        @return:
        '''
        stop_training = False
        batch_preds = self.model(batch_q_doc_vectors)
        '''


        if 'epoch_k' in kwargs and kwargs['epoch_k'] % self.stop_check_freq == 0:
            stop_training = self.stop_training(batch_preds)
        '''
        batch_loss =  _loss_fn(batch_preds, batch_std_labels, **kwargs)

        _optimizer.zero_grad()
        batch_loss.backward()
        _optimizer.step()


        return batch_loss, stop_training

    def train(self, train_set, validate_set=None, optimizer=None, loss=None, extra_data={}):
        self._optimizer = optimizer
        self._loss_fn = loss
        sample_num = len(train_set)
        train_letor_sampler = LETORSampler(data_source=train_set, rough_batch_size=self.batch_size)
        train_loader = t.utils.data.DataLoader(train_set, batch_sampler=train_letor_sampler, num_workers=0)
        aggregator = None
        if self.fed_mode:
            aggregator = SecureAggregatorClient(True, aggregate_weight=sample_num,
                                                communicate_match_suffix='fedprox')  # initialize aggregator

        # set dataloader

        for epoch in range(self.epochs):
            LOGGER.debug('running epoch {}'.format(epoch))
            num_queries = 0
            epoch_loss = torch.tensor([0.0], device=self.device)

            for batch_ids, batch_q_doc_vectors, batch_std_labels in train_loader:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
                num_queries += len(batch_ids)
                if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(
                    self.device), batch_std_labels.to(self.device)

                batch_loss, stop_training = self.train_op(self._optimizer, self._loss_fn,batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids,
                                                          epoch_k=epoch,)

                if stop_training:
                    break
                else:
                    epoch_loss += batch_loss.item()

            epoch_loss = epoch_loss / num_queries


            LOGGER.debug('epoch loss is {}'.format(epoch_loss))
            '''
            验证，好麻烦，先不写
            torch_vali_metric_value = ndcg_at_k(test_data=vali_data, k=5, label_type=label_type, presort=presort)
            vali_metric_value = torch_vali_metric_value.squeeze(-1).data.numpy()
            if epoch_k > 1:  # report and buffer currently optimal model
                if (metric_value > self.optimal_metric_value) \
                        or (epoch_k == self.num_epochs and metric_value == self.optimal_metric_value):
                    # we need at least a reference, in case all zero
                    print('\t', epoch_k, '- {}@{} - '.format(self.validation_metric, self.validation_at_k),
                          metric_value)
                    self.optimal_epoch_value = epoch_k
                    self.optimal_metric_value = metric_value
                    ranker.save(dir=self.dir_run + self.fold_optimal_checkpoint + '/',
                                name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')
                else:
                    print('\t\t', epoch_k, '- {}@{} - '.format(self.validation_metric, self.validation_at_k),
                          metric_value)
            '''

            # the aggregation process
            if aggregator is not None:
                self.model = aggregator.model_aggregation(self.model)
                loss_value = epoch_loss.item()  # 将 torch.Tensor 转换为 float

                converge_status = aggregator.loss_aggregation(loss_value)
        metric_string = evaluation (self,file_test = '/data/Corpus/MQ2008/MQ2008/Fold1/test.txt')

    # implement the aggregation function, this function will be called by the sever side
    def server_aggregate_procedure(self, extra_data={}):

        # initialize aggregator
        if self.fed_mode:
            aggregator = SecureAggregatorServer(communicate_match_suffix='fedprox')

        # the aggregation process is simple: every epoch the server aggregate model and loss once
        for i in range(self.epochs):
            aggregator.model_aggregation()
            merge_loss, _ = aggregator.loss_aggregation()

    def predict(self, batch_q_doc_vectors):
        """
        实现预测逻辑
        """

        batch_preds = self.model(batch_q_doc_vectors)

        return batch_preds



def adhoc_performance_at_ks(self, test_data=None, ks=[1, 5, 10],  max_label=None,
                                presort=False, device='cpu', need_per_q=False):
    '''
    Compute the performance using multiple metrics
    '''
    #self.eval()  # switch evaluation mode

    num_queries = 0
    sum_ndcg_at_ks = torch.zeros(len(ks))
    '''
    sum_nerr_at_ks = torch.zeros(len(ks))
    sum_ap_at_ks = torch.zeros(len(ks))
    sum_p_at_ks = torch.zeros(len(ks))


    if need_per_q: list_per_q_p, list_per_q_ap, list_per_q_nerr, list_per_q_ndcg = [], [], [], []
    '''

    for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
        if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
        batch_preds = self.predict(batch_q_doc_vectors)
        if self.gpu: batch_preds = batch_preds.cpu()

        _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
        batch_pred_desc_inds = batch_pred_desc_inds.squeeze(-1)
        batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
        if presort:
            batch_ideal_rankings = batch_std_labels
        else:
            batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

        batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                            batch_ideal_rankings=batch_ideal_rankings,
                                            ks=ks, device=device)
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
        '''


        batch_nerr_at_ks = torch_nerr_at_ks(batch_predict_rankings=batch_predict_rankings,
                                            batch_ideal_rankings=batch_ideal_rankings, max_label=max_label,
                                            ks=ks,  device=device)
        sum_nerr_at_ks = torch.add(sum_nerr_at_ks, torch.sum(batch_nerr_at_ks, dim=0))

        batch_ap_at_ks = torch_ap_at_ks(batch_predict_rankings=batch_predict_rankings,
                                        batch_ideal_rankings=batch_ideal_rankings, ks=ks, device=device)
        sum_ap_at_ks = torch.add(sum_ap_at_ks, torch.sum(batch_ap_at_ks, dim=0))

        batch_p_at_ks = torch_precision_at_ks(batch_predict_rankings=batch_predict_rankings, ks=ks, device=device)
        sum_p_at_ks = torch.add(sum_p_at_ks, torch.sum(batch_p_at_ks, dim=0))

        if need_per_q:
            list_per_q_p.append(batch_p_at_ks)
            list_per_q_ap.append(batch_ap_at_ks)
            list_per_q_nerr.append(batch_nerr_at_ks)
            avg_nerr_at_ks = sum_nerr_at_ks / num_queries
    avg_ap_at_ks = sum_ap_at_ks / num_queries
    avg_p_at_ks = sum_p_at_ks / num_queries
            list_per_q_ndcg.append(batch_ndcg_at_ks)
                if need_per_q:
        return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks, \
            list_per_q_ndcg, list_per_q_nerr, list_per_q_ap, list_per_q_p
        '''

        num_queries += len(batch_ids)

    avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
    return avg_ndcg_at_ks
def metric_results_to_string(list_scores=None, list_cutoffs=None, split_str=', ', metric='nDCG'):
    """
    Convert metric results to a string representation
    :param list_scores:
    :param list_cutoffs:
    :param split_str:
    :return:
    """
    list_str = []
    for i in range(len(list_scores)):
        list_str.append(metric + '@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
    return split_str.join(list_str)

def evaluation(self, file_test=None):
    vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
    self.cutoffs = cutoffs

    _test_data = LTRDataset(file=file_test, split_type=SPLIT_TYPE.Test,data_id="MQ2008_Super",
                                 data_dict=None, eval_dict=None)
    test_letor_sampler = LETORSampler(data_source=_test_data,
                                      rough_batch_size=128)
    test_loader = torch.utils.data.DataLoader(_test_data, batch_sampler=test_letor_sampler, num_workers=0)
    avg_ndcg_at_ks = adhoc_performance_at_ks(self,test_data=test_loader, ks=self.cutoffs, device='cpu', max_label=4)
    fold_ndcg_ks = avg_ndcg_at_ks.data.numpy()

    #ndcg_cv_avg_scores = np.add(self.ndcg_cv_avg_scores, fold_ndcg_ks)

    list_metric_strs = []
    list_metric_strs.append(metric_results_to_string(list_scores=fold_ndcg_ks, list_cutoffs=self.cutoffs,
                                                          metric='nDCG'))

    metric_string = '\n\t'.join(list_metric_strs)
    print("\n{} on Fold - {}\n".format('fed', metric_string))


    return metric_string



    def server_aggregate_procedure(self, extra_data={}):

        # initialize aggregator
        if self.fed_mode:
            aggregator = SecureAggregatorServer(communicate_match_suffix='fedprox')

        # the aggregation process is simple: every epoch the server aggregate model and loss once
        for i in range(self.epochs):
            aggregator.model_aggregation()
            merge_loss, _ = aggregator.loss_aggregation()
