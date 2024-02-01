import pickle
from sklearn.metrics import accuracy_score, classification_report
import torch as t
import numpy as np

from federatedml.evaluation.metrics.ranking_metric import torch_ndcg_at_ks
from federatedml.nn.dataset.table_LTR import LTRDataset, LETORSampler
from federatedml.nn.homo.trainer.fedavg_ltr_trainer import adhoc_performance_at_ks, metric_results_to_string

# 加载模型
# 加载模型
model_weights = np.load('/home/user/Workbench/tan_haonan/FATE_Ltr/saved_model/aggregated_model.npy', allow_pickle=True)

# 提取权重和偏置
weights_0 = model_weights[0][0]._weights  # 提取第一层的权重
bias_0 = model_weights[0][1]._weights    # 提取第一层的偏置
weights_1 = model_weights[0][2]._weights  # 提取第二层的权重
bias_1 = model_weights[0][3]._weights    # 提取第二层的偏置

# 创建模型
model = t.nn.Sequential(
    t.nn.Linear(46, 32),
    t.nn.ReLU(),
    t.nn.Linear(32, 1),
    t.nn.Softmax(dim=1)
)

# 将权重和偏置分配给模型的对应层
model[0].weight.data = t.from_numpy(weights_0).type(t.float32)
model[0].bias.data = t.from_numpy(bias_0).type(t.float32)
model[2].weight.data = t.from_numpy(weights_1).type(t.float32)
model[2].bias.data = t.from_numpy(bias_1).type(t.float32)

# 设置为评估模式
model.eval()

def evaluation( file_test=None,data_id = None):
    vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
    file_path = str(file_test)
    # 创建 LTRDataset 的实例
    _test_data_instance = LTRDataset(data_id= "MQ2008_Super")

    _test_data_instance.load(file_path='/data/Corpus/MQ2008/MQ2008/Fold1/test.txt', split_type=None, data_dict=None,
                             eval_dict=None, presort=False, hot=False, buffer=True)

    test_letor_sampler = LETORSampler(data_source=_test_data_instance,
                                      rough_batch_size=128)
    test_loader = t.utils.data.DataLoader(_test_data_instance, batch_sampler=test_letor_sampler, num_workers=0)
    avg_ndcg_at_ks = adhoc_performance_at_ks(test_data=test_loader, ks=cutoffs, device='cpu', max_label=4)
    fold_ndcg_ks = avg_ndcg_at_ks.data.numpy()

    #ndcg_cv_avg_scores = np.add(self.ndcg_cv_avg_scores, fold_ndcg_ks)

    list_metric_strs = []
    list_metric_strs.append(metric_results_to_string(list_scores=fold_ndcg_ks, list_cutoffs=cutoffs,
                                                          metric='nDCG'))

    metric_string = '\n\t'.join(list_metric_strs)
    print("\n{} on Fold - {}\n".format('fed', metric_string))


    return metric_string
def adhoc_performance_at_ks(test_data=None, ks=[1, 5, 10],  max_label=None,
                                presort=False, device='cpu', need_per_q=False):
    '''
    Compute the performance using multiple metrics
    '''
    #self.eval()  # switch evaluation mode

    num_queries = 0
    sum_ndcg_at_ks = t.zeros(len(ks))
    '''
    sum_nerr_at_ks = torch.zeros(len(ks))
    sum_ap_at_ks = torch.zeros(len(ks))
    sum_p_at_ks = torch.zeros(len(ks))


    if need_per_q: list_per_q_p, list_per_q_ap, list_per_q_nerr, list_per_q_ndcg = [], [], [], []
    '''

    for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]

        batch_preds = model(batch_q_doc_vectors)


        _, batch_pred_desc_inds = t.sort(batch_preds, dim=1, descending=True)
        batch_pred_desc_inds = batch_pred_desc_inds.squeeze(-1)
        batch_predict_rankings = t.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
        if presort:
            batch_ideal_rankings = batch_std_labels
        else:
            batch_ideal_rankings, _ = t.sort(batch_std_labels, dim=1, descending=True)

        batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                            batch_ideal_rankings=batch_ideal_rankings,
                                            ks=ks, device=device)
        sum_ndcg_at_ks = t.add(sum_ndcg_at_ks, t.sum(batch_ndcg_at_ks, dim=0))
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
evaluation(file_test='/data/Corpus/MQ2008/MQ2008/Fold1/test.txt', data_id="MQ2008_Super")

#