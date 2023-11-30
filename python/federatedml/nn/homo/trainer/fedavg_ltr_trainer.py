import torch
from torch.utils.data import DataLoader

from federatedml.nn.dataset.table_LTR import LTRDataset, SPLIT_TYPE, LETORSampler
from federatedml.nn.homo.trainer.trainer_base import TrainerBase
from federatedml.util import LOGGER



def load_data(self, eval_dict, data_dict, fold_k):
    """
    Load the dataset correspondingly.
    :param eval_dict:
    :param data_dict:
    :param fold_k:
    :param model_para_dict:
    :return:
    """
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
class LTRTrainer(TrainerBase):
    def __init__(self, epochs, batch_size, model, optimizer, loss_fn, scheduler=None):
        super(LTRTrainer, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train(self, train_set, validate_set=None, extra_data={}):
        """
        实现 LTR 的训练逻辑
        """
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            LOGGER.debug(f"Epoch {epoch+1}/{self.epochs}")
            epoch_loss = 0.0

            for data, target in dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss_val = self.loss_fn(output, target)
                loss_val.backward()
                self.optimizer.step()
                epoch_loss += loss_val.item()

            LOGGER.debug(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")

            if self.scheduler:
                self.scheduler.step()

            # 可选的验证逻辑
            if validate_set is not None:
                self._validate(validate_set)

        # 联邦学习特定的聚合逻辑

    def _validate(self, validate_set):
        """
        验证逻辑
        """
        dataloader = DataLoader(validate_set, batch_size=self.batch_size)
        validate_loss = 0.0

        with torch.no_grad():
            for data, target in dataloader:
                output = self.model(data)
                loss_val = self.loss_fn(output, target)
                validate_loss += loss_val.item()

        LOGGER.debug(f"Validation Loss: {validate_loss/len(dataloader)}")

    # predict 和 server_aggregate_procedure 方法的实现略
    def predict(self, dataset):
        """
        实现预测逻辑
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        predictions = []

        with torch.no_grad():
            for data in dataloader:
                output = self.model(data)
                predictions.append(output)

        return predictions

    def server_aggregate_procedure(self, extra_data={}):
        """
        服务器聚合过程
        """
        # 如果在联邦模式下，实现模型聚合逻辑
        pass
