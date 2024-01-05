
from federatedml.nn.dataset.table_LTR import LTRDataset, SPLIT_TYPE
from federatedml.nn.loss.lambda_loss import LambdaLoss


ds = LTRDataset(data_id="MQ2008_Super")
ds.load(file_path='/home/user/Workbench/tan_haonan/FATE_Ltr/examples/data/mq2008_homo_guest.csv', split_type=SPLIT_TYPE.Train, presort=False,
                                 data_dict=None, eval_dict=None)
# load MNIST data and check
# ds.load('/home/user/Workbench/tan_haonan/FATE_Ltr/examples/data/mq2008_homo_guest.csv')

from federatedml.nn.homo.trainer.fedavg_ltr_trainer import LTRTrainer

import torch as t
from pipeline import fate_torch_hook
fate_torch_hook(t)
# our simple classification model:
model = t.nn.Sequential(
    t.nn.Linear(46, 32),
    t.nn.ReLU(),
    t.nn.Linear(32, 1),
    t.nn.Softmax(dim=1)
)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)  # optimizer

#loss = t.nn.CustLoss(loss_module_name='mse_loss', class_name='MSELoss')
#loss = t.nn.CustLoss(loss_module_name='lambda_loss', class_name='LambdaLoss')
loss = LambdaLoss()
#loss = MSELoss()
trainer = LTRTrainer(epochs=3, batch_size=128) # set parameter
trainer.local_mode() # !! Be sure to enable local_mode to skip the federation process !!
trainer.set_model(model) # set model

trainer.train(train_set=ds, optimizer=optimizer, loss=loss)  # use dataset we just developed