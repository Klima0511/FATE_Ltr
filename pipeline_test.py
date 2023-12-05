# torch
import torch as t
from torch import nn
from pipeline import fate_torch_hook

from federatedml.nn.loss.lambda_loss import LambdaLoss

fate_torch_hook(t)
# pipeline
from pipeline.component.homo_nn import HomoNN, TrainerParam  # HomoNN Component, TrainerParam for setting trainer parameter
from pipeline.backend.pipeline import PipeLine  # pipeline class
from pipeline.component import Reader, DataTransform, Evaluation # Data I/O and Evaluation
from pipeline.interface import Data  # Data Interaces for defining data flow


# create a pipeline to submitting the job
guest = 9999
host = 10000
arbiter = 10000
pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

# read uploaded dataset
train_data_0 = {"name": "mq2008_homo_guest", "namespace": "experiment"}
train_data_1 = {"name": "mq2008_homo_host", "namespace": "experiment"}
reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=train_data_0)
reader_0.get_party_instance(role='host', party_id=host).component_param(table=train_data_1)

# The transform component converts the uploaded data to the DATE standard format
data_transform_0 = DataTransform(name='data_transform_0')
data_transform_0.get_party_instance(
    role='guest', party_id=guest).component_param(
    with_label=True, output_format="dense")
data_transform_0.get_party_instance(
    role='host', party_id=host).component_param(
    with_label=True, output_format="dense")

"""
Define Pytorch model/ optimizer and loss
"""
model = nn.Sequential(
    nn.Linear(30, 1),
    nn.Sigmoid()
)
loss = t.nn.CustLoss(loss_module_name='lambda_loss', class_name='LambdaLoss')
optimizer = t.optim.Adam(model.parameters(), lr=0.01)


"""
Create Homo-NN Component
"""
nn_component = HomoNN(name='nn_0',
                      model=model, # set model
                      loss=loss, # set loss
                      optimizer=optimizer, # set optimizer
                      # Here we use fedavg trainer
                      # TrainerParam passes parameters to fedavg_trainer, see below for details about Trainer
                      trainer=TrainerParam(trainer_name='fedavg_ltr_trainer', epochs=3, batch_size=128, u=0.5),
                      torch_seed=100 # random seed
                      )

# define work flow
pipeline.add_component(reader_0)
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
pipeline.add_component(nn_component, data=Data(train_data=data_transform_0.output.data))
pipeline.add_component(Evaluation(name='eval_0'), data=Data(data=nn_component.output.data))

pipeline.compile()
pipeline.fit()