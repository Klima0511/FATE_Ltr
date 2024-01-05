# torch
import torch as t
import torch.nn as nn
# pipeline
from pipeline.component.homo_nn import HomoNN, TrainerParam  # HomoNN Component, TrainerParam for setting trainer parameter
from pipeline.backend.pipeline import PipeLine  # pipeline class
from pipeline.component import Reader, DataTransform, Evaluation # Data I/O and Evaluation
from pipeline.interface import Data  # Data Interaces for defining data flow
from pipeline import fate_torch_hook
t = fate_torch_hook(t)
# create a pipeline to submitting the job
guest = 9999
host = 10000
arbiter = 10000
# read uploaded dataset
guest_data = {"name": "web30k_homo_guest_2", "namespace": "experiment"}
host_data = {"name": "web30k_homo_host_2", "namespace": "experiment"}

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

data_path_0 = '/data/Corpus/MSLR-WEB30K/Fold2/train.txt'
data_path_1 = '/data/Corpus/MSLR-WEB30K/Fold2/vali.txt'

pipeline.bind_table(name=guest_data['name'], namespace=guest_data['namespace'], path=data_path_0)
pipeline.bind_table(name=host_data['name'], namespace=host_data['namespace'], path=data_path_1)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_data)
reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_data)


from pipeline.component.nn import DatasetParam

dataset_param = DatasetParam(dataset_name='table_LTR',data_id= "MSLRWEB30K")


model = t.nn.Sequential(
    t.nn.CustModel(module_name='Lambdarank', class_name='MyCustomModel')
)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
loss = t.nn.CustLoss(loss_module_name='lambda_loss', class_name='LambdaLoss')
#loss = t.nn.CustLoss(loss_module_name='mse_loss', class_name='MSELoss')
#trainer = LTRTrainer(epochs=30, batch_size=128,model = model,optimizer=optimizer,loss_fn= loss) # set parameter

"""
Create Homo-NN Component
"""
nn_component = HomoNN(name='nn_0',
                      model=model, # set model
                      loss=loss, # set loss
                      optimizer=optimizer,
                      dataset=dataset_param,
                      # TrainerParam passes parameters to fedavg_trainer, see below for details about Trainer
                      trainer=TrainerParam(trainer_name='fedavg_ltr_trainer', epochs=30, batch_size=1024),
                      torch_seed=100 # random seed
                      )
# define work flow
pipeline.add_component(reader_0)
pipeline.add_component(nn_component, data=Data(data=reader_0.output.data))
pipeline.compile()
pipeline.fit()
# get predict scores
"""
def get_AF(activation_function):
    if activation_function == "GE":
        return nn.GELU()  # 使用 GELU 作为代替
    else:
        return nn.ReLU()  # 默认返回 ReLU

# 定义 LTRBatchNorm 类
class LTRBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, affine=True, track_running_stats=False):
        super(LTRBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, X):
        if X.dim() == 3:
            return self.bn(X.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return self.bn(X)

    def to_dict(self):
        return {
            'num_features': 136,
            'momentum': 0.1,
            'affine': True,
            'track_running_stats': False
        }

# 模型参数
N = 136  # 输入特征数量
H = 100  # 隐藏层维度
O = 1    # 输出层维度
ff_dims = [N, H, H, H, O]  # 网络维度

# 创建模型
model = t.nn.Sequential()

for i in range(1, len(ff_dims)):
    model.add_module(f'ff_{i}', nn.Linear(ff_dims[i - 1], ff_dims[i]))  # 线性层
    if i < len(ff_dims) - 1:  # 除了输出层外的层
        model.add_module(f'dr_{i}', nn.Dropout(0.1))  # Dropout层
        model.add_module(f'bn_{i}', LTRBatchNorm(ff_dims[i], affine=True))  # 批量归一化层
        model.add_module(f'act_{i}', get_AF("GE"))  # GELU激活函数层

# 添加尾部激活函数
model.add_module(f'act_tl', get_AF("GE"))
# 加载训练好的模型
pipeline.dump("pipeline_saved.pkl");
pipeline = PipeLine.load_model_from_file('pipeline_saved.pkl')

# 部署需要的组件
pipeline.deploy_component([pipeline.nn_0])

# 创建读取测试数据的Reader组件
test_data = {"name": "mq2008_homo_test", "namespace": "experiment"}
data_path_2 = '/data/Corpus/MQ2008/MQ2008/Fold1/test.txt'
pipeline.bind_table(name=test_data['name'], namespace=test_data['namespace'], path=data_path_2)
reader_1 = Reader(name="reader_1")
# 这里需要根据您的测试数据设置正确的参数
# 例如，如果测试数据是文件，您需要指定文件路径p
reader_1.get_party_instance(role="guest", party_id=9999).component_param(table={"name": "mq2008_homo_guest", "namespace": "experiment"})
reader_1.get_party_instance(role="host", party_id=10000).component_param(table={"name": "mq2008_homo_host", "namespace": "experiment"})

# 创建评估组件，根据您的模型类型选择合适的eval_type
#evaluation_0 = Evaluation(name="evaluation_0", eval_type="ranking")

# 构建预测Pipeline
predict_pipeline = PipeLine()
predict_pipeline.add_component(reader_1)
predict_pipeline.add_component(pipeline, data=Data(predict_input={"nn_0.train_data": reader_1.output.data}))
predict_pipeline.predict()

predict_pipeline.add_component(evaluation_0, data=Data(data=pipeline.nn_0.output.data))

# 执行预测

"""

