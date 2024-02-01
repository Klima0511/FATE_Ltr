from pipeline.backend.pipeline import PipeLine  # pipeline Class

# [9999(guest), 10000(host)] as client
# [10000(arbiter)] as server

guest = 9999
host = 10000
arbiter = 10000
pipeline_upload = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

partition = 4

# upload a dataset

path_to_fate_project = '/home/user/Workbench/tan_haonan/FATE_Ltr/'
test_data = {"name": "mq2008_homo_test", "namespace": "experiment"}
pipeline_upload.add_upload_data(file="/data/Corpus/MSLR-WEB30K/Fold2/train.txt", # file in the example/data
                                table_name=test_data["name"],             # table name
                                namespace=test_data["namespace"],         # namespace
                                head=1, partition=partition)               # data info


pipeline_upload.upload(drop=1)



"""
# upload a dataset
path_to_fate_project = '../../../../'
guest_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_data = {"name": "breast_hetero_host", "namespace": "experiment"}
pipeline_upload.add_upload_data(file="/home/user/Workbench/tan_haonan/FATE_Ltr/examples/data/breast_hetero_guest.csv", # file in the example/data
                                table_name=guest_data["name"],             # table name
                                namespace=guest_data["namespace"],         # namespace
                                head=1, partition=partition)               # data info
pipeline_upload.add_upload_data(file="/home/user/Workbench/tan_haonan/FATE_Ltr/examples/data/breast_hetero_host.csv", # file in the example/data
                                table_name=host_data["name"],             # table name
                                namespace=host_data["namespace"],         # namespace
                                head=1, partition=partition)               # data info


pipeline_upload.upload(drop=1)
"""