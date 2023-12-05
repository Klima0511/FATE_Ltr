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
print("1")
guest_data = {"name": "web30k_homo_guest_2", "namespace": "experiment"}
host_data = {"name": "web30k_homo_host_2", "namespace": "experiment"}
pipeline_upload.add_upload_data(file="/data/Corpus/MSLR-WEB30K/Fold2/train.txt", # file in the example/data
                                table_name=guest_data["name"],             # table name
                                namespace=guest_data["namespace"],         # namespace
                                head=1, partition=partition)               # data info
pipeline_upload.add_upload_data(file="/data/Corpus/MSLR-WEB30K/Fold2/vali.txt", # file in the example/data
                                table_name=host_data["name"],             # table name
                                namespace=host_data["namespace"],         # namespace
                                head=1, partition=partition)               # data info


pipeline_upload.upload(drop=1)