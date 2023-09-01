from tf_client.tf_dnn import MyTensorflowDNN

MyTensorflowDNN.FEDERATED_MODE = False
# MyTensorflowDNN.train("ALL", False, airline="PUBLIC")

# MyTensorflowDNN.evaluate_global_fed()

MyTensorflowDNN.evaluate_global()
