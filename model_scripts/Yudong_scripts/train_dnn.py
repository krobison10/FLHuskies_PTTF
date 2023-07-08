from constants import ALL_AIRPORTS

# from pytorch_dnn import MyTorchDNN
from tf_dnn import MyTensorflowDNN

if __name__ == "__main__":
    # for airport in [*ALL_AIRPORTS, "ALL"]:
    # MyTensorflowDNN.train(airport)
    # MyTensorflowDNN.evaluate(airport)
    MyTensorflowDNN.evaluate_individual_in_global_setting()
