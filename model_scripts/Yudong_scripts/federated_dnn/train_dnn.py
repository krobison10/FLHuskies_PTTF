from constants import ALL_AIRPORTS

# from pytorch_dnn import MyTorchDNN
from tf_dnn import MyTensorflowDNN

if __name__ == "__main__":
    # for theAirport in ALL_AIRPORTS:
    #    MyTensorflowDNN.train(theAirport)
    MyTensorflowDNN.train("ALL")
    # MyTensorflowDNN.evaluate_global()
