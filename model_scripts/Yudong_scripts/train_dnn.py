from constants import ALL_AIRPORTS
from tf_dnn import MyTensorflowDNN
from pytorch_dnn import MyTorchDNN

if __name__ == "__main__":
    for theAirport in ALL_AIRPORTS:
        MyTensorflowDNN.train_dnn(theAirport)
