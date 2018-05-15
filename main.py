import nn.mlp as mlp
import nn.lstm as lstm
import nn.cnn as cnn
import nn.cnn6 as cnn6


def main():
   file_names = ["WIKI-GE.csv", "WIKI-HPQ.csv", "WIKI-KO.csv", "WIKI-ORCL.csv", "WIKI-S.csv", "WIKI-T.csv", "WIKI-TWX.csv", "WIKI-UAA.csv", "WIKI-VZ.csv"]

   for file_name in file_names:
       mlp.MLP(file_name).train()
       lstm.LSTM(file_name).train()
       cnn.CNN(file_name).train()
       cnn6.CNN6(file_name).train()


if __name__ == '__main__':
    main()
