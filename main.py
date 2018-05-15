#import import_data as id
# def main():
#     # input util
#     ticker = 'AAPL'
#     period = 60
#     days = 365
#     exchange = 'NYSE'
#
#     df = id.get_google_finance_intraday(ticker, period=period, days=days)
#     print(df)
#
#     apple_data = id.google_stocks('AAPL')
#
#     print(apple_data)

import nn.mlp as mlp
import nn.lstm as lstm
import nn.cnn as cnn
import nn.cnn6 as cnn7
import nn.cnnbatch as cnnbatch
import nn.cnnpool as cnnpool
import nn.cnnpoolbatch as cnnpoolbatch


def main():
   file_names = ["WIKI-GE.csv", "WIKI-HPQ.csv", "WIKI-KO.csv", "WIKI-ORCL.csv", "WIKI-S.csv", "WIKI-T.csv", "WIKI-TWX.csv", "WIKI-UAA.csv", "WIKI-VZ.csv"]

   for file_name in file_names:
       mlp.MLP(file_name).train()
       lstm.LSTM(file_name).train()
       cnn.CNN(file_name).train()
       cnn7.CNN7(file_name).train()

       #cnnpool.CNNPool(file_name).train()
       #cnnpoolbatch.CNNPoolBatch(file_name).train()


if __name__ == '__main__':
    main()
