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
import nn.cnn as cnn


def main():
   # mlp.MLP().train()
    cnn.CNN().train()


if __name__ == '__main__':
    main()
