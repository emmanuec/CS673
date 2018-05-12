import csv
import datetime
import re
import codecs
import requests
import time
import io
import pandas as pd

def get_google_finance_intraday(ticker, period=60, days=1, exchange='NASD'):
    """
    Retrieve intraday stock util from Google Finance.

    Parameters
    ----------------
    ticker : str
        Company ticker symbol.
    period : int
        Interval between stock values in seconds.
        i = 60 corresponds to one minute tick util
        i = 86400 corresponds to daily util
    days : int
        Number of days of util to retrieve.
    exchange : str
        Exchange from which the quotes should be fetched

    Returns
    ---------------
    df : pandas.DataFrame
        DataFrame containing the opening price, high price, low price,
        closing price, and volume. The index contains the times associated with
        the retrieved price values.
    """

    # build url
    url = 'https://finance.google.com/finance/getprices' + \
          '?p={days}d&f=d,o,h,l,c,v&q={ticker}&i={period}&x={exchange}'.format(ticker=ticker,
                                                                               period=period,
                                                                               days=days,
                                                                               exchange=exchange)

    page = requests.get(url)
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    rows = []
    times = []
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start + datetime.timedelta(seconds=period * int(row[0])))
            rows.append(map(float, row[1:]))
    if len(rows):
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'), columns=columns)
    else:
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))


def google_stocks(symbol, startdate=(4, 26, 2018), enddate=(4, 27, 2018)):
    startdate = str(startdate[0]) + '+' + str(startdate[1]) + '+' + str(startdate[2])

    if not enddate:
        enddate = time.strftime("%m+%d+%Y")
    else:
        enddate = str(enddate[0]) + '+' + str(enddate[1]) + '+' + str(enddate[2])

    stock_url = "https://finance.google.com/historical?q=" + symbol + \
                "&startdate=" + startdate + "&enddate=" + enddate + "&output=csv"

    raw_response = requests.get(stock_url).content

    stock_data = pd.read_csv(io.StringIO(raw_response.decode('utf-8')))

    return stock_data