# -*- coding: utf-8 -*-

import time

t0 = time.clock()

import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import datetime as dt
from copy import copy
import warnings

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
# ================================================================== #
# datetime management

d = dt.date.today()
# ---------- Days ----------
l10 = d - 10 * BDay()
l21 = d - 21 * BDay()
l63 = d - 63 * BDay()
l252 = d - 252 * BDay()
# ---------- Years ----------
l252_x2 = d - 252 * 2 * BDay()
l252_x3 = d - 252 * 3 * BDay()
l252_x5 = d - 252 * 5 * BDay()
l252_x7 = d - 252 * 7 * BDay()
l252_x10 = d - 252 * 10 * BDay()
l252_x20 = d - 252 * 20 * BDay()
l252_x25 = d - 252 * 25 * BDay()
# ================================================================== #
# filepath management

project_dir = r'D:\\Files\\Box Sync\\classes\\Spring2018\\CS673\\final\\util\\'
price_path = project_dir + r'Stock_Price_Data\\'
# ================================================================== #
apikey = 'insert_your_api_key'


def construct_barChart_url(sym, start_date, freq, api_key=apikey):
    '''Function to construct barchart api url'''

    url = 'http://marketdata.websol.barchart.com/getHistory.csv?' + \
          'key={}&symbol={}&type={}&startDate={}'.format(api_key, sym, freq, start_date)
    return url


# ================================================================== #

# header=3 to skip unnecesary file metadata included by State Street
spy_components = pd.read_excel(project_dir + \
                               'SPDR_Holdings/holdings-spy.xls', header=3)
syms = spy_components.Identifier.dropna()
print(syms)


def get_minute_data():
    '''Function to Retrieve <= 3 months of minute util for SP500 components'''

    # This is the required format for datetimes to access the API
    # You could make a function to translate datetime to this format
    start = '20180427000000'
    # end = d
    freq = 'minutes'
    prices = {}
    symbol_count = len(syms)
    N = copy(symbol_count)
    try:
        for i, sym in enumerate(syms, start=1):
            api_url = construct_barChart_url(sym, start, freq, api_key=apikey)
            try:
                csvfile = pd.read_csv(api_url, parse_dates=['timestamp'])
                csvfile.set_index('timestamp', inplace=True)
                prices[sym] = csvfile
            except:
                continue
            N -= 1
            pct_total_left = (N / symbol_count)
            print('{}..[done] | {} of {} symbols collected | percent remaining: {:>.2%}'.format( \
                sym, i, symbol_count, pct_total_left))
    except Exception as e:
        print(e)
    finally:
        pass
    px = pd.Panel.from_dict(prices)
    # convert timestamps to EST
    px.major_axis = px.major_axis.tz_localize('utc').tz_convert('US/Eastern')
    return px


pxx = get_minute_data()
print(pxx)
print(pxx['AAL'].tail())
print(pxx['ZTS'].tail())

try:
    store = pd.HDFStore(price_path + 'Minute_Symbol_Data.h5')
    store['minute_prices'] = pxx
    store.close()
except Exception as e:
    print(e)
finally:
    pass

# ================================================================== #
# timer looking clean #
secs = np.round((time.clock() - t0), 4)
time_secs = "{timeSecs} seconds to run".format(timeSecs=secs)
mins = np.round(((time.clock()) - t0) / 60, 4)
time_mins = "| {timeMins} minutes to run".format(timeMins=mins)
hours = np.round((time.clock() - t0) / 60 / 60, 4)
time_hrs = "| {timeHrs} hours to run".format(timeHrs=hours)
print(time_secs, time_mins, time_hrs)