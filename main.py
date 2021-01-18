from alphaVinage import Apha
import pandas as pd
import json

apikey = 'AWRI5P1983OG0FXN'
delay = 30
bank = 5000
start_date = '2018-03-07'
result_data = {}

listing = ['AAPL', 'MSFT', 'AMZN', 'FB', 'JPM', 'BRK.B', 'JNJ', 'GOOG', 'GOOGL', 'XOM',
           'BAC', 'WFC', 'INTC', 'T', 'V', 'CSCO', 'CVX', 'UNH', 'PFE', 'HD',
           'PG', 'VZ', 'C', 'ABBV', 'BA', 'KO', 'CMCSA', 'MA', 'PM', 'DWDP',
           'PEP', 'ORCL', 'DIS', 'MRK', 'NVDA', 'MMM', 'AMGN', 'IBM', 'NFLX', 'WMT',
           'MO', 'MCD', 'GE', 'HON', 'MDT', 'ABT', 'TXN', 'BMY', 'ADBE', 'UNP']
           # 'PRO', 'MCBC', 'ENOC', 'LINC', 'SYF', 'FLDM', 'ECHO', 'DISH', 'NRCIA', 'NAT',
           # 'THOR', 'FTNT', 'FMD', 'GIS', 'ASC', 'WRE', 'QRHC', 'ROST', 'HIBB', 'HIL']

for i in listing:
    try:
        ibm = Apha(ticker='{}'.format(i), apikey=apikey)
    except Exception:
        continue

    # получаем максимальную цену предсказания
    max_value = ibm.predict_future(days=60)

    max_price_pred_day = max_value['Date']
    try:
        for key, value in max_price_pred_day.items():
            # day of max price pred
            day_of_week = int(value.dayofweek)
            if day_of_week == 6:
                # value = value - pd.offsets.Day(1)
                # day_pred = str(value.date())
                raise Exception
            elif day_of_week == 7:
                # value = value + pd.offsets.Day(1)
                # day_pred = str(value.date())
                raise Exception
            else:
                day_pred = str(value.date())
            break
    except Exception:
        continue

    max_price_pred_value = max_value['estimate']
    for key, value in max_price_pred_value.items():
        # max price pred
        price_pred = value
        break

    # получаем все данные
    data = ibm.stock_real

    # получаем данные за день покупки
    buy_day = data[data['Date'].astype('datetime64[ns]') == start_date]

    # получаем данные за день реальной продажи
    sell_day_real = data[data['Date'].astype('datetime64[ns]') == day_pred]

    if max(float(buy_day['Adj. Close']), price_pred) != price_pred:
        continue

    g = float(buy_day['Adj. Close'])
    # покупаем акции и получаем их количество
    count_ticker = bank // float(buy_day['Adj. Close'])
    count_ticker_ost = bank % float(buy_day['Adj. Close'])

    # продаем акции по цене предсказания
    sell_pred = count_ticker * price_pred

    # продаем акции по цене реальной
    sell_real = count_ticker * float(sell_day_real['Adj. Close'])

    # result = 'Ticker: {}'\
    #       'Date buy:   {}\n'\
    #       'Price in buy day: {} \n\n'\
    #       'Date sell:  {}\n'\
    #       'Real Price in sell day: {}\n'\
    #       'Pred Price in sell day: {}\n\n'\
    #       'Result for ticker =>   \n'\
    #       'Real profit: {}        \n'\
    #       'Pred profit: {}\n'. format(
    #                                 ibm.symbol,
    #                                 start_date,
    #                                 buy_day['Adj. Close'],
    #                                 max_value['Date'],
    #                                 sell_day_real['Adj. Close'],
    #                                 price_pred,
    #                                 sell_real,
    #                                 sell_pred)

    result_data['Ticker: {}'.format(ibm.symbol)] = {
            'Date buy': start_date,
            'Price in buy day': float(buy_day['Adj. Close']),
            'Tickers count': count_ticker,
            'Ostatok ot buy': count_ticker_ost,
            'Date sell': str(max_value['Date']),
            'Real Price in sell day': float(sell_day_real['Adj. Close']),
            'Pred Price in sell day': price_pred,
            'Real profit': sell_real,
            'Pred profit': sell_pred
        }

print(result_data)

count_filter_ticker = len(result_data)
ticker_list = [k for k, v in result_data.items()]

summa_vloj = bank * count_filter_ticker

prog_bank = 0
real_bank = 0
ost_by_buy = 0

for k, v in result_data.items():
    prog_bank = prog_bank + float(result_data[k]['Pred profit'])
    real_bank = real_bank + float(result_data[k]['Real profit'])
    ost_by_buy = ost_by_buy + float(result_data[k]['Ostatok ot buy'])

win = max(prog_bank, real_bank)
pogreshnost = max(prog_bank, real_bank) - min(prog_bank, real_bank)

s = 'Стартовый бюджет: {}\n ' \
    'Прогнозируемый капитал: {}\n ' \
    'Реальный капитал: {}\n ' \
    'Остаток после покупок акций: {}\n ' \
    'Погрешность: {}'.format(summa_vloj, prog_bank, real_bank, ost_by_buy, pogreshnost)

print(s)



with open('result.txt', 'w', encoding='utf-8') as file:
    file.write(s)

with open('result.json', 'w', encoding="utf-8") as file:
    json.dump(result_data, file)





