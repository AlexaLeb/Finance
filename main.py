import datetime as dt
from datetime import datetime as dt2
from finfun import *
from graph import EF_graph

stocklist = ['V', 'DVA', 'HPQ', 'MCK', 'AON', 'MCO', 'KR', 'PFE', 'CVX', 'KO']  # Лист акций
# stocklist = ['KO', 'CVX', 'MCD', 'V', 'HPQ', 'MCO', 'DVA', 'KR', 'MCK', 'AON']
endDate = dt2(2020, 12, 31)
startDate = endDate - dt.timedelta(days=365)
weight = np.ones(12)
meant, cov, returns = get_data(stocks=stocklist, start=startDate, end=endDate)
print(meant)
print(returns)


def market(start, end):
    df = pd.read_csv('NYSE Composite Historical Data.csv')  # Прочитали файл с данными по облигациям
    df['Date'] = df['Date'].apply(lambda x: x.replace('.', '/'))  # заменили точки на палочки
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df[(df.Date > start) & (df.Date < end)]
    df = df['Price'].apply(lambda x: x.replace(',', ''))  # Заменили запятые на точки
    markets = df.apply(lambda x: float(x[0:-1]) if "%" in x else float(x))  # убрали знак процента
    lis = []
    for i in range(len(markets)):
        if i == 0:
            pass
        else:
            x = np.log(markets[i] / markets[i - 1])
            lis.append(x)
    x = np.array(lis)
    market_m = x.mean()  # Годовая доходность рынка
    market_m = ((1 + market_m) ** 252) - 1
    market_var = x.var()
    return market_m, market_var, x


def bill_rate_mean(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df['52 WEEKS BANK DISCOUNT']
    rates = []
    for i in range(len(df)):
        if i == 0:
            pass
        else:
            x = np.log(df[i] / df[i-1])
            rates.append(x)
    x = np.array(rates)
    return x.mean()


bill2020 = bill_rate_mean('daily-treasury-rates.csv')
bill2021 = bill_rate_mean('daily-treasury-rates-2.csv')
bill2022 = bill_rate_mean('daily-treasury-rates-3.csv')
list_of_bills = [bill2020, bill2021, bill2022]
market_m, market_var, df = market(startDate, endDate)
print(market_m, market_var)
print()
print('годовая доходность')
for i in meant:
    print(((1 + i) ** 252 - 1) * 1)
calculatedResults(dataframe=returns, meanReturns=meant, covMatrix=cov, riskFreeRate=(bill2020), dfm=df,
                  constraintSet=(0.05, 0.25), rm=market_m, mvar=market_var)
# EF_graph(returns, meant, cov, 1, (0.05, 0.20))


