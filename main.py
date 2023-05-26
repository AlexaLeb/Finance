import datetime as dt
from datetime import datetime as dt2
from finfun import *
from graph import EF_graph

stocklist = ['V', 'DVA', 'HPQ', 'MCK', 'AON', 'MCO', 'KR', 'PFE', 'PLTR']  # Лист акций

endDate = dt2(2020, 12, 31)
startDate = endDate - dt.timedelta(days=365 * 3)
weight = np.ones(12)
meant, cov, returns = get_data(stocks=stocklist, start=startDate, end=endDate)


def market(start, end):
    df = pd.read_csv('NYSE Composite Historical Data.csv')  # Прочитали файл с данными по облигациям
    df['Date'] = df['Date'].apply(lambda x: x.replace('.', '/'))  # заменили точки на палочки
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df[(df.Date > start) & (df.Date < end)]
    df = df['Change %'].apply(lambda x: x.replace(',', '.'))  # Заменили запятые на точки
    markets = df.apply(lambda x: float(x[0:-1]) if "%" in x else float(x))  # убрали знак процента
    market_m = markets.mean() * 252  # Годовая доходность рынка
    market_var = markets.var()
    print(market_m)
    return market_m, market_var


def bill_rate_mean(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    return df['52 WEEKS BANK DISCOUNT'].mean()


bill2020 = bill_rate_mean('daily-treasury-rates.csv')
bill2021 = bill_rate_mean('daily-treasury-rates-2.csv')
bill2022 = bill_rate_mean('daily-treasury-rates-3.csv')
list_of_bills = [bill2020, bill2021, bill2022]
print(list_of_bills)
market(startDate, endDate)


calculatedResults(returns, meant, cov, 1, (0.05, 0.25))
# EF_graph(returns, meant, cov, 1, (0.05, 0.20))

