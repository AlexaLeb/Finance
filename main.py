import datetime as dt
from datetime import datetime as dt2
from finfun import *
from graph import EF_graph

stocklist = ['KO', 'CVX', 'MCD', 'V', 'HPQ', 'MCO', 'DVA', 'KR', 'MCK', 'AON']  # Лист акций

endDate = dt2(2020, 12, 31)
startDate = endDate - dt.timedelta(days=365)

weight = np.ones(12)
meant, cov = get_data(stocks=stocklist, start=startDate, end=endDate)


#  Индекс
# df = pd.read_csv('Proshlye_dannye_-_S_amp_amp_P_500.csv')  # Прочитали файл с данными по облигациям
# df['Дата'] = df['Дата'].apply(lambda x: x.replace('.', '/'))  # заменили точки на палочки
# df['Date'] = pd.to_datetime(df['Дата'], dayfirst=True)
# df = df.set_index(df['Date'])
# df = df['Изм. %'].apply(lambda x: x.replace(',', '.'))  # Заменили запятые на точки
# market = df.apply(lambda x: float(x[0:-1]) if "%" in x else float(x))  # убрали знак процента


# print(fun(meant))
# print(meant)
# a, b, c, d, e, f, g, h = calculatedResults(meant, cov, 4)
# print(c)
# print(f)
calculatedResults(meant, cov, 1, (0.05, 0.20))
# EF_graph(meant, cov, 1, (0.07, 0.20))
# print(maxMSR(meant, cov, 4))
