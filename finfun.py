import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import scipy.optimize as sco
import scipy.stats as scs
import datetime as dt
from datetime import datetime as dt2
import random
yf.pdr_override()


class Color:
    """
    Класс добавляет названия цветов для оформления вывода текста
    """
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def market(start, end):
    """
    Функция берет нужный промежуток из файла с индексом NYSE Composite.
    :param start: Дата начала периода.
    :param end: Дата конца периода.
    :return: среднюю доходность рынка, его дисперсию и таблицу с изменениями цен.
    """
    df = pd.read_csv('NYSE Composite Historical Data.csv')  # Прочитали файл с данными по рынку
    df['Date'] = df['Date'].apply(lambda date: date.replace('.', '/'))  # заменили точки на палочки
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df[(df.Date > start) & (df.Date < end)]
    df = df['Price'].apply(lambda price: price.replace(',', ''))  # Заменили запятые на точки
    markets = df.apply(lambda price: float(price[0:-1]) if "%" in price else float(price))  # убрали знак процента
    lis = []
    for i in range(len(markets)):  # Находим изменение индекса
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
    """
    Находит безрисковую ставку доходности.
    :param file: Файл с данными по облигациям.
    :return:
    """
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


def get_data(stocks, start, end):
    """
    Импортирует данные с сайта Yahoo и берет данные по безрисковым активам с сайта.
    :param stocks: список акций.
    :param start: время начальное.
    :param end: время конечное.
    :return: среднюю ежедневную доходность и ковариационную матрицу
    """
    # Импорт данных
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockdata = stockdata['Close']  # Получили цены закрытия
    files = ['XLE Historical Data.csv', 'Proshlye_dannye_po_SPY.csv']  # Список с нашими файлами
    result = pd.DataFrame()

    # ETF
    df = pd.read_csv(files[1])  # Прочитали файл с данными по ETF
    df['Дата'] = df['Дата'].apply(lambda date: date.replace('.', '/'))  # заменили точки на палочки
    df['Date'] = pd.to_datetime(df['Дата'], dayfirst=True)
    df = df.set_index(df['Date'])
    df = df['Цена'].apply(lambda price: price.replace(',', '.'))  # Заменили запятые на точки
    df = df.apply(lambda price: float(price[0:-1]) if "%" in price else float(price))  # убрали знак процента
    stockdata['ETF'] = df

    # xle
    df = pd.read_csv(files[0])  # Прочитали файл с данными по XLE
    df['Date'] = df['Date'].apply(lambda date: date.replace('.', '/'))  # заменили точки на палочки
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df['Price']
    stockdata['XLE'] = df

    stockdata = stockdata.dropna()
    for i in range(len(stockdata.columns)):
        li = []
        # print(stockdata[stockdata.columns[i]])
        for it in range(len(stockdata[stockdata.columns[i]])):
            if it == 0:
                pass
            else:
                x = np.log(stockdata[stockdata.columns[i]][it] / stockdata[stockdata.columns[i]][it-1])
                li.append(x)
        result[stockdata.columns[i]] = li

    mean = result.mean()  # Получили среднее доходности
    covMatrix = result.cov()  # Получили матрицу ковариации
    return mean, covMatrix, result


def portfolioPerformance(weights, meanReturns, covMatrix, dataframe):
    """
    Функция находит доходность портфеля и его среднеквадратичное отклонение.
    :param weights: веса активов.
    :param meanReturns: средняя доходность в сутки.
    :param covMatrix: матрица ковариации.
    :param dataframe: таблица доходностей.
    :return: доходность за год, стандартное отклонение.
    """
    returns = 0
    for i in range(len(meanReturns)):
        returns += (((1 + meanReturns[i]) ** len(dataframe)) - 1) * weights[i]
    std = np.sqrt(
            np.dot(weights.T, np.dot(covMatrix, weights))
           )
    return returns, std


def negativePP(weights, meanReturns, covMatrix, dataframe):
    """
    Функция находит отрицательную доходность портфеля, минус нужен для оптимизации.
    :param weights: веса активов.
    :param meanReturns: средняя доходность в сутки.
    :param covMatrix: ковариационная матрица.
    :param dataframe: таблица доходностей.
    :return: доходность за год.
    """
    returns = portfolioPerformance(weights, meanReturns, covMatrix, dataframe)[0]
    return - returns


def maxPPerformance(meanReturns, covMatrix, dataframe, constraintSet=(0, 1)):
    """
    Минимизирует негативную доходность портфеля.
    :param meanReturns: средняя доходность активов.
    :param constraintSet: параметр необходимый для решения уравнения.
    :param covMatrix ковариационная матрица.
    :param dataframe: таблица доходностей.
    :return: большой словарь по оптимизации заданных параметров.
    """
    numAssets = len(meanReturns)  # количество активов
    args = (meanReturns, covMatrix, dataframe)  # Известные аргументы функции (вес активов неизвестен, его будут искать)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Сумма весов должна быть равна единице
    bound = constraintSet  # Границы для каждого актива
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))  # Модифицирование границ
    # noinspection PyTypeChecker
    result = sco.minimize(negativePP, numAssets * [1. / numAssets], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)  # Оптимизатор
    return result


def gbound(bounds, assets, asset):
    """
    Функция задает персонализированные границы для последнего и предпоследнего актива.
    :param bounds: общие границы.
    :param assets: длина списка активов.
    :param asset: номер актива.
    :return: модифицированные границы.
    """
    if asset == assets - 1 or asset == assets - 2:
        return 0.1, bounds[1]
    else:
        return bounds


def negativeSR(weights, meanReturns, covMatrix, dataframe, riskFreeRate=1):
    """
    В библиотеки scipy нет функции максимизации, зато есть функция минимизации. Так как мы хотим найти максимальный
    коэффициент Шарпа, то сначала мы сделаем его отрицательным, чтобы минимизировать отрицательный. Так мы найдем и
    положительный.
    :param weights: веса активов.
    :param meanReturns: средняя доходность активов.
    :param covMatrix: матрица ковариации.
    :param riskFreeRate: безрисковая ставка доходности.
    :param dataframe: таблица доходностей.
    :return: негативный коэффициент шарпа.
    """
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix, dataframe)
    return - (pReturns - riskFreeRate)/pStd


def maxSRatio(meanReturns, covMatrix, dataframe, riskFreeRate=1, constraintSet=(0.04, 0.5)):
    """
    Минимизирует негативный К шарпа балансируя веса портфеля.
    :param meanReturns: средняя доходность активов.
    :param covMatrix: матрица ковариации.
    :param riskFreeRate: безрисковая ставка доходности.
    :param constraintSet: параметр необходимый для решения уравнения.
    :param dataframe: таблица доходностей.
    :return: большой словарь по оптимизации заданных параметров.
    """
    numAssets = len(meanReturns)  # количество активов
    args = (meanReturns, covMatrix, dataframe, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))
    # noinspection PyTypeChecker
    result = sco.minimize(negativeSR, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def portfolioVariance(weights, meanReturns, covMatrix, dataframe):
    """
    Выдает отклонение портфеля (вариацию) (волатильность)
    :param weights: веса
    :param meanReturns: среднее
    :param covMatrix: ковариация
    :param dataframe: таблица доходностей.
    :return: отклонения
    """
    return portfolioPerformance(weights, meanReturns, covMatrix, dataframe)[1]


def minimizeVariance(meanReturns, covMatrix, dataframe, constraintSet=(0.04, 0.5)):
    """
    Выдает веса для минимальной вариации
    :param meanReturns: средняя доходность
    :param covMatrix: матрица ковариации
    :param constraintSet: параметр для решения уравнения
    :param dataframe: таблица доходностей.
    :return:
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, dataframe)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))
    # noinspection PyTypeChecker
    result = sco.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def negativeMSR(weights, dataframe, meanReturns, covMatrix, riskFreeRate=1):
    """
    Считает модифицированный коэффициент Шарпа и делает его отрицательным
    :param dataframe: таблица доходностей.
    :param weights: веса активов.
    :param meanReturns: средние доходности актива.
    :param covMatrix: матрица ковариации.
    :param riskFreeRate: безрисковая ставка.
    :return:
    """
    data = dataframe.copy()
    data['bew'] = np.zeros(data.shape[0])
    for i in range(len(dataframe.columns) - 1):
        data['bew'] += data[dataframe.columns[i]] * weights[i]  # Складываем произведение дневной
        # доходности актива на его вес
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix, dataframe)
    data = data['bew'].dropna()
    data = data.apply(lambda x: x + 1)
    geom = scs.gmean(data) - 1  # Находим геометрическое среднее
    vol = data.std()  # считаем волатильность
    z = scs.norm.ppf(0.99)    # Z оценка для 99% интервала
    skew = scs.skew(data)  # skewness Асимметрия (смещение распределения)
    kurt = scs.kurtosis(data)  # Kurtosis Эксцесс (выпуклость графика)
    zmvar = z + (1 / 6 * ((z ** 2) - 1) * skew) + (1 / 24 * ((z ** 3) - 3 * z)
                                                   * kurt) - (1 / 36 * (2 * (z ** 3) - 5 * z) * skew ** 2)
    # Считаем z оценку для модифицированной стоимостной оценки риска
    mvar = geom - zmvar * vol  # Находим модифицированную стоимостную оценку риска
    msr = (returns - riskFreeRate) / abs(mvar)  # Находим модифицированный коэффициент шарпа
    return - msr


def maxMSR(dataframe, meanReturns, covMatrix, riskFreeRate=1, constraintSet=(0.04, 0.5)):
    """
    Выдает веса для максимального модифицироованного коэффициента шарпа
    :param meanReturns: средняя доходность
    :param covMatrix: матрица ковариации
    :param constraintSet: параметр для решения уравнения.
    :param dataframe: таблица доходностей.
    :param riskFreeRate: безрисковая ставка.
    :return:
    """
    numAssets = len(meanReturns)  # Находим количество членов уравнения
    args = (dataframe, meanReturns, covMatrix, riskFreeRate)  # Задаем аргументы для функции
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet  # Задаем ограничения для основного массива активов
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))
    # noinspection PyTypeChecker
    result = sco.minimize(negativeMSR, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def negativeCSR(weights, dataframe, mean, cov, riskFreeRate=1):
    """
    Рассчитывает отрицательный коэффициент шарпа при условной стоимости под риском.
    :param weights: веса
    :param dataframe: данные с доходностями активов по дням.
    :param mean: средняя доходность актива за период.
    :param cov: ковариационная матрица активов.
    :param riskFreeRate: безрисковая ставка
    :return:
    """
    data = dataframe.copy()
    data['Gew'] = np.zeros(data.shape[0])
    for i in range(len(dataframe.columns) - 1):
        data['Gew'] += data[dataframe.columns[i]] * weights[i]  # Складываем произведение дневной
        # доходности актива на его вес
    returns, std = portfolioPerformance(weights, mean, cov, dataframe)
    data = data['Gew'].dropna()
    cvar = np.percentile(data, 1)  # Условная стоимость под риском или ожидаемый дефицит
    csr = (returns - riskFreeRate) / abs(cvar)
    return - csr


def maxCSR(dataframe, meanReturns, covMatrix, riskFreeRate=1, constraintSet=(0.04, 0.5)):
    """
    Выдает веса для коэффициента Шарпа при условной стоимости под риском.
    :param meanReturns: средняя доходность.
    :param covMatrix: матрица ковариации
    :param constraintSet: ограничения активов для решения уравнения.
    :param dataframe: данные с доходностями активов по дням.
    :param riskFreeRate: безрисковая ставка
    :return:
    """
    numAssets = len(meanReturns)  # Находим количество членов уравнения
    args = (dataframe, meanReturns, covMatrix, riskFreeRate)  # Задаем аргументы для функции
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet  # Задаем ограничения для основного массива активов
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))  #
    # noinspection PyTypeChecker
    result = sco.minimize(negativeCSR, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def calculatedResults(dataframe, meanReturns, covMatrix, dfm, riskFreeRate=1,  constraintSet=(0.04, 0.20), rm=10, mvar=5):
    """
    Общая функция, которая вызывает остальные функции расчета, обрабатывает их результат.
    :param meanReturns: средняя
    :param covMatrix: матрица ковариации
    :param riskFreeRate: безрисковая ставка доходности.
    :param constraintSet: ограничения весов активов.
    :param dataframe: данные с доходностями активов по дням.
    :param dfm: данные по рынку.
    :param rm: рыночная доходность.
    :param mvar: рыночная дисперсия.
    :return:
    1. Доходность максимального к Шарпа.
    2. Волатильность максимального к Шарпа.
    3. Веса портфеля максимального к Шарпа.
    4. Доходность минимальной волатильности.
    5. Волатильность минимальной волатильности.
    6. Веса минимальной волатильности.
    7. Доходность промежуточных параметров для построения графика.
    8. Волатильность промежуточных параметров для построения графика.
    """

    # Максимальный коэффициент Шарпа
    maxSR_Portfolio = maxSRatio(meanReturns, covMatrix, dataframe, riskFreeRate, constraintSet)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix, dataframe)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['вес в %'])
    maxSR_allocation['вес в %'] = [round(i * 100, 3) for i in maxSR_allocation['вес в %']]
    printer(maxSR_Portfolio['x'], riskFreeRate, rm, mvar, 'Максимальный к Шарпа', dataframe, meanReturns, covMatrix, dfm, maxSR_allocation)

    # Минимальная волатильность
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix, dataframe, constraintSet)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix, dataframe)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['вес в %'])
    minVol_allocation['вес в %'] = [round(i * 100, 3) for i in minVol_allocation['вес в %']]
    printer(allocation=minVol_Portfolio['x'], rf=riskFreeRate, rm=rm, mvar=mvar,
            string='Минимальная волатильность', dataframe=dataframe, mean=meanReturns, cov=covMatrix, dfm=dfm, view=minVol_allocation)

    # Максимальная доходность
    maxPerf_Portfolio = maxPPerformance(meanReturns, covMatrix, dataframe, constraintSet)
    maxPerf_returns, maxPerf_std = portfolioPerformance(maxPerf_Portfolio['x'], meanReturns, covMatrix, dataframe)
    maxPerf_allocation = pd.DataFrame(maxPerf_Portfolio['x'], index=meanReturns.index, columns=['вес в %'])
    maxPerf_allocation['вес в %'] = [round(i * 100, 3) for i in maxPerf_allocation['вес в %']]
    printer(maxPerf_Portfolio['x'], riskFreeRate, rm, mvar, 'Максимальная доходность', dataframe, meanReturns, covMatrix, dfm, maxPerf_allocation)

    # Максимальный кондиционный коэффициент Шарпа
    maxCSRatio = maxCSR(dataframe, meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxCSR_return, maxCSR_std = portfolioPerformance(maxCSRatio['x'], meanReturns, covMatrix, dataframe)
    maxCSR_allocation = pd.DataFrame(maxCSRatio['x'], index=meanReturns.index, columns=['вес в %'])
    maxCSR_allocation['вес в %'] = [round(i * 100, 3) for i in maxCSR_allocation['вес в %']]
    printer(maxCSRatio['x'], riskFreeRate, rm, mvar, 'Максимальный коэффициент шарпа при условной стоимости под риском', dataframe, meanReturns, covMatrix, dfm, maxCSR_allocation)

    # Максимальный модифицированный коэффициент шарпа
    maxMSRatio = maxMSR(dataframe, meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxMSR_return, maxMSR_std = portfolioPerformance(maxMSRatio['x'], meanReturns, covMatrix, dataframe)
    maxMSR_allocation = pd.DataFrame(maxMSRatio['x'], index=meanReturns.index, columns=['вес в %'])
    maxMSR_allocation['вес в %'] = [round(i * 100, 3) for i in maxMSR_allocation['вес в %']]
    printer(maxMSRatio['x'], riskFreeRate, rm, mvar, 'Максимальный модифицированный коэффициент шарпа', dataframe, meanReturns, covMatrix, dfm, maxMSR_allocation)

    # Генератор случайных портфелей.
    target_returns = []
    efficientList = []
    c = 0
    while c < 15000:  # Счетчик на 15000 портфелей
        random_list = []
        for i in range(len(meanReturns)):
            random_list.append(random.randrange(100))  # генерирует случайное число столько раз сколько есть активов
        random_weight = []
        for i in random_list:
            random_weight.append(i / sum(random_list))  # делит каждое число на сумму всех сгенерированных портфелей
        target_return = portfolioReturn(np.array(random_weight), meanReturns, covMatrix, dataframe)  # находит доходность
        target_returns.append(target_return)
        efficient = portfolioVariance(np.array(random_weight),  meanReturns, covMatrix, dataframe)  # находит вариацию
        efficientList.append(efficient)
        c += 1  # увеличение счетчика на 1

    # Возвращения функции
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, maxPerf_returns, \
        maxPerf_std, maxPerf_allocation, maxMSR_return, maxMSR_std, maxMSR_allocation, maxCSR_return, maxCSR_std, \
        maxCSR_allocation, efficientList, target_returns


def portfolioReturn(weights, meanReturns, covMatrix, dataframe):
    """Считает доходность портфеля"""
    return portfolioPerformance(weights, meanReturns, covMatrix, dataframe)[0]


def Jensens_Alpha(rp, rf, rm, b):
    """
    Считает Альфу Йенса
    :param rp: portfolio return
    :param rf: risk-free rate
    :param rm: expected market return
    :param b: portfolio beta
    :return:
    """
    coeffJensensAlpha = rp - (rf + b*(rm - rf))
    return coeffJensensAlpha


def Treynor(rp, rf, b):
    """
    Считает коэффициент Тейнора
    :param rp: - portfolio return
    :param rf: - risk-free rate
    :param b: - portfolio beta
    :return:
    """
    coeffTreynor = (rp - rf) / b
    return coeffTreynor


def beta(dataframe, weight, mvar, dfm):
    """
    Бета коэффициент
    """
    if len(dataframe) > len(dfm):
        dataframe = dataframe[(len(dataframe)-len(dfm)):]
    l = []
    betta = []
    for i in range(len(dataframe.columns)):
        ar = dataframe[dataframe.columns[i]].to_numpy()
        cova = np.cov(ar, dfm)[0][1]
        cova = cova / mvar
        l.append(cova)
    for i in range(len(l)):
        betta.append(l[i] * weight[i])
    betta = sum(betta)
    return betta


def M2(rp, rf, pstd, mstd, rm):
    """
    Считает коэффициент М2
    :param rp: доходность портфеля
    :param rf: доходность без рискового актива
    :param pstd: стандартное отклонение портфеля
    :param mstd: стандартное отклонение индекса
    :param rm: доходность индекса
    :return:
    """
    return (((rp - rf) * np.sqrt(mstd)) / pstd) - (rm - rf)


def printer(allocation, rf, rm, mvar, string, dataframe, mean, cov, dfm, view, year=None):
    """Функция занимается выводом результатов на экран"""
    print(Color.BOLD + Color.GREEN + '\nПоказатели' + Color.END)
    if year:
        year = str(year)
        print(Color.BOLD + Color.GREEN + 'за ' + year + Color.END)
    print(Color.GREEN + string + ', веса активов' + Color.END)
    if year is None:
        print(view)
    nrp = portfolioReturn(allocation, meanReturns=mean, covMatrix=cov, dataframe=dataframe)
    nvr = portfolioVariance(allocation, mean, cov, dataframe)
    b = beta(dataframe, allocation, mvar, dfm)
    print(Color.DARKCYAN + '\nдоходность -' + Color.END, round(nrp * 100, 2), '%')
    print(Color.DARKCYAN + 'волатильность - ' + Color.END, round(nvr * 100, 2), '%')
    print(Color.DARKCYAN + 'коэффициент Шарпа - ' + Color.END, round(((nrp - rf)/nvr), 2))
    print(Color.DARKCYAN + 'модифицированный коэффициент Шарпа - ' + Color.END, - round(negativeMSR(allocation, dataframe, mean, cov, rf), 2))
    print(Color.DARKCYAN + 'кондиционный коэффициент Шарпа - ' + Color.END, - round(negativeCSR(allocation, dataframe, mean, cov, rf), 2))
    print(Color.DARKCYAN + 'коэффициент Бета - ' + Color.END, round(b, 2))
    print(Color.DARKCYAN + 'коэффициент Тейнора - ' + Color.END, round(Treynor(nrp, rf, b), 4))
    print(Color.DARKCYAN + 'Альфа Йенса - ' + Color.END, round(Jensens_Alpha(nrp, rf, rm, b) * 100, 2), '%')
    print(Color.DARKCYAN + 'M2 - ' + Color.END, round(M2(nrp, rf, nvr, mvar, rm) * 100, 2), '%')
    print(Color.RED + 'в разработке - ' + Color.END)


def conclude(allocation, date, safe_return, stocklist, str):
    """
    Функция выводящая данные существующего портфеля по другим годам
    :param allocation: таблица с весами активов
    :param date: год
    :param safe_return: безрисковая ставка
    :param stocklist: лист акций
    :param str:
    :return:
    """
    endDate = dt2(date, 12, 31)
    startDate = endDate - dt.timedelta(days=365)
    mean, cov, data = get_data(stocklist, startDate, endDate)
    l = []
    for i in allocation['вес в %']:
        l.append(i / 100)
    x = np.array(l)
    market_p, market_var, marke = market(startDate, endDate)
    printer(allocation=x, rf=safe_return, rm=market_p, mvar=market_var, string=str, dataframe=data, mean=mean,
            cov=cov, dfm=marke, view=allocation, year=date)
