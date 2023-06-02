import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import scipy.optimize as sco
import scipy.stats as scs
# import openpyxl as xl

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


def get_data(stocks, start, end):
    """
    Импортирует данные
    :param stocks: список акций
    :param start: время начальное
    :param end: время конечное
    :return: среднюю ежедневную доходность и ковариационную матрицу
    """
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockdata = stockdata['Close']  # Получили цены закрытия
    files = ['daily-treasury-rates.csv', 'Proshlye_dannye_po_SPY.csv']  # Список с нашими файлами
    result = pd.DataFrame()
    # ETF
    df = pd.read_csv(files[1])  # Прочитали файл с данными по облигациям
    df['Дата'] = df['Дата'].apply(lambda x: x.replace('.', '/'))  # заменили точки на палочки
    df['Date'] = pd.to_datetime(df['Дата'], dayfirst=True)
    df = df.set_index(df['Date'])
    df = df['Цена'].apply(lambda x: x.replace(',', '.'))  # Заменили запятые на точки
    df = df.apply(lambda x: float(x[0:-1]) if "%" in x else float(x))  # убрали знак процента
    stockdata['ETF'] = df

    df = pd.read_csv(files[0])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df['52 WEEKS BANK DISCOUNT'].apply(lambda x: x / 1000)
    stockdata['Bounds'] = df
    # # подвал с девочками
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

    result['Bounds'] = df.mean()
    mean = result.mean()  # Получили среднее доходности
    # mean[-1] = - mean[-1] / 10
    covMatrix = result.cov()  # Получили матрицу ковариации
    return mean, covMatrix, result


def portfolioPerformance(weights, meanReturns, covMatrix, dataframe):
    """
    Функция находит доходность портфеля и его среднеквадратичное отклонение.
    :param weights: веса активов.
    :param meanReturns: средняя доходность в сутки.
    :param covMatrix: матрица ковариации.
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
    Функция находит доходность портфеля.
    :param weights: веса активов.
    :param meanReturns: средняя доходность в сутки.
    :param covMatrix: ковариационная матрица
    :return: доходность за год.
    """
    returns = portfolioPerformance(weights, meanReturns, covMatrix, dataframe)[0]
    return - returns


def maxPPerformance(meanReturns, covMatrix, dataframe, constraintSet=(0, 1)):
    """
    Минимизирует негативную доходность портфеля.
    :param meanReturns: средняя доходность активов.
    :param constraintSet: параметр необходимый для решения уравнения.
    :param covMatrix ковариационная матрица
    :return: большой словарь по оптимизации заданных параметров.
    """
    numAssets = len(meanReturns)  # количество активов
    args = (meanReturns, covMatrix, dataframe)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))
    # noinspection PyTypeChecker
    result = sco.minimize(negativePP, numAssets * [1. / numAssets], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def gbound(bounds, assets, asset):
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
    :return: негативный коэф шарпа.
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
    :return: большой словарь по оптимизации заданных параметров .
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
    Выдает отклонение портфеля (вариацию)
    :param weights: веса
    :param meanReturns: среднее
    :param covMatrix: ковариация
    :return: отклонения
    """
    return portfolioPerformance(weights, meanReturns, covMatrix, dataframe)[1]


def minimizeVariance(meanReturns, covMatrix, dataframe, constraintSet=(0.04, 0.5)):
    """
    Выдает веса для минимальной вариации
    :param meanReturns: средняя доходность
    :param covMatrix: матрица ковариации
    :param constraintSet: параметрр для решения уравнения
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
    Считает модифицированный коэф шарпа и делает его отрицательным
    :param dataframe
    :param weights:
    :param meanReturns:
    :param covMatrix:
    :param riskFreeRate:
    :return:
    """
    data = dataframe.copy()
    data['bew'] = np.zeros(data.shape[0])
    for i in range(len(dataframe.columns) - 1):
        data['bew'] += data[dataframe.columns[i]] * weights[i]  # Складываем произведение дневной
        # доходности актива на его вес
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix, dataframe)
    data = data['bew'].dropna()
    data = data.apply(lambda x: abs(x))  # Находим взвешенную доходность портфелея по дням
    geom = scs.gmean(data)  # Находим геометрическое среднее
    vol = data.std()  # считаем волатильность
    z = scs.norm.ppf(0.99)    # Z оценка для 99% интервала
    skew = scs.skew(data)  # skewness Ассиметрия (смещение распределения)
    kurt = scs.kurtosis(data)  # Kurtosis  Экцесс (выпуклость графика)
    zmvar = z + (1/6 * ((z ** 2) - 1) * skew) + (1/24 * ((z ** 3) - 3 * z) * kurt) - (1/36 * (2 * (z ** 3) - 5 * z) *
                                                                              skew ** 2)
    # Считаем z оценку для модифицированной стоимостной оценки риска
    mvar = geom - zmvar * vol  # Находим модифицированную стоимостную оценку риска
    msr = (returns - riskFreeRate) / abs(mvar)  # Находим модифицированный коэффициент шарпа
    return - msr

def maxMSR(dataframe, meanReturns, covMatrix, riskFreeRate=1, constraintSet=(0.04, 0.5)):
    """
    Выдает веса для максимальногоо модифицироованного коэфициента шарпа
    :param meanReturns: средняя доходность
    :param covMatrix: матрица ковариации
    :param constraintSet: параметрр для решения уравнения
    :return:
    """
    numAssets = len(meanReturns)  # Находим количество членов уравнения
    args = (dataframe, meanReturns, covMatrix, riskFreeRate)  # Задаем аргументы для функции
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet  # Задаем ограничения для основного массива активов
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))  #
    # noinspection PyTypeChecker
    result = sco.minimize(negativeMSR, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def negativeCSR(weights, dataframe, mean, cov, riskFreeRate=1):
    """
    Расчитывает отрицательный коэффициент шарпа при условной стоимости под риском
    :param weights: веса
    :param dataframe: данные с доходностями активов по дням
    :param mean: средняя доходность актива за период
    :param cov: ковариационная матрица активов
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
    data = data.apply(lambda x: abs(x))  # Находим взвешенную доходность портфелея по дням
    cvar = np.percentile(data, 1) * 100  # Условная стоимость под риском или ожидаемый дефицит
    csr = (returns - riskFreeRate) / abs(cvar)
    return - csr

def maxCSR(dataframe, meanReturns, covMatrix, riskFreeRate=1, constraintSet=(0.04, 0.5)):
    """
    Выдает веса для коэффициент шарпа при условной стоимости под риском.
    :param meanReturns: средняя доходность
    :param covMatrix: матрица ковариации
    :param constraintSet: параметрр для решения уравнения
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
    Общая функция, которая вызывает остальные функции расчета, обрабатывает их результат
    :param meanReturns: средняя
    :param covMatrix: матрица ковариации
    :param riskFreeRate: безрисковая ставка доходноости.
    :param constraintSet: параметр для решения уравнения.
    :return:
    1. Доходность максимального к Шапра
    2. Волатильность максимального к Шарпа
    3. Веса портфеля максимальног к Шарпа
    4. Доходность минимальной волатильности
    5. Волотильность минимальной волатильности
    6. Веса минимальной волатильности
    7. Доходность промежуточных параметров для построения графика
    8. Валотильноость промежуточных параметров для построения графика
    """

    # Максимальный к Шарпа
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
            str='Минимальная волатильность', dataframe=dataframe, mean=meanReturns, cov=covMatrix, dfm=dfm, view=minVol_allocation, file=None)

    # Максимальная доходность
    maxPerf_Portfolio = maxPPerformance(meanReturns, covMatrix, dataframe, constraintSet)
    maxPerf_returns, maxPerf_std = portfolioPerformance(maxPerf_Portfolio['x'], meanReturns, covMatrix, dataframe)
    maxPerf_allocation = pd.DataFrame(maxPerf_Portfolio['x'], index=meanReturns.index, columns=['вес в %'])
    maxPerf_allocation['вес в %'] = [round(i * 100, 3) for i in maxPerf_allocation['вес в %']]
    printer(maxPerf_Portfolio['x'], riskFreeRate, rm, mvar, 'Максимальная доходность', dataframe, meanReturns, covMatrix, dfm, maxPerf_allocation)

    # Максимальный коэффициент шарпа при условной стоимости под риском.
    maxCSRatio = maxCSR(dataframe, meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxCSR_return, maxCSR_std = portfolioPerformance(maxCSRatio['x'], meanReturns, covMatrix, dataframe)
    maxCSR_allocation = pd.DataFrame(maxCSRatio['x'], index=meanReturns.index, columns=['вес в %'])
    maxCSR_allocation['вес в %'] = [round(i * 100, 3) for i in maxCSR_allocation['вес в %']]
    printer(maxCSRatio['x'], riskFreeRate, rm, mvar, 'Максимальный коэффициент шарпа при условной стоимости под риском.', dataframe, meanReturns, covMatrix, dfm, maxCSR_allocation)

    # Максимальный модифицированный коэффициент шарпа
    maxMSRatio = maxMSR(dataframe, meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxMSR_return, maxMSR_std = portfolioPerformance(maxMSRatio['x'], meanReturns, covMatrix, dataframe)
    maxMSR_allocation = pd.DataFrame(maxMSRatio['x'], index=meanReturns.index, columns=['вес в %'])
    maxMSR_allocation['вес в %'] = [round(i * 100, 3) for i in maxMSR_allocation['вес в %']]
    printer(maxMSRatio['x'], riskFreeRate, rm, mvar, 'Максимальный модифицированный коэффициент шарпа', dataframe, meanReturns, covMatrix, dfm, maxMSR_allocation)

    with open('results.txt', 'w') as file:
        print('Итог', file=file)

        print('\nМаксимальный к Шарпа, веса активов', file=file)
        print(maxSR_allocation, file=file)
        print('\nдоходность -', round(maxSR_returns, 3), 'волатильность - ', round(maxSR_std, 3), file=file)

        print('\nМинимальная волатильность, веса активов', file=file)
        print(minVol_allocation, file=file)
        print('\nдоходность -', round(minVol_returns, 3), 'волатильность - ', round(minVol_std, 3), file=file)

        print('\nМаксимальная доходность, веса активов', file=file)
        print(maxPerf_allocation, file=file)
        print('\nдоходность -', round(maxPerf_returns, 3), 'волатильность - ', round(maxPerf_std, 3), file=file)

        print('\nМаксимальный коэффициент шарпа при условной стоимости под риском., веса активов', file=file)
        print(maxCSR_allocation, file=file)
        print('\nдоходность -', round(maxCSR_return, 3),'волатильность - ', round(maxCSR_std, 3), file=file)

        print('\nМаксимальный модифицированный коэффициент шарпа, веса активов', file=file)
        print(maxMSR_allocation, file=file)
        print('\nдоходность -', round(maxMSR_return, 3), 'волатильность - ', round(maxMSR_std, 3), file=file)


    # print(efficientList)
    targetReturns = (maxPerf_returns - minVol_returns) * np.random.random_sample(50) + minVol_returns
    efficientList = (maxPerf_std - minVol_std) * np.random.random_sample(50) + minVol_std
    # print(targetReturns)
    # print(efficientList)

    # Возвращения функции
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, maxPerf_returns, \
        maxPerf_std, maxPerf_allocation, maxMSR_return, maxMSR_std, maxMSR_allocation, maxCSR_return, maxCSR_std, \
        maxCSR_allocation, efficientList, targetReturns


def portfolioReturn(weights, meanReturns, covMatrix, dataframe):
    """Считает доходность портфеля"""
    return portfolioPerformance(weights, meanReturns, covMatrix, dataframe)[0]


def efficientOpt(meanReturns, covMatrix, returnTarget, dataframe, constraintSet=(0, 0.3)):
    """Считает параметры для построения границы эффективности"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix, dataframe) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(gbound(bound, numAssets, asset) for asset in range(numAssets))
    # noinspection PyTypeChecker
    effOpt = sco.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return effOpt


def JensensAlpha(rp, rf, rm, b):
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

def Beta(ra, rp):
    """
    Находит Бетта коэфициент
    :param ra: - доходность оцениваемого актива
    :param rp: - доходность индекса
    :return:
    """
    coeffBeta = (np.cov(ra, rp)) / (np.std(ra, rp))
    return coeffBeta

def JensensAlpha(rp, rf, rm, b):
    """
    Альфа Йенса
    :param rp: - portfolio return
    :param rf: - risk-free rate
    :param rm: - expected market return
    :param b: - portfolio beta
    :return:
    """
    coeffJensensAlpha = rp - (rf + b*(rm - rf))
    return coeffJensensAlpha

def Treynor(rp, rf, b):
    """
    К Трейнора
    :param rp: - portfolio return
    :param rf: - risk-free rate
    :param b: - portfolio beta
    :return:
    """
    coeffTreynor = (rp - rf) / b
    return coeffTreynor

def beta(dataframe, weight, rm, mvar, dfm):
    """
    Бетта коэфициент
    :param ra: - доходность оцениваемого актива
    :param rp: - доходность индекса
    :return:
    """
    l = []
    betta = []
    w = []
    for i in range(len(dataframe.columns)):
        print(dataframe[dataframe.columns[i]])
        print(dfm)
        cova = dataframe[dataframe.columns[i]].cov(dfm)
        cova = cova / ((mvar ** 2) / 100)
        print(cova)
        l.append(cova)
    for i in range(len(l)):
        betta.append(l[i] * weight[i])
    beta = sum(betta)
    print('betta', beta)
    return beta

def M2(rp, rf, pstd, mstd, rm):
    """
    Считает коэффициент Модельятти
    :param rp: доходность портфеля
    :param rf: доходность безрискового актива
    :param pstd: стандартное отклонение портфеля
    :param mstd: стандартное отклонение индекса
    :param rm: доходность индекса
    :return:
    """
    return ((rp - rf) * mstd ** 2) / pstd ** 2 - (rm - rf)

def printer(allocation, rf, rm, mvar, str, dataframe, mean, cov, dfm, view, file=None):
    print(Color.BOLD + Color.GREEN + 'Показатели' + Color.END)
    print(Color.GREEN + '\n' + str + ', веса активов' + Color.END)
    print(view)
    # b = beta(dataframe, allocation, rm, mvar, dfm)
    nrp = portfolioReturn(allocation, meanReturns=mean, covMatrix=cov, dataframe=dataframe)
    nvr = portfolioVariance(allocation, mean, cov, dataframe)
    print(Color.DARKCYAN + '\nдоходность -' + Color.END, nrp * 100)
    print(Color.DARKCYAN + 'волатильность - ' + Color.END, nvr * 100)
    print(Color.DARKCYAN + 'коэффициент Шарпа - ' + Color.END, ((nrp - rf)/nvr))
    print(Color.DARKCYAN + 'модифицированный коэффициент Шарпа - ' + Color.END, - negativeMSR(allocation, dataframe, mean, cov, rf))
    print(Color.DARKCYAN + 'кондиционный коэффициент Шарпа - ' + Color.END, - negativeCSR(allocation, dataframe, mean, cov, rf))
    # print(Color.DARKCYAN + 'коэффициент Тейнора - ' + Color.END, Treynor(rp, rf, b))
    # print(Color.DARKCYAN + 'Альфа Йенса - ' + Color.END, JensensAlpha(rp, rf, rm, b))
    print(Color.DARKCYAN + 'M2 - ' + Color.END, M2(nrp, rf, nvr, mvar, rm))
    print(Color.RED + 'в разработке - ' + Color.END)