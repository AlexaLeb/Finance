import pandas as pd
import datetime as dt
from datetime import datetime as dt2
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import scipy.optimize as sc
import plotly.graph_objects as go

yf.pdr_override()

# fdf dij
def get_data(stocks: list, start, end):
    """
    Импортирует данные
    :param stocks: список акций
    :param start: время начальное
    :param end: время конечное
    :return: среднюю ежедневную доходность и ковариационную матрицу
    """
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockdata = stockdata['Close']  # Получили цены закрытия
    returns = stockdata.pct_change()  # Получили дневную доходность
    mean = returns.mean()  # Получили среднее доходности
    covMatrix = returns.cov()  # Получили матрицу ковариации
    return mean, covMatrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    """
    Функция находит доходность портфеля и его среднеквадратичное отклонение.
    :param weights: веса активов.
    :param meanReturns: средняя доходность в сутки.
    :param covMatrix: матрица ковариации.
    :return: доходность за год, стандартное отклонение.
    """
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(
            np.dot(weights.T, np.dot(covMatrix, weights))
           )*np.sqrt(252)
    return returns, std


def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 3):
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
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate)/pStd


def maxSRatio(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0, 1)):
    """
    Минимизирует негативный К шарпа балансируя веса портфеля.
    :param meanReturns: средняя доходность активов.
    :param covMatrix: матрица ковариации.
    :param riskFreeRate: безрисковая ставка доходности.
    :param constraintSet: параметр необходимый для решения уравнения.
    :return: большой словарь по оптимизации заданных параметров .
    """
    numAssets = len(meanReturns)  # количество активов
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolioVariance(weights, meanReturns, covMatrix):
    """
    Выдает отклонение портфеля (вариацию)
    :param weights: веса
    :param meanReturns: среднее
    :param covMatrix: ковариация
    :return: отклонения
    """
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]


def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """
    Выдает веса для минимальной вариации
    :param meanReturns: средняя доходность
    :param covMatrix: матрица ковариации
    :param constraintSet: параметрр для решения уравнения
    :return:
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    """
    Общая функция, которая вызывает остальные функции расчета, обрабатывает их результат
    :param meanReturns: средняя
    :param covMatrix: матрица ковариации
    :param riskFreeRate: безрисковая ставка доходноости
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
    maxSR_Portfolio = maxSRatio(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i * 100, 0) for i in maxSR_allocation.allocation]
    print(maxSR_allocation.allocation)

    # Минимальная волатильность
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i * 100, 0) for i in minVol_allocation.allocation]
    # Граница эффективности
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 15)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, \
        minVol_allocation, efficientList, targetReturns


def portfolioReturn(weights, meanReturns, covMatrix):
    """Считает доходность портфеля"""
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]


def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0, 1)):
    """Считает параметры для построения границы эффективности"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                         constraints=constraints)
    return effOpt


def EF_graph(meanReturns, covMatrix, riskFreeRate=8, constraintSet=(0, 1)):
    """Строит границу эффективности"""
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, \
        efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    # Максимальный к Шарпа
    MaxSharpeRatio = go.Scatter(
        name='Максимальный к Шарпа',
        mode='markers',
        x=[round(100*maxSR_std, 4)],
        y=[round(100*maxSR_returns, 4)],
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )
    # Минимальная волатильность
    MinVol = go.Scatter(
        name='Минимальная волатильность',
        mode='markers',
        x=[round(100 * minVol_std, 4)],
        y=[round(100 * minVol_returns, 4)],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )
    # Граница эффективности
    EF_curve = go.Scatter(
        name='Граница эффективности',
        mode='lines',
        x=[round(ef_std * 100, 4) for ef_std in efficientList],
        y=[round(100 * target, 4) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    data = [MaxSharpeRatio, MinVol, EF_curve]
    layout = go.Layout(
        title='Оптимизация портфеля с границей эффективности',
        yaxis=dict(title='Годовой доход (%)'),
        xaxis=dict(title='Годовая волатильность (%)'),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)

    fig = go.Figure(data=data, layout=layout)
    return fig.show()

stocklist = ['AAPL', 'BHP', 'TLS']
# stock = [stock + '.AX' for stock in stocklist]

endDate = dt2(2020,12,31)
startDate = endDate - dt.timedelta(days=365)

weight = np.array([0.3, 0.3, 0.4])
meant, cov = get_data(stocklist, start=startDate, end=endDate)


print(calculatedResults(meant, cov))

# print(efficientOptim(meant, cov, 0.09))

print(EF_graph(meant, cov))
