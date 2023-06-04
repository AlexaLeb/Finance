from finfun import *
from graph import EF_graph

stocklist = ['V', 'DVA', 'HPQ', 'MCK', 'AON', 'MCO', 'KR', 'PFE', 'CVX', 'KO']  # Лист акций
# stocklist = ['KO', 'CVX', 'MCD', 'V', 'HPQ', 'MCO', 'DVA', 'KR', 'MCK', 'AON']
endDate = dt2(2020, 12, 31)
startDate = endDate - dt.timedelta(days=365)
meant, cov, returns = get_data(stocks=stocklist, start=startDate, end=endDate)
# print(meant)
# print(returns)

bill2020 = bill_rate_mean('daily-treasury-rates.csv')
bill2021 = bill_rate_mean('daily-treasury-rates-2.csv')
bill2022 = bill_rate_mean('daily-treasury-rates-3.csv')
list_of_bills = [bill2020, bill2021, bill2022]
market_m, market_var, df = market(startDate, endDate)
# print(market_m, market_var)
# print()
# print('годовая доходность')
# for i in meant:
#     print(((1 + i) ** 252 - 1) * 1)
list_by_calc = calculatedResults(dataframe=returns, meanReturns=meant, covMatrix=cov, riskFreeRate=(bill2020), dfm=df,
                  constraintSet=(0.06, 0.15), rm=market_m, mvar=market_var)

di = {
    'SR ratio': [list_by_calc[0], list_by_calc[1], list_by_calc[2]],
    'Min vol': [list_by_calc[3], list_by_calc[4], list_by_calc[5]],
    'Max PP': [list_by_calc[6], list_by_calc[7], list_by_calc[8]],
    'Max MSR': [list_by_calc[9], list_by_calc[10], list_by_calc[11]],
    'Max CSR': [list_by_calc[12],list_by_calc[13], list_by_calc[14]]
}

print(Color.BOLD + Color.GREEN + 'Выберите портфель и введите его номер, чтобы узнать его показатели за 2021 и 2022. Чтобы остановить работу программы введите любое другое слово' + Color.END)
x = int(input("""
                                1 - для портфеля максимального коэффициента Шарпа
                                2 - для портфеля минимальной волатильности 
                                3 - для портфеля максимальной доходности
                                4 - для портфеля максимального модифицированного коэффициента Шарпа 
                                5 - для портфеля максимального кондиционного коэффициента Шарпа 
                                Поле ввода: 

"""))

if x == 1:
    conclude(di['SR ratio'][2], 2021, bill2021, stocklist, list(di.keys())[0])
    conclude(di['SR ratio'][2], 2022, bill2021, stocklist, list(di.keys())[0])
elif x == 2:
    conclude(di['Min vol'][2], 2021, bill2021, stocklist, list(di.keys())[1])
    conclude(di['Min vol'][2], 2022, bill2021, stocklist, list(di.keys())[1])
elif x == 3:
    conclude(di['Max PP'][2], 2021, bill2021, stocklist, list(di.keys())[2])
    conclude(di['Max PP'][2], 2022, bill2021, stocklist, list(di.keys())[2])
elif x == 4:
    conclude(di['Max MSR'][2], 2021, bill2021, stocklist, list(di.keys())[3])
    conclude(di['Max MSR'][2], 2022, bill2021, stocklist, list(di.keys())[3])
elif x == 5:
    conclude(di['Max CSR'][2], 2021, bill2021, stocklist, list(di.keys())[4])
    conclude(di['Max CSR'][2], 2022, bill2021, stocklist, list(di.keys())[4])
else:
    print(Color.RED + Color.BOLD + 'Выход' + Color.END)




