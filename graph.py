from finfun import calculatedResults
import plotly.graph_objects as go


def EF_graph(meanReturns, covMatrix, riskFreeRate=1, constraintSet=(0.04, 0.3)):
    """Строит границу эффективности"""
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, maxPerf_returns, \
        maxPerf_std, maxPerf_allocation, maxMSR_return, maxMSR_std, maxMSR_allocation, efficientList, targetReturns = \
        calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    # Максимальный к Шарпа
    MaxSharpeRatio = go.Scatter(
        name='Максимальный к Шарпа',
        mode='markers',
        x=[round(maxSR_std, 4)],
        y=[round(maxSR_returns, 4)],
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )
    # Минимальная волатильность
    MinVol = go.Scatter(
        name='Минимальная волатильность',
        mode='markers',
        x=[round(minVol_std, 4)],
        y=[round(minVol_returns, 4)],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )
    # Максимальная доходность
    MaxPP = go.Scatter(
        name='Максимальная доходность',
        mode='markers',
        x=[round(maxPerf_std, 4)],
        y=[round(maxPerf_returns, 4)],
        marker=dict(color='blue', size=14, line=dict(width=3, color='black'))
    )
    # Максимальный модифицированный коэффициент Шарпа
    MaxMSR = go.Scatter(
        name='Максимальный модифицированный коэффициент Шарпа',
        mode='markers',
        x=[round(maxMSR_std, 4)],
        y=[round(maxMSR_return, 4)],
        marker=dict(color='yellow', size=14, line=dict(width=3, color='black'))
    )
    # Граница эффективности
    EF_curve = go.Scatter(
        name='Граница эффективности',
        mode='lines',
        x=[round(ef_std, 4) for ef_std in efficientList],
        y=[round(target, 4) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    data = [MaxSharpeRatio, MinVol, EF_curve, MaxPP, MaxMSR]
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
