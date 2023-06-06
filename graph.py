from finfun import calculatedResults
import plotly.graph_objects as go


def EF_graph(list_by_calc):
    """Строит границу эффективности"""
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, maxPerf_returns, \
        maxPerf_std, maxPerf_allocation, maxMSR_return, maxMSR_std, maxMSR_allocation, maxCSR_return, maxCSR_std, \
        maxCSR_allocation, efficientList, targetReturns = list_by_calc
    # Максимальный к Шарпа
    MaxSharpeRatio = go.Scatter(
        name='Максимальный к Шарпа',
        mode='markers',
        x=[round(maxSR_std * 100, 4)],
        y=[round(maxSR_returns * 100, 4)],
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )
    # Минимальная волатильность
    MinVol = go.Scatter(
        name='Минимальная волатильность',
        mode='markers',
        x=[round(minVol_std * 100, 4)],
        y=[round(minVol_returns * 100, 4)],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )
    # Максимальная доходность
    MaxPP = go.Scatter(
        name='Максимальная доходность',
        mode='markers',
        x=[round(maxPerf_std * 100, 4)],
        y=[round(maxPerf_returns * 100, 4)],
        marker=dict(color='blue', size=14, line=dict(width=3, color='black'))
    )
    # Максимальный модифицированный коэффициент Шарпа
    MaxMSR = go.Scatter(
        name='Максимальный модифицированный коэффициент Шарпа',
        mode='markers',
        x=[round(maxMSR_std * 100, 4)],
        y=[round(maxMSR_return * 100, 4)],
        marker=dict(color='yellow', size=14, line=dict(width=3, color='black'))
    )

    MaxCSR = go.Scatter(
        name='Максимальный кондиционный коэффициент Шарпа',
        mode='markers',
        x=[round(maxCSR_std * 100, 4)],
        y=[round(maxCSR_return * 100, 4)],
        marker=dict(color='orange', size=14, line=dict(width=3, color='black'))
    )
    # Граница эффективности
    EF_curve = go.Scatter(
        name='Случайный портфель',
        mode='markers',
        x=[round(ef_std * 100, 4) for ef_std in efficientList],
        y=[round(target * 100, 4) for target in targetReturns],
        marker=dict(color='black', size=4)
    )
    data = [MaxSharpeRatio, MinVol, EF_curve, MaxPP, MaxMSR, MaxCSR]
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
        width=1500,
        height=800)
    fig = go.Figure(data=data, layout=layout)
    return fig.show(), maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, maxPerf_returns, \
        maxPerf_std, maxPerf_allocation, maxMSR_return, maxMSR_std, maxMSR_allocation, maxCSR_return, maxCSR_std, \
        maxCSR_allocation, efficientList, targetReturns
