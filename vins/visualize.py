import sys
import numpy as np
import torch
from visdom import Visdom
from termcolor import colored

d = {}

viz = Visdom()

def get_line(x, y, name, color='#000', isFilled=False, fillcolor='transparent', width=2, showlegend=False):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(
                color=color,
                width=width),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=showlegend
        )


def plot_loss(epoch, loss, name, color='#000'):
    win = name
    title = name + ' Loss'

    if name not in d:
        d[name] = []
    d[name].append((epoch, loss.item()))

    x, y = zip(*d[name])
    data = [get_line(x, y, 'loss', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def plot(t, r, data_type, name, color='#000'):
    win = data_type
    title = data_type

    if data_type not in d:
        d[data_type] = {}

    if name not in d[data_type]:
        d[data_type][name] = {'points': [], 'color': color}
    d[data_type][name]['points'].append((t, float(r)))

    data = []
    for name in d[data_type]:
        x, y = zip(*d[data_type][name]['points'])
        data.append(get_line(x, y, name, color=d[data_type][name]['color'], showlegend=True))

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': data_type}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def map(v, low, high, name):
    points = np.zeros((int(high[0]) - int(low[0]) + 1, int(high[1]) - int(low[1]) + 1))
    for x in range(int(low[0]), int(high[0]) + 1):
        for y in range(int(low[1]), int(high[1]) + 1):
            points[(x, y)] = v(torch.FloatTensor([x, y]))

    viz.heatmap(
        X=points.T,
        win=f'map-{name}',
        opts=dict(
            title=f'Values - {name}'
        )
    )


def alert(text, done=False):
    if done:
        text = '\r' + colored('\U00002714 ', 'green') + text + '\n'
    else:
        text = '\r' + colored('\U00002714 ', 'yellow') + text
    sys.stdout.write(text)
