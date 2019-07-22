from math import isnan
import numpy as np
import torch
from visdom import Visdom

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


def plot_reward(t, r, name, color='#000'):
    win = name
    title = name + ' Episodic Reward'

    if name not in d:
        d[name] = []
    d[name].append((t, float(r)))

    x, y = zip(*d[name])
    data = [get_line(x, y, 'reward', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Episodes'},
        yaxis={'title': 'Cumulative Reward'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def plot(t, r, name, color='#000'):
    win = name
    title = name

    if name not in d:
        d[name] = []
    d[name].append((t, float(r)))

    x, y = zip(*d[name])
    data = [get_line(x, y, name, color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': name}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})
