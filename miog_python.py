#!/usr/bin/env python

import numpy as np
import struct
import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm

plt.style.use('ggplot')


# Задаем параметры:
# * window_size -- размер окна для вычисления оконной дисперсии
# * window_size_2 -- размер окна для сравнения средних
# * min_dist -- минимальное расстояние между двумя соседними точками

window_size = 32
window_size_2 = 64

min_dist = 4 * window_size_2

# Функция, которая загружает файл.
# 
# Файл хранится в бинарном формате следующей структуры:
# * Байты 0-3: содержат размер временного ряда, тип _unsigned int_.
# * Байты (4 + 8i)-(4 + 8(i+1)): содержат i-ый член временного ряда, тип _double_.


def load_bin(fname):
    raw = open(fname,'rb').read()

    length, = struct.unpack('I', raw[:4])
    data = np.array(struct.unpack('d' * length, raw[4:]))

    return data


# Считываем временной ряд и печатаем первые 10 его элементов, чтобы убедиться, что данные загружены корректно.


miog = load_bin('data/miog.bin')
print(miog[:10])


# Вычисляем оконную дисперсию. Сначала необходимо посчитать оконное среднее, затем вторым проходом вычисляется оконная дисперсия.
# 
# Заметим, что для значение оконной дисперсии будет храниться по индексу левой точки окна.


rolling_mean = np.zeros(miog.shape[0] - (window_size - 1))
rolling_var  = np.zeros(miog.shape[0] - (window_size - 1))

for i in tqdm(range(miog.shape[0] - (window_size - 1))):
    # считаем среднее
    loc = 0
    for k in range(window_size):
        loc += miog[i+k]
    rolling_mean[i] = loc / window_size
    
    # считаем дисперсию
    loc = 0
    for k in range(window_size):
        loc += np.power(miog[i+k] - rolling_mean[i], 2)
    rolling_var[i] = loc / (window_size - 1)

print(rolling_mean[:10])
print(rolling_var [:10])


# Теперь вычисляем квантиль q_alpha заданного уровня.
# 
# Выборочной оценкой квантили является k-ая порядковая статистика, где
#     
#     k = round(alpha * N)
# 
# Здесь N -- длина временного ряда.


alpha = 0.70

# копируем оконную дисперсию и сортируем, чтобы получить вариационный ряд.
var_cdf = rolling_var.copy()
var_cdf.sort()

print(var_cdf.shape[0], int(alpha * var_cdf.shape[0]))

q_alpha = var_cdf[int(alpha * var_cdf.shape[0])]
q_alpha


# Теперь применяем метод сравнения средних для окон размера window_size_2.
# 
# Если средние в них отличаются более чем на q_alpha, то в точки правого окна помечаем как точки движения.


moving = [0]
moving_x = [0]

for i in tqdm(range(1, int(len(rolling_var) / window_size_2))):
    w1 = rolling_var[window_size_2 * (i-1):window_size_2 * (i)]
    w2 = rolling_var[window_size_2 * i:window_size_2 * (i+1)]
    
    m1 = w1.mean()
    m2 = w2.mean()

    if np.abs(m1 - m2) > q_alpha:
        moving += [1]
    else:
        moving += [0]
    moving_x += [window_size_2 * i]

moving = np.array(moving)


# Помечаем все точки ряда, которые стоят на границе между промежутками нулей и единиц.


support_points = []

for i in tqdm(range(1,len(moving))):
    if (moving[i] == 1 and moving[i-1] == 0
        and(len(support_points) == 0 or moving_x[i] - support_points[-1] > min_dist)):
        support_points.append(moving_x[i])


# Строим график


_,ax = plt.subplots(1,1,figsize=(100,5))

scaler = miog.max() / rolling_var.max()
print(scaler)

ax.plot(miog,color='blue',lw=0.2)

rolling_x = list(range(window_size-1, miog.shape[0]))
ax.plot(rolling_x, rolling_var * scaler,color='purple')

for pt in support_points:
    ax.axvline(window_size + pt,color='red',lw=0.5)

ax.set_xbound(0,miog.shape[0])
plt.show()




