from __future__ import print_function, division
from builtins import range, input

import pickle
import numpy as np
import os
from sortedcontainers import SortedList

# 1. Carga de datos
if not os.path.exists('user2movie.json') or \
        not os.path.exists('movie2user.json') or \
        not os.path.exists('usermovie2rating.json') or \
        not os.path.exists('usermovie2rating_test.json'):
    import preprocess2dict

with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)
with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)
with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

# Dimensiones
N = np.max(list(user2movie.keys())) + 1
m_test = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(np.max(list(usermovie2rating.keys())), m_test) + 1

print(f"N: {N}, M: {M}")

if N > 10000:
    print("N es muy grande, esto podría tardar.")

# ---------------------------------------------------------
# 2. PRE-CALCULO: La clave de la velocidad
# ---------------------------------------------------------
print("Pre-calculando promedios y desviaciones...")
averages = np.zeros(N)
deviations = {}
sigmas = np.zeros(N)
user2movie_sets = {}

for i in range(N):
    movies_i = user2movie.get(i, [])
    if movies_i:
        ratings_i = [usermovie2rating[(i, m)] for m in movies_i]
        avg_i = np.mean(ratings_i)
        averages[i] = avg_i

        # Desviaciones (rating - promedio)
        dev_dict = {m: (usermovie2rating[(i, m)] - avg_i) for m in movies_i}
        deviations[i] = dev_dict

        # Sigma para el denominador de Pearson (norma del vector de desviaciones)
        dev_values = np.array(list(dev_dict.values()))
        sigmas[i] = np.sqrt(dev_values.dot(dev_values))

        # Guardamos el set para intersecciones rápidas
        user2movie_sets[i] = set(movies_i)
    else:
        # Para usuarios sin ratings en el set de entrenamiento
        averages[i] = 3.0  # Valor neutral
        sigmas[i] = 1e-8
        user2movie_sets[i] = set()

# ---------------------------------------------------------
# 3. CÁLCULO DE VECINOS (Similitud de Pearson)
# ---------------------------------------------------------
K = 25
limit = 5
neighbors = []

print("Calculando similitudes entre usuarios...")
for i in range(N):
    sl = SortedList()
    movies_i_set = user2movie_sets[i]
    sig_i = sigmas[i]
    dev_i = deviations.get(i, {})

    # Solo comparamos si el usuario i tiene ratings
    if sig_i > 1e-7:
        for j in range(N):
            if i == j: continue

            # Intersección rápida de sets
            common_movies = movies_i_set.intersection(user2movie_sets[j])

            if len(common_movies) > limit:
                # Numerador: producto punto de las desviaciones comunes
                num = sum(dev_i[m] * deviations[j][m] for m in common_movies)
                w_ij = num / (sig_i * sigmas[j] + 1e-8)

                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    neighbors.append(sl)
    if i % 500 == 0:
        print(f"Usuario {i}/{N} procesado")


# ---------------------------------------------------------
# 4. PREDICCIÓN Y EVALUACIÓN
# ---------------------------------------------------------
def predict(i, m):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        w = -neg_w
        if m in deviations[j]:
            numerator += w * deviations[j][m]
            denominator += abs(w)

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = (numerator / denominator) + averages[i]

    # Clipping
    return min(5, max(0.5, prediction))


def calculate_mse(data_dict):
    predictions = []
    targets = []
    for (i, m), target in data_dict.items():
        predictions.append(predict(i, m))
        targets.append(target)
    return np.mean((np.array(predictions) - np.array(targets)) ** 2)


print('Train MSE:', calculate_mse(usermovie2rating))
print('Test MSE:', calculate_mse(usermovie2rating_test))