# Импорт библиотек
import numpy as np
import cv2, random, os
import time
from matplotlib import pyplot as plt
from PIL import Image

# Функция чтения изображения
def readData(src:str) -> np.ndarray:
    img = cv2.imread(src, cv2.COLOR_RGB2GRAY)/255
    bin = []
    for i in range(80):
        ap = []
        for j in range(80):
            ap.append(np.sum(img[i, j])/3)
        bin.append(ap)
    bin = np.array(bin)
    bin = np.ndarray.flatten(bin)
    return bin

# Функция активации гипреболический тангенс
def tanh(x:np.ndarray) -> np.ndarray:
    return (np.power(np.e, x) - np.power(np.e, -x))/(np.power(np.e, x) + np.power(np.e, -x))

# Производная гипреболического тангенс
def tanh_derivative(x:np.ndarray) -> np.ndarray:
    sech = 1 / np.cosh(x)
    tanh_deriv = sech**2
    return tanh_deriv

# Инициализация весов
W1 = np.array([[random.choice([-1, 1]) * random.random() for _ in range(156)] for _ in range(80 * 80)])
W2 = np.array([[random.choice([-1, 1]) * random.random() for _ in range(96)] for _ in range(156)])
W3 = np.array([[random.choice([-1, 1]) * random.random() for _ in range(20)] for _ in range(96)])
W4 = np.array([[random.choice([-1, 1]) * random.random() for _ in range(1)] for _ in range(20)])

# Сохранение изначальных весов
W1_copy = W1.copy()
W2_copy = W2.copy()
W3_copy = W3.copy()
W4_copy = W4.copy()

# Функция сохранения послойного вывода сети
def prediction(inputData:np.ndarray) -> list:
    layers = [inputData]
    layers.append(layers[-1] @ W1)
    layers.append(tanh(layers[-1]) @ W2)
    layers.append(tanh(layers[-1]) @ W3)
    layers.append(tanh(layers[-1] @ W4))
    layers.append(tanh(layers[-1]))
    return layers

# Функция распределния локальных градиентов по нейронам
def spreadGradient(correct:int, layers:list) -> list:
    sigma = []
    error = layers[-1] - correct
    sigma.append(error * tanh_derivative(layers[-2]))
    sigma.append(W4 @ sigma[-1] * tanh_derivative(layers[-3]))
    sigma.append(W3 @ sigma[-1] * tanh_derivative(layers[-4]))
    sigma.append(W2 @ sigma[-1] * tanh_derivative(layers[-5]))
    sigma.reverse()
    return sigma

LEARNING_RATE = 10**-4 # Инициализация скорости обучения

# Функция создания матриц градиентов по каждому весу
def changeW(src:str, correct:int) -> list:
    layers = prediction(readData(src))
    sigmas = spreadGradient(correct, layers)
    changeList = []
    for i in range(len(sigmas)):
        changeList.append(np.outer(layers[i], sigmas[i]))
    return np.array(changeList)

# Чтение корректных ответов на каждое избражение
with open("dataset\data_answers.txt") as f:
    lines = f.readlines()

W1, W2, W3, W4 = W1_copy, W2_copy, W3_copy, W4_copy

# Инициализация переменных для оптимизатора Adam и постороения графиков
mt, vt = 0, 0
b1, b2 = 0.99, 0.999
x_data = np.arange(0, 1200)
y_data = [[], []]
start = time.time()
bach = [0, 0, 0, 0]
bach_size = 1

# Кол-во верных ответов
corrects = 0

# Цикл обучения
for epoch in range(17):
    if epoch%4 == 0:
        x_data = np.arange(0, 1200)
        y_data = [[], []]
        corrects = 0
        for m in range(0, 1200):
            correct = lines[m]
            AI_ans = prediction(readData(f"dataset/test{m}.png"))
            if int(np.sign(AI_ans[-1][0])) == int(correct):
                corrects += 1
            y_data[0].append(corrects/(m + 1))
            os.system("cls||clear") 
            print(f"График обучения №{epoch//4 + 1} построен на {round(m/12, 1)}%")
        plt.plot(x_data, y_data[0], label = f"LEARNING {epoch}")я
    for i in range(1200):
        
        # Получение градиентов по каждому весу
        change = changeW(f"dataset/test{i}.png", int(lines[i]))
        
        # Вычисление градиента по Adam
        mt = b1 * mt + (1 - b1) * change
        vt = b2 * vt + (1 - b1) * change ** 2
        M_t = mt/(1 - b1)
        V_t = vt/(1 - b2)

        # Обновление mini-bach
        bach[0] += LEARNING_RATE/(np.sqrt(V_t[0]) + 10 ** -10) * M_t[0]
        bach[1] += LEARNING_RATE/(np.sqrt(V_t[1]) + 10 ** -10) * M_t[1]
        bach[2] += LEARNING_RATE/(np.sqrt(V_t[2]) + 10 ** -10) * M_t[2]
        bach[3] += LEARNING_RATE/(np.sqrt(V_t[3]) + 10 ** -10) * M_t[3]
        if i%(bach_size) == 0:
            # Обновление весов
            W1 -= bach[0]
            W2 -= bach[1]
            W3 -= bach[2]
            W4 -= bach[3]
            bach = [0, 0, 0, 0]
        os.system("cls||clear") 
        
        # Вывод данных нейросети
        print(f"""
Обучен на {round((epoch * 1200 + i) / (17 * 1200) * 100)}%, прошло - {round(time.time() - start)}с
Изучил {epoch * 1200 + i + 1} фотографий
Эпоха {epoch +1}
{W4}
        """)

# Вывод отношения обученности нейросети к кол-ву эпох
plt.legend()
plt.show()
