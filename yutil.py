import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


path = "Digits/ZipDigits.train"
test_path = "Digits/ZipDigits.test"

#symmetry
def calc_sym(df):
        
    r, c = df.shape
    symmetry_list = [[0] for _ in range(r)]
    for j in range(r):

        get_gray = [[0]*16 for _ in range(16)]

        for i in range(256):
            x = i // 16
            y = i % 16
            gray = df.iloc[j,i+1] + 1
            get_gray[x][y] = gray


        x_symmetry = 0
        y_symmetry = 0

        for i in range(256):

            x = i // 16
            y = i % 16
            gray = df.iloc[j,i+1] + 1

            gray_sym_x = get_gray[15-x][y]
            x_symmetry += abs((gray - gray_sym_x)) #这里计算出来的是不对称度

            gray_sym_y = get_gray[x][15-y]
            y_symmetry += abs((gray - gray_sym_y))

        #归一化
        x_symmetry = (1- x_symmetry/512)
        y_symmetry = (1- y_symmetry/512)
        symmetry_list[j] = 0.5 * (x_symmetry+y_symmetry)
    symmetry_list = pd.core.frame.DataFrame(symmetry_list)
    return symmetry_list


#intensity
def calc_intensity(df):
    r, c = df.shape
    sum_row = [0 for _ in range(r)]
    for j in range(r):
        for i in range(256):
            sum_row[j] += df.iloc[j,i+1] + 1
        sum_row[j] /= 512
    sum_row = pd.core.frame.DataFrame(sum_row)
    return sum_row


#load data
def load_preprocess(path):
    
    df = pd.read_csv(path, delimiter=" ")

    #preprocess
    names = [str(i) for i in range(258)]
    names[0] = 'value'
    df.columns=names
    df = df.drop(labels='257', axis=1)

    #get data of 1 and 5
    df1 = df.query("value == 1")
    df5 = df.query("value == 5")

    symmetry_1 = calc_sym(df1)
    symmetry_5 = calc_sym(df5)

    intensity_1 = calc_intensity(df1)
    intensity_5 = calc_intensity(df5)

    N = df1.shape[0] + df5.shape[0]
    col_sym = np.row_stack((symmetry_1, symmetry_5))
    col_intens = np.row_stack((intensity_1, intensity_5))
    # X = np.column_stack((np.ones(N), col_intens, col_sym))
    X = np.column_stack((col_intens, col_sym))
    # print(X)
    y = np.ones(N)
    y[0:df1.shape[0]] = -1
    return X, y, df1.shape[0], df5.shape[0]

def draw_points(X1, X2):
    plt.scatter(X1[:,0],X1[:,1],marker='o',c='orange')
    plt.scatter(X2[:,0],X2[:,1],marker='o', c='blue')
    plt.xlabel("X1")
    plt.ylabel("X2")