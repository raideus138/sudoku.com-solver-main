import cv2
import skimage
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pyautogui
from SudokuSolver import SudokuSolver

def main():
    pyautogui.hotkey("alt", "tab", interval=0.1)
    
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    preprocesada = preprocess(screenshot)
    contorno_sudoku = find_sudoku_contour(preprocesada)
    
    if contorno_sudoku is None:
        print("No se encontró ningún sudoku")
        return
    
    cuadrícula_recortada = crop_grid(screenshot, contorno_sudoku)
    
    imágenes_cuadrados = split_grid(cuadrícula_recortada)
    sudoku = squares_images_to_sudoku(imágenes_cuadrados)
    
    solver = SudokuSolver(sudoku)
    resuelto = solver.solve()
    
    solve_on_website(contorno_sudoku, resuelto)

def preprocess(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def find_sudoku_contour(preprocesada):
    contornos, _ = cv2.findContours(preprocesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuadrados = []
    for contorno in contornos:
        if is_square(contorno):
            cuadrados.append(contorno)
    cuadrados = sorted(cuadrados, key=cv2.contourArea, reverse=True)
    if len(cuadrados) == 0:
        return None
    return cuadrados[0]

def is_square(contorno):
    approx = cv2.approxPolyDP(contorno, 0.01 * cv2.arcLength(contorno, True), True)
    _, _, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    return len(approx) == 4 and abs(aspect_ratio - 1) < 0.1

def crop_grid(imagen, cuadrado):
    x, y, w, h = cv2.boundingRect(cuadrado)
    recortada = imagen[y:y+h, x:x+w]
    return recortada

def split_grid(cuadrícula_recortada):
    img = preprocess(cuadrícula_recortada)
    img = skimage.segmentation.clear_border(img)
    img = 255 - img
    altura, _ = img.shape
    tamaño_cuadrado = altura // 9
    cuadrados = []
    for i in range(9):
        for j in range(9):
            cuadrados.append(img[i*tamaño_cuadrado:(i+1)*tamaño_cuadrado, j*tamaño_cuadrado:(j+1)*tamaño_cuadrado])
    return cuadrados

def squares_images_to_sudoku(imágenes_cuadrados):
    knn = create_knn_model()
    sudoku = np.zeros((81), dtype=int)
    for i, imagen in enumerate(imágenes_cuadrados):
        sudoku[i] = predict_digit(imagen, knn)
    return sudoku.reshape(9, 9)

def predict_digit(img, knn):
    img = img.reshape(1, -1)
    return knn.predict(img)[0]

def create_knn_model():
    df = pd.read_csv("dataset.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn

def solve_on_website(contorno_sudoku, resuelto):
    x, y, w, h = cv2.boundingRect(contorno_sudoku)
    tamaño_cuadrado = h // 9
    for i in range(9):
        for j in range(9):
            pyautogui.click(x + j*tamaño_cuadrado + tamaño_cuadrado//2, y + i*tamaño_cuadrado + tamaño_cuadrado//2, _pause=False)
            pyautogui.press(str(resuelto[i, j]), _pause=False)

if __name__ == '__main__':
    main()
