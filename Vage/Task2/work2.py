import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_data(file_path):
    """Загрузка данных из CSV файла"""
    x_values = []
    y_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаем заголовок
        for row in reader:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))
    return x_values, y_values

def plot_graph(x_values, y_values, width=10, height=5):
    """Построение графика с указанными размерами окна"""
    plt.figure(figsize=(width, height))
    plt.plot(x_values, y_values, label="f(x)", color="red")
    plt.title("График функции")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.xticks(np.arange(min(x_values), max(x_values) + 1, 1.0))
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Построение графика из CSV файла')
    parser.add_argument('file', help='Путь к CSV файлу с данными')
    parser.add_argument('-w', '--width', type=float, default=10, 
                       help='Ширина окна графика (по умолчанию: 10)')
    parser.add_argument('-ht', '--height', type=float, default=5,
                       help='Высота окна графика (по умолчанию: 5)')

    args = parser.parse_args()

    try:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Файл {args.file} не найден")
            
        x, y = load_data(args.file)
        plot_graph(x, y, args.width, args.height)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == '__main__':
    main()
#команда для запуска python plot_graph.py data.csv -w 20 -ht 5