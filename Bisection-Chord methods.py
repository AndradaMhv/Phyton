from sympy import *
import numpy as np
import sys
import csv
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QMessageBox
) 
import time
import pandas as pd
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon



from scipy.optimize import bisect, newton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog
import random
from sympy import Symbol, diff, lambdify


class Window(QMainWindow):
    def __init__(self, parent=None):
        x = Symbol('x')
        
        super(Window, self).__init__(parent)
        self.n=1
        self.fs = 'sin(x)'
        self.root_points = []

        self.setWindowTitle("Metode de căutare a rădăcinilor")
        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)

        
        self.toolbar = NavigationToolbar(self.canvas, self)

        
        self.button = QPushButton('Plot')
        self.bisect_button = QRadioButton('Metoda Bisecției')
        self.chord_button = QRadioButton('Metoda Coardei')

        self.button.clicked.connect(self.on_plot_button_clicked)
        self.bisect_button.clicked.connect(lambda: self.on_button_clicked(self.bisection_method_root))
        self.chord_button.clicked.connect(lambda: self.on_button_clicked(self.chord_method_root))

        self.setGeometry(200, 200, 1000, 800)
        icon = QIcon('AndradaAndreea.jpg')   
        self.setWindowIcon(icon)
        layout = QGridLayout()

        layout.addWidget(self.toolbar, 0, 0, 1, 1)

        layout.addWidget(self.canvas, 1, 0)

        layout.addWidget(self.button, 2, 0)
        layout.addWidget(self.bisect_button, 5, 1)
        layout.addWidget(self.chord_button, 5, 2)

        fl = QLabel('f = ')
        layout.addWidget(fl, 0, 1)
        self.fe = QLineEdit('sin(x)')
        layout.addWidget(self.fe, 0, 2)
        al = QLabel('a = ')
        layout.addWidget(al, 1, 1)
        self.ae = QLineEdit('2')
        layout.addWidget(self.ae, 1, 2)
        bl = QLabel('b = ')
        layout.addWidget(bl, 2, 1)
        self.be = QLineEdit('4')
        layout.addWidget(self.be, 2, 2)

        el = QLabel('Eroarea absolută = ')
        layout.addWidget(el, 3, 1)
        self.error_input = QLineEdit('0.001')
        layout.addWidget(self.error_input, 3, 2)
        iter_l = QLabel('Număr de iterații = ')
        layout.addWidget(iter_l, 4, 1)
        self.iter_e = QLineEdit('30')
        self.iterations_activated = True
        layout.addWidget(self.iter_e, 4, 2)
        self.open_file_button = QPushButton('Deschide Fișier')
        self.open_file_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.open_file_button, 6, 0)
        self.generate_button = QPushButton('Generare Automată')
        self.generate_button.clicked.connect(self.on_generate_button_clicked)
        layout.addWidget(self.generate_button, 7, 0)
        self.execution_time_label = QLabel("Timpul de executie: ")
        layout.addWidget(self.execution_time_label)
        self.keyboard_input_button = QPushButton('Introducere Tastatură')
        self.keyboard_input_button.clicked.connect(self.enable_keyboard_input)
        layout.addWidget(self.keyboard_input_button, 5, 0)
        
        

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
    
        layout.addWidget(self.scroll_area, 8, 0, 1, 3)
    
        self.roots_label = QLabel()
        self.roots_label.setText("Radacini:")
        self.scroll_layout.addWidget(self.roots_label)
        
        self.fe.setReadOnly(False)
        self.ae.setReadOnly(False)
        self.be.setReadOnly(False)
        self.error_input.setReadOnly(False)
        self.iter_e.setReadOnly(False)

        widget = QWidget()
        widget.setLayout(layout)

        
        self.setCentralWidget(widget)

        fs = self.fe.text()
        x = Symbol('x')
        self.f = lambdify(x, fs)
        self.a = float(self.ae.text())
        self.b = float(self.be.text())
        self.error = float(self.error_input.text())
        self.x_val = np.linspace(self.a, self.b, 100)
        self.t_val = (self.x_val - self.a) / (self.b - self.a) 
        ax = self.figure.add_subplot(111)
        ax.plot(self.x_val, self.f(self.x_val))

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.stop()
        self.show()
        
    def on_button_clicked(self, func):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.stop()
        self.timer.timeout.connect(lambda: self.update_plot(func))
        
        

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Deschide Fișier', '', 'Fișiere Text (*.txt);;Fișiere CSV (*.csv);;Fișiere Excel (*.xlsx)')

        if file_path:
            if file_path.endswith('.csv'):
                with open(file_path, 'r') as file:
                    file_contents = file.readlines()
                    if not file_contents:
                        self.show_error_message("Fisierul este gol. Va rugam sa alegeti o alta metoda de introducere a datelor!")
                        return
                    values = self.parse_csv_contents(file_contents)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                if df.empty:
                    self.show_error_message("Fisierul este gol. Va rugam sa alegeti o alta metoda de introducere a datelor!")
                    return
                values = self.parse_excel_data(df)
            else:
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    if not file_contents:
                        self.show_error_message("Fisierul este gol. Va rugam sa alegeti o alta metoda de introducere a datelor!")
                        return
                    values = self.parse_file_contents(file_contents)
            
            self.populate_text_fields(values)
            self.set_text_fields_read_only()

    
    
    def parse_csv_contents(self, file_contents):
        values = []
        csv_reader = csv.reader(file_contents)
        for row in csv_reader:
            values.append(row)
        return values
    
    def parse_excel_data(self, df):
        
        values = df.values.tolist()
        return values

    
    def set_text_fields_read_only(self):
        self.fe.setReadOnly(True)
        self.ae.setReadOnly(True)
        self.be.setReadOnly(True)
        self.error_input.setReadOnly(True)
        self.iter_e.setReadOnly(True)
            
    def parse_file_contents(self, file_contents):
        lines = file_contents.split('\n')
        values = [line.split(',') if ',' in line else line.split() for line in lines]
        return values
    
    def populate_text_fields(self, values=None):
        if values is None:
            function, a, b, error, max_iterations = self.generate_values()
    
            self.fe.setText(str(function))
            self.fe.setReadOnly(True) 
    
            self.ae.setText(str(a))
            self.ae.setReadOnly(True)
            self.be.setText(str(b))
            self.be.setReadOnly(True) 
    
            self.error_input.setText(str(error))
            self.error_input.setReadOnly(True)  
    
            self.iter_e.setText(str(max_iterations))
            self.iter_e.setReadOnly(True)  
        else:
            num_fields = len(values)
            for i in range(num_fields):
                text_field = self.get_text_field(i)
                text_field.setText(str(values[i][0]))
    
    def get_text_field(self, index):
        if index == 0:
            return self.fe
        elif index == 1:
            return self.ae
        elif index == 2:
            return self.be
        elif index == 3:
            return self.error_input
        elif index == 4:
            return self.iter_e
    
    def generate_function(self):
        x = symbols('x')
        coefficients = [round(random.uniform(-5, 5), 1) for _ in range(random.randint(1, 4))]
        
        function_expr = sum(coef * x ** i for i, coef in enumerate(coefficients))
        function_str = str(function_expr)
        
        return function_str
    


    def generate_values(self):
        while True:
            a = round(random.uniform(-10, 10), 4)
            b = round(random.uniform(-10, 10), 4)
    
            function = self.generate_function()
    
            if self.has_root(function, a, b):
                break
    
        use_error = random.choice([True, False])  
        use_iterations = random.choice([True, False]) 
    
        if use_error and not use_iterations:
            error = round(random.uniform(0.001, 0.1), 4)
            max_iterations = ("")
        elif use_iterations and not use_error:
            error = 0
            max_iterations = random.randint(10, 100)
        else:
            error = round(random.uniform(0.001, 0.1), 4)
            max_iterations = random.randint(10, 100)
    
        self.fe.setText(function)
        self.ae.setText(str(a))
        self.be.setText(str(b))
        self.error_input.setText(str(error))
        self.iter_e.setText(str(max_iterations))
    
        return function, a, b, error, max_iterations
        
    def has_root(self, function, a, b):
        expr = sympify(function)
        x = symbols('x')
        expr_eval = lambdify(x, expr)
        if expr_eval(a) * expr_eval(b) < 0:
            return True
        else:
            return False
    
    def enable_keyboard_input(self):
        self.fe.setReadOnly(False)
        self.ae.setReadOnly(False)
        self.be.setReadOnly(False)
        self.error_input.setReadOnly(False)
        self.iter_e.setReadOnly(False)
    
    def on_generate_button_clicked(self):
        self.populate_text_fields()

    def update_plot(self, root_finding_method):
        self.root_points = []
        start_time = time.time()

        try:
            max_iterations = int(self.iter_e.text())
            self.iterations_activated = True
        except ValueError:
            self.iterations_activated = False

        if self.iterations_activated:
            if self.n > max_iterations - 2:
                self.timer.stop()
        else:
            try:
                self.error = float(self.error_input.text())
            except ValueError:
                self.show_error_message("Valoarea erorii trebuie să fie un număr real")
                return
            if abs(self.f(root_finding_method())) < self.error * 10:
                self.timer.stop()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(self.x_val, self.f(self.x_val))

        self.n += 1
        n = self.n

        root = root_finding_method()

        self.recalculate_values()

        self.roots_label.setText(f"{self.roots_label.text()}Radacina iteratiei {n}: {root}\n")

        self.root_points.append((root, self.f(root)))

        for point in self.root_points:
            ax.plot(point[0], point[1], 'ro', label='Root')
            ax.axhline(0, color='black', linewidth=0.5)

            ax.text(point[0], point[1], f'({point[0]}, {point[1]})', ha='center', va='bottom')

        plt.title('Iteratia ' + str(n))

        self.canvas.draw()

        end_time = time.time()
        execution_time = end_time - start_time

        self.execution_time_label.setText(f"Timpul de executie: {execution_time:.4f} secunde")

        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
    
    def on_plot_button_clicked(self):
        a_input = self.ae.text()
        b_input = self.be.text()
        error_input = self.error_input.text()
        iter_e_input = self.iter_e.text()
        function_input = self.fe.text()

        if not function_input and not a_input and not b_input and not error_input and not iter_e_input:
            self.show_error_message("Fieldurile sunt goale! Introduce-ți valorile corespunzătoare.")
            return
        if not function_input:
            self.show_error_message("Introduceți funcția")
            return
        
        if not a_input and not b_input and not error_input and not iter_e_input:
            self.show_error_message("Introduceți pe rând valorile pentru a, b, eroarea absolută și/sau numărul de iterații")
            return

        if not a_input and not b_input:
            self.show_error_message("Introduceți valorile pentru a și b")
            return
        a_input = self.ae.text()
        if not a_input:
            self.show_error_message("Introduceți valoarea lui a")
            return
        try:
            self.a = float(self.ae.text())
        except ValueError:
            self.show_error_message("Valoarea lui a trebuie să fie un număr real")
            return
        
        b_input = self.be.text()
        if not b_input:
            self.show_error_message("Introduceți valoarea lui b")
            return
        
        try:
            self.b = float(b_input)
        except ValueError:
            self.show_error_message("Valoarea lui b trebuie să fie un număr real")
            return
        
        error_input = self.error_input.text()
        iter_e_input = self.iter_e.text()
        
        if not error_input and not iter_e_input:
            self.show_error_message("Introduceți eroarea absolută sau numărul de iterații. Ambele câmpuri nu pot fi goale.")
            return
        
        if error_input:
            try:
                self.error = float(error_input)
            except ValueError:
                self.show_error_message("Valoarea erorii trebuie să fie un număr real")
                return
        else:
            self.error = 0
        
        if iter_e_input:
            try:
                self.max_iterations = int(iter_e_input)
            except ValueError:
                self.show_error_message("Valoarea numărului de iterații trebuie să fie un număr întreg")
                return
        else:
            self.max_iterations = float('inf')
        if not (self.bisect_button.isChecked() or self.chord_button.isChecked()):
            self.show_error_message("Vă rugăm să alegeți o metodă de calcul (Metoda Bisecției sau Metoda Coardei)")
            return
        
        self.n = 0
        fs = self.fe.text()
        x = Symbol('x')
        try:
            self.f = lambdify(x, fs)
        except SyntaxError:
            self.show_error_message("Funcția introdusă nu este validă")
            return
        
        self.recalculate_values()
        
        ax = self.figure.add_subplot(111)
        ax.plot(self.x_val, self.f(self.x_val))
        
        self.timer.start()


    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(message)
        msg.setWindowTitle("Error")
        msg.exec_()
    
    def bisection_method_root(self):
        if np.sign(self.f(self.a)) == np.sign(self.f(self.b)):
            print("Intervalul specificat nu conține o rădăcină.")
            return 0

        for i in range(self.n):
            m = (self.a + self.b) / 2
            f_m = self.f(m)

            if np.sign(self.f(self.a)) == np.sign(f_m):
                self.a = m
            elif np.sign(self.f(self.b)) == np.sign(f_m):
                self.b = m

        return (self.a + self.b) / 2
    
    def chord_method_root(self):
        x = Symbol('x')
        f2 = diff(self.fe.text(), x, 2)
        f2 = lambdify(x, f2)
        if self.f(self.a) * f2(self.a) < 0:
            x = self.a
            for i in range(self.n):
                x = x - self.f(x) / (self.f(x) - self.f(self.b)) * (x - self.b)
        else:
            x = self.b
            for i in range(self.n):
                x = x - self.f(x) / (self.f(x) - self.f(self.a)) * (x - self.a)
        self.a = x - 1
        self.b = x + 1
        return x

    def recalculate_values(self):
        self.x_val = np.linspace(self.a, self.b, 100)
        self.t_val = (self.x_val - self.a) / (self.b - self.a) 
    


    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('AndradaAndreea.jpg'))
    main = Window()
    main.show()
    sys.exit(app.exec_())
