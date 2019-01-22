#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import tkinter as tk
import tkinter.messagebox as msg
import numpy as np
import re

ui = tk.Tk( )
ui.title("Caculator")
ui.resizable(False, False)
ui.geometry('360x480')
result = tk.StringVar( )
expression = tk.StringVar( )
result.set(' ')
expression.set('0')

ui.mainloop()

expressionLabel = tk.Label(ui, bg='white', fg='black', font=('Arail','15'),\
                           bd='0', textvariable=expression, anchor='se')
resultLabel = tk.Label(ui, bg='white', fg='black', font=('Arail','30'),\
                       bd='0', textvariable=result, anchor='se')
expressionLabel.place(x='10', y='10', width='300', height='50')
resultLabel.place(x='10', y='60', width='300', height='50')

digits = list('1234567890.=')
index = 0
for row in range(4):
    for col in range(3):
        d = digi