#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import tkinter as TK
import tkinter.messagebox as msg
import numpy as np
import re


root = TK.Tk( )
root.title("Caculator")
root.resizable(0,0)
root.geometry('320x420')
result = TK.StringVar( )
equation = TK.StringVar( )
result.set(' ')
equation.set('0')
# 获得按下的数字或者符号
def getnum(num):
    temp = equation.get( )
    temp2 = result.get( )
    print(temp)
    print(temp2)
    if temp2 != ' ' :
        temp = '0'
        temp2 = ' '
        result.set(temp2)
    if (temp=='0'):
        temp = ''
    temp = temp + num
    equation.set( temp )
    print(equation)
# 按下退格键时，去除最后一个字符
def back( ):
    temp = equation.get( )
    equation.set(temp[:-1])
# 按下MC时，清空算式行与结果行
def clear( ):
    equation.set('0')
    result.set(' ')
# 按下等于号时计算结果
def run( ):
    temp = equation.get( )
    temp = temp.replace('x','*')
    temp = temp.replace('÷','/')
    # 写一个小彩蛋，可以用于表白哦
    if temp == '5+2+0+1+3+1+4':               # 暗号
        result.set('xxx我爱你')               # 彩蛋或者表白语
        return 0
    print(temp)
    answer = caculator.caculator(temp)
    answer = '%.2f'%answer
    result.set(str(answer))
# 结果显示框
show_uresult = TK.Label(root,bg='white',fg = 'black',font = ('Arail','15'),bd='0',textvariable =equation,anchor='se')
show_dresult = TK.Label(root,bg='white',fg = 'black',font = ('Arail','30'),bd='0',textvariable=result,anchor='se')
show_uresult.place(x='10',y='10',width='300',height='50')
show_dresult.place(x='10',y='60',width='300',height='50')
# 按钮
# 第一行按钮
button_back =TK.Button(root,text='←',bg='DarkGray',command=back)
button_back.place(x = '10',y='150',width = '60',height='40')
button_lbracket=TK.Button(root,text='(',bg='DarkGray',command= lambda : getnum('('))
button_lbracket.place(x = '90',y='150',width = '60',height='40')
button_rbracket=TK.Button(root,text=')',bg='DarkGray',command= lambda : getnum(')'))
button_rbracket.place(x = '170',y='150',width = '60',height='40')
button_division =TK.Button(root,text='÷',bg='DarkGray',command= lambda : getnum('÷'))
button_division.place(x = '250',y='150',width = '60',height='40')
# 第二行按钮
button_7 =TK.Button(root,text='7',bg='DarkGray',command= lambda : getnum('7'))
button_7.place(x = '10',y='205',width = '60',height='40')
button_8 =TK.Button(root,text='8',bg='DarkGray',command= lambda : getnum('8'))
button_8.place(x = '90',y='205',width = '60',height='40')
button_9 =TK.Button(root,text='9',bg='DarkGray',command= lambda : getnum('9'))
button_9.place(x = '170',y='205',width = '60',height='40')
button_multiplication =TK.Button(root,text='X',bg='DarkGray',command= lambda : getnum('x'))
button_multiplication.place(x = '250',y='205',width = '60',height='40')
# 第三行按钮
button_4 =TK.Button(root,text='4',bg='DarkGray',command= lambda : getnum('4'))
button_4.place(x = '10',y='260',width = '60',height='40')
button_5 =TK.Button(root,text='5',bg='DarkGray',command= lambda : getnum('5'))
button_5.place(x = '90',y='260',width = '60',height='40')
button_6 =TK.Button(root,text='6',bg='DarkGray',command= lambda : getnum('6'))
button_6.place(x = '170',y='260',width = '60',height='40')
button_minus =TK.Button(root,text='—',bg='DarkGray',command= lambda : getnum('-'))
button_minus.place(x = '250',y='260',width = '60',height='40')
# 第四行按钮
button_1 =TK.Button(root,text='1',bg='DarkGray',command= lambda :getnum('1'))
button_1.place(x = '10',y='315',width = '60',height='40')
button_2 =TK.Button(root,text='2',bg='DarkGray',command= lambda : getnum('2'))
button_2.place(x = '90',y='315',width = '60',height='40')
button_3 =TK.Button(root,text='3',bg='DarkGray',command= lambda : getnum('3'))
button_3.place(x = '170',y='315',width = '60',height='40')
button_plus =TK.Button(root,text='+',bg='DarkGray',command= lambda : getnum('+'))
button_plus.place(x = '250',y='315',width = '60',height='40')
# 第五行按钮
button_MC =TK.Button(root,text='MC',bg='DarkGray',command = clear)
button_MC.place(x = '10',y='370',width = '60',height='40')
button_0 =TK.Button(root,text='0',bg='DarkGray',command= lambda : getnum('0'))
button_0.place(x = '90',y='370',width = '60',height='40')
button_point =TK.Button(root,text='.',bg='DarkGray',command= lambda : getnum('.'))
button_point.place(x = '170',y='370',width = '60',height='40')
button_equal=TK.Button(root,text='=',bg='DarkGray',command= run)
button_equal.place(x = '250',y='370',width = '60',height='40')
root.mainloop()
def eq_format(eq):
    '''
    :param eq: 输入的算式字符串
    :return: 格式化以后的列表，如['60','+','7','*','8']
    '''
    format_list = re.findall('[\d\.]+|\(|\+|\-|\*|\/|\)',eq)
    return format_list
def change(eq,count):
    '''
    :param eq: 刚去完括号或者乘除后的格式化列表
    :param count: 发生变化的元素的索引
    :return: 返回一个不存在 '+-' ,'--'类的格式化列表
    '''
    if eq[count] == '-':
        if eq[count-1] == '-':
            eq[count-1] = '+'
            del eq[count]
        elif eq[count-1] == '+':
            eq[count-1] = '-'
            del eq[count]
    return eq
def remove_multiplication_division(eq):
    '''
    :param eq: 带有乘除号的格式化列表
    :return: 去除了乘除号的格式化列表
    '''
    count = 0
    for i in eq:
        if i == '*':
            if eq[count+1] != '-':
                eq[count-1] = float(eq[count-1]) * float(eq[count+1])
                del(eq[count])
                del(eq[count])
            elif eq[count+1] == '-':
                eq[count] = float(eq[count-1]) * float(eq[count+2])
                eq[count-1] = '-'
                del(eq[count+1])
                del(eq[count+1])
            eq = change(eq,count-1)
            return remove_multiplication_division(eq)
        elif i == '/':
            if eq[count+1] != '-':
                eq[count-1] = float(eq[count-1]) / float(eq[count+1])
                del(eq[count])
                del(eq[count])
            elif eq[count+1] == '-':
                eq[count] = float(eq[count-1]) / float(eq[count+2])
                eq[count-1] = '-'
                del(eq[count+1])
                del(eq[count+1])
            eq = change(eq,count-1)
            return remove_multiplication_division(eq)
        count = count + 1
    return eq
def remove_plus_minus(eq):
    '''
    :param eq: 只带有加减号的格式化列表
    :return: 计算出整个列表的结果
    '''
    count = 0
    if eq[0] != '-':
        sum = float(eq[0])
    else:
        sum = 0.0
    for i in eq:
        if i == '-':
            sum = sum - float(eq[count+1])
        elif i == '+':
            sum = sum + float(eq[count+1])
        count = count + 1
    if sum >= 0:
        eq = [str(sum)]
    else:
        eq = ['-',str(-sum)]
    return eq
def calculate(s_eq):
    '''
    :param s_eq: 不带括号的格式化列表
    :return: 计算结果
    '''
    if '*' or '/' in s_eq:
        s_eq = remove_multiplication_division(s_eq)
    if '+' or '-' in s_eq:
        s_eq = remove_plus_minus(s_eq)
    return s_eq
def simplify(format_list):
    '''
    :param format_list: 输入的算式格式化列表如['60','+','7','*','8']
    :return: 通过递归去括号，返回简化后的列表
    '''
    bracket = 0     # 用于存放左括号在格式化列表中的索引
    count = 0
    for i in format_list:
        if i == '(':
            bracket = count
        elif i == ')':
            temp = format_list[bracket + 1 : count]
            # print(temp)
            new_temp = calculate(temp)
            format_list = format_list[:bracket] + new_temp + format_list[count+1:]
            format_list = change(format_list,bracket)     # 解决去括号后会出现的--  +- 问题
            return simplify(format_list)            # 递归去括号
        count = count + 1
    return format_list                     # 当递归到最后一层的时候，不再有括号，因此返回列表
def caculator(eq):
    format_list = eq_format(eq)
    s_eq = simplify(format_list)
    ans = calculate(s_eq)
    if len(ans) == 2:
        ans = -float(ans[1])
    else:
        ans = float(ans[0])
    return ans
'''
if __name__ == '__main__':
    equation = '1-2*((60-30+(-40/5)*(9-2*5/3+7/3*99/4*2998+10*568/14))-(-4*3)/(16-3*2))'
    ans = caculator(equation)
    print('eval运算结果：',eval(equation))
    print('程序运算结果：',ans)
'''