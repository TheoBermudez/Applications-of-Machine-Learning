# Theo Bermudez
# ITP 449
# HW2
# Question 2

'''
2. Ask the user to enter two positive integers between 1 and 100. Read those integers. Then
output a multiplication table of the first number times the second number.
Please enter an integer: 5
Please enter another integer: 20
5 x 1 = 5
5 x 2 = 10
5 x 3 = 15
…
5 x 20 = 100
'''

num1 = int(input("Please enter an integer: "))
num2 = int(input("Please enter another integer: "))
for i in range(1, num2 + 1):
    print(num1, "x", i, "=", num1 * i)

