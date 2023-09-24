# Theo Bermudez
# ITP 449
# HW1
# Question 5

'''
Write a program that prompts the user to enter their first name. It then prints whether or not their name is a palindrome.
What is your name? Tommy
Tommy, your name is not a palindrome!
'''
name = input('What is your name? ')
if name.lower() == name[::-1].lower():
    print(name + ', your name is a palindrome!')
else:
    print(name + ', your name is not a palindrome!')