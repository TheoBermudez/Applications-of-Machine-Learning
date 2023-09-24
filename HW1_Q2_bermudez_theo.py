# Theo Bermudez
# ITP 449
# HW1
# Question 2

'''
Write a program that prompts the user to enter their full name then prints the number of characters in their name (do not count spaces).
What is your name? Tommy Trojan
Tommy Trojan your name has 11 characters.
'''

name = input('What is your name? ')
print(name, 'your name has', len(name.replace(' ', '')), 'characters.')