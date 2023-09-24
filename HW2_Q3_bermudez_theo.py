# Theo Bermudez
# ITP 449
# HW2
# Question 3

'''
3. Write a program to ask the user to enter a password. Then check to see if it is a valid password
based on these requirements
a. Must be at least 8 characters long
b. Must contain both uppercase and lowercase letters
c. Must contain at least one number between 0-9
d. Must contain a special character: !, @, #, $
If the password is not valid, ask the user to re-enter. This should continue until the user enters a
valid password. After a valid password is entered, print Access Granted.
Please enter a password. Follow these requirements
a. Must be at least 8 characters long
b. Must contain both uppercase and lowercase letters
c. Must contain at least one number between 0-9
d. Must contain a special character: !, @, #, $
Password: HelloWorld1
Invalid password. Try again!
Password: Hello@World1
Access Granted!
'''

import re
password = input("Password: ")
while True:
    if len(password) < 8:
        print("Invalid password. Try again!")
        password = input("Password: ")
    elif not re.search("[a-z]", password):
        print("Invalid password. Try again!")
        password = input("Password: ")
    elif not re.search("[A-Z]", password):
        print("Invalid password. Try again!")
        password = input("Password: ")
    elif not re.search("[0-9]", password):
        print("Invalid password. Try again!")
        password = input("Password: ")
    elif not re.search("[!@#$]", password):
        print("Invalid password. Try again!")
        password = input("Password: ")
    else:
        print("Access Granted!")
        break




