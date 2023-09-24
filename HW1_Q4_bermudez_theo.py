# Theo Bermudez
# ITP 449
# HW1
# Question 4

'''
Write a program that prompts the user to enter a loan amount, annual interest rate, and number of years for a car loan. Then it prints the monthly payment amount.
Loan Amount: 30000.00
Annual Interest Rate: 4.00
Years: 5
Your monthly payment is: $552.50
Note:
𝑃𝑀𝑇= 𝑃𝑉𝑖(1+𝑖)𝑛 (1+𝑖)𝑛 −1
PMT is the monthly payment
PV is the loan amount
i is the interest rate per month in decimal form (interest rate percentage divided by 12) n is the number of months (term of the loan in months)
'''
loan_amount = float(input('Loan Amount: '))
annual_interest_rate = float(input('Annual Interest Rate: '))
years = int(input('Years: '))
monthly_interest_rate = annual_interest_rate / 12 / 100
months = years * 12
monthly_payment = loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** months / ((1 + monthly_interest_rate) ** months - 1)
print('Your monthly payment is: $' + str(round(monthly_payment, 2)))