# Theo Bermudez
# ITP 449
# HW2
# Question 1

'''
1. Write a program to compute and print all possible combinations of change for $1.
Denominations to be considered – quarter, dime, nickel, penny
Hint: Use nested loops (loops within loops for the various denominations of coins)
Change for $1
0 quarters, 0 dimes, 0 nickels, 100 pennies
…
4 quarters, 0 dimes, 0 nickels, 0 pennie
'''

for q in range(5):
    for d in range(11):
        for n in range(21):
            for p in range(101):
                if q * 25 + d * 10 + n * 5 + p == 100:
                    print(q, "quarters,", d, "dimes,", n, "nickels,", p, "pennies")