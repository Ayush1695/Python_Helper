# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:57:07 2019

@author: ayush.oturkar
"""

import pandas as pd
import numpy as np
import re
import time
from datetime import timedelta

#Note '/' gives division result in the form of float while '//' gives the result in the form of int
# 09/10/1995
#1. Python dictionary - "fromkeys"
#ex: vowel check

def check_vow(string, vowels):
    string = string.casefold()  #casefold is a function which ignores cases
    print(string)
    count = {}.fromkeys(vowels, 0) #fromkeys automatically adds keys with specified user values
    for charac in string:
        if charac in count:
            count[charac]+= 1
    return count

vowels = 'aeiou'
string = 'Geeks for GeEks'

string.casfold()
check_vow(string, vowels)

#2. To check the substring in a string
# string- "find"

''' 
NOTE: "find" function is used to find the location of specific substring, 
if not present it returns -1
'''

#Example
str1 = "geeks for geeks"
str2 = "for"
str1.find(str2)


#3
# Function to find Longest Common Sub-string - "SequenceMatcher" function
  
from difflib import SequenceMatcher 
  

def longestSubstring(str1,str2): 
  
     # initialize SequenceMatcher object with  
     # input string 
     seqMatch = SequenceMatcher(None,str1,str2)  #None implies that it will not ignore any character 
  
     # find match of longest sub-string 
     # output will be like Match(a=0, b=0, size=5) 
     match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
  
     # print longest substring 
     if (match.size!=0): 
          print (str1[match.a: match.a + match.size])  
     else: 
          print ('No longest common sub-string found') 
  
# Driver program 
if __name__ == "__main__": 
    str1 = 'GeeksforGeeks'
    str2 = 'GeeksQuiz'
    longestSubstring(str1,str2) 

#4. getting close matches of word and the pattern
# difflib.get_close_matches(word, possibilities, n=3, cutoff=0.6)
# Where n: maximum number of close matches to return, 
# & cutoff: (default 0.6) is a float in the range [0, 1]. Possibilities that don’t score at least that similar to word are ignored.

from difflib import get_close_matches 
get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy'])
#Output:
['apple', 'ape']   
    
#5. Filtering a dictionary:
# Using lambda: say we need to filter the dict such that value is equal to 1
dc = {'a':1,'b':2,'c':1}
dc_fil = dict(filter(lambda x: x[1]==1, dc.items())) #x[1] is the value on which we are applying the filters

#6. Adding commas at every 1000th place
# To add commas at every thousanth place use "{:,}" with format

#Example:
def place_value(number): 
    return ("{:,}".format(number)) 
  
print(place_value(1000000)) 

#6 "exec" function in python:
# "exec" function is useful in running the string as code

# Ex.
def exec_code(): 
    LOC = """ 
def factorial(num): 
    fact=1 
    for i in range(1,num+1): 
        fact = fact*i 
    return fact 
print(factorial(5)) 
"""
    exec(LOC) 
    
exec_code()
#Output:
#>>> 120

#7 'find' function:
# 'find' in string to find the location of pattern in a string
# Ex:
inp = 'GEEKSFORGEEKS'

      
# Driver Code 
exec_code() 

#==================================================
t1 = time.time()
#SORTING ALGORITHMS:

#1. Insertion sort: swaping from right to left one by one
def insertion_sort(A):
    for i in range(1, len(A)):
        for j in range(i-1, -1, -1):
            print('A[j]', A[j])
            print('A[j+1]', A[j+1])
            if A[j] > A[j+1]:
                A[j], A[j+1] = A[j+1], A[j]
                print('A[j] now', A[j])
                print('A[j+1] now', A[j+1])
            else:
                break
    return A
 
lst = [5,8,9,3,10, 40, 20, 22, 1000, 200, 39, 100, 4,1 , 77, 99, 22, 33, 55, 101, 33, 44, 11, 14, 23, 99, 101, 22, 33, 333, 444, 555, 666, 16, 12]           

insertion_sort(lst)   
t2 = time.time()
elapsed_time_secs = t2 - t1
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg) 


# Complexity: O[n^2]


#2. Selection sort  : Swapping based upon min values starting from the first element to the last element
def selection_sort(A):   
    for i in range(0, len(A) - 1):
        minIndex = i
        for j in range(i+1, len(A)):
            if A[j] < A[minIndex]:
                minIndex = j
        if minIndex!=i:
            
            A[i], A[minIndex] = A[minIndex], A[i]
        
    return A

t1 = time.time()
lst = [5,8,9,3,10, 40, 20, 22, 1000, 200, 39, 100, 4,1 , 77, 99, 22, 33, 55, 101, 33, 44, 11, 14, 23, 99, 101, 22, 33, 333, 444, 555, 666, 16, 12]           
selection_sort(lst)   
t2 = time.time()
elapsed_time_secs = t2 - t1
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg) 

#Complexity = O[n^2]

#3. Bubble sort:
def bubbleSort(myList):
    for i in range(0, len(myList) - 1):
        for j in range(0, len(mylist) - 1 - i):
            if myList[j] > myList[j + 1]:
                myList[j], myList[j+1] = mylist[j+1], myList[j]
                
    return myList

t1 = time.time()
lst = [5,8,9,3,10, 40, 20, 22, 1000, 200, 39, 100, 4,1,77, 99, 22, 33, 55, 101, 33, 44, 11, 14, 23, 99, 101, 22, 33, 333, 444, 555, 666, 16, 12]           
selection_sort(lst)   
t2 = time.time()
elapsed_time_secs = t2 - t1
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg) 

#MERGE SORT
#Advantages: 
'''
Merge Sort is recursive(method call itself)
Divide and conquer algo
Very efficient for large datasets
'''
#Complexity = O(n*logn) logn because each step doubles the list size, 1,2,4,8,16 hence log with base 2

def merge_sort(A):
    merge_sort2(A, 0 , len(A) - 1)

def merge_sort2(A, first, last):
    if first<last:
        middle = (first + last)//2
        merge_sort2(A, first, middle)
        merge_sort2(A, middle + 1, last)
        merge(A, first , middle, last)        

def merge(A, first ,middle, last):
    L = A[first: middle]
    R = A[middle: last + 1]
    L.append(999999999)  #so that we know we reached last number
    R.append(999999999)
    i = j = 0
    for k in range(first, last + 1):
        if L[i]<=R[j]:
            A[k] = L[i]
            i+=1   
        else:
            A[k] = R[j]
            j+=1
    return A

merge_sort(lst)
print(lst)
    
#===================================================================================================
# Write a code to count number of occurences of numbers in the string
l1 = ['11337723511','44321272777']
from collections import Counter
#Counter returns dictionary of count of all distinct values in the 
dict(Counter(l1[0]))

             
Q2
# E.g: "John Wick"
list2 = ['John Wick', 'Mukil Doss']
S = pd.Series(list2)
D = pd.DataFrame(S, columns = ['name'])
D['First_name'] = ""
D['Last_name'] = ""

def first_name(z):
    lst = z.split()
    return lst[0]

def last_name(z2):
    lst2 = z2.split()
    return lst2[1]

D['First_name'] = D.name.apply(first_name)
D['Last_name'] = D.name.apply(last_name)

#Important
D['name'].str.split().str.get(1)

Q3

d = {"Class" : 10, "Score" : 50}
for p,y in d.items():
    print(p, y, sep = " ", end= "\n", flush = True)

#CONCEPT
# np.logical_not
#It computes the truth value of NOT x element 
import numpy as np
x = np.array([9, 5])
print(np.logical_not(x<4))


#CONCEPT
#Creating a dataframe
import pandas as pd
dt =[[1,1000, 300], [2,1200,800], [3,800,500]] 
dataframe = pd.DataFrame(dt, columns = ['ID', 'Views', 'Clicks'])
dataframe.iloc[2]

import matplotlib.pyplot as plt
plt.plot(50, 50)
plt.text(49, 49, text = 'Histogram')

checking = np.array([[5, 0, 1],[6, 4, 4]])
list(checking[0]) + list(checking[1])

#CONCEPT
#To check the booleans by applying some conditions
import numpy as np
costs = [1,2,3,4]
#INCORRECT costs<=2 this doesnot work on list
print(np.array(costs)<=2)

#CONCEPT: One more method same as dict
#CASE 1
#So within a function defination , the parameter **args is turned into a dictionary
def easy_print(**x):
    for key, value in x.items():
        print('The value of ' + 
              str(key) + " is " + 
              str(value))  
        
(easy_print(a = 16), easy_print(b = 12))
i, j = **a=16.items()

#CASE2 input size is changing
#So within a function defination , the parameter *args is turned into a tuple
def mean(*x):
    """Returns the mean of all the numbers"""
    total_sum = 0 # Intial sum
    n = len(x) # Number of arguments
    for i in x:
        print(i)
        total_sum = total_sum + i
    return total_sum/n

print((mean(1, 2), mean(20, 25, 30)))


#CONCEPT str.capatalize only keeps the first alphabet in the string as caps
"HELLO".capitalize()
#Output  : "Hello"

#CONCEPT OF global and non local
#CASE 1:
temp = 15 #This is a global variable as out oof any function
def convert_temp(x):
    """Converts the temperature from Celsius to Fahrenheit"""
    global temp   #Adding global will update the global value from memory if any changes done
    temp = (x * 1.8) + 32 #For example here the temp 15 in memory will be changed to 59
convert_temp(temp)
print(temp)
#Output = 59 in fehrenheit

#CASE 2:
def add_zeros(string):
    """Returns a string padded with zeros 
       to ensure consistent length"""
    updated_string = string + '0' # A variable inside inside a function is nonlocal
    def add_more():
        """Adds more zeros if necessary"""
        nonlocal updated_string   #Adding global will update the nonlocal value from memory if any changes done
        updated_string = updated_string + '0' 
    while len(updated_string) < 6:
        add_more()
    return updated_string
(add_zeros('3.4'), add_zeros('2.345'))
#Output = (3.4000, 2.3450)
len('3.4')

#==============================================================
#CONCEPT: Giving mutiple value for function in function
#Ex: Cube root
def nth_root(n):
    def actual_root(x):
        root = x ** (1/n)
        return root
    return actual_root

print(nth_root(3)(27))

#CONCEPT
str('3')*3
#Output = '333' similarly for all

#CONCEPT: Print in descing order
print(dataframe[2:0:-1]) #Prints 3nd row first then 2nd row then we need to add this ":-1"

#================================================================
#CONCEPT
#df = 
'''
              eggs  salt  spam
apr portugal    44    18     5
    usa         36    95    63
feb france      93    56    10
    spain       67    20    88
jan england     90    33    93
    ireland     62    21    94
jun germany     96    75    79
    italy       52    99     7
mar china       83    43    19
    india        5     6    98
    
'''
#We need output:
'''
            eggs  ...   spam
feb france    93  ...     10
'''

#Solution
print(df.loc[[('feb', 'france')]])
df[["Country"]]

#=====================================
#CONCEPT: To change the spelling of indexes
D.set_index(D['Last_name'], inplace = True)

D.index = D.index.str.replace('Wick', 'wick')

#======================================
#CONCEPT: to calculate various quantile of a columns
#df = 
'''
  Month Count
0   Jan    52
1   Apr    29
2   Mar    46
3   Feb     3
'''
#Now to calculate the 5th and 95th percentile:
print(df['Count'].quantile([0.05, 0.95]))
#Output:
'''
0.05   6.9
0.95   51.1
Name:Count, dtype: float64
'''
#==============================================================
#CONCEPT: zip function
import pandas as pd
list_keys = ['Country', 'Total']
list_values = [['United States', 'India', 'UK'], [100, 1000, 10]]

zipped = list(zip(list_keys, list_values)) #Maps first value of first list to first of second and so on..
data = dict(zipped)
df = pd.DataFrame(data)
df.head()
#Output
'''
          Country  Total
0   United States   1118
1   Soviet Union    473
2   United Kingdom    273
'''

#==============================================================
#CONCEPT: using pd.DataFrame.from_items()
#Converts lists of tuples into dataframe
pd.DataFrame.from_items([('Country', ['india', 'UK'])])
    
#CONCEPT: COnvert list of time into list of datetime
l = ['2017-01-01 091234','2017-01-01 091234']
print((pd.to_datetime(l)))

#================================================================
#CONCEPT pop .pop() if no argument passed returns last value of list and removes it from the main list
#CASE.1 = For list
l = [1,2,3,4]
l.pop()
#Output : 4
l.pop()
#Output : 3
#and so on...

#CASE.2 = For Dict. .popitem()
from collections import OrderedDict

od = [('cc',30), ('bb', 20), ('aa', 10)]
d = OrderedDict(od)
d.popitem()
print(d.items())
#================================================================
#Adding values in set
s = set([3,5,6,8])
s.add(5)
print(s)

#================================================================
#Modifying for ahead work
import select
st = select([dataframe])
df.loc[4, "Country"] = "UK"

#================================================================
#CONCEPT: Calculating the distinct counts
#df = 
'''
Country	Total
UK	10.0
United States	100.0
UK	300.0
India	1000.0
'''
df['Country'].value_counts(dropna = False) #False will also count number of NA

l1 = ['Country', 'Total']
l2 = ['UK', 100]
l3 = list(zip(l1, l2))
l4 = dict(l3)
df = df.append(l4, ignore_index = True)

#================================================================
#CONCEPT: To strip part of string directly
#CASE 1
#ex
df = 
'''
           A
0  time.1.15
1  time.1.16
2  time.2.15
3  time.3.14
'''
df["new_col"] = df["A"].str[-2:] #Strips last 2 from it
print(df)

#Output = 
'''
           A  new_col
0  time.1.15   15
1  time.1.16   16 
2  time.2.15   15
3  time.3.14   14
'''
#CASE 2
df['A'].str.split(".").str.get(2)

#Output:
'''
           A  new_col
0  time.1.15   15
1  time.1.16   16 
2  time.2.15   15
3  time.3.14   14
'''
#========================================================================
#CONCEPT: pivot_table
#Consider a df:
'''
   class     name  test1  test2
0      1     Nick     11     32
1      2    Sarah     12     45
2      1  Jasmine     92     62
3      2   Martin     56     34
'''
print(df.pivot_table([index = ['class']])) #It will take mean of class = 1 values and class = 2 to be a unique index
#Output:

'''
       test1  test2
class              
1       51.5   47.0
2       34.0   39.5
'''

#======================================================================
#CONCEPT: notnull check with any()
pd.notnull(df) #Gives boolean for each value
pd.notnull(df).any() #Gives boolean column wise
pd.notnull(df).any().any() #Gives a boolean for entire dataframe
#========================================================================

#CONCEPT: Appending series ignoring index
s = pd.Series([2,4,6])
s2 = pd.Series([1,2,3])
print(s.append(s2, ignore_index = True)) #Similarly for pandas dataframe
#Output:
'''
0    2
1    4
2    6
3    1
4    2
5    3
dtype: int64
'''
#========================================================================
#CONCEPT: Changing the index of the dataframe using "reindex"
#Consider the dataframe df:
'''
       eggs
month                  
a      10   
c      15   
e       6
'''

#required output:
'''
       eggs
month      
a      10.0
b      10.0
c      15.0
d      15.0
e       6.0
'''
#CODE:
cols = []
cols = ['a', 'b', 'c', 'd', 'e']
df.reindex(cols).ffill()

check =  [['Harry', 37.21], ['Berry', 37.21], ['Tina', 37.2], ['Akriti', 41], ['Harsh', 39]]
" ".join(['Hello', 'brother'])
string
#===========================================================
#CONCEPT: ljust and rjust in string
#Ex:
w = len('HELLO') 
"HELLO"[2:].ljust(w, ' ') #Returns 'LLO   ' same len of character replaced by spaces or any values ypu put
"HELLO"[2:].rjust(w, ' ') #Returns '   LLO' same len of character replaced by spaces or any values ypu put


#===========================================================
#CONCEPT: Printing octal, hexal etc of input that you give:

st=int(input())

w=len(bin(st)[2:])

for i in range(1,st+1):
    print (str(i).rjust(w,' '),str(oct(i)[2:]).rjust(w,' '),str(hex(i)[2:].upper()).rjust(w,' '),str(bin(i)[2:]).rjust(w,' '),sep=' ')

#=============================================================

#CONCEPT: Reversing the string
x = "hello"
x[::-1]
#Output: 'olleh'

#=============================================================
#CONCEPT: Rangoli code
import string
alpha = string.ascii_lowercase

n = int(input())
L = []
for i in range(n):
    s = "-".join(alpha[i:n])
    L.append(s[::-1]+s[1:])


width = len(L[0])

for i in range(n-1, 0, -1):
    print(L[i].center(width, "-")) #Puts the string in center and fills rest of width by "-"

for i in range(n):
    print(L[i].center(width, "-"))


#=============================================================================
" ".join(['a','b'])

def solve(s):
    lst = s.split()
    lst_capitalize = [str(i).capitalize() for i in lst]
    return " ".join(lst_capitalize)

#=======================================================================
#MINION GAME

s = input()

vowels = 'AEIOU'

kevsc = 0
stusc = 0
list(range(len(s)))
for i in range(len(s)):
    if s[i] in vowels:
        kevsc += (len(s)-i)
        print("Kevsc",kevsc)
    else:
        stusc += (len(s)-i)
        print("Stusc",stusc)
        

if kevsc > stusc:
    print("Kevin", kevsc)
elif kevsc < stusc:
    print("Stuart", stusc)
else:
    print("Draw")
    
#==================================================================================
#CONCEPT: SQL in python
x = pd.Series([3,1000,1000], index = ['ID','Views', 'Clicks'])
dataframe = dataframe.append(x, ignore_index = True)

#SQL
dataframe.groupby('ID').agg({'Views': np.sum,'Clicks' : np.sum}).reset_index(drop = True)

#CASE
dataframe.groupby(['ID']).agg({'Clicks':[np.size, np.max, np.mean]})
dataframe.query('Clicks = 1000')

df['Country'].value_counts()
#CASE
'''
UPDATE tips
SET tip = tip*2
WHERE tip < 2;
'''
#In python:
tips.loc[tips['tip'] < 2, 'tip'] *= 2
    
#=================================================================================
#PANDAS 
#  Checking the number of duplicates present in the data:
pokemon_dup.duplicated().sum() # 70 indicates the num of duplicates present in the data

#Doubt
nba_csv.drop_duplicates(subset=["Name", "Team", "Number"]) # level on which duplicates can be allowed

# if more than one variable is used for sorting we sort one by descending and other by ascending if req
nba_csv.sort_values(by=["Team", "College"], ascending=["True", "False"])

#
D['name'] = D['name'].str.capitalize()

#=================================================================================
#REPLACING THE STRING in df
#Consider a df:
'''
Name	Team	Number	Position	Age	Height	Weight	College	Salary	Age_status
0	Avery Bradley	Boston Celtics	0.0	PG	25.0	6-2	180.0	Texas	7730337.0	Less than 30
1	Jae Crowder	Boston Celtics	99.0	SF	25.0	6-6	235.0	Marquette	6796117.0	Less than 30
2	John Holland	Boston Celtics	30.0	SG	27.0	6-5	205.0	Boston University	NaN	Less than 30
3	R.J. Hunter	Boston Celtics	28.0	SG	22.0	6-5	185.0	Georgia State	1148640.0	Less than 30
'''
#=================================================================================
nba_csv["Age_status_replaced"] = nba_csv["Age_status"].str.replace("Less than 30", "LESS THAN 30")
nba_csv.head()

'''
Name	Team	Number	Position	Age	Height	Weight	College	Salary	Age_status	Salary_Bonus	Age_status_replaced
0	Avery Bradley	Boston Celtics	0.0	PG	25.0	6-2	180.0	Texas	7730337.0	Less than 30	7740337.0	LESS THAN 30
1	Jae Crowder	Boston Celtics	99.0	SF	25.0	6-6	235.0	Marquette	6796117.0	Less than 30	6806117.0	LESS THAN 30
2	John Holland	Boston Celtics	30.0	SG	27.0	6-5	205.0	Boston University	NaN	Less than 30	NaN	LESS THAN 30
3	R.J. Hunter	Boston Celtics	28.0	SG	22.0	6-5	185.0	Georgia State	1148640.0	Less than 30	1158640.0	LESS THAN 30
'''

#=================================================================================
#5.3.28.4  Fitlering rows using contains:
# Similar to like in SQL:
nba_csv = nba_csv[0:457]
nba_csv[nba_csv["Name"][0:457].str.contains("John")]

#Output:
'''
Name	Team	Number	Position	Age	Height	Weight	College	Salary	Age_status	Salary_Bonus	Age_status_replaced	Frist_Name	Last_Name
2	John Holland	Boston Celtics	30.0	SG	27.0	6-5	205.0	Boston University	NaN	Less than 30	NaN	LESS THAN 30	John	Holland
5	Amir Johnson	Boston Celtics	90.0	PF	29.0	6-9	240.0	NaN	12000000.0	Less than 30	12010000.0	LESS THAN 30	Amir	Johnson
65	James Johnson	Toronto Raptors	3.0	PF	29.0	6-9	250.0	Wake Forest	2500000.0	Less than 30	2510000.0	LESS THAN 30	James	Johnson'''

#Getting dataframe after groupby operation
nba_csv.groupby(["College"], as_index=False)["Name"].count().sort_values(by="Name", ascending =True)

#=================================================================================
#CONCEPT: Pivoting and Melting 
#df:
'''
         date variable     value
0  2000-01-03        A  0.469112
1  2000-01-04        A -0.282863
2  2000-01-05        A -1.509059
3  2000-01-03        B -1.135632
4  2000-01-04        B  1.212112
5  2000-01-05        B -0.173215
6  2000-01-03        C  0.119209
7  2000-01-04        C -1.044236
8  2000-01-05        C -0.861849
9  2000-01-03        D -2.104569
10 2000-01-04        D -0.494929
11 2000-01-05        D  1.071804
'''
#================================================================================
df.pivot(index='date', columns='variable', values='value')
#"index" = the index, "columns" =  the columns to become the header, "values" = value
#Output:

'''
variable           A         B         C         D
date                                              
2000-01-03  0.469112 -1.135632  0.119209 -2.104569
2000-01-04 -0.282863  1.212112 -1.044236 -0.494929
2000-01-05 -1.509059 -0.173215 -0.861849  1.071804
'''

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],['one', 'two', 'one', 'two',  'one', 'two', 'one', 'two']])) 
tuples
#*arg converts it into tuples
#=================================================================================
#Melting:
keys = ['first', 'last', 'height', 'weight']
values = [['John', 'Mary'], ['Doe', 'Bo'], [5.5, 6.0], [130,150]]

dfs = dict(list(zip(keys, values)))
df2 = pd.DataFrame(dfs)

#df2 : 
'''
first	last	height	weight
John	Doe	    5.5	    130
Mary	Bo	    6.0	    150
'''

#=================================================================================

df2.melt(id_vars=['first', 'last'], var_name='quantity')
'''
  first last quantity  value
0  John  Doe   height    5.5
1  Mary   Bo   height    6.0
2  John  Doe   weight  130.0
3  Mary   Bo   weight  150.0
'''

#==================================================================================
    
df2.pivot_table(values='weight', index='last', columns='first', aggfunc='mean', fill_value=0)
'''
first  John  Mary
last             
Bo        0   150
Doe     130     0
'''
#aggfunc ex. 'measn', 'max', 'min', ['max', 'min', 'mean']

#==================================================================================
#Filter in python
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)

#=================================================================================
#Joining two dataframes in Python
>>> df1.merge(df2, how='left', left_on='lkey', right_on='rkey', suffixes=('_left', '_right'))


#=================================================================================
#"eval" function in python
'''
The eval() method parses the expression passed to it and runs python expression(code) within the program.

The syntax of eval is:
eval(expression, globals=None, locals=None)

For more details vist: https://www.geeksforgeeks.org/eval-in-python/
'''

#=================================================================================
#Itertools package
#1. "product" function
from itertools import product
>>>
>>> print list(product([1,2,3],repeat = 2`))
[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
>>>
>>> print(list(product([1,2,3],[4,5,6])))
[(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4)]
>>>
A = [[1,2,3],[3,4,5]]
print(list(product(*A)))

[(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]
>>>
>>> B = [[1,2,3],[3,4,5],[7,8]]
>>> print list(product(*B))
[(1, 3, 7), (1, 3, 8), (1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8), (2, 3, 7), (2, 3, 8), (2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8), (3, 3, 7), (3, 3, 8), (3, 4, 7), (3, 4, 8), (3, 5, 7), (3, 5, 8)]

#2. "permutation" function
>>> from itertools import permutations
>>> print permutations(['1','2','3'])
<itertools.permutations object at 0x02A45210>
>>> 
>>> print list(permutations(['1','2','3']))
[('1', '2', '3'), ('1', '3', '2'), ('2', '1', '3'), ('2', '3', '1'), ('3', '1', '2'), ('3', '2', '1')]
>>> 
>>> print list(permutations(['1','2','3'],2))
[('1', '2'), ('1', '3'), ('2', '1'), ('2', '3'), ('3', '1'), ('3', '2')]
>>>
>>> print list(permutations('abc',3))
[('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]

#3. Similary for "combination" for no order sequences

#4. "combinations_with_replacement
>>> from itertools import combinations_with_replacement
>>> 
>>> print list(combinations_with_replacement('12345',2))
[('1', '1'), ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '2'), ('2', '3'), ('2', '4'), ('2', '5'), ('3', '3'), ('3', '4'), ('3', '5'), ('4', '4'), ('4', '5'), ('5', '5')]
>>> 
>>> A = [1,1,3,3,3]
>>> print list(combinations(A,2))
[(1, 1), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (3, 3), (3, 3), (3, 3)]

#5. grpupby function in itertools
from itertools import groupby
groups = []
uniquekeys = []
for k, g in groupby(1222311"\):
    groups.append(list(g))      # Store group iterator as a list
    uniquekeys.append(k)
#Output:
>>>groups we get are [[1],[2,2,2],[3],[1,1]]
le
    
# Itertools package guidance Link: https://docs.python.org/2/library/itertools.html#itertools.groupby
#=====================================================================================
#Printing in horizontal line while looping using end and flush
cart_prod = [1,2,3,4]
for i in cart_prod:
    print(i, sep = " ", end =" ", flush = True)

