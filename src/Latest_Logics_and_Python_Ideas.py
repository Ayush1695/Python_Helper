# 1. Usage of python decorators
#Let us assume our function is:
def somefunc(a,b):
    output = a+b
    return output

import time
def somefunc(a,b):
    print("somefunc begins")
    start_time = time.time()
    output = a+b
    print("somefunc ends in ",time.time()-start_time,"secs")
    return output
out = somefunc(4,5)

#Output
'''
somefunc begins
somefunc ends in  0.0 secs
'''

# Using decorators
from functools import wraps
def timer(func):
    @wraps(func)
    def wrapper(a,b):
        print(f"{func.__name__!r} begins")
        start_time = time.time()
        result = func(a,b)
        print(f"{func.__name__!r} ends in {time.time()-start_time} secs")
        return result
    return wrapper

@timer
def somefunc(a,b):
    output = a+b
    return output

a = somefunc(4,5)

'''
somefunc begins
somefunc ends in  0.0 secs
'''

# If function takes multiple inputs
from functools import wraps
def timer(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        print(f"{func.__name__!r} begins")
        start_time = time.time()
        result = func(*args,**kwargs)
        print(f"{func.__name__!r} ends in {time.time()-start_time} secs")
        return result
    return wrapper

def timer2(func):
    @wraps(func)
    def wrapper(*args):
        print(f"{func.__name__!r} begins")
        start_time = time.time()
        result = func(*args)
        print(f"{func.__name__!r} ends in {time.time()-start_time} secs")
        return result
    return wrapper


# 2. How Yield is used:
def check_prime(number):
    for divisor in range(2, int(number ** 0.5) + 1):    
        if number % divisor == 0:
            return False
        else:
            return True
        
check_prime(19)


def Primes(max):
    number = 1
    generated = 0
    while generated < max:
        number += 1
        if check_prime(number):
            generated+=1
            yield number
            
prime_generator = Primes(10)
for x in prime_generator:
    # Process Here
    pass

# 3. Generators expression example:
import time
def triplets(n):
    for a in range(n):
        for b in range(a):
            for c in range(b):
                if a*a == b*b + c*c:
                    yield(a, b, c)

triplet_generator = triplets(1000)
for x in triplet_generator:
    print(x)

    
# Generators operations
x= time.time()
triplet_generator = [(a,b,c) for a in range(1000) for b in range(a) for c in range(b) if a*a == b*b + c*c]
for x in triplet_generator:
    print(x)
print(time.time() - x)