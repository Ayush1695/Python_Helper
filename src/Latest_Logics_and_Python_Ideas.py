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


