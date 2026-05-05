from time import time

#-------------------------#
#-----Timer decorator-----#
#-------------------------#
def timer_func(func):
    def wrap_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        return result, end_time - start_time
    return wrap_func