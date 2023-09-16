import time

def time_counter(last_time, motion_str, show_flag):
    now_time = time.time()
    if show_flag:
        print(motion_str, ':', str(round(now_time - last_time,7)),'s')
    last_time = time.time()
    return last_time
