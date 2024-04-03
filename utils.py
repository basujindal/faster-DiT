import torch, time, gc

def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return time.time()

def end_timer_and_print(start_time, local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} GB".format(torch.cuda.max_memory_allocated() / 1e9))
    
    
class TimeCuda:
    def __init__(self, start_msg = "", end_msg = ""):
    
        self.start_msg = start_msg
        self.end_msg = end_msg
        
    def __enter__(self):
        if self.start_msg != "":
            print(self.start_msg)
        self.start_time = start_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_timer_and_print(self.start_time, self.end_msg)


class Timer:
    def __init__(self, start_msg = "", end_msg = ""):
    
        self.start_msg = start_msg
        self.end_msg = end_msg
        
    def __enter__(self):
        if self.start_msg != "":
            print(self.start_msg)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(self.end_msg, f"{elapsed_time:.3f} sec")