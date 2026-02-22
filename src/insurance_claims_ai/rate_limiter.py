from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def wait_if_needed(self):
        now = time.time()
        
        # Step 1: remove expired timestamps from the left
        if self.requests:
            while self.requests and self.requests[0] <= now - self.window_seconds:
                self.requests.popleft()
        # Step 2: check if at limit
        if len(self.requests) >= self.max_requests:
            wait_time = self.window_seconds - (now - self.requests[0])
            time.sleep(wait_time)
            
        self.requests.append(time.time())
