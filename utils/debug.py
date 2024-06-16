import os
import time
from typing import Callable, Any

def log_time(func: Callable) -> Callable:
	"""
	A decorator that logs the time a function takes to execute.

	Args:
		func (Callable): The function to be wrapped and timed.

	Returns:
		Callable: The wrapped function with added logging.
	"""
	def wrapper(*args: Any, **kwargs: Any) -> Any:
		"""
		Wrapper for functions to log execution time if in DEBUG mode.

		Args:
			*args (Any): Positional arguments for the function.
			**kwargs (Any): Keyword arguments for the function.

		Returns:
			Any: The result of the function execution.
		"""
		if os.getenv('DEBUG') == None:
			return func(*args, **kwargs)

		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		duration = end_time - start_time 
		print(f"{func.__name__} took {duration:.2f} seconds")
		return result

	return wrapper
