from typing import *


def printOnFirstCall(message: str) -> Callable:
    """ Decorator to print a message only once for a method, for each instance of a class.

    :param method: the class method to decorate
    :param message: the print message, defaults to None
    :return: the decorated method
    """

    def decorator(method: Callable) -> Callable:
        # Here process message to make it more obvious by adding color
        # method: \033[<color>some thing you want to print\033[0m
        # color: 31: red, 32: green, 33: yellow, 34: blue, 35: purple, 36: cyan, 37: white
        if message is None:
            inner_message = f"\033[35mFirst call of {method.__name__}\033[0m"
        else:
            inner_message = f"\033[35m{message}\033[0m"
                
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_printOnceRegister"):
                self._printOnceRegister = set()
            if method.__name__ not in self._printOnceRegister:
                print(inner_message)
                self._printOnceRegister.add(method.__name__)
            return method(self, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    class Test:
        @printOnFirstCall("First call")
        def test(self):
            pass

    t = Test()
    t.test()
    t.test()