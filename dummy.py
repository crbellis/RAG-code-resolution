# dummy_file.py


class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")


def square(x):
    return x**2


def cube(x):
    return x**2


if __name__ == "__main__":
    obj = MyClass("John")
    obj.greet()

    num = 5
    squared = square(num)
    cubed = cube(num)

    print(f"Square of {num}: {squared}")
    print(f"Cube of {num}: {cubed}")
