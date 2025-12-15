# In[2]:
people = [("Ana", 25), ("Mihai", 30), ("Ioana", 22)]

people.sort(key=lambda x: x[1])

print(people)

# In[4]:
scores = [10, 20, 35, 50, 75, 100]

def binary_search(scores, target):
    left = 0
    right = len(scores)
    sel = left + ((right - left) // 2)

    while scores[sel] != target:
        if right - left <= 1:
            return -1
        if scores[sel] > target:
            right = sel
        else:
            left = sel
        sel = left + ((right - left) // 2)
    return sel
    
print(binary_search(scores, 35))
    

# In[6]:
class Product:
    def __init__(self, name, price):
        self.name: str = name
        self.price: float = price

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.price == other
        if isinstance(other, Product):
            return self.price == other.price
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.price < other
        if isinstance(other, Product):
            return self.price < other.price
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.price > other
        if isinstance(other, Product):
            return self.price > other.price
        return NotImplemented

    def __str__(self):
        return f"({self.__class__.__name__}) {self.name.capitalize()}: {self.price}$"

    def __repr__(self):
        return str(self)


def bubble_sort(products):
    n = len(products)
    for i in range(n):
        for j in range(0, n - i - 1):
            if products[j] > products[j + 1]:
                products[j], products[j + 1] = products[j + 1], products[j]


def quicksort(products):
    if len(products) <= 1:
        return products

    pivot = products[len(products) // 2]
    left, mid, right = [], [], []

    for x in products:
        if x < pivot:
            left.append(x)
        elif x == pivot:
            mid.append(x)
        else:
            right.append(x)

    return quicksort(left) + mid + quicksort(right)


def merge_two(a: list, b: list):
    i = j = 0
    new_full = []

    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            new_full.append(a[i])
            i += 1
        else:
            new_full.append(b[j])
            j += 1

    new_full.extend(a[i:])
    new_full.extend(b[j:])

    return new_full


def insertion_sort(a):
    for i in range(1, len(a)):
        for j in range(i, 0, -1):
            if a[j] < a[j - 1]:
                a[j], a[j - 1] = a[j - 1], a[j]
            else:
                break


def merge_sort(a):
    if len(a) <= 4:
        insertion_sort(a)
        return

    mid = len(a) // 2
    left = a[:mid]
    right = a[mid:]

    merge_sort(left)
    merge_sort(right)
    
    a[:] = merge_two(left, right)


products = {
    "apple": 1.2,
    "bread": 2.5,
    "milk": 1.8,
    "eggs": 3.0
}

print("--- bubble sort")

product_list = [Product(name, value) for name, value in products.items()]
print(product_list)

bubble_sort(product_list)
print(product_list)

print("--- quick sort")

product_list = [Product(name, value) for name, value in products.items()]
print(product_list)

new_list = quicksort(product_list)
print(new_list)

print("--- merge + insertion sort")
product_list = [Product(name, value) for name, value in products.items()]
print(product_list)

merge_sort(product_list)
print(product_list)

print("--- timsort")
product_list = [Product(name, value) for name, value in products.items()]
print(product_list)

product_list.sort()
print(product_list)

# In[8]:
from multiprocessing import Process
import random
import time


def run_with_timeout(func, timeout):
    p = Process(target=func)
    p.start()
    p.join(timeout)
    if p.is_alive():
        print("Timeout reached — killing process.")
        p.terminate()
        p.join()
        return False
    return True


big_list = [random.randint(0, 1_000_000) for _ in range(1_000_000)]

start = time.time()
run_with_timeout(lambda: bubble_sort(big_list[:]), 5)
print("bubble sort: ", time.time() - start)

start = time.time()
run_with_timeout(lambda: quicksort(big_list[:]), 5)
print("quick sort: ", time.time() - start)

start = time.time()
run_with_timeout(lambda: merge_sort(big_list[:]), 5)
print("merge sort: ", time.time() - start)

start = time.time()
run_with_timeout(lambda: big_list[:].sort(), 5)
print("timsort sort: ", time.time() - start)