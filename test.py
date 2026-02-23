def make_counter():
    count = 0  # this lives in the enclosing scope

    def counter():
        nonlocal count      # <-- THIS is the key line
        count = count + 1
        return count

    return counter

my_counter = make_counter()

print(my_counter())  # 1
print(my_counter())  # 2
print(my_counter())  # 3