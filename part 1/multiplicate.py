def multiplicate(_list: list) -> list:
    product = 1
    for element in _list:
        product *= element
    return[product // x for x in _list]


_list = [1, 2, 3, 4]
print(multiplicate(_list))

#######################################################################
# Или, если ограничение не распространяется на встроенные модули python

from functools import reduce

def multiplicate(_list: list) -> list:
    product = reduce(lambda x, y: x * y, _list)
    return[product // x for x in _list]

print(multiplicate(_list))
