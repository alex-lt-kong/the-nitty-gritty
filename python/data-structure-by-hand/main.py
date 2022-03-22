# * It is possible to get the address of an object so that it is still possible to construct data structures, such as linked list or tree, by hand.
# * The memory addresses of a linked list, as expected, are not continuous.
# * If you directly print the object, you get a string which is generated from the repr() method, which is an "official" string representation of an object.
# * Even though one cannot find the address of the 2nd element of a list and cast the address back to a variable, one can do this to a traditional object.
#   (For whatever reasons, the value can only be accessed with .val.val ¯\_(ツ)_/¯



import ctypes

class node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next

def construct_list(arr):
    
    root, prev_node, curr_node = None, None, None
    
    for i in range(len(arr)):
        curr_node = node(arr[i])
        if i == 0:
            root = curr_node
        if i > 0:
            prev_node.next = curr_node
        prev_node = curr_node
        
    return root



arr = [1, 1, 2, 3, 5, 8]
root = construct_list(arr)
while root is not None:
    addr = id(root)
    print(f'value: {root.val}\naddr(dec): {addr}, addr(hex): {hex(addr)}, repr: {root}')
    print('From address to value: {}\n'.format(node(ctypes.cast(addr, ctypes.py_object).value).val.val))
    root = root.next    

