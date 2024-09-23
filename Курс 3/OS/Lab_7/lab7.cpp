#include <dlfcn.h>
#include <iostream>
#include <unistd.h>

void print_maps()
{
    char command[256];
    snprintf(command, sizeof(command), "cat /proc/%d/maps", getpid());
    system(command);
}

int main()
{
    std::cout << "Before loading the library:" << std::endl;
    print_maps();

    void* handle = dlopen("./libdoubly_linked_list.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "\nAfter loading the library:" << std::endl;
    print_maps();

    // Получение указателей на функции
    typedef struct DoublyLinkedList* (*create_list_func)();
    typedef void (*add_node_func)(struct DoublyLinkedList*, int);
    typedef void (*remove_node_func)(struct DoublyLinkedList*, int);
    typedef void (*swap_nodes_func)(struct DoublyLinkedList*, int, int);
    typedef void (*print_list_func)(struct DoublyLinkedList*);
    typedef void (*free_list_func)(struct DoublyLinkedList*);

    create_list_func create_list = (create_list_func)dlsym(handle, "create_list");
    add_node_func add_node = (add_node_func)dlsym(handle, "add_node");
    remove_node_func remove_node = (remove_node_func)dlsym(handle, "remove_node");
    swap_nodes_func swap_nodes = (swap_nodes_func)dlsym(handle, "swap_nodes");
    print_list_func print_list = (print_list_func)dlsym(handle, "print_list");
    free_list_func free_list = (free_list_func)dlsym(handle, "free_list");

    // Проверка наличия всех функций
    if (!create_list || !add_node || !remove_node || !swap_nodes || !print_list || !free_list) {
        std::cerr << "Error locating one or more symbols: " << dlerror()
                  << std::endl;
        dlclose(handle);
        return 1;
    }

    DoublyLinkedList* list = create_list();
    add_node(list, 10);
    add_node(list, 20);
    add_node(list, 30);
    add_node(list, 40);

    std::cout << "\nInitial list: ";
    print_list(list);

    remove_node(list, 20);
    std::cout << "\nAfter removing 20: ";
    print_list(list);

    swap_nodes(list, 10, 40);
    std::cout << "\nAfter swapping 10 and 40: ";
    print_list(list);

    free_list(list);

    dlclose(handle);

    std::cout << "\nAfter unloading the library:" << std::endl;
    print_maps();

    return 0;
}
