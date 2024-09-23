#include "doubly_linked_list.h"
#include <iostream>

DoublyLinkedList* create_list()
{
    DoublyLinkedList* list = new DoublyLinkedList;
    list->head = nullptr;
    list->tail = nullptr;
    return list;
}

void add_node(DoublyLinkedList* list, int value)
{
    Node* newNode = new Node { value, nullptr, nullptr };
    if (list->tail == nullptr) {
        list->head = list->tail = newNode;
    } else {
        list->tail->next = newNode;
        newNode->prev = list->tail;
        list->tail = newNode;
    }
}

void remove_node(DoublyLinkedList* list, int value)
{
    Node* current = list->head;
    while (current != nullptr) {
        if (current->data == value) {
            if (current == list->head) {
                list->head = current->next;
                if (list->head)
                    list->head->prev = nullptr;
            } else if (current == list->tail) {
                list->tail = current->prev;
                if (list->tail)
                    list->tail->next = nullptr;
            } else {
                current->prev->next = current->next;
                current->next->prev = current->prev;
            }
            delete current;
            return;
        }
        current = current->next;
    }
}

void swap_nodes(DoublyLinkedList* list, int value1, int value2)
{
    Node* node1 = nullptr;
    Node* node2 = nullptr;
    Node* current = list->head;

    while (current != nullptr) {
        if (current->data == value1)
            node1 = current;
        if (current->data == value2)
            node2 = current;
        current = current->next;
    }

    if (node1 == nullptr || node2 == nullptr) {
        std::cout << "One or both nodes not found." << std::endl;
        return;
    }

    // Swap data
    int temp = node1->data;
    node1->data = node2->data;
    node2->data = temp;
}

void print_list(DoublyLinkedList* list)
{
    Node* current = list->head;
    while (current != nullptr) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << std::endl;
}

void free_list(DoublyLinkedList* list)
{
    Node* current = list->head;
    while (current != nullptr) {
        Node* next = current->next;
        delete current;
        current = next;
    }
    delete list;
}
