#ifdef __cplusplus
extern "C" {
#endif

struct Node {
    int data;
    Node* prev;
    Node* next;
};

struct DoublyLinkedList {
    Node* head;
    Node* tail;
};

DoublyLinkedList* create_list();
void add_node(DoublyLinkedList* list, int value);
void remove_node(DoublyLinkedList* list, int value);
void swap_nodes(DoublyLinkedList* list, int value1, int value2);
void print_list(DoublyLinkedList* list);
void free_list(DoublyLinkedList* list);

#ifdef __cplusplus
}
#endif
