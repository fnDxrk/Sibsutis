#ifndef LIST_H
#define LIST_H

typedef struct Node
{
  int data;
  struct Node* prev;
  struct Node* next;
} Node;

Node*
insertAtHead(Node* head, int data);
Node*
insertAtTail(Node* head, int data);
Node*
deleteNode(Node* head, int data);
void
printList(Node* head);
void
serializeList(Node* head, const char* filename);
Node*
deserializeList(const char* filename);
void
freeList(Node* head);

#endif
