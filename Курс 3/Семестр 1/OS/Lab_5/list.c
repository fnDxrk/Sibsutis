#include "list.h"
#include <stdio.h>
#include <stdlib.h>

Node*
insertAtHead(Node* head, int data)
{
  Node* newNode = (Node*)malloc(sizeof(Node));
  newNode->data = data;
  newNode->prev = NULL;
  newNode->next = head;

  if (head != NULL) {
    head->prev = newNode;
  }
  return newNode;
}

Node*
insertAtTail(Node* head, int data)
{
  Node* newNode = (Node*)malloc(sizeof(Node));
  newNode->data = data;
  newNode->next = NULL;

  if (head == NULL) {
    newNode->prev = NULL;
    return newNode;
  }

  Node* temp = head;
  while (temp->next != NULL) {
    temp = temp->next;
  }
  temp->next = newNode;
  newNode->prev = temp;

  return head;
}

Node*
deleteNode(Node* head, int data)
{
  Node* temp = head;

  while (temp != NULL) {
    if (temp->data == data) {
      if (temp->prev != NULL) {
        temp->prev->next = temp->next;
      }
      if (temp->next != NULL) {
        temp->next->prev = temp->prev;
      }
      if (temp == head) {
        head = temp->next;
      }
      free(temp);
      return head;
    }
    temp = temp->next;
  }
  return head;
}

void
printList(Node* head)
{
  Node* temp = head;
  while (temp != NULL) {
    printf("%d ", temp->data);
    temp = temp->next;
  }
  printf("\n");
}

void
serializeList(Node* head, const char* filename)
{
  FILE* file = fopen(filename, "wb");
  if (file == NULL) {
    perror("Failed to open file");
    return;
  }

  Node* temp = head;
  while (temp != NULL) {
    fwrite(&(temp->data), sizeof(int), 1, file);
    temp = temp->next;
  }

  fclose(file);
}

Node*
deserializeList(const char* filename)
{
  FILE* file = fopen(filename, "rb");
  if (file == NULL) {
    perror("Failed to open file");
    return NULL;
  }

  Node* head = NULL;
  Node* tail = NULL;
  int data;

  while (fread(&data, sizeof(int), 1, file)) {
    if (head == NULL) {
      head = (Node*)malloc(sizeof(Node));
      head->data = data;
      head->prev = NULL;
      head->next = NULL;
      tail = head;
    } else {
      Node* newNode = (Node*)malloc(sizeof(Node));
      newNode->data = data;
      newNode->next = NULL;
      newNode->prev = tail;
      tail->next = newNode;
      tail = newNode;
    }
  }

  fclose(file);
  return head;
}

void
freeList(Node* head)
{
  Node* temp;
  while (head != NULL) {
    temp = head;
    head = head->next;
    free(temp);
  }
}
