#include "list.h"
#include <stdio.h>

int
main()
{
  Node* list = NULL;

  list = insertAtHead(list, 1);
  list = insertAtTail(list, 2);
  list = insertAtTail(list, 3);

  printf("Лист : ");
  printList(list);

  serializeList(list, "list.bin");

  list = deleteNode(list, 2);

  printf("Лист после удаления элемента : ");
  printList(list);

  Node* deserializedList = deserializeList("list.bin");
  printf("Восстановленный список : ");
  printList(deserializedList);

  freeList(list);
  freeList(deserializedList);

  return 0;
}
