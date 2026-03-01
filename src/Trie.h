#ifdef TRIE_H
#define TRIE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structDef.h" 

void initNode(Node* node);
Node* createNode();
Node* searchAndInsNode(char bitStream[], int symbol, Node* curNode);
Node* buildTrie(HufTable table[], size_t tableSize);
void printTrie(Node* root);
void freeTrie(Node* root);
#endif
