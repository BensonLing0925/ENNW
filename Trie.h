#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structDef.h" 

void initNode(Node* node) {
    for ( int i = 0 ; i < MAX_CHILD ; ++i ) {
        node->children[i] = NULL;
    }    
    memset(node->val, 0, MAX_HEIGHT);
} 

Node* createNode() {
    Node* node = (Node*)malloc(sizeof(Node));
    initNode(node);
    return node;
}    

Node* searchNode(char bitStream[], int bitLen, Node* curNode) {
    if (!curNode) return NULL;
    else if (bitStream[0] == '\0') return curNode;
    else {
        switch(bitStream[0]) {
            case '0':
                searchNode(bitStream+1, curNode->children[0]);
                break;
            case '1':
                searchNode(bitStream+1, curNode->children[1]);
                break;
            default:
                return NULL;
        }    
        
    }    
}    

void insertNode(Node* root) {
    // root case
    Node* node = createNode();
    if (root == NULL) {
        root = node;
    }    
    else {
         

    }    
}    

void buildTree() {



}    
