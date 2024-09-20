#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structDef.h" 

void initNode(Node* node) {
    for ( int i = 0 ; i < MAX_CHILD ; ++i ) {
        node->children[i] = NULL;
    }    
    memset(node->bitVal, 0, MAX_HEIGHT);
    node->val = -1;
    node->bitLen = 0;
} 

Node* createNode() {
    Node* node = (Node*)malloc(sizeof(Node));
    initNode(node);
    return node;
}    

Node* searchAndInsNode(char bitStream[], int symbol, Node* curNode) {
    if ((!curNode->children[0] && bitStream[0] == '0') || 
        (!curNode->children[1] && bitStream[0] == '1')) {
        Node* newNode = createNode();
        newNode->bitLen = curNode->bitLen+1;
        strcpy(newNode->bitVal, curNode->bitVal);
        if ( bitStream[0] == '0' ) {
            curNode->children[0] = newNode;
            strcat(newNode->bitVal, "0");
        }    
        else if ( bitStream[0] == '1' ) {
            curNode->children[1] = newNode;
            strcat(newNode->bitVal, "1");
        }    
        return searchAndInsNode(bitStream+1, symbol, newNode);
    }
    else if (!curNode->children[0] && !curNode->children[1] && bitStream[0] == '\0') {
        curNode->val = symbol;
        return curNode;
    }    
    else {
        switch(bitStream[0]) {
            case '0':
                return searchAndInsNode(bitStream+1, symbol, curNode->children[0]);
            case '1':
                return searchAndInsNode(bitStream+1, symbol, curNode->children[1]);
            default:
                return NULL;
        }    
    }    
}    


// return root
Node* buildTrie(HufTable table[], size_t tableSize) {
    Node* root = createNode();
    for ( size_t i = 0 ; i < tableSize ; ++i ) {
        /*
        printf("%d\n", i);
        printf("table[%d]\n", i);
        printf("table[%d].val: %d\n", i, table[i].val);
        printf("table[%d].bitLen: %d\n", i, table[i].bitLen);
        printf("table[%d].bitStr: %s\n", i, table[i].bitStr);
        */
        searchAndInsNode(table[i].bitStr, table[i].val, root); 
    }
    return root;
}    

void printTrie(Node* root) {
    if ( root == NULL )  return;
    else {
        printf("val: %d\n", root->val);
        printf("bitLen: %d\n", root->bitLen);
        printf("bitVal: %s\n", root->bitVal);
        for ( int i = 0 ; i < MAX_CHILD ; ++i ) {
            printTrie(root->children[i]);
        }    
    }    
}    

void freeTrie(Node* root) {
    if ( root == NULL )  return;
    else {
        for ( int i = 0 ; i < MAX_CHILD ; ++i ) {
             freeTrie(root->children[i]);
        }    
        free(root);
    }    
}    
