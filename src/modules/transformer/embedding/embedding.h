#ifndef TK_EMB_H
#define TK_EMB_H

// the most basic structure used by almost all transformer's embedding layer
struct tk_embedding {
    struct tk_tensor* weights;  // points to weight matrix [vocab_size, hidden_dim]
};

#endif
