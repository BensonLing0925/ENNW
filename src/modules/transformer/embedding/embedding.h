#ifndef TK_EMB_H
#define TK_EMB_H

struct tk_emb_config {
    int vocab_size;
    int hidden_dim;
    int max_seq_len;
	int use_ln;
};

// the most basic structure used by almost all transformer's embedding layer
struct tk_embedding {
    struct tk_tensor* weights;  // points to weight matrix [vocab_size, hidden_dim]
};

struct tk_emb_block {
	struct tk_emb_config config;
    struct tk_embedding* word_emb;
    struct tk_embedding* pos_emb;
    struct tk_tensor* ln_gamma;
    struct tk_tensor* ln_beta;
	
	// temporary
	struct tk_tensor* out_buf;
};


#endif
