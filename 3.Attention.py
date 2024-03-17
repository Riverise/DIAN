import numpy as np


def softmax(X, axis):
    X_exp = np.exp(X)
    # 对列进行softmax
    partition = np.sum(X_exp, axis=axis, keepdims=True)  # 保持shape不变
    return X_exp / partition  # 利用广播机制


def attention(Q, K, V):
    # Q/K/V shape:(batch_size, seq_length, hidden_dim)
    # scaled_dot_product
    K_T = K.transpose(0, 2, 1)
    result = Q @ K_T  # (batch_size, seq_length, seq_length)
    return softmax(result / np.sqrt(K.shape[2]), -1) @ V
    # (batch_size, seq_length, hidden_dim)


class MultiHeadAttention:
    def __init__(self, embed_dim, hidden_dim, head_num):
        # input_shape: (batch_size, seq_length, embed_dim)。
        assert hidden_dim % head_num == 0, "Input dim mismatch"
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.subdim = int(self.hidden_dim / self.head_num)

        # network parameter to be learned
        self.W_Q = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_K = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_V = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_O = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.W_Q_is = []
        self.W_K_is = []
        self.W_V_is = []
        for i in range(self.head_num):
            self.W_Q_is.append(np.random.randn(self.hidden_dim, self.subdim))
            self.W_K_is.append(np.random.randn(self.hidden_dim, self.subdim))
            self.W_V_is.append(np.random.randn(self.hidden_dim, self.subdim))

    def forward(self, x):
        # input_shape: (batch_size, seq_length, embed_dim)
        Q = x @ self.W_Q  # Q_shape:(batch_size, seq_length, hidden_dim)
        K = x @ self.W_K
        V = x @ self.W_V

        heads = []

        for i in range(self.head_num):
            Q_i = Q @ self.W_Q_is[i]  # Q_i_shape: (batch_size, seq_length, subdim)
            K_i = K @ self.W_K_is[i]
            V_i = V @ self.W_V_is[i]

            head_i = attention(Q_i, K_i, V_i)  # (batch_size, seq_length, subdim)
            heads.append(head_i)

        head_merged = np.concatenate(heads, axis=2)  # (batch_size, seq_length, hidden_dim == head_num*subdim)
        outputs = head_merged @ self.W_O
        return outputs  # (batch_size, seq_length, hidden_dim)


class MultiQueryAttention:
    def __init__(self, embed_dim, hidden_dim, head_num):
        # input_shape: (batch_size, seq_length, embed_dim)。
        assert hidden_dim % head_num == 0, "Input dim mismatch"
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.subdim = int(self.hidden_dim / self.head_num)
        # network parameter to be learned
        self.W_Q = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_K = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_V = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_O = np.random.randn(self.hidden_dim, self.hidden_dim)

        self.W_Q_is = []
        self.W_K_i = np.random.randn(self.hidden_dim, self.subdim)
        self.W_V_i = np.random.randn(self.hidden_dim, self.subdim)
        for i in range(self.head_num):
            self.W_Q_is.append(np.random.randn(self.hidden_dim, self.subdim))

    def forward(self, x):
        # input_shape: (batch_size, seq_length, embed_dim)
        Q = x @ self.W_Q  # Q_shape:(batch_size, seq_length, hidden_dim)
        K = x @ self.W_K
        V = x @ self.W_V

        heads = []
        K_i = K @ self.W_K_i
        V_i = V @ self.W_V_i
        for i in range(self.head_num):
            Q_i = Q @ self.W_Q_is[i]  # Q_i_shape: (batch_size, seq_length, subdim)
            head_i = attention(Q_i, K_i, V_i)  # (batch_size, seq_length, subdim)
            heads.append(head_i)

        head_merged = np.concatenate(heads, axis=2)  # (batch_size, seq_length, hidden_dim == head_num*subdim)
        outputs = head_merged @ self.W_O
        return outputs  # (batch_size, seq_length, hidden_dim)


class GroupQueryAttention:
    def __init__(self, embed_dim, hidden_dim, head_num, group_num):
        # input_shape: (batch_size, seq_length, embed_dim)。
        assert hidden_dim % head_num == 0, "Input dim mismatch"
        assert head_num % group_num == 0, "num of groups and heads mismatch"

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.group_num = group_num
        self.subdim = int(self.hidden_dim / self.head_num)
        self.shared_num = int(self.head_num / self.group_num)
        # network parameter to be learned
        self.W_Q = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_K = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_V = np.random.randn(self.embed_dim, self.hidden_dim)
        self.W_O = np.random.randn(self.hidden_dim, self.hidden_dim)

        self.W_Q_is = []
        self.W_K_is = []
        self.W_V_is = []
        for i in range(self.head_num):
            self.W_Q_is.append(np.random.randn(self.hidden_dim, self.subdim))
            if (i + 1) % self.shared_num == 0:  # 假设head_num / group_num == 4, 则每逢四次， 计算一个K、V矩阵，让4个Q共享
                self.W_K_is.append(np.random.randn(self.hidden_dim, self.subdim))
                self.W_V_is.append(np.random.randn(self.hidden_dim, self.subdim))

    def forward(self, x):
        # input_shape: (batch_size, seq_length, embed_dim)
        Q = x @ self.W_Q  # Q_shape:(batch_size, seq_length, hidden_dim)
        K = x @ self.W_K
        V = x @ self.W_V

        heads = []
        for i in range(self.head_num):
            j = i // self.shared_num
            Q_i = Q @ self.W_Q_is[i]  # Q_i_shape: (batch_size, seq_length, subdim)
            K_i = K @ self.W_K_is[j]
            V_i = V @ self.W_V_is[j]
            head_i = attention(Q_i, K_i, V_i)  # (batch_size, seq_length, subdim)
            heads.append(head_i)

        head_merged = np.concatenate(heads, axis=2)  # (batch_size, seq_length, hidden_dim == head_num*subdim)
        outputs = head_merged @ self.W_O
        return outputs  # (batch_size, seq_length, hidden_dim)


mha = MultiHeadAttention(2, 6, 2)
mqa = MultiQueryAttention(2, 6, 2)
gqa = GroupQueryAttention(2, 16, 8, 2)
X = np.random.randn(1, 5, 2)

print(X)
print(mha.forward(X))
print(mqa.forward(X))
print(gqa.forward(X))
