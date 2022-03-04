from copy import deepcopy


class Strategy:
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
        self.idxs_lb = idxs_lb
        self.device = device
        self.model = model
        self.args = args
        self.n_pool = len(Y)

        self.writer = writer

        self.query_count = 0

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def train(self, name):
        self.model.train(name, self.X, self.Y, self.idxs_lb, self.X_val, self.Y_val)

    def predict(self, X, Y):
        return self.model.predict(X, Y)

    def predict_prob(self, X, Y):
        return self.model.predict_prob(X, Y)

    def predict_prob_embed(self, X, Y, eval=True):
        return self.model.predict_prob_embed(X, Y, eval)

    def predict_all_representations(self, X, Y):
        return self.model.predict_all_representations(X, Y)

    def predict_embedding_prob(self, X_embedding):
        return self.model.predict_embedding_prob(X_embedding)

    def predict_prob_dropout(self, X, Y, n_drop):
        return self.model.predict_prob_dropout(X, Y, n_drop)

    def predict_prob_dropout_split(self, X, Y, n_drop):
        return self.model.predict_prob_dropout_split(X, Y, n_drop)

    def predict_prob_embed_dropout_split(self, X, Y, n_drop):
        return self.model.predict_prob_embed_dropout_split(X, Y, n_drop)

    def get_embedding(self, X, Y):
        return self.model.get_embedding(X, Y)

    def get_grad_embedding(self, X, Y, is_embedding=False):
        return self.model.get_grad_embedding(X, Y, is_embedding)
