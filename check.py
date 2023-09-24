import torch
from torch import nn
from torch.nn import functional as F


# Я решил использовать модель на основе Transformer
# Плюсы трансформера в сравнении с рекурентными сетями это отсутствие взрывные/исчезающих градиентов
class PasswordAI(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 nhead: int,
                 max_seq_length: int) -> None:
        super().__init__()
        self._positional_encoding(embedding_dim, max_seq_length)
        self.properties = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "nhead": nhead,
            "max_seq_length": max_seq_length,
        }
        # Вложения слов
        self.emb = nn.Embedding(num_embeddings,
                                embedding_dim,
                                padding_idx=0)
        # Слой энкодера
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            batch_first=True,
        )
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 3), nn.LogSoftmax(-1),
        )

    def forward(self, passwords) -> torch.Tensor:
        x = self.emb(passwords) + self.pe
        out = self.transformer(x)[:, -1, :]

        return self.classifier(out)

    def _positional_encoding(self, embedding_dim: int, max_seq_length: int) -> None:
        # Обучаемые позиционные кодировки для вложений слов
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(9.21 / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe)

    def save(self, patch: str):
        torch.save([self.properties, self.state_dict()], patch)

        return self

    @staticmethod
    def load(patch: str):
        properties, parameters = torch.load(patch)
        model = PasswordAI(**properties)
        model.load_state_dict(parameters)

        return model

class Vocab:
    def __init__(self, data: list = None) -> None:
        self.tokens = {"<unk>": 0}
        if data is not None:
            self.addNewTokens(data)

    def __call__(self, tokens: list[list[str]]) -> torch.Tensor:
        replace = lambda token: self[token]

        for index, word in enumerate(tokens):
            word = list(map(replace, word)) # Заменить каждый токен-сомвол на его числовую форму
            tokens[index] = torch.tensor(word, dtype=torch.long)

        # Заполнить каждый вектор с числовым представлением нулями, неизвестный токен == 0.
        # Длинна заполнения == самый длинный пароль
        return nn.utils.rnn.pad_sequence(tokens, padding_value=0)

    def __getitem__(self, token: str) -> int:
        if token not in self.tokens:
            return 0

        return self.tokens[token]

    def addNewTokens(self, tokens: list[list[str]]):
        unique_tokens = set(merged_list(tokens)) # Уникальные токены
        for token in unique_tokens:
            # Добавляем каждый уникальный, ещё не добавленый, токен в словарь
            if token in self.tokens:
                continue
            self.tokens[token] = len(self.tokens)

        return self

    def save(self, patch: str):
        torch.save(self.tokens, patch)

        return self

    @staticmethod
    def load(patch: str):
        tokens = torch.load(patch)
        vocab = Vocab()
        vocab.tokens = tokens

        return vocab

# Разделение слов-паролей на токены-символы
def tokenizer(words: list[str]) -> list[list[str]]:
    for index, word in enumerate(words):
        words[index] = list(word)

    return words

# Перевод из 3 классов надёжности в одно непрерывное число
def class_to_reliability(classes: torch.Tensor) -> float:
    device = classes.device
    weight = torch.tensor([0.0, 0.5, 1.0]).to(device)
    return (classes * weight).sum(-1).item()

# Показать в удобном виде надёжность пароля
def print_reliability(strength: float) -> None:
    txt = "Надёжность пароля - "
    if strength < 0.5: txt += "слабый"
    elif strength >= 0.5 and strength <= 0.75: txt += "средний"
    else: txt += "сильный"

    x = int(20 * strength)
    scale = "#" * x + "~" * (20 - x)
    print(txt)
    print(scale + f" ({strength*100:.1f}%)\n")

# Функция для проверки своего пароля на надёжность
def check_password(password: str,
                   model: PasswordAI,
                   vocab: Vocab,
                   device: str = "cpu") -> float:
    model.eval().to(device)
    max_len = model.properties["max_seq_length"]
    password = vocab(tokenizer([password]))[:max_len]
    if password.size(0) < max_len:
        zero = torch.zeros(max_len-password.size(0), 1, dtype=torch.long)
        password = torch.cat([password, zero], 0)

    password = password.transpose(1, 0).to(device)
    predict = F.softmax(model(password)[0], -1)

    strength = class_to_reliability(predict)
    print_reliability(strength)

    return strength


if __name__ == '__main__':
    path_model = input("Введите путь до обученной модели: ")
    path_vocab = input("Введите путь до словаря: ")

    model = PasswordAI.load(path_model)
    vocab = Vocab.load(path_vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        password = input("Введите пароль для проверки: ")
        check_password(password, model, vocab, device)


