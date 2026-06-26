import torch
from collections import Counter
from tqdm import tqdm
import os
import torch
import re
import sqlparse
from sqlparse.tokens import Number, String
import pandas as pd

class SQLTokenizer:

    def __init__(self, vocab_path="generated_data/final_encoded_data.pt"):
        """
        Load the vocabulary from a saved .pt file.
        vocab_path can be 'final_encoded_data.pt' (the full training data) or
        'tokenizer.pt' (only the token2id/id2token dict).
        """
        data = torch.load(vocab_path, weights_only=False)
        if 'token2id' in data:
            self.token2id = data['token2id']
            self.id2token = data['id2token']
        else:
            # Assume it's the bare dict
            self.token2id = data
            self.id2token = {v: k for k, v in data.items()}
        self.unk_id = self.token2id.get("<UNK>", 2)
        self.sneaky_number_pattern = re.compile(
            r'^[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?$'
        )

    @staticmethod
    def normalize_sql(sql):
        """Lowercase and collapse all whitespace to single spaces."""
        if not isinstance(sql, str):
            return ""
        sql = sql.lower().strip()
        return re.sub(r'\s+', ' ', sql)

    def tokenize_sql(self, sql):

        sql = self.normalize_sql(sql)
        if not sql:
            return []
        parsed = sqlparse.parse(sql)
        if not parsed:
            return []
        tokens = []
        for token in parsed[0].flatten():
            if token.is_whitespace:
                continue
            if token.ttype in Number:
                tokens.append('NUM_LITERAL')
            elif token.ttype in String:
                tokens.append('STR_LITERAL')
            elif token.value.startswith('0x') and len(token.value) > 2:
                tokens.append('HEX_LITERAL')
            else:
                tokens.append(token.value)
        return tokens

    def _fix_sneaky_numbers(self, tokens):
        fixed = []
        for token in tokens:
            if self.sneaky_number_pattern.fullmatch(token) or re.fullmatch(r'^[-+]?\d+$', token):
                fixed.append('NUM_LITERAL')
            else:
                fixed.append(token)
        return fixed

    def encode(self, sql):
        """
        Convert a single raw SQL string into a list of integer token IDs.
        Returns empty list for invalid/empty input.
        """
        tokens = self.tokenize_sql(sql)
        tokens = self._fix_sneaky_numbers(tokens)
        return [self.token2id.get(tok, self.unk_id) for tok in tokens]

    def encode_batch(self, sql_list):

        rows = []
        for sql in sql_list:
            tokens = self.tokenize_sql(sql)
            tokens = self._fix_sneaky_numbers(tokens)
            token_ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
            rows.append({
                'sql': sql,
                'tokens': tokens,
                'token_ids': token_ids,
            })

        df = pd.DataFrame(rows)
        return df, self.token2id, self.id2token



def load_queries_from_folder(folder_path):
    queries = []

    files = sorted(os.listdir(folder_path))

    for file in files:
        if file.endswith(".sql"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                query = f.read().strip()

                if query:
                    queries.append(query)
    return queries

def load_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    statements = content.split("#$@")
    statements = [q.replace("!#@", "").strip() for q in statements if q.strip()]
    return statements


def run_pipeline(file_path):
    statements = load_queries(file_path)
    tokenizer=SQLTokenizer("generated_data/final_encoded_data.pt")
    encoded=tokenizer.encode_batch(statements)
    return encoded

