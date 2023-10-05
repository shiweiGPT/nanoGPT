from tokenizer import Tokenizer

raw_text = "'antsi w0z h'api_:_: and Si: \n f'aUnd a# p'i:sf@L kw'aI@t sp'0t t@ r'Est \n"
tokenizer_path = './data/tinystory/tok2048.model'
tokenizer_instance = Tokenizer(tokenizer_path)

tokens = tokenizer_instance.encode(raw_text, bos=True, eos=False)
print(tokens)
decode_text = tokenizer_instance.decode(tokens)
print(decode_text)