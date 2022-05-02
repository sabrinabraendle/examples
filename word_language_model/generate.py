###############################################################################
# Language Modeling on Pride and Prejudice
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import torch
import sys

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--input', type=str, required=False,
                    help='input sequence as basis of the generation')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

# The --input argument should be optional and text generation should still work without this argument
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        input_seq = []
        if args.input:
            input_seq = args.input.split(' ')
        else:
            output, hidden = model(input, hidden)

        # Count each word in the prompt as part of the total number of words to be generated
        for i in range(args.words):

            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                # Fill input with word index of input sequence and train model with this sequence
                if len(input_seq) > i:
                    # Make sure that each word in the prompt is part of the vocabulary and give feedback to the user if one of them is not
                    if input_seq[i] not in corpus.dictionary.word2idx:
                        print(f'The word "{input_seq[i]}" is not part of the vocabulary, please choose another input.')
                        sys.exit()
                    # Specify the first word of the sequence as initial model input
                    word_idx = corpus.dictionary.word2idx[input_seq[i]]
                    input.fill_(word_idx)

                # Get next word from weights trained on input sequence
                else:
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                output, hidden = model(input, hidden)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
