#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
run
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-1下午7:58
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from config import *
from Encoder.EncoderTutorial import EncoderRNN
from Decoder.AttnDecoderTutorial import AttnDecoderRNN

def main():
    # input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    input_lang, output_lang, pairs = load_obj_from_pickle(PROCESSED_DATA_PATH)
    print(random.choice(pairs))

    hidden_size = 256
    embedding_size = 256

    encoder = EncoderRNN(input_lang.n_words, embedding_size, hidden_size)
    attn_decoder = AttnDecoderRNN(output_lang.n_words, embedding_size, hidden_size,
                                  max_length=MAX_LENGTH, n_layers=1, dropout_p=0.1)

    if use_cuda:
        encoder = encoder.cuda()
        attn_decoder = attn_decoder.cuda()

    # train_iters(encoder, attn_decoder, n_iters=750, pairs=pairs,
    #             input_lang=input_lang, output_lang=output_lang,
    #             max_length=MAX_LENGTH, print_every=50)
    train_iters(encoder, attn_decoder, n_iters=75000, pairs=pairs,
                input_lang=input_lang, output_lang=output_lang,
                max_length=MAX_LENGTH, print_every=5000)

    torch.save(encoder.state_dict(), './models/encoder.params.pkl')
    torch.save(attn_decoder.state_dict(), './models/attn_decoder.params.pkl')
    print('model saved successfully!')

    evaluate_randomly(encoder, attn_decoder, input_lang, output_lang, pairs, MAX_LENGTH)

    output_words, attentions = evaluate(encoder, attn_decoder,
                                        input_lang, output_lang, "je suis trop froid .",
                                        max_length=MAX_LENGTH)
    plt.matshow(attentions.numpy())

    evaluate_and_show_attention(encoder, attn_decoder, input_lang, output_lang,
                                "elle a cinq ans de moins que moi .", MAX_LENGTH)

    evaluate_and_show_attention(encoder, attn_decoder, input_lang, output_lang,
                                "elle est trop petit .", MAX_LENGTH)

    evaluate_and_show_attention(encoder, attn_decoder, input_lang, output_lang,
                                "je ne crains pas de mourir .", MAX_LENGTH)

    evaluate_and_show_attention(encoder, attn_decoder, input_lang, output_lang,
                                "c est un jeune directeur plein de talent .", MAX_LENGTH)


if __name__ == "__main__":
    main()