# Chessformer

The goal of this project is to predict the result of a chess game and the Elo
ratings of the players involved solely from the sequence of moves. No additional
information, like captures, checks, or checkmate, is provided. Only the piece
that was moved, from which square, and to which square. The architectures of the
models are based on the encoder section of a Transformer. For more details,
including results, see the [report](report/chessformer_report.pdf).

## Quick start

Clone the repository and install the necessary Python packages using the
following command:

```sh
pip install -r requirements.txt
```
