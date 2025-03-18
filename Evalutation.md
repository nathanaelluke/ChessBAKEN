# Evaluation of a Position
### Resources:
- [Chess Programming Wiki](https://www.chessprogramming.org/Evaluation)
- [Implementation of evaluation function](https://www.chessprogramming.org/CPW-Engine_eval)
- [Easier implementation of evaluation function](https://www.chessprogramming.org/Evaluation_Function_Draft)
- [Transportation table](https://www.chessprogramming.org/Transposition_Table) [Evaluation table](https://www.chessprogramming.org/Evaluation_Hash_Table#Forum_Posts)
- [Various evaluation functions](https://www.chessprogramming.org/Evaluation_Overlap)
- [Lazy evaluation](https://www.chessprogramming.org/Lazy_Evaluation)
- [Very basic evaluation](https://www.chessprogramming.org/Simplified_Evaluation_Function)
- [Piece-Square Tables Only evaluation](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function)
### What is a winning position?
- Development
- Threats
- Tactics (pins, discovered attacks, forks, etc.)
- [Material](https://www.chessprogramming.org/Material "Material")
- [Piece-Square Tables](https://www.chessprogramming.org/Piece-Square_Tables "Piece-Square Tables")
- [Pawn Structure](https://www.chessprogramming.org/Pawn_Structure "Pawn Structure")
- [Evaluation of Pieces](https://www.chessprogramming.org/Evaluation_of_Pieces "Evaluation of Pieces")
- [Evaluation Patterns](https://www.chessprogramming.org/Evaluation_Patterns "Evaluation Patterns")
- [Mobility](https://www.chessprogramming.org/Mobility "Mobility")
- [Center Control](https://www.chessprogramming.org/Center_Control "Center Control")
- [Connectivity](https://www.chessprogramming.org/Connectivity "Connectivity")
- [Trapped Pieces](https://www.chessprogramming.org/Trapped_Pieces "Trapped Pieces")
- [King Safety](https://www.chessprogramming.org/King_Safety "King Safety")
- [Space](https://www.chessprogramming.org/Space "Space")
- [Tempo](https://www.chessprogramming.org/Tempo "Tempo")

### We want to find the probability that a given board position is "winning" for each side.
Ways to determine who is winning:
- Use the information above as features to input into a neural network.
	- We could have the neural network return a number in some range (maybe -10 to 10) to represent if Black or White is winning a position
	- Would be a black box so it would be hard to determine why it is or isn't working
- We could also apply a weight to each feature and sum them to form a probability of winning in a position.
	- It would be clearer what caused a position to be seen a winning
	- Might take a while to play with weights to find a "good" weighted sum

### Another point to consider (if we have enough time) is what "phase" of the game we are in
- Early game
	- Very little development
	- Can rely on book moves
- Middle game
	- More opportunities to make mistakes and capitalize on them
	- More developed pieces
	- King safety is key
- Endgame
	- Less moves
	- Look deeper
	- Pawn moves can be very important

