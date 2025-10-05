<p align="center">
  <img src="Images/BaconHorse.webp" width="150" height="150" />
</p>

<h1 align="center">♞ ChessBAKEN 🥓</h1>

<p align="center">
  A Python-powered chess engine inspired by AlphaGo
  <br />
</p>

#### 

## About

ChessBAKEN is a chess engine built in Python. It is compatible with UCI GUIs and uses `python-chess` for move validation and game logic. 

## Features

- 🕸️ Neural net powered move selector and position evaluator  

- ✅ UCI compatible

- ♟️ Plays legal chess moves  

## Dependencies
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install python-chess
pip install numpy
pip install PyInstaller
```

## Installation 
```sh
git clone https://github.com/averyrair/ChessBAKEN
cd ChessBAKEN
python -m PyInstaller --onefile BAKEN.py
cp dist/BAKEN .
```
Import BAKEN binary as an engine in a UCI compatible GUI or run from the command line! 😀
```sh
./BAKEN
```

## Recommended UCI GUI
[Arena Chess](http://www.playwitharena.de/downloads/arena_3.5.1.zip)

## Games played by v0.1.3 against Chess.com's Martin Bot
### Victory for Martin (White) against BAKEN (Black)
![Chess Board](/Images/Martin%20vs%20BAKEN%20(Game%201).gif)
### Draw between Martin (Black) and BAKEN (White)
![Chess Board](/Images/Martin%20vs%20BAKEN%20(Game%202).gif)

## Lichess Account
https://lichess.org/@/ChessBAKEN

Instructions for setting up lichess-bot: https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install
### Instructions to play against ChessBAKEN on Lichess
1. Navigate to your lichess-bot directory
2. Start up the python virtual environment: `source ./venv/bin/activate`
3. Start the bot: `python3 lichess-bot.py`
4. Log in to another Lichess account
5. Go to the ChessBAKEN account and challenge it to a game
