import numpy as np
from collections import Counter
from PIL import Image
from glob import glob

class CardStack(type):
    def __iter__(cls):
        return iter(cls._allCards)


class Card(metaclass=CardStack):
    _allCards = []

    def __init__(self,grid,serial):
        self._allCards.append(self)

        self.grid = grid
        self.serial = serial
        self.tokens = {'22'}

    def check_for_bingo(self,round_number):
        position_dict = {1:{'00','11','22','32','42','13','04'},
                         2:{'00','11','22','33','44','40','31','13','04'},
                         3:{'00','01','02','03','04','12','22','32','42'},
                         4:{str(i)+str(j) for i in range(0,5) for j in range(0,5)}}
        if position_dict[round_number].issubset(self.tokens):
            print(f"Bingo! Round {round_number}. Carte numéro {self.serial[0]} dans l'image qui va s'ouvrir.")
            print(self.serial[2])
            self.serial[1].show()

    def draw(self,number):
        position = np.where(self.grid==number)
        if position[0].size==0:
            return
        else:
            position_str = str(position[0][0])+str(position[1][0])
            self.tokens.add(position_str)


if __name__ == "__main__":
    #Welcome players and ask for the current round's number.
    print(r"C'est l'heure de jouer aux bingo!!! À quelle ronde sommes-nous rendus?")
    round_number = int(input())
    print(f"Ok c'est parti pour la ronde {round_number}!")

    #Get file with cards, populate CardStack.
    for file in glob(r"Fichiers_npy\*.npy"):
        four_cards = np.load(file, allow_pickle=True)
        i = 1
        for grid in four_cards[0]:
            serial = [i,four_cards[1],file]
            card = Card(grid,serial)
            i += 1

    #Draw numbers until bingo.
    while True:
        print("Entrer le prochain numéro tiré:")
        number_drawn = int(input())
        for card in Card:
            card.draw(number_drawn)
            card.check_for_bingo(round_number)
