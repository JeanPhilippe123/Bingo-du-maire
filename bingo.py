import numpy as np
from collections import Counter

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
            print(f'Bingo! Round {round_number}. Numéro de série {self.serial}.')


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
    print("Ok c'est parti pour la ronde 1!")

    #Get file with cards, populate CardStack.
    with open("cards.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            numbers_list = [int(string) for string in line.rstrip().split(' ')]
            arr = np.array(numbers_list[:-1]).reshape(5,5)
            serial = numbers_list[-1]
            card = Card(arr,serial)

    #Draw numbers until bingo.
    while True:
        print("Entrer le prochain numéro tiré:")
        number_drawn = int(input())
        for card in Card:
            card.draw(number_drawn)
            card.check_for_bingo(round_number)
