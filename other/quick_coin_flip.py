import random

class Coin:
    def __init__(self, debug=True):
        self.face = False # Trie is heads, false is tails
        self.chance = 0.60 # % after decimal point

        self.max_flips = 0
        self.counter = 0

        self.sys_rand = random.SystemRandom()
        self.debug = debug

    def flip(self):
        addr = True if self.face else False
        self.face = self.sys_rand.random() < self.chance
        self.counter = self.counter + 1 if addr and self.face else 0
        self.max_flips = self.counter if self.counter > self.max_flips else self.max_flips

        if self.debug: print("heads" if self.face else "tails")


coin = Coin(debug=False)

for i in range(2161):
    coin.flip()

print(coin.max_flips)

