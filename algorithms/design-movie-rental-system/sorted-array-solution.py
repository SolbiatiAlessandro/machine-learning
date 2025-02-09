# ACCEPTED 
# https://leetcode.com/problems/design-movie-rental-system/


from dataclasses import dataclass


class SortedList:
    def __init__(self):
        self.array = []

    def insert(self, x):
        l, r = 0, len(self.array) 
        while l < r:
            m = (l + r) // 2
            if x > self.array[m]: l = m + 1 
            else: r = m 
        self.array.insert(l, x)
            
    def remove(self, x):
        l, r = 0, len(self.array) 
        while l < r: 
            m = (l + r) // 2 
            if x > self.array[m] : l = m + 1 
            else: r = m
        if 0 <= l and l < len(self.array) and x == self.array[l]:
            self.array.pop(l)
            
@dataclass
class MovieAtShop:
    shop_index: int
    movie_index: int
    movie_price: int

    def __lt__(self, other):
        if self.movie_price != other.movie_price:
            return self.movie_price < other.movie_price
        if self.shop_index != other.shop_index:
            return self.shop_index < other.shop_index
        return self.movie_index < other.movie_index 

@dataclass
class ShopWithMovie:
    shop_index: int
    movie_index: int
    movie_price: int

    def __lt__(self, other):
        if self.movie_price != other.movie_price:
            return self.movie_price < other.movie_price
        return self.shop_index < other.shop_index

class MovieRentingSystem:

    def __init__(self, n: int, entries: List[List[int]]):
        # cheapest_shops_w_availability = sorted list of cheapest shops with availability per movie
        # cheapest_rented_movies = sorted list of cheapest rented rented movie
        
        # drop() for each entry - O(NlogN)

        self.cheapest_shops_w_availability = defaultdict(SortedList)
        self.cheapest_rented_movies = SortedList()
        self.shops_number = n
        self.prices = {k: defaultdict(int) for k in range(self.shops_number)}

        for (shop, movie, price) in entries:
            self.cheapest_shops_w_availability[movie].insert(ShopWithMovie(shop, movie, price))
            self.prices[shop][movie] = price

    def search(self, movie: int) -> List[int]:
        return [shop.shop_index for shop in self.cheapest_shops_w_availability[movie].array[:5]]
        

    def rent(self, shop: int, movie: int) -> None:
        self.cheapest_shops_w_availability[movie].remove(ShopWithMovie(shop, movie, self.prices[shop][movie]))
        movie = MovieAtShop(shop, movie, self.prices[shop][movie])
        self.cheapest_rented_movies.insert(movie)
    

    def drop(self, shop: int, movie: int) -> None:
        self.cheapest_shops_w_availability[movie].insert(ShopWithMovie(shop, movie, self.prices[shop][movie]))
        self.cheapest_rented_movies.remove(MovieAtShop(shop, movie, self.prices[shop][movie]))


    def report(self) -> List[List[int]]:
        return [[movie.shop_index, movie.movie_index] for movie in self.cheapest_rented_movies.array[:5]]
        


# Your MovieRentingSystem object will be instantiated and called as such:
# obj = MovieRentingSystem(n, entries)
# param_1 = obj.search(movie)
# obj.rent(shop,movie)
# obj.drop(shop,movie)
# param_4 = obj.report()
