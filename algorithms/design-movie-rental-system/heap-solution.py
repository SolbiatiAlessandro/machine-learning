# TLE
# https://leetcode.com/problems/design-movie-rental-system/

from collections import defaultdict
from dataclasses import dataclass
import heapq

@dataclass
class ShopWithMovie:
    shop_index: int
    movie_index: int
    price: int

    def __lt__(self, other):
        if self.price == other.price:
            return self.shop_index < other.shop_index
        return self.price < other.price


@dataclass
class MovieWithShop:
    shop_index: int
    movie_index: int
    price: int

    def __lt__(self, other):
        if self.price != other.price:
            return self.price < other.price
        if self.shop_index != other.shop_index:
            return self.shop_index < other.shop_index
        return self.movie_index < other.movie_index



class MovieRentingSystem:

    def __init__(self, n: int, entries: List[List[int]]):
        self.shops_number = n

        # prices[shop=3][movie=2] = 3
        self.prices = [{} for _ in range(n)] 

        # available[shop=3][movie=2]
        self.available = [defaultdict(lambda: True) for _ in range(n)]

        # shops_with_available_movie[movie=2] = minheap(shops)
        self.shops_by_movie = defaultdict(list)

        # rented_movies_by_shops[shop=3] = minheap(shops)
        self.movies_by_shops = defaultdict(list)

        for entry in entries:
            shop, movie, price = entry
            self.prices[shop][movie] = price

            heapq.heappush(
                self.shops_by_movie[movie],
                ShopWithMovie(shop_index=shop, movie_index=movie, price=price))
            heapq.heappush(
                self.movies_by_shops[shop],
                MovieWithShop(shop_index=shop, movie_index=movie, price=price)
            )

    def search(self, movie: int) -> List[int]:
        top_shops = []
        popped_shops = []
        while len(top_shops) < 5:
            try:
                next_shop = heapq.heappop(self.shops_by_movie[movie])
                if self.available[next_shop.shop_index][movie]:
                    top_shops.append(next_shop)
                popped_shops.append(next_shop)
            except IndexError:
                break
        for shop in popped_shops:
            heapq.heappush(
                self.shops_by_movie[movie],
                shop
            )
        return [shop.shop_index for shop in top_shops]

    def rent(self, shop: int, movie: int) -> None:
        self.available[shop][movie] = 0
 
    def drop(self, shop: int, movie: int) -> None:
        self.available[shop][movie] = 1
        
    def report(self) -> List[List[int]]:
        res = []
        for shop_index in range(self.shops_number):
            popped_movies = []
            top_5_from_this_shop, empty_heap = [], False
            while not empty_heap and len(top_5_from_this_shop) < 5:
                try:
                    movie: MovieWithShop = heapq.heappop(self.movies_by_shops[shop_index])
                    popped_movies.append(movie)
                    if not self.available[shop_index][movie.movie_index]:
                        top_5_from_this_shop.append(movie)
                except IndexError:
                    empty_heap=True
            res += top_5_from_this_shop
            res = sorted(res)[:5]
            for movie in popped_movies:
                heapq.heappush(self.movies_by_shops[shop_index], movie)
        return [(movie.shop_index, movie.movie_index) for movie in res]
        


# Your MovieRentingSystem object will be instantiated and called as such:
# obj = MovieRentingSystem(n, entries)

# param_1 = obj.search(movie)
# obj.rent(shop,movie)
# obj.drop(shop,movie)
# param_4 = obj.report()
