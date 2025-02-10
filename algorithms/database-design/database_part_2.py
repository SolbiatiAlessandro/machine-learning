# primary_key is id column

# Q: data types?
# Q: machines? one machine 10GB

# string: 100, 100B
# column: 10, 1kB
# 10M rows, 10e7

# basic solution
# O(N) filter it  -> 1sec
# string, int

# O(N * k) filter by k conditions

# O(1) insert

# ordered O(nlogn) -> 10sec, too slow
# ordered by multiple column?
# O(nlogn ^ k) for each column -> really really slow, 100s + with several columns

# faster solution

# ordered? keep it ordered
# O(N) return the ordered list with filter
# each column keep a list of ordered idx
# I will just return 

# age
# 15 30 20 15
# index
# 1 2 3 4
# age sorted index = 1, 4, 3, 2

# I just return rows  in age sorted index order

# name
# b a c d
# 1 2 3 4
# name sorted index 2 1 3 4

# insert becomes O(logN * K) for k columns

# memory is similar, with added indexes is 

# string: 100, 100B
# column: 10, 1kB
# + 1 index per column: 1.01kB negligible
# 10M rows, 10e7

# more complex if there is order by multiple columns
# when there is a tie order by the other column
# I should achieve O(N * k)

# age
# 15 30 20 15
# index
# 1 2 3 4
# age sorted index = 1, 4, 3, 2

# name
# b a c d
# 1 2 3 4
# name sorted index 2 1 3 4

# sort by age and then name

# 1, 4, 3, 2
# tie (1,4), tie (1,4), 3, 2
# would need Nlog(n) to sort the tie
# O(N) + Nlog(n) ^ k
# I can't really get  abetter solution than this


from collections import defaultdict
from bisect import bisect_left

class SortedList:
    def __init__(self, key_function):
        self.list = []
        self.key_function = key_function # lambda value: value.key

    def insert(self, value): # logN
        insert_ix = bisect_left(self.list, self.key_function(value), key=self.key_function)
        self.list.insert(insert_ix, value)

class Row:
    def __init__(self, row_id, columns, values):
        self.row_id = row_id
        self.data = {col: values[ix] for ix, col in enumerate(columns)}

        #{"id": 1, "name": "Alice", "age": 30},
        #{"id": 2, "name": "Bob", "age": 25},
        #{"id": 3, "name": "Charlie", "age": 30},
        #{"id": 4, "name": "Alice", "age": 25}

        # column_sorted_index['name'] = 1, 4, 2, 3

class InMemoryDatabase:
    def __init__(self, columns):
        self.columns: list[str] = columns
        self.rows: dict[int, Row] = {} 

    def insert(self, row_values): # 1, alice, 30
        row_id = row_values[0] # 1
        self.rows[row_id] = Row(row_id, self.columns, row_values)

    def delete(self, row_id): 
        del self.rows[row_id]

    def update(self, row_id, values):
        self.rows[row_id] = Row(row_id, self.columns, values)

    def query(self, where=None, order_by=None): 
        rows  = self.rows.values()
        if order_by:
            rows = sorted(rows, key=lambda row: list(row.data[col] for col in order_by))

        def where_filter(row, where):
            if not where: return True
            return all([row.data[k] == v for k,v in where.items()]) 
        rows = filter(lambda row: where_filter(row, where), rows) 
        return [row.data for row in rows]


def run_tests():
    db = InMemoryDatabase(["id", "name", "age"])
    db.insert([1, "Alice", 30])
    db.insert([2, "Bob", 25])
    db.insert([3, "Charlie", 30])
    db.insert([4, "Alice", 25])
    result1 = db.query()
    expected1 = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 3, "name": "Charlie", "age": 30},
        {"id": 4, "name": "Alice", "age": 25}
    ]
    assert result1 == expected1
    print("PASSED 1 TEST")
    result2 = db.query(where={"age": 30})
    expected2 = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 3, "name": "Charlie", "age": 30}
    ]
    assert result2 == expected2
    print("PASSED 1 TEST")
    result3 = db.query(where={"age": 30, "name": "Alice"})
    expected3 = [
        {"id": 1, "name": "Alice", "age": 30}
    ]
    print("PASSED 1 TEST")
    assert result3 == expected3
    result4 = db.query(where={"age": 30}, order_by=["name"])
    expected4 = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 3, "name": "Charlie", "age": 30}
    ]
    print("PASSED 1 TEST")
    assert result4 == expected4
    result5 = db.query(order_by=["age", "name"])
    expected5 = [
        {"id": 4, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 3, "name": "Charlie", "age": 30}
    ]
    assert result5 == expected5
    print("All test cases passed!")

def run_new_tests():
    # Create a new database with three columns.
    db = InMemoryDatabase(["id", "name", "age"])

    # Insert sample rows.
    db.insert([1, "Alice", 30])
    db.insert([2, "Bob", 25])
    db.insert([3, "Charlie", 30])
    db.insert([4, "Alice", 25])

    # ----------------------
    # Test 1: Deletion
    # ----------------------
    # Delete row with primary key 2 (Bob)
    db.delete(2)
    # After deletion, querying ordered by "id" should not include row with id 2.
    result_del = sorted(db.query(order_by=["id"]), key=lambda x: x["id"])
    expected_del = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 3, "name": "Charlie", "age": 30},
        {"id": 4, "name": "Alice", "age": 25}
    ]
    assert result_del == expected_del, f"Delete test failed: expected {expected_del}, got {result_del}"
    print("PASSED 1 TEST")

    # ----------------------
    # Test 2: Update
    # ----------------------
    # Update row with id 3: change age from 30 to 35.
    db.update(3, [3, "Charlie", 35])
    result_update = sorted(db.query(order_by=["id"]), key=lambda x: x["id"])
    expected_update = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 3, "name": "Charlie", "age": 35},
        {"id": 4, "name": "Alice", "age": 25}
    ]
    assert result_update == expected_update, f"Update test failed: expected {expected_update}, got {result_update}"
    print("PASSED 1 TEST")

    # ----------------------
    # Test 3: Range Query
    # ----------------------
    # Query for rows with age between 26 and 40 (inclusive).
    result_range = sorted(db.query_range("age", lower_bound=26, upper_bound=40), key=lambda x: x["id"])
    expected_range = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 3, "name": "Charlie", "age": 35}
    ]
    assert result_range == expected_range, f"Range query test failed: expected {expected_range}, got {result_range}"
    print("PASSED 1 TEST")

    print("All new test cases passed!")

if __name__ == "__main__":
    run_tests()
    run_new_tests()

