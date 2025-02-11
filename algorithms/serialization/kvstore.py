from collections import defaultdict

class KVStore:
    def __init__(self):
        """Initialize the in-memory store."""
        self.data = defaultdict(None)

    def set(self, key: str, value: str) -> None:
        """Store or update the key-value pair."""
        self.data[key] = value

    def get(self, key: str):
        """Return the value associated with the key, or None if not found."""
        if key in self.data.keys():
            return self.data[key]
        return None

    def delete(self, key: str) -> bool:
        """Delete the key-value pair and return True if successful, otherwise False."""
        if key in self.data.keys():
            del self.data[key]
            return True
        return None

    # {'a': '123\n'}
    def save(self, filename: str) -> None:
        """Serialize the in-memory store and save it to the specified file."""
        serialized_string = ""
        for k, v in self.data.items():
            serialized_string += f"{len(k)}key{k}{len(v)}value{v}" 
            #1keya4value123\n2key..
        with open(filename, "w") as f:
            f.write(serialized_string)

    def _deserialize(self, ix, splitword, serialized_string): #5, value
            #1keya45value123\n2key..
        splitword_index = serialized_string[ix:].index(splitword) + ix # 7
        value_length = int(serialized_string[ix:splitword_index]) # 5:7 = 45
        value_start = splitword_index+len(splitword) # 7 + 5 = 12
        value_end = value_start + value_length
        value = serialized_string[value_start:value_end] # 12, 12+45
        return value,value_end


    def load(self, filename: str) -> None:
        """Load key-value pairs from the specified file into the in-memory store."""
        with open(filename, "r") as f:
            serialized_string = f.read()
            ix = 0
            while ix < len(serialized_string):
                key, ix = self._deserialize(ix, "key", serialized_string)
                value, ix = self._deserialize(ix, "value", serialized_string)
                self.data[key] = value




# Some initial test cases for verifying your implementation
def main():
    # Test basic set and get
    store = KVStore()
    store.set("name", "Alice")
    store.set("greeting", "Hello\nWorld!")
    assert store.get("name") == "Alice", "Expected 'Alice'"
    assert store.get("greeting") == "Hello\nWorld!", "Value with newline did not match!"


    print("1 Test Passed")

    # Test update and deletion
    store.set("name", "Bob")
    assert store.get("name") == "Bob", "Update did not work correctly."
    assert store.delete("name") is True, "Deletion should return True when key exists."
    assert store.get("name") is None, "Deleted key should return None."

    # test mutiple digit keys, len(keys) > 10
    store.set("greeting"*100, "Hello\nWorld!"*100)
    store.set("!123...{}---==!@%!@#\t\n\p$}greeting"*100, "1")
    # test key value length is zero, not allowed

    print("2 Test Passed")
    # Test persistence: saving and loading from a file
    filename = "kvstore_data.txt"
    store.save(filename)

    # Create a new KVStore instance and load data from the file
    new_store = KVStore()
    new_store.load(filename)
    assert new_store.get("greeting") == "Hello\nWorld!", "Loaded value did not match expected."
    assert new_store.get("greeting"*100) == "Hello\nWorld!"*100, "Loaded value did not match expected."
    assert new_store.get("!123...{}---==!@%!@#\t\n\p$}greeting"*100) == "1"

    print("All tests passed!")


if __name__ == "__main__":
    main()

